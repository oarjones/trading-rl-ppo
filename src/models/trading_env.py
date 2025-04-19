import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class TradingEnv(gym.Env):
    """
    Entorno de trading para RL usando solo el timeframe de 1H
    """
    
    def __init__(
        self,
        data_1d: Dict[str, pd.DataFrame],  # Datos diarios por activo (los mantenemos pero no los usamos)
        data_1h: Dict[str, pd.DataFrame],  # Datos horarios por activo (los usamos)
        data_15m: Dict[str, pd.DataFrame], # Datos de 15 minutos por activo (los mantenemos pero no los usamos)
        features: List[str],               # Lista de características (indicadores)
        window_size: int = 30,             # Tamaño de ventana para el estado
        commission: float = 0.001,         # Comisión por operación (0.1%)
        initial_balance: float = 10000,    # Balance inicial
        reward_window: int = 20,           # Ventana para calcular zonas calientes
        min_price_change: float = 0.05,    # Cambio mínimo significativo (5%)
    ):
        super(TradingEnv, self).__init__()
        
        self.data_1d = data_1d  # Mantener para futuro uso
        self.data_1h = data_1h  # Usamos solo esto por ahora
        self.data_15m = data_15m  # Mantener para futuro uso
        self.assets = list(data_1h.keys())
        self.features = features
        self.window_size = window_size
        self.commission = commission
        self.initial_balance = initial_balance
        self.reward_window = reward_window
        self.min_price_change = min_price_change
        
        # Definir espacios de acción y observación
        # Acciones: 0=mantener, 1=comprar, 2=vender
        self.action_space = spaces.Discrete(3)
        
        # Filtrar para usar solo features de 1H (excluir los que comienzan con 1d_, 15m_, cross_tf_)
        self.hourly_features = [f for f in features if not (f.startswith('1d_') or f.startswith('15m_') or f.startswith('cross_tf_'))]
        n_features = len(self.hourly_features)
        n_assets = len(self.assets)
        
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.window_size, n_features), 
                dtype=np.float32
            ),
            'account_state': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(3,),  # posición, balance, beneficio
                dtype=np.float32
            ),
            'asset_id': spaces.Box(
                low=0, high=1, 
                shape=(n_assets,),  # one-hot para identificar el activo
                dtype=np.float32
            )
        })
        
        # Variables de estado
        self.current_step = 0
        self.current_asset = None
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0
        self.profit = 0
        self.history = []
        self.data_aligned = None
        self.hot_zones = None
    
    def _align_timeframes(self, asset):
        """Alinea los datos, usando solo el timeframe 1H"""
        # Obtener datos base
        hourly_data = self.data_1h[asset].copy()
        
        # Verificar que el índice es DatetimeIndex
        if not isinstance(hourly_data.index, pd.DatetimeIndex):
            hourly_data.index = pd.to_datetime(hourly_data.index, utc=True)
        
        # Normalizar el precio de cierre para features (manteniendo original para cálculos)
        if 'Close' in hourly_data.columns:
            first_close = hourly_data['Close'].iloc[0]
            if first_close != 0:  # Evitar división por cero
                hourly_data['Close_norm'] = hourly_data['Close'] / first_close
            else:
                hourly_data['Close_norm'] = hourly_data['Close']
        
        # También normalizar otros precios OHLC si existen
        for col in ['Open', 'High', 'Low']:
            if col in hourly_data.columns:
                first_value = hourly_data[col].iloc[0]
                if first_value != 0:  # Evitar división por cero
                    hourly_data[f'{col}_norm'] = hourly_data[col] / first_value
                else:
                    hourly_data[f'{col}_norm'] = hourly_data[col]
        
        # Manejar valores NaN
        # Para indicadores de tendencia, usar forward fill y luego backward fill
        trend_cols = [col for col in hourly_data.columns if any(ind in col.lower() for ind in 
                                                    ['ema', 'sma', 'macd', 'bb_', 'adx', 'obv', 'market_phase'])]
        for col in trend_cols:
            if col in hourly_data.columns:
                hourly_data[col] = hourly_data[col].ffill().bfill()
        
        # Para otros indicadores, usar 0 como valor neutral
        hourly_data = hourly_data.fillna(0)
        
        return hourly_data

    def _calculate_hot_zones(self, asset):
        """Calcula zonas óptimas de compra/venta basadas en cambios significativos"""
        # Usar el precio de cierre original, no el normalizado
        data = self.data_1h[asset]['Close'].copy()
        hot_zones = pd.DataFrame(index=data.index)
        
        # Inicializar columnas para zonas calientes
        hot_zones['buy_zone'] = 0.0  # 1.0 para zonas óptimas de compra
        hot_zones['sell_zone'] = 0.0  # 1.0 para zonas óptimas de venta
        
        for i in range(len(data) - self.reward_window):
            # Mirar hacia adelante para la ventana de recompensa
            window = data.iloc[i:i+self.reward_window]
            
            # Calcular el cambio porcentual máximo en la ventana
            min_price = window.min()
            max_price = window.max()
            
            min_idx = window.idxmin()
            max_idx = window.idxmax()
            
            price_change = (max_price - min_price) / min_price if min_price > 0 else 0
            
            # Si el cambio es significativo (≥5%)
            if price_change >= self.min_price_change:
                # Si el mínimo viene antes que el máximo, es una zona de compra
                if min_idx < max_idx:
                    # Crear una zona caliente alrededor del mínimo
                    # Peso máximo en el mínimo, decreciente a medida que nos alejamos
                    idx_min = data.index.get_loc(min_idx)
                    window_size = 5  # Tamaño de la zona caliente
                    
                    for j in range(max(0, idx_min-window_size), min(len(data), idx_min+window_size+1)):
                        distance = abs(j - idx_min)
                        current_idx = data.index[j]
                        
                        if distance == 0:
                            hot_zones.loc[current_idx, 'buy_zone'] = 1.0
                        else:
                            current_value = hot_zones.loc[current_idx, 'buy_zone']
                            hot_zones.loc[current_idx, 'buy_zone'] = max(
                                current_value,
                                1.0 - (distance / window_size)
                            )
                
                # Si el máximo viene antes que el mínimo, es una zona de venta
                elif max_idx < min_idx:
                    # Crear una zona caliente alrededor del máximo
                    idx_max = data.index.get_loc(max_idx)
                    window_size = 5
                    
                    for j in range(max(0, idx_max-window_size), min(len(data), idx_max+window_size+1)):
                        distance = abs(j - idx_max)
                        current_idx = data.index[j]
                        
                        if distance == 0:
                            hot_zones.loc[current_idx, 'sell_zone'] = 1.0
                        else:
                            current_value = hot_zones.loc[current_idx, 'sell_zone']
                            hot_zones.loc[current_idx, 'sell_zone'] = max(
                                current_value,
                                1.0 - (distance / window_size)
                            )
        
        return hot_zones
    
    def reset(self, seed=None, options=None):
        """Reinicia el entorno y selecciona un activo aleatorio"""
        super().reset(seed=seed)
        
        # Seleccionar un activo aleatorio
        self.current_asset = np.random.choice(self.assets)
        
        # Alinear timeframes para el activo actual
        self.data_aligned = self._align_timeframes(self.current_asset)
        
        # Calcular zonas calientes
        self.hot_zones = self._calculate_hot_zones(self.current_asset)
        
        # Reiniciar estado
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.profit = 0
        self.history = []
        
        # Obtener observación inicial
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self):
        """Devuelve la observación actual del entorno"""
        # Obtener datos de mercado (usando valores normalizados para los precios)
        market_data = self.data_aligned.iloc[self.current_step - self.window_size:self.current_step]
        
        # Crear lista de columnas a usar, reemplazando columnas de precio por sus versiones normalizadas
        use_columns = []
        for col in self.hourly_features:
            if col in ['Open', 'High', 'Low', 'Close']:
                normalized_col = f'{col}_norm'
                if normalized_col in market_data.columns:
                    use_columns.append(normalized_col)
                else:
                    use_columns.append(col)  # Fallback si no existe versión normalizada
            else:
                use_columns.append(col)
                
        # Seleccionar solo las columnas relevantes
        market_features = [col for col in use_columns if col in market_data.columns]
        market_data = market_data[market_features].values
        
        # Preparar one-hot encoding para el activo actual
        asset_id = np.zeros(len(self.assets))
        asset_idx = self.assets.index(self.current_asset)
        asset_id[asset_idx] = 1
        
        # Estado de la cuenta
        current_price = self.data_1h[self.current_asset]['Close'].iloc[self.current_step]
        account_state = np.array([
            self.position,  # 1 si tiene posición larga, 0 si no tiene, -1 si tiene posición corta
            self.balance / self.initial_balance,  # Balance normalizado
            self.profit / self.initial_balance if self.position != 0 else 0  # Beneficio normalizado
        ], dtype=np.float32)
        
        return {
            'market_data': market_data.astype(np.float32),
            'account_state': account_state,
            'asset_id': asset_id.astype(np.float32)
        }
    
    def _get_info(self):
        """Devuelve información adicional del entorno"""
        return {
            'current_asset': self.current_asset,
            'current_step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'timestamp': self.data_aligned.index[self.current_step]
        }
    
    def _calculate_reward(self, action):
        """Calcula la recompensa basada en la acción tomada y las zonas calientes"""
        current_price = self.data_1h[self.current_asset]['Close'].iloc[self.current_step]
        timestamp = self.data_aligned.index[self.current_step]
        
        # Obtener valores de zonas calientes
        buy_zone_value = self.hot_zones.loc[timestamp, 'buy_zone'] if timestamp in self.hot_zones.index else 0
        sell_zone_value = self.hot_zones.loc[timestamp, 'sell_zone'] if timestamp in self.hot_zones.index else 0
        
        reward = 0
        
        # Recompensa basada en zonas calientes
        if action == 1:  # Comprar
            if self.position == 0:  # Si no tenemos posición
                # Recompensa por comprar en zona caliente de compra
                reward = buy_zone_value * 2 - 0.5  # Escala de -0.5 a 1.5
                # Penalización por comprar en zona caliente de venta
                reward -= sell_zone_value * 0.5
            else:
                # Penalización por operación innecesaria
                reward = -0.5
                
        elif action == 2:  # Vender
            if self.position == 1:  # Si tenemos posición larga
                # Recompensa por vender en zona caliente de venta
                reward = sell_zone_value * 2 - 0.5  # Escala de -0.5 a 1.5
                
                # Recompensa adicional por beneficio realizado
                if current_price > self.entry_price:
                    profit_pct = (current_price - self.entry_price) / self.entry_price
                    # Escalar según la magnitud del movimiento
                    if profit_pct > self.min_price_change:
                        reward += profit_pct * 5
                    else:
                        reward += profit_pct * 2
            else:
                # Penalización por operación innecesaria
                reward = -0.5
                
        elif action == 0:  # Mantener
            # Pequeña recompensa por mantener si no estamos en ninguna zona
            if buy_zone_value < 0.3 and sell_zone_value < 0.3:
                reward = 0.1
            # Pequeña penalización por mantener si estamos en una zona clara de acción
            elif (buy_zone_value > 0.7 and self.position == 0) or (sell_zone_value > 0.7 and self.position == 1):
                reward = -0.3
        
        return reward
    
    def step(self, action):
        """Ejecuta una acción en el entorno"""
        # Obtener precio actual
        current_price = self.data_1h[self.current_asset]['Close'].iloc[self.current_step]
        
        # Variable para seguir si hubo operación
        trade_executed = False
        
        # Ejecutar acción
        if action == 1:  # Comprar
            if self.position == 0:  # Si no tenemos posición
                # Calcular coste de la operación con comisión
                cost = current_price * (1 + self.commission)
                self.balance -= cost
                self.position = 1
                self.entry_price = current_price
                trade_executed = True
                
        elif action == 2:  # Vender
            if self.position == 1:  # Si tenemos posición larga
                # Calcular beneficio de la operación con comisión
                revenue = current_price * (1 - self.commission)
                self.balance += revenue
                self.profit = revenue - self.entry_price
                self.position = 0
                self.entry_price = 0
                trade_executed = True
        
        # Registrar historia
        self.history.append({
            'step': self.current_step,
            'timestamp': self.data_aligned.index[self.current_step],
            'asset': self.current_asset,
            'price': current_price,
            'action': action,
            'position': self.position,
            'balance': self.balance,
            'profit': self.profit
        })
        
        # Avanzar al siguiente paso
        self.current_step += 1
        
        # Calcular recompensa
        reward = self._calculate_reward(action)
        
        # Verificar si el episodio ha terminado (final de datos o bancarrota)
        done = self.current_step >= len(self.data_aligned) - 1 or self.balance <= 0
        
        # Obtener nueva observación
        observation = self._get_observation()
        info = self._get_info()
        
        # Añadir información sobre la operación
        info['trade_executed'] = trade_executed
        info['reward'] = reward
        
        return observation, reward, done, False, info
    
    def render(self):
        """Visualiza el entorno actual"""
        if not self.history:
            return
        
        # Obtener datos para visualización
        data = self.data_1h[self.current_asset].copy()
        start_idx = max(0, self.current_step - 100)
        end_idx = self.current_step
        
        plot_data = data.iloc[start_idx:end_idx]
        
        # Obtener operaciones en el rango visible
        trades = [h for h in self.history if h['step'] >= start_idx and h['step'] <= end_idx]
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Graficar precio
        ax.plot(plot_data.index, plot_data['Close'], label='Precio', color='blue')
        
        # Marcar operaciones
        buy_points = [t['price'] for t in trades if t['action'] == 1 and t['position'] == 1]
        buy_idx = [data.index[t['step']] for t in trades if t['action'] == 1 and t['position'] == 1]
        
        sell_points = [t['price'] for t in trades if t['action'] == 2 and t['position'] == 0]
        sell_idx = [data.index[t['step']] for t in trades if t['action'] == 2 and t['position'] == 0]
        
        ax.scatter(buy_idx, buy_points, color='green', marker='^', s=100, label='Compra')
        ax.scatter(sell_idx, sell_points, color='red', marker='v', s=100, label='Venta')
        
        # Añadir información adicional
        ax.set_title(f'Trading de {self.current_asset} - Balance: {self.balance:.2f}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Precio')
        ax.legend()
        
        plt.tight_layout()
        plt.show()