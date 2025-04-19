import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class TradingEnv(gym.Env):
    """
    Entorno de trading para RL con soporte para múltiples timeframes y activos
    """
    
    def __init__(
        self,
        data_1d: Dict[str, pd.DataFrame],  # Datos diarios por activo
        data_1h: Dict[str, pd.DataFrame],  # Datos horarios por activo
        data_15m: Dict[str, pd.DataFrame], # Datos de 15 minutos por activo
        features: List[str],               # Lista de características (indicadores)
        window_size: int = 30,             # Tamaño de ventana para el estado
        commission: float = 0.001,         # Comisión por operación (0.1%)
        initial_balance: float = 10000,    # Balance inicial
        reward_window: int = 20,           # Ventana para calcular zonas calientes
        min_price_change: float = 0.05,    # Cambio mínimo significativo (5%)
    ):
        super(TradingEnv, self).__init__()
        
        self.data_1d = data_1d
        self.data_1h = data_1h
        self.data_15m = data_15m
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
        
        # Observación: [características * window_size] + [posición actual, balance]
        # + [identificador de activo (one-hot)]
        n_features = len(features)  # Características de los 3 timeframes pero seleccionadas correctamente
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
        """Alinea los datos de diferentes timeframes, tomando solo las características correctas de cada uno"""
        # Obtener datos base
        data_1h = self.data_1h[asset].copy()
        data_1d = self.data_1d[asset].copy()
        data_15m = self.data_15m[asset].copy()
        
        # Verificar que los índices son DatetimeIndex
        if not isinstance(data_1h.index, pd.DatetimeIndex):
            data_1h.index = pd.to_datetime(data_1h.index, utc=True)
        if not isinstance(data_1d.index, pd.DatetimeIndex):
            data_1d.index = pd.to_datetime(data_1d.index, utc=True)
        if not isinstance(data_15m.index, pd.DatetimeIndex):
            data_15m.index = pd.to_datetime(data_15m.index, utc=True)
        
        # Reindexar para tener todos los timeframes en cada punto horario
        data_1h_idx = data_1h.index
        
        # CORRECCIÓN: Solo usar OHLC del timeframe de 1 hora
        # Preparar datos horarios (incluidos OHLC)
        hourly_data = data_1h.copy()
        
        # Lista de indicadores diarios específicos
        daily_indicators = [
            'SMA20_slope', 'Market_Phase', 'Distance_52W_High', 'Distance_52W_Low', 
            'ADX_normalized'
        ]
        
        # Lista de indicadores de 15 minutos específicos
        min15_indicators = [
            'Momentum', 'RSI_divergence', 'Volume_acceleration', 'BB_squeeze'
        ]
        
        # Forwardfillear los indicadores diarios a cada hora
        daily_on_hourly = pd.DataFrame(index=data_1h_idx)
        
        # Iterar por las columnas del dataframe diario
        for col in data_1d.columns:
            # Solo incluir indicadores específicos del timeframe diario
            if any(indicator in col for indicator in daily_indicators):
                col_name = f'1d_{col}'
                daily_on_hourly[col_name] = np.nan
                
                # Para cada fecha en el índice diario
                for date in data_1d.index:
                    # CORRECCIÓN: Verificar que estamos usando objetos datetime
                    date_only = date.date() if hasattr(date, 'date') else pd.Timestamp(date).date()
                    
                    # Crear mascara para todos los índices horarios que corresponden a esta fecha
                    mask = pd.Series(False, index=data_1h.index)
                    for i, idx in enumerate(data_1h.index):
                        idx_date = idx.date() if hasattr(idx, 'date') else pd.Timestamp(idx).date()
                        if idx_date == date_only:
                            mask.iloc[i] = True
                            
                    # Asignar el valor del indicador diario a todas las horas de ese día
                    if mask.any():
                        daily_on_hourly.loc[mask, col_name] = data_1d.loc[date, col]
        
        # Rellenar valores NaN con forward fill
        daily_on_hourly = daily_on_hourly.ffill()
        
        # Agregar los indicadores de 15 minutos (último valor de cada hora)
        min15_on_hourly = pd.DataFrame(index=data_1h_idx)
        
        for col in data_15m.columns:
            # Solo incluir indicadores específicos del timeframe de 15 minutos
            if any(indicator in col for indicator in min15_indicators):
                col_name = f'15m_{col}'
                min15_on_hourly[col_name] = np.nan
                
                # Para cada hora en el índice horario
                for hour in data_1h_idx:
                    # Convertir a timestamp si es necesario
                    hour_ts = pd.Timestamp(hour)
                    
                    # Tomar los datos de 15m hasta la hora actual
                    mask = data_15m.index <= hour_ts
                    if mask.any():
                        last_15m_data = data_15m[mask].iloc[-1]
                        min15_on_hourly.loc[hour_ts, col_name] = last_15m_data[col]
        
        # Rellenar valores NaN
        min15_on_hourly = min15_on_hourly.ffill()
        
        # Obtener características cross-timeframe si están presentes
        cross_tf_cols = [col for col in data_1h.columns if 'cross_tf_' in col]
        cross_tf_data = hourly_data[cross_tf_cols].copy() if cross_tf_cols else pd.DataFrame(index=data_1h_idx)
        
        # Unir todos los datos
        aligned_data = pd.concat([hourly_data, daily_on_hourly, min15_on_hourly, cross_tf_data], axis=1)
        
        # Manejar valores NaN
        # Para indicadores de tendencia, usar forward fill y luego backward fill
        trend_cols = [col for col in aligned_data.columns if any(ind in col.lower() for ind in 
                                                        ['ema', 'sma', 'macd', 'market_phase'])]
        for col in trend_cols:
            aligned_data[col] = aligned_data[col].ffill().bfill()
            
        # Para otros indicadores, usar 0 como valor neutral
        aligned_data = aligned_data.fillna(0)
        
        return aligned_data

    def _calculate_hot_zones(self, asset):
        """Calcula zonas óptimas de compra/venta basadas en cambios significativos"""
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
            
            price_change = (max_price - min_price) / min_price
            
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
                            # CORRECCIÓN: Usar .loc para evitar chained assignment
                            hot_zones.loc[current_idx, 'buy_zone'] = 1.0
                        else:
                            # CORRECCIÓN: Usar .loc para evitar chained assignment
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
                            # CORRECCIÓN: Usar .loc para evitar chained assignment
                            hot_zones.loc[current_idx, 'sell_zone'] = 1.0
                        else:
                            # CORRECCIÓN: Usar .loc para evitar chained assignment
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
        # Obtener datos de mercado
        market_data = self.data_aligned.iloc[self.current_step - self.window_size:self.current_step].values
        
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

# Ejemplo de uso:
# if __name__ == "__main__":
#     # Datos de ejemplo (en la práctica, usar datos reales de IBKR)
#     import yfinance as yf
    
#     # Descargar datos
#     data_1d = {}
#     data_1h = {}
#     data_15m = {}
    
#     symbols = ['AAPL', 'MSFT', 'GOOGL']
#     for symbol in symbols:
#         data_1d[symbol] = yf.download(symbol, period='1y', interval='1d')
#         data_1h[symbol] = yf.download(symbol, period='60d', interval='1h')
#         data_15m[symbol] = yf.download(symbol, period='30d', interval='15m')
    
#     # Crear entorno
#     env = TradingEnv(
#         data_1d=data_1d,
#         data_1h=data_1h,
#         data_15m=data_15m,
#         features=['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal',
#                  'MACD_histogram', 'EMA8', 'EMA21', 'EMA50', 'BB_middle', 'BB_upper',
#                  'BB_lower', 'BB_width', 'ATR', 'Stoch_K', 'Stoch_D', 'OBV', 'ADX',
#                  '1d_SMA20_slope', '1d_Market_Phase', '1d_Distance_52W_High', '1d_Distance_52W_Low',
#                  '1d_ADX_normalized', '15m_Momentum', '15m_RSI_divergence',
#                  '15m_Volume_acceleration', '15m_BB_squeeze', 'cross_tf_trend_alignment',
#                  'cross_tf_multi_tf_momentum', 'cross_tf_inflection_point',
#                  'cross_tf_multi_tf_strength', 'cross_tf_trend_change_index',
#                  'cross_tf_volatility_ratio'],
#         window_size=30,
#         commission=0.001,
#         initial_balance=10000,
#         reward_window=20,
#         min_price_change=0.05
#     )
    
#     # Resetear entorno
#     obs, info = env.reset()
#     print(f"Entorno iniciado con activo: {info['current_asset']}")
    
#     # Ejecutar algunas acciones aleatorias
#     for i in range(100):
#         action = np.random.randint(0, 3)
#         obs, reward, done, truncated, info = env.step(action)
#         print(f"Paso {i}, Acción: {action}, Recompensa: {reward:.4f}, Balance: {info['balance']:.2f}")
        
#         if done:
#             print("Episodio terminado")
#             break
    
#     # Renderizar resultados
#     env.render()