import os
import json
from ib_insync import OrderStatus
import pandas as pd
import numpy as np
import torch
import datetime
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt

# Importaciones de QuantConnect
import clr
import sys

# Modificar estas rutas a la ubicación de tu instalación de LEAN
LEAN_PATH = "Lean/Launcher/bin/Debug"
sys.path.append(LEAN_PATH)

# Cargar las DLLs de QuantConnect
clr.AddReference("QuantConnect.Common")
clr.AddReference("QuantConnect.Algorithm")
clr.AddReference("QuantConnect.Algorithm.Framework")
clr.AddReference("QuantConnect.Indicators")

# Importar clases de QuantConnect
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Algorithm.Framework.Alphas import *
from QuantConnect.Algorithm.Framework.Portfolio import *
from QuantConnect.Algorithm.Framework.Risk import *
from QuantConnect.Indicators import *
from QuantConnect.Data.Market import TradeBar
from QuantConnect.Brokerages import BrokerageName  # Agregar esta línea

# Importar clases propias
from src.models.ppo_agent import PPOAgent, ActorNetwork, CriticNetwork
from src.models.feature_engineering import FeatureEngineering

class RLModelIntegrator:
    """
    Clase para integrar un modelo de RL entrenado con QuantConnect LEAN
    
    Esta clase carga un modelo PPO entrenado y proporciona métodos para
    evaluar el estado actual del mercado y tomar decisiones de trading.
    """
    
    def __init__(self, 
                model_dir: str,
                actor_file: str = "actor_final.pth",
                critic_file: str = "critic_final.pth",
                config_file: str = "config.json"):
        """
        Inicializa el integrador cargando el modelo y la configuración
        
        Args:
            model_dir: Directorio donde se encuentra el modelo guardado
            actor_file: Nombre del archivo del modelo del actor
            critic_file: Nombre del archivo del modelo del crítico
            config_file: Nombre del archivo de configuración
        """
        self.model_dir = model_dir
        self.actor_file = os.path.join(model_dir, actor_file)
        self.critic_file = os.path.join(model_dir, critic_file)
        self.config_file = os.path.join(model_dir, config_file)
        
        # Cargar configuración
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
            
        # Configurar dispositivo para PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Variables para almacenar el estado del mercado
        self.window_size = self.config.get("window_size", 30)
        self.features = None  # Se establecerá al cargar el modelo
        self.market_data_buffer = {}  # Buffer para almacenar datos históricos por símbolo
        self.current_position = {}  # Posición actual por símbolo
        self.entry_price = {}  # Precio de entrada por símbolo
        
        # Inicializar ingeniería de características
        self.feature_engineering = FeatureEngineering()
        
        # Cargar modelo
        self._load_model()
        
    def _load_model(self):
        """Carga el modelo de RL pre-entrenado"""
        try:
            # Obtener número de características y activos de la configuración o inferirlo
            if "n_features" in self.config and "n_assets" in self.config:
                self.n_features = self.config["n_features"]
                self.n_assets = self.config["n_assets"]
            else:
                # Valores predeterminados razonables
                self.n_features = 30  # Número típico de características
                self.n_assets = 5     # Número típico de activos en entrenamiento
                
            # Reconstruir arquitectura del actor
            input_dims = (self.window_size, self.n_features)
            n_actions = 3  # hold, buy, sell
            
            # Obtener arquitectura de capas ocultas
            actor_hidden_dims = self.config.get("actor_hidden_dims", [512, 256, 128])
            critic_hidden_dims = self.config.get("critic_hidden_dims", [512, 256, 128])
            
            # Crear modelos
            self.actor = ActorNetwork(
                input_dims=input_dims,
                n_assets=self.n_assets,
                n_actions=n_actions,
                hidden_dims=actor_hidden_dims
            ).to(self.device)
            
            self.critic = CriticNetwork(
                input_dims=input_dims,
                n_assets=self.n_assets,
                hidden_dims=critic_hidden_dims
            ).to(self.device)
            
            # Cargar pesos pre-entrenados
            self.actor.load_state_dict(torch.load(self.actor_file, map_location=self.device))
            self.critic.load_state_dict(torch.load(self.critic_file, map_location=self.device))
            
            # Establecer en modo evaluación
            self.actor.eval()
            self.critic.eval()
            
            print(f"Modelo cargado correctamente desde {self.model_dir}")
            
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            raise
            
    def initialize_symbol(self, symbol: str, asset_id: int):
        """
        Inicializa un nuevo símbolo para ser monitoreado por el modelo
        
        Args:
            symbol: Símbolo del activo
            asset_id: ID del activo en el modelo (índice en el one-hot encoding)
        """
        self.market_data_buffer[symbol] = []
        self.current_position[symbol] = 0  # 0: sin posición, 1: long
        self.entry_price[symbol] = 0
        
    def update_market_data(self, symbol: str, bar: TradeBar, features: Dict[str, float]):
        """
        Actualiza los datos del mercado con una nueva barra y características calculadas
        
        Args:
            symbol: Símbolo del activo
            bar: Barra actual del mercado
            features: Características calculadas para esta barra
        """
        if symbol not in self.market_data_buffer:
            self.initialize_symbol(symbol, len(self.market_data_buffer))
            
        # Combinar datos OHLCV con características calculadas
        bar_data = {
            'open': bar.Open,
            'high': bar.High,
            'low': bar.Low,
            'close': bar.Close,
            'volume': bar.Volume,
            'datetime': bar.Time
        }
        
        # Agregar características calculadas
        bar_data.update(features)
        
        # Agregar al buffer
        self.market_data_buffer[symbol].append(bar_data)
        
        # Mantener solo las últimas window_size barras
        if len(self.market_data_buffer[symbol]) > self.window_size:
            self.market_data_buffer[symbol].pop(0)
            
    def _prepare_observation(self, symbol: str, asset_id: int) -> Dict[str, np.ndarray]:
        """
        Prepara una observación para el modelo a partir de los datos actuales
        
        Args:
            symbol: Símbolo del activo
            asset_id: ID del activo
            
        Returns:
            Observación en el formato esperado por el modelo
        """
        # Verificar si tenemos suficientes datos
        if len(self.market_data_buffer[symbol]) < self.window_size:
            return None
            
        # Extraer características en un tensor
        feature_keys = list(self.market_data_buffer[symbol][0].keys())
        # Eliminar 'datetime' y otras no numéricas
        feature_keys = [k for k in feature_keys if k not in ['datetime']]
        
        # Crear matriz de características
        market_data = np.zeros((self.window_size, len(feature_keys)))
        
        for i, bar_data in enumerate(self.market_data_buffer[symbol]):
            for j, key in enumerate(feature_keys):
                market_data[i, j] = bar_data[key]
                
        # Normalizar datos (importante para redes neuronales)
        # Usar normalización simple: (x - mean) / std
        market_data = (market_data - np.mean(market_data, axis=0)) / (np.std(market_data, axis=0) + 1e-8)
        
        # Crear one-hot encoding para el activo
        asset_one_hot = np.zeros(self.n_assets)
        asset_one_hot[asset_id] = 1
        
        # Estado de la cuenta para este activo
        account_state = np.array([
            self.current_position[symbol],  # 0 o 1
            1.0,  # Balance normalizado (no usado en predicción)
            0.0   # Beneficio (no usado en predicción)
        ])
        
        # Crear observación completa
        observation = {
            'market_data': market_data.astype(np.float32),
            'asset_id': asset_one_hot.astype(np.float32),
            'account_state': account_state.astype(np.float32)
        }
        
        return observation
        
    def predict_action(self, symbol: str, asset_id: int) -> Tuple[int, float]:
        """
        Predice la mejor acción para un símbolo dado
        
        Args:
            symbol: Símbolo del activo
            asset_id: ID del activo
            
        Returns:
            Tupla (acción, probabilidad)
            Acción: 0=hold, 1=buy, 2=sell
        """
        # Preparar observación
        observation = self._prepare_observation(symbol, asset_id)
        
        if observation is None:
            # No tenemos suficientes datos, mantener posición actual
            return 0, 1.0
            
        with torch.no_grad():
            # Convertir a tensores
            market_data = torch.tensor(observation['market_data'], dtype=torch.float32).unsqueeze(0).to(self.device)
            asset_id_tensor = torch.tensor(observation['asset_id'], dtype=torch.float32).unsqueeze(0).to(self.device)
            account_state = torch.tensor(observation['account_state'], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Obtener distribución de acciones
            action_probs = self.actor(market_data, asset_id_tensor, account_state)
            
            # Seleccionar la acción con mayor probabilidad
            action = torch.argmax(action_probs, dim=1).item()
            prob = action_probs[0, action].item()
            
            return action, prob
            
    def update_position(self, symbol: str, action: int, current_price: float):
        """
        Actualiza el estado interno de la posición
        
        Args:
            symbol: Símbolo del activo
            action: Acción tomada (0=hold, 1=buy, 2=sell)
            current_price: Precio actual del activo
        """
        if action == 1 and self.current_position[symbol] == 0:  # Comprar
            self.current_position[symbol] = 1
            self.entry_price[symbol] = current_price
        elif action == 2 and self.current_position[symbol] == 1:  # Vender
            self.current_position[symbol] = 0
            self.entry_price[symbol] = 0
            
    def get_trading_signal(self, symbol: str, asset_id: int, current_price: float) -> Tuple[int, float]:
        """
        Obtiene una señal de trading basada en el modelo RL
        
        Args:
            symbol: Símbolo del activo
            asset_id: ID del activo
            current_price: Precio actual del activo
            
        Returns:
            Tupla (señal, confianza)
            Señal: 0=no hacer nada, 1=comprar, -1=vender
        """
        # Obtener acción del modelo
        action, confidence = self.predict_action(symbol, asset_id)
        
        # Convertir acción a señal de trading
        signal = 0
        
        if action == 1:  # Comprar
            signal = 1
        elif action == 2:  # Vender
            signal = -1
            
        # Actualizar posición interna
        self.update_position(symbol, action, current_price)
        
        return signal, confidence


class RLTradingAlgorithm(QCAlgorithm):
    """
    Algoritmo de trading usando un modelo de RL pre-entrenado
    
    Esta clase implementa un algoritmo de trading en QuantConnect LEAN
    que utiliza un modelo de RL para tomar decisiones de trading.
    """
    
    def Initialize(self):
        """Inicializa el algoritmo"""
        # Configurar parámetros del algoritmo
        self.SetStartDate(2023, 1, 1)  # Fecha de inicio
        self.SetEndDate(2023, 12, 31)  # Fecha de fin
        self.SetCash(100000)          # Capital inicial
        
        # Establecer comisiones y deslizamiento realistas
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        
        # Configurar universo de activos
        self.symbols = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]
        self.symbol_data = {}
        
        for idx, symbol_str in enumerate(self.symbols):
            symbol = self.AddEquity(symbol_str, Resolution.Hour).Symbol
            self.symbol_data[symbol] = {
                "asset_id": idx,  # ID para el modelo
                "indicators": {}  # Almacenará los indicadores
            }
            
        # Indicadores a calcular por activo
        for symbol in self.symbol_data.keys():
            indicators = self.symbol_data[symbol]["indicators"]
            
            # Agregar indicadores
            indicators["RSI"] = self.RSI(symbol, 14)
            indicators["EMA8"] = self.EMA(symbol, 8)
            indicators["EMA21"] = self.EMA(symbol, 21)
            indicators["EMA50"] = self.EMA(symbol, 50)
            indicators["BB"] = self.BB(symbol, 20, 2)
            indicators["MACD"] = self.MACD(symbol, 12, 26, 9)
            indicators["ATR"] = self.ATR(symbol, 14)
            indicators["Stoch"] = self.Stochastic(symbol, 14, 3, 3)
            indicators["ADX"] = self.ADX(symbol, 14)
            
        # Ventana para almacenar datos históricos para características cross-timeframe
        self.daily_window = {}
        self.hourly_window = {}
        for symbol in self.symbol_data.keys():
            self.daily_window[symbol] = self.CreateRollingWindow(TradeBar, 30)  # 30 días
            self.hourly_window[symbol] = self.CreateRollingWindow(TradeBar, 24*5)  # 5 días
            
        # Inicializar la clase FeatureEngineering
        self.feature_engineering = FeatureEngineering()
        
        # Ruta al modelo entrenado
        model_path = self.ObjectStore.GetFilePath("rl_model")
        
        # Verificar si el modelo existe, si no, usar un modelo por defecto
        if model_path is None or not os.path.exists(model_path):
            self.Debug("Modelo no encontrado en ObjectStore, usando path por defecto")
            model_path = "models/best_model"
            
        # Cargar el integrador del modelo RL
        self.rl_integrator = RLModelIntegrator(
            model_dir=model_path,
            actor_file="actor_final.pth",
            critic_file="critic_final.pth",
            config_file="config.json"
        )
        
        # Inicializar integradores para cada símbolo
        for symbol, data in self.symbol_data.items():
            self.rl_integrator.initialize_symbol(symbol.Value, data["asset_id"])
            
        # Programar la función de trading para ejecutarse cada hora
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(datetime.timedelta(hours=1)), self.TradeBasedOnModel)
        
    def OnData(self, data):
        """
        Evento llamado cuando llegan nuevos datos
        
        Args:
            data: Slice con los nuevos datos
        """
        # Actualizar ventanas de datos históricos
        for symbol in self.symbol_data.keys():
            if data.ContainsKey(symbol) and data[symbol] is not None:
                # Actualizar ventana horaria
                self.hourly_window[symbol].Add(data[symbol])
                
                # Solo agregar datos diarios una vez al día
                if data[symbol].Time.hour == 16:  # Cierre del mercado
                    self.daily_window[symbol].Add(data[symbol])
                    
    def CalculateFeatures(self, symbol):
        """
        Calcula todas las características necesarias para un símbolo
        
        Args:
            symbol: Símbolo para calcular características
            
        Returns:
            Diccionario con todas las características calculadas
        """
        indicators = self.symbol_data[symbol]["indicators"]
        features = {}
        
        # Indicadores básicos si están listos
        if indicators["RSI"].IsReady:
            features["RSI"] = float(indicators["RSI"].Current.Value)
            
        if indicators["EMA8"].IsReady:
            features["EMA8"] = float(indicators["EMA8"].Current.Value)
            
        if indicators["EMA21"].IsReady:
            features["EMA21"] = float(indicators["EMA21"].Current.Value)
            
        if indicators["EMA50"].IsReady:
            features["EMA50"] = float(indicators["EMA50"].Current.Value)
            
        if indicators["BB"].IsReady:
            features["BB_middle"] = float(indicators["BB"].MiddleBand.Current.Value)
            features["BB_upper"] = float(indicators["BB"].UpperBand.Current.Value)
            features["BB_lower"] = float(indicators["BB"].LowerBand.Current.Value)
            # Calcular BB width
            features["BB_width"] = (features["BB_upper"] - features["BB_lower"]) / features["BB_middle"]
            # Calcular %B (posición dentro de las bandas)
            current_price = self.Securities[symbol].Price
            features["BB_pct_b"] = (current_price - features["BB_lower"]) / (features["BB_upper"] - features["BB_lower"])
            
        if indicators["MACD"].IsReady:
            features["MACD"] = float(indicators["MACD"].Current.Value)
            features["MACD_signal"] = float(indicators["MACD"].Signal.Current.Value)
            features["MACD_histogram"] = features["MACD"] - features["MACD_signal"]
            
        if indicators["ATR"].IsReady:
            features["ATR"] = float(indicators["ATR"].Current.Value)
            
        if indicators["Stoch"].IsReady:
            features["Stoch_K"] = float(indicators["Stoch"].StochK.Current.Value)
            features["Stoch_D"] = float(indicators["Stoch"].StochD.Current.Value)
            
        if indicators["ADX"].IsReady:
            features["ADX"] = float(indicators["ADX"].Current.Value)
            
        # Calcular características adicionales basadas en ventanas de datos
        # Esto simula las características cross-timeframe
        
        # Verificar si tenemos suficientes datos
        if self.hourly_window[symbol].IsReady and self.daily_window[symbol].IsReady:
            # Convertir datos de ventanas a pandas DataFrames para cálculos más complejos
            hourly_bars = [bar for bar in self.hourly_window[symbol]]
            daily_bars = [bar for bar in self.daily_window[symbol]]
            
            # Crear DataFrames para diario y horario
            df_1h = pd.DataFrame({
                'open': [bar.Open for bar in hourly_bars],
                'high': [bar.High for bar in hourly_bars],
                'low': [bar.Low for bar in hourly_bars],
                'close': [bar.Close for bar in hourly_bars],
                'volume': [bar.Volume for bar in hourly_bars],
                'datetime': [bar.Time for bar in hourly_bars]
            }).set_index('datetime')
            
            df_1d = pd.DataFrame({
                'open': [bar.Open for bar in daily_bars],
                'high': [bar.High for bar in daily_bars],
                'low': [bar.Low for bar in daily_bars],
                'close': [bar.Close for bar in daily_bars],
                'volume': [bar.Volume for bar in daily_bars],
                'datetime': [bar.Time for bar in daily_bars]
            }).set_index('datetime')
            
            # Calcular características de tendencia diaria
            try:
                # SMA 20 días y su pendiente
                df_1d['SMA20'] = df_1d['close'].rolling(20).mean()
                # Pendiente simple
                if len(df_1d) > 5:
                    features["SMA20_slope"] = (df_1d['SMA20'].iloc[-1] - df_1d['SMA20'].iloc[-5]) / df_1d['SMA20'].iloc[-5]
                
                # Distancia a máximos/mínimos de 20 días
                if len(df_1d) >= 20:
                    features["Distance_20D_High"] = (df_1d['close'].iloc[-1] - df_1d['high'].rolling(20).max().iloc[-1]) / df_1d['high'].rolling(20).max().iloc[-1]
                    features["Distance_20D_Low"] = (df_1d['close'].iloc[-1] - df_1d['low'].rolling(20).min().iloc[-1]) / df_1d['low'].rolling(20).min().iloc[-1]
                
                # Fase del mercado (simplificado)
                if "EMA8" in features and "EMA21" in features:
                    if features["EMA8"] > features["EMA21"] and features["ADX"] > 25:
                        features["Market_Phase"] = 1.0  # Tendencia alcista fuerte
                    elif features["EMA8"] > features["EMA21"] and features["ADX"] <= 25:
                        features["Market_Phase"] = 0.5  # Tendencia alcista débil
                    elif features["EMA8"] < features["EMA21"] and features["ADX"] > 25:
                        features["Market_Phase"] = -1.0  # Tendencia bajista fuerte
                    elif features["EMA8"] < features["EMA21"] and features["ADX"] <= 25:
                        features["Market_Phase"] = -0.5  # Tendencia bajista débil
                    else:
                        features["Market_Phase"] = 0.0  # Rango
                        
                # Características de volatilidad y momentum
                features["Volatility_1h"] = df_1h['close'].pct_change().std() * np.sqrt(252 * 6.5)
                features["Momentum_1h"] = df_1h['close'].pct_change(5).iloc[-1]
                
                # Alineación de tendencias (simplificado)
                if "EMA8" in features and "EMA21" in features and len(df_1d) > 1:
                    trend_1h = 1 if features["EMA8"] > features["EMA21"] else -1
                    trend_1d = 1 if df_1d['close'].iloc[-1] > df_1d['close'].iloc[-2] else -1
                    features["Trend_Alignment"] = 1.0 if trend_1h == trend_1d else -1.0
                    
            except Exception as e:
                self.Debug(f"Error al calcular características: {e}")
                
        return features
        
    def TradeBasedOnModel(self):
        """Ejecuta operaciones basadas en las predicciones del modelo"""
        for symbol, data in self.symbol_data.items():
            # Verificar si tenemos datos suficientes
            if not self.hourly_window[symbol].IsReady:
                self.Debug(f"No hay suficientes datos para {symbol}")
                continue
                
            # Obtener datos actuales
            current_bar = self.hourly_window[symbol][0]  # Barra más reciente
            
            # Calcular características
            features = self.CalculateFeatures(symbol)
            
            # Actualizar datos del mercado en el integrador
            self.rl_integrator.update_market_data(symbol.Value, current_bar, features)
            
            # Obtener señal de trading
            signal, confidence = self.rl_integrator.get_trading_signal(
                symbol.Value, 
                data["asset_id"],
                self.Securities[symbol].Price
            )
            
            # Ejecutar operaciones basadas en la señal
            self.ExecuteSignal(symbol, signal, confidence)
            
    def ExecuteSignal(self, symbol, signal, confidence):
        """
        Ejecuta una operación basada en la señal del modelo
        
        Args:
            symbol: Símbolo a operar
            signal: Señal (1=comprar, -1=vender, 0=mantener)
            confidence: Confianza de la señal (0-1)
        """
        # Solo operar si la confianza es alta
        if confidence < 0.7:
            return
            
        # Obtener posición actual
        current_position = self.Portfolio[symbol].Quantity
        current_price = self.Securities[symbol].Price
        
        # Calcular tamaño de la posición (fijo para este ejemplo)
        position_size = self.CalculateOrderQuantity(symbol, 0.1)  # 10% del portfolio
        
        if signal == 1 and current_position <= 0:  # Comprar
            self.SetHoldings(symbol, 0.1)  # Asignar 10% del portfolio
            self.Debug(f"Comprando {symbol} a {current_price} (Confianza: {confidence:.2f})")
            
        elif signal == -1 and current_position > 0:  # Vender
            self.Liquidate(symbol)
            self.Debug(f"Vendiendo {symbol} a {current_price} (Confianza: {confidence:.2f})")
            
    def OnOrderEvent(self, orderEvent):
        """
        Evento llamado cuando se produce un evento de orden
        
        Args:
            orderEvent: Evento de orden
        """
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Orden {orderEvent.OrderId} ejecutada: {orderEvent.FillQuantity} @ {orderEvent.FillPrice}")
            
    def OnEndOfAlgorithm(self):
        """Evento llamado al finalizar el algoritmo"""
        self.Debug("Algoritmo finalizado")
        self.Debug(f"Rendimiento total: {self.Portfolio.TotalPortfolioValue - self.StartingPortfolioValue}")
        self.Debug(f"Rendimiento %: {(self.Portfolio.TotalPortfolioValue / self.StartingPortfolioValue - 1) * 100:.2f}%")

# Clase para Paper Trading con IBKR
class IBKRPaperTradingIntegration:
    """
    Clase para integrar el modelo RL con paper trading en IBKR
    
    Esta clase utiliza QuantConnect LEAN junto con la conexión a
    IBKR para realizar paper trading con el modelo RL.
    """
    
    def __init__(self, 
                model_dir: str,
                symbols: List[str],
                ibkr_port: int = 7497):
        """
        Inicializa la integración para paper trading
        
        Args:
            model_dir: Directorio donde se encuentra el modelo guardado
            symbols: Lista de símbolos a operar
            ibkr_port: Puerto para conectar con IBKR (7497 para paper trading)
        """
        self.model_dir = model_dir
        self.symbols = symbols
        self.ibkr_port = ibkr_port
        
        # Código para iniciar un algoritmo de QuantConnect con IBKR
        # Este código debería ejecutarse desde la interfaz de línea de comandos
        # de QuantConnect LEAN
        
        lean_command = f"""
        lean live --algorithm-id="RLTradingAlgorithm" \\
                  --brokerage=InteractiveBrokers \\
                  --environment=paper \\
                  --ib-user=$IB_USER \\
                  --ib-password=$IB_PASSWORD \\
                  --ib-trading-mode=paper \\
                  --ib-port={ibkr_port} \\
                  --data-feed=InteractiveBrokers
        """
        
        print("Para iniciar el paper trading con IBKR, ejecuta el siguiente comando:")
        print(lean_command)
        
        # Instrucciones adicionales
        print("\nAntes de ejecutar el comando:")
        print("1. Asegúrate de que IBKR TWS o Gateway esté en ejecución")
        print("2. Configura las variables de entorno IB_USER e IB_PASSWORD")
        print("3. Sube el modelo a QuantConnect ObjectStore o especifica la ruta en el código")
        print("4. Asegúrate de que el algoritmo RLTradingAlgorithm esté registrado en QuantConnect")

# Función principal para crear y ejecutar un backtest
def run_backtest():
    """Ejecuta un backtest con el algoritmo RL Trading"""
    # Esta función debería ejecutarse desde la CLI de QuantConnect
    
    lean_backtest_command = """
    lean backtest --algorithm-id="RLTradingAlgorithm" \\
                  --data-folder=./data \\
                  --start-date=2023-01-01 \\
                  --end-date=2023-12-31 \\
                  --results-destination-folder=./results
    """
    
    print("Para ejecutar un backtest, usa el siguiente comando:")
    print(lean_backtest_command)

# Ejemplo de uso
if __name__ == "__main__":
    # Esta parte solo es informativa, ya que normalmente
    # el código se ejecutaría dentro del entorno de QuantConnect
    
    # Inicializar la integración con paper trading
    ibkr_integration = IBKRPaperTradingIntegration(
        model_dir="models/best_model",
        symbols=["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"],
        ibkr_port=7497
    )
    
    # Mostrar comandos para backtest
    run_backtest()