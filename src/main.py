import os
import argparse
import logging
import pandas as pd
import numpy as np
import json
import sys
import shutil
import datetime
from typing import Dict, List, Optional, Union, Tuple, Any

# Importar módulos propios
from src.data.ibkr_data import IBDataManager
from src.models.trading_env import TradingEnv
from src.models.feature_engineering import FeatureEngineering
from src.models.ppo_agent import PPOAgent
from src.models.hp_optimization import OptimizationManager
from src.utils.quantconnect_integration import RLModelIntegrator, IBKRPaperTradingIntegration

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_rl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingRLPipeline:
    """
    Pipeline completo para entrenamiento y despliegue de modelos RL para trading
    
    Esta clase orquesta todo el flujo de trabajo:
    1. Descarga de datos históricos de IBKR
    2. Ingeniería de características
    3. Entrenamiento del modelo
    4. Optimización de hiperparámetros
    5. Evaluación del modelo
    6. Integración con QuantConnect para backtesting y trading en vivo
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Inicializa el pipeline con un archivo de configuración
        
        Args:
            config_file: Ruta al archivo de configuración JSON
        """
        # Cargar configuración
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
        # Crear directorios necesarios
        os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(self.config["models_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)
        
        # Inicializar componentes según la configuración
        self.ibkr_config = self.config.get("ibkr", {})
        self.training_config = self.config.get("training", {})
        self.optimization_config = self.config.get("optimization", {})
        self.env_config = self.config.get("environment", {})
        self.features_config = self.config.get("features", {})
        self.quantconnect_config = self.config.get("quantconnect", {})
        
        # Validar configuración mínima
        self._validate_config()
        
        # Componentes principales (se inicializarán según sea necesario)
        self.data_manager = None
        self.feature_engineering = None
        self.environment = None
        self.agent = None
        self.optimization_manager = None
        self.model_integrator = None
        
    def _validate_config(self):
        """Valida la configuración mínima necesaria"""
        required_sections = ["ibkr", "training", "environment", "features"]
        missing_sections = [section for section in required_sections if section not in self.config]
        
        if missing_sections:
            raise ValueError(f"Faltan secciones en el archivo de configuración: {missing_sections}")
            
        # Validar símbolos
        if "symbols" not in self.config or not self.config["symbols"]:
            raise ValueError("No se especificaron símbolos para trading")
            
    def download_historical_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Descarga datos históricos de IBKR
        
        Returns:
            Datos en formato {símbolo: {timeframe: DataFrame}}
        """
        logger.info("Iniciando descarga de datos históricos de IBKR")
        
        # Inicializar gestor de datos si es necesario
        if self.data_manager is None:
            self.data_manager = IBDataManager(
                host=self.ibkr_config.get("host", "127.0.0.1"),
                port=self.ibkr_config.get("port", 7497),
                client_id=self.ibkr_config.get("client_id", 0)
            )
            
        # Obtener datos
        symbols = self.config["symbols"]
        timeframes = self.env_config.get("timeframes", ["1D", "1H", "15M"])
        duration = self.ibkr_config.get("duration", "1 Y")
        use_rth = self.ibkr_config.get("use_rth", True)
        
        try:
            # Verificar si los datos ya existen en disco
            data_dir = self.config["data_dir"]
            data = None
            
            if os.path.exists(data_dir) and self.ibkr_config.get("use_cached_data", True):
                # Intentar cargar desde disco
                data = self.data_manager.load_data_from_disk(symbols, timeframes, data_dir)
                
                # Verificar si tenemos todos los datos
                all_data_present = all(
                    symbol in data and all(tf in data[symbol] for tf in timeframes)
                    for symbol in symbols
                )
                
                if all_data_present:
                    logger.info("Datos cargados desde disco correctamente")
                else:
                    logger.info("Datos incompletos en disco, descargando de IBKR")
                    data = None
                    
            if data is None:
                # Conectar y descargar datos
                connected = self.data_manager.connect()
                
                if not connected:
                    raise ConnectionError("No se pudo conectar a IBKR")
                    
                # Descargar datos
                data = self.data_manager.get_data_for_multiple_assets(
                    symbols=symbols,
                    timeframes=timeframes,
                    duration=duration,
                    use_rth=use_rth,
                    sec_type=self.ibkr_config.get("sec_type", "STK"),
                    exchange=self.ibkr_config.get("exchange", "SMART"),
                    currency=self.ibkr_config.get("currency", "USD")
                )
                
                # Guardar datos en disco
                self.data_manager.save_data_to_disk(data, output_dir=data_dir)
                
            # Alinear datos
            aligned_data = self.data_manager.align_timeframes(data)
            
            # Desconectar de IBKR
            if self.data_manager.connected:
                self.data_manager.disconnect()
                
            return aligned_data
            
        except Exception as e:
            logger.error(f"Error al descargar datos: {e}")
            if self.data_manager and self.data_manager.connected:
                self.data_manager.disconnect()
            raise
            
    def process_features(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], List[str]]:
        """
        Procesa los datos y calcula todas las características
        
        Args:
            data: Datos en formato {símbolo: {timeframe: DataFrame}}
            
        Returns:
            Tupla (datos_procesados, lista_de_características)
        """
        logger.info("Procesando características para todos los símbolos")
        
        # Inicializar ingeniería de características si es necesario
        if self.feature_engineering is None:
            self.feature_engineering = FeatureEngineering()
            
        processed_data = {}
        all_features = []
        
        for symbol, timeframes_data in data.items():
            logger.info(f"Procesando características para {symbol}")
            processed_data[symbol] = {}
            
            # Procesar cada timeframe
            features_1d = self.feature_engineering.calculate_indicators(
                timeframes_data["1D"], "1d"
            )
            features_1h = self.feature_engineering.calculate_indicators(
                timeframes_data["1H"], "1h"
            )
            features_15m = self.feature_engineering.calculate_indicators(
                timeframes_data["15M"], "15m"
            )
            
            # Calcular características cross-timeframe
            cross_tf_features = self.feature_engineering._calculate_cross_timeframe_features(
                features_1d, features_1h, features_15m
            )
            
            # Almacenar datos procesados
            processed_data[symbol]["1D"] = features_1d
            processed_data[symbol]["1H"] = features_1h
            processed_data[symbol]["15M"] = features_15m
            processed_data[symbol]["cross_tf"] = cross_tf_features
            
            # Recopilar todas las características
            if not all_features:
                all_features = list(features_1d.columns) + list(features_1h.columns) + \
                             list(features_15m.columns) + list(cross_tf_features.columns)
                all_features = list(set(all_features))  # Eliminar duplicados
                
        return processed_data, all_features
        
    def create_environment(self, data: Dict[str, Dict[str, pd.DataFrame]], features: List[str]) -> TradingEnv:
        """
        Crea el entorno de trading con los datos procesados
        
        Args:
            data: Datos procesados
            features: Lista de características
            
        Returns:
            Entorno de trading inicializado
        """
        logger.info("Creando entorno de trading")
        
        # Preparar datos en formato adecuado para el entorno
        data_1d = {}
        data_1h = {}
        data_15m = {}
        
        for symbol in data.keys():
            data_1d[symbol] = data[symbol]["1D"]
            data_1h[symbol] = data[symbol]["1H"]
            data_15m[symbol] = data[symbol]["15M"]
            
        # Crear entorno
        self.environment = TradingEnv(
            data_1d=data_1d,
            data_1h=data_1h,
            data_15m=data_15m,
            features=features,
            window_size=self.env_config.get("window_size", 30),
            commission=self.env_config.get("commission", 0.001),
            initial_balance=self.env_config.get("initial_balance", 10000),
            reward_window=self.env_config.get("reward_window", 20),
            min_price_change=self.env_config.get("min_price_change", 0.05)
        )
        
        return self.environment
        
    def train_model(self, env: TradingEnv) -> PPOAgent:
        """
        Entrena un modelo PPO en el entorno de trading
        
        Args:
            env: Entorno de trading inicializado
            
        Returns:
            Agente PPO entrenado
        """
        logger.info("Iniciando entrenamiento del modelo")
        
        # Crear agente
        self.agent = PPOAgent(
            env=env,
            learning_rate=self.training_config.get("learning_rate", 3e-4),
            gamma=self.training_config.get("gamma", 0.99),
            gae_lambda=self.training_config.get("gae_lambda", 0.95),
            policy_clip=self.training_config.get("policy_clip", 0.2),
            batch_size=self.training_config.get("batch_size", 64),
            n_epochs=self.training_config.get("n_epochs", 10),
            entropy_coef=self.training_config.get("entropy_coef", 0.01),
            value_coef=self.training_config.get("value_coef", 0.5),
            actor_hidden_dims=self.training_config.get("actor_hidden_dims", [512, 256, 128]),
            critic_hidden_dims=self.training_config.get("critic_hidden_dims", [512, 256, 128]),
            checkpoint_dir=os.path.join(self.config["models_dir"], "checkpoints")
        )
        
        # Entrenar modelo
        n_episodes = self.training_config.get("n_episodes", 100)
        max_steps = self.training_config.get("max_steps_per_episode", 1000)
        save_freq = self.training_config.get("save_freq", 10)
        
        training_metrics = self.agent.train(
            n_episodes=n_episodes,
            max_steps_per_episode=max_steps,
            save_freq=save_freq
        )
        
        # Guardar métricas de entrenamiento
        metrics_file = os.path.join(self.config["results_dir"], "training_metrics.json")
        with open(metrics_file, 'w') as f:
            # Convertir arrays numpy a listas para serialización JSON
            serializable_metrics = {}
            for key, value in training_metrics.items():
                if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                    serializable_metrics[key] = [float(v) for v in value]
                elif isinstance(value, list) and value and isinstance(value[0], np.float32):
                    serializable_metrics[key] = [float(v) for v in value]
                else:
                    serializable_metrics[key] = value
                    
            json.dump(serializable_metrics, f, indent=4)
            
        # Guardar modelo final
        final_model_dir = os.path.join(self.config["models_dir"], "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        
        # Copiar archivos del modelo
        for filename in os.listdir(os.path.join(self.config["models_dir"], "checkpoints")):
            if filename.startswith(f"actor_ep{n_episodes}") or filename.startswith(f"critic_ep{n_episodes}"):
                src = os.path.join(self.config["models_dir"], "checkpoints", filename)
                dst = os.path.join(final_model_dir, filename.replace(f"_ep{n_episodes}", "_final"))
                shutil.copy(src, dst)
                
        logger.info(f"Entrenamiento completado. Modelo guardado en {final_model_dir}")
        
        return self.agent
        
    def run_optimization(self, data: Dict[str, Dict[str, pd.DataFrame]], features: List[str]) -> Dict[str, Any]:
        """
        Ejecuta la optimización de hiperparámetros
        
        Args:
            data: Datos procesados
            features: Lista de características
            
        Returns:
            Mejores hiperparámetros encontrados
        """
        logger.info("Iniciando optimización de hiperparámetros")
        
        # Preparar datos en formato adecuado
        data_1d = {}
        data_1h = {}
        data_15m = {}
        
        for symbol in data.keys():
            data_1d[symbol] = data[symbol]["1D"]
            data_1h[symbol] = data[symbol]["1H"]
            data_15m[symbol] = data[symbol]["15M"]
            
        # Crear gestor de optimización
        self.optimization_manager = OptimizationManager(
            data_1d=data_1d,
            data_1h=data_1h,
            data_15m=data_15m,
            features=features,
            study_name=self.optimization_config.get("study_name", "ppo_trading_optimization"),
            n_trials=self.optimization_config.get("n_trials", 100),
            n_startup_trials=self.optimization_config.get("n_startup_trials", 10),
            n_evaluations=self.optimization_config.get("n_evaluations", 3),
            n_episodes_per_eval=self.optimization_config.get("n_episodes_per_eval", 20),
            max_episode_steps=self.optimization_config.get("max_episode_steps", 10000),
            output_dir=os.path.join(self.config["results_dir"], "optimization")
        )
        
        # Ejecutar optimización
        study = self.optimization_manager.run_optimization()
        
        # Entrenar modelo final con los mejores hiperparámetros
        best_agent = self.optimization_manager.train_best_model(
            n_episodes=self.optimization_config.get("final_training_episodes", 200)
        )
        
        # Guardar modelo optimizado como modelo "best"
        best_model_dir = os.path.join(self.config["models_dir"], "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        
        # Copiar archivos del mejor modelo
        optim_dir = os.path.join(self.config["results_dir"], "optimization", "best_model")
        for filename in os.listdir(optim_dir):
            if filename.endswith(".pth") or filename == "config.json":
                src = os.path.join(optim_dir, filename)
                dst = os.path.join(best_model_dir, filename)
                shutil.copy(src, dst)
                
        logger.info(f"Optimización completada. Mejor modelo guardado en {best_model_dir}")
        
        # Devolver mejores hiperparámetros
        return study.best_params
        
    def evaluate_model(self, model_dir: str, data: Dict[str, Dict[str, pd.DataFrame]], features: List[str]) -> Dict[str, Any]:
        """
        Evalúa un modelo entrenado
        
        Args:
            model_dir: Directorio del modelo a evaluar
            data: Datos procesados
            features: Lista de características
            
        Returns:
            Métricas de evaluación
        """
        logger.info(f"Evaluando modelo en {model_dir}")
        
        # Preparar datos
        data_1d = {}
        data_1h = {}
        data_15m = {}
        
        for symbol in data.keys():
            data_1d[symbol] = data[symbol]["1D"]
            data_1h[symbol] = data[symbol]["1H"]
            data_15m[symbol] = data[symbol]["15M"]
            
        # Cargar configuración del modelo
        config_file = os.path.join(model_dir, "config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                model_config = json.load(f)
        else:
            # Usar configuración por defecto
            model_config = self.env_config
            
        # Crear entorno
        env = TradingEnv(
            data_1d=data_1d,
            data_1h=data_1h,
            data_15m=data_15m,
            features=features,
            window_size=model_config.get("window_size", 30),
            commission=model_config.get("commission", 0.001),
            initial_balance=model_config.get("initial_balance", 10000),
            reward_window=model_config.get("reward_window", 20),
            min_price_change=model_config.get("min_price_change", 0.05)
        )
        
        # Crear agente
        agent = PPOAgent(
            env=env,
            learning_rate=0.0,  # No se necesita para evaluación
            gamma=0.99,
            checkpoint_dir=model_dir
        )
        
        # Cargar modelo
        agent.load_models(
            actor_path=os.path.join(model_dir, "actor_final.pth"),
            critic_path=os.path.join(model_dir, "critic_final.pth")
        )
        
        # Evaluar modelo
        n_episodes = self.training_config.get("eval_episodes", 20)
        render = self.training_config.get("render_evaluation", True)
        
        eval_results = agent.evaluate(n_episodes=n_episodes, render=render)
        
        # Guardar resultados
        results_file = os.path.join(self.config["results_dir"], "evaluation_results.json")
        with open(results_file, 'w') as f:
            # Convertir a tipos serializables
            serializable_results = {}
            for key, value in eval_results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, list) and value and isinstance(value[0], np.float32):
                    serializable_results[key] = [float(v) for v in value]
                else:
                    serializable_results[key] = value
                    
            json.dump(serializable_results, f, indent=4)
            
        logger.info(f"Evaluación completada. Resultados guardados en {results_file}")
        
        return eval_results
        
    def setup_quantconnect(self, model_dir: str = None):
        """
        Configura la integración con QuantConnect
        
        Args:
            model_dir: Directorio del modelo a utilizar (si None, usa el mejor modelo)
        """
        logger.info("Configurando integración con QuantConnect")
        
        if model_dir is None:
            model_dir = os.path.join(self.config["models_dir"], "best_model")
            
        # Verificar si el modelo existe
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"No se encontró el modelo en {model_dir}")
            
        # Integración con QuantConnect para backtesting
        symbols = self.config["symbols"]
        
        # Mostrar ruta para colocar el modelo
        lean_dir = self.quantconnect_config.get("lean_dir", "~/Lean")
        model_dest = os.path.join(lean_dir, "Data/models/rl_model")
        
        logger.info(f"Para usar el modelo en QuantConnect, copia los archivos de {model_dir} a {model_dest}")
        
        # Configuración para paper trading
        ibkr_port = self.ibkr_config.get("port", 7497)
        
        # Crear integración con IBKR para paper trading
        ibkr_integration = IBKRPaperTradingIntegration(
            model_dir=model_dir,
            symbols=symbols,
            ibkr_port=ibkr_port
        )
        
        return ibkr_integration
        
    def run_pipeline(self, steps: List[str] = None):
        """
        Ejecuta el pipeline completo o los pasos especificados
        
        Args:
            steps: Lista de pasos a ejecutar. Si es None, ejecuta todos los pasos.
                  Pasos disponibles: 'download', 'features', 'train', 'optimize', 'evaluate', 'quantconnect'
        """
        # Pasos por defecto
        all_steps = ['download', 'features', 'train', 'optimize', 'evaluate', 'quantconnect']
        
        # Si no se especifican pasos, ejecutar todos
        if steps is None:
            steps = all_steps
            
        # Validar pasos
        invalid_steps = [step for step in steps if step not in all_steps]
        if invalid_steps:
            raise ValueError(f"Pasos no válidos: {invalid_steps}")
            
        # Variables para almacenar resultados intermedios
        data = None
        processed_data = None
        features = None
        env = None
        best_params = None
        
        # Ejecutar pasos
        try:
            # Descargar datos
            if 'download' in steps:
                logger.info("=== PASO 1: Descarga de datos históricos ===")
                data = self.download_historical_data()
                
            # Procesar características
            if 'features' in steps:
                logger.info("=== PASO 2: Procesamiento de características ===")
                if data is None and 'download' not in steps:
                    logger.info("Cargando datos desde disco para procesamiento de características")
                    symbols = self.config["symbols"]
                    timeframes = self.env_config.get("timeframes", ["1D", "1H", "15M"])
                    data_dir = self.config["data_dir"]
                    
                    self.data_manager = IBDataManager()
                    data = self.data_manager.load_data_from_disk(symbols, timeframes, data_dir)
                    data = self.data_manager.align_timeframes(data)
                    
                processed_data, features = self.process_features(data)
                
                # Guardar lista de características
                features_file = os.path.join(self.config["results_dir"], "features.json")
                with open(features_file, 'w') as f:
                    json.dump(features, f, indent=4)
                    
            # Entrenar modelo
            if 'train' in steps:
                logger.info("=== PASO 3: Entrenamiento del modelo ===")
                if processed_data is None and 'features' not in steps:
                    logger.warning("No se pueden cargar datos procesados, saltando entrenamiento")
                else:
                    env = self.create_environment(processed_data, features)
                    agent = self.train_model(env)
                    
            # Optimizar hiperparámetros
            if 'optimize' in steps:
                logger.info("=== PASO 4: Optimización de hiperparámetros ===")
                if processed_data is None and 'features' not in steps:
                    logger.warning("No se pueden cargar datos procesados, saltando optimización")
                else:
                    best_params = self.run_optimization(processed_data, features)
                    
            # Evaluar modelo
            if 'evaluate' in steps:
                logger.info("=== PASO 5: Evaluación del modelo ===")
                if processed_data is None and 'features' not in steps:
                    logger.warning("No se pueden cargar datos procesados, saltando evaluación")
                else:
                    # Determinar qué modelo evaluar
                    model_to_evaluate = None
                    
                    if os.path.exists(os.path.join(self.config["models_dir"], "best_model")):
                        model_to_evaluate = os.path.join(self.config["models_dir"], "best_model")
                    elif os.path.exists(os.path.join(self.config["models_dir"], "final_model")):
                        model_to_evaluate = os.path.join(self.config["models_dir"], "final_model")
                        
                    if model_to_evaluate:
                        eval_results = self.evaluate_model(model_to_evaluate, processed_data, features)
                    else:
                        logger.warning("No se encontró ningún modelo para evaluar")
                        
            # Configurar QuantConnect
            if 'quantconnect' in steps:
                logger.info("=== PASO 6: Configuración de QuantConnect ===")
                self.setup_quantconnect()
                
            logger.info("Pipeline completado exitosamente")
            
        except Exception as e:
            logger.error(f"Error en el pipeline: {e}")
            raise
            
def create_default_config():
    """Crea un archivo de configuración por defecto"""
    default_config = {
        "data_dir": "data",
        "models_dir": "models",
        "results_dir": "results",
        "symbols": ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"],
        "ibkr": {
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 0,
            "duration": "1 Y",
            "use_rth": True,
            "use_cached_data": True
        },
        "environment": {
            "timeframes": ["1D", "1H", "15M"],
            "window_size": 30,
            "commission": 0.001,
            "initial_balance": 10000,
            "reward_window": 20,
            "min_price_change": 0.05
        },
        "features": {
            "include_standard_indicators": True,
            "include_cross_timeframe": True
        },
        "training": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "policy_clip": 0.2,
            "batch_size": 64,
            "n_epochs": 10,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "n_episodes": 100,
            "max_steps_per_episode": 1000,
            "save_freq": 10,
            "eval_episodes": 20,
            "render_evaluation": True
        },
        "optimization": {
            "study_name": "ppo_trading_optimization",
            "n_trials": 100,
            "n_startup_trials": 10,
            "n_evaluations": 3,
            "n_episodes_per_eval": 20,
            "max_episode_steps": 10000,
            "final_training_episodes": 200
        },
        "quantconnect": {
            "lean_dir": "~/Lean"
        }
    }
    
    # Guardar configuración
    with open("config.json", "w") as f:
        json.dump(default_config, f, indent=4)
        
    print("Archivo de configuración por defecto creado: config.json")
    
def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Pipeline para trading con RL")
    parser.add_argument("--config", type=str, default="config.json", help="Archivo de configuración")
    parser.add_argument("--steps", type=str, nargs="+", 
                      choices=["download", "features", "train", "optimize", "evaluate", "quantconnect", "all"],
                      default=["all"], help="Pasos a ejecutar")
    parser.add_argument("--create-config", action="store_true", help="Crear archivo de configuración por defecto")
    
    args = parser.parse_args()
    
    # Crear configuración por defecto si se solicita
    if args.create_config:
        create_default_config()
        return
        
    # Verificar si existe el archivo de configuración
    if not os.path.exists(args.config):
        print(f"No se encontró el archivo de configuración: {args.config}")
        print("Ejecute con --create-config para crear uno por defecto")
        return
        
    # Convertir 'all' a todos los pasos
    if "all" in args.steps:
        steps = None
    else:
        steps = args.steps
        
    # Inicializar pipeline
    pipeline = TradingRLPipeline(config_file=args.config)
    
    # Ejecutar pipeline
    pipeline.run_pipeline(steps=steps)
    
if __name__ == "__main__":
    main()