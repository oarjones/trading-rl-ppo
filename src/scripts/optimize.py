#!/usr/bin/env python
"""
Script para ejecutar una optimización de hiperparámetros para el modelo de trading con RL

Este script:
1. Carga datos históricos de IBKR o desde disco
2. Procesa características con FeatureEngineering
3. Ejecuta la optimización de hiperparámetros con Optuna
4. Entrena un modelo final con los mejores hiperparámetros
5. Guarda y visualiza los resultados
"""

import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

# Importar componentes del sistema
from src.data.ibkr_data import IBDataManager
from src.models.feature_engineering import FeatureEngineering
from src.models.trading_env import TradingEnv
from src.models.hp_optimization import OptimizationManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("optimization")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo JSON
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Configuración como diccionario
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_data(config: Dict[str, Any]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Carga datos históricos desde IBKR o desde disco
    
    Args:
        config: Configuración del sistema
        
    Returns:
        Datos en formato {símbolo: {timeframe: DataFrame}}
    """
    data_dir = config.get("data_dir", "data")
    symbols = config.get("symbols", ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"])
    timeframes = config.get("environment", {}).get("timeframes", ["1D", "1H", "15M"])
    use_cached_data = config.get("ibkr", {}).get("use_cached_data", True)
    
    logger.info(f"Cargando datos para símbolos: {symbols}")
    
    # Inicializar gestor de datos
    data_manager = IBDataManager(
        host=config.get("ibkr", {}).get("host", "127.0.0.1"),
        port=config.get("ibkr", {}).get("port", 7497),
        client_id=config.get("ibkr", {}).get("client_id", 1)
    )
    
    # Intentar cargar datos desde disco si se permite
    data = None
    if use_cached_data and os.path.exists(data_dir):
        try:
            data = data_manager.load_data_from_disk(symbols, timeframes, data_dir)
            
            # Verificar si tenemos todos los datos necesarios
            all_data_present = True
            for symbol in symbols:
                if symbol not in data:
                    all_data_present = False
                    break
                for tf in timeframes:
                    if tf not in data[symbol]:
                        all_data_present = False
                        break
                if not all_data_present:
                    break
                    
            if all_data_present:
                logger.info("Datos cargados correctamente desde disco")
            else:
                logger.info("Datos incompletos en disco, descargando desde IBKR")
                data = None
        except Exception as e:
            logger.error(f"Error al cargar datos desde disco: {e}")
            data = None
            
    # Si no tenemos datos, descargar desde IBKR
    if data is None:
        try:
            # Conectar con IBKR
            connected = data_manager.connect()
            if not connected:
                raise ConnectionError("No se pudo conectar a IBKR")
                
            # Descargar datos
            logger.info("Descargando datos desde IBKR...")
            data = data_manager.get_data_for_multiple_assets(
                symbols=symbols,
                timeframes=timeframes,
                duration=config.get("ibkr", {}).get("duration", "1 Y"),
                use_rth=config.get("ibkr", {}).get("use_rth", True)
            )
            
            # Guardar datos en disco
            logger.info("Guardando datos en disco...")
            data_manager.save_data_to_disk(data, output_dir=data_dir)
            
        except Exception as e:
            logger.error(f"Error al descargar datos: {e}")
            raise
        finally:
            # Desconectar de IBKR
            if data_manager.connected:
                data_manager.disconnect()
                
    # Alinear timeframes
    logger.info("Alineando datos de diferentes timeframes...")
    aligned_data = data_manager.align_timeframes(data)
    
    return aligned_data

def process_features(data: Dict[str, Dict[str, pd.DataFrame]], config: Dict[str, Any]) -> tuple:
    """
    Procesa los datos y calcula características técnicas
    
    Args:
        data: Datos en formato {símbolo: {timeframe: DataFrame}}
        config: Configuración del sistema
        
    Returns:
        Tupla de (datos procesados, lista de características)
    """
    logger.info("Procesando características técnicas...")
    
    # Inicializar ingeniería de características
    feature_engineering = FeatureEngineering()
    
    processed_data = {}
    all_features = []
    
    for symbol, timeframes_data in data.items():
        logger.info(f"Procesando características para {symbol}")
        processed_data[symbol] = {}
        
        # Procesar cada timeframe
        features_1d = feature_engineering.calculate_indicators(
            timeframes_data["1D"], "1d"
        )
        features_1h = feature_engineering.calculate_indicators(
            timeframes_data["1H"], "1h"
        )
        features_15m = feature_engineering.calculate_indicators(
            timeframes_data["15M"], "15m"
        )
        
        # Calcular características cross-timeframe
        cross_tf_features = feature_engineering._calculate_cross_timeframe_features(
            features_1d, features_1h, features_15m
        )
        
        # Almacenar datos procesados
        processed_data[symbol]["1D"] = features_1d
        processed_data[symbol]["1H"] = features_1h
        processed_data[symbol]["15M"] = features_15m
        processed_data[symbol]["cross_tf"] = cross_tf_features
        
        # Recopilar todas las características
        if not all_features:
            # Obtener características por timeframe
            # Nota: Solo queremos OHLC del timeframe horario
            hourly_features = list(features_1h.columns)
            
            # Obtener específicamente solo los indicadores del timeframe diario
            daily_indicators = [col for col in features_1d.columns if 
                               any(ind in col for ind in ['SMA20_slope', 'Market_Phase', 
                                                         'Distance_52W', 'ADX_normalized'])]
            daily_features = [f"1d_{col}" for col in daily_indicators]
            
            # Obtener específicamente solo los indicadores del timeframe de 15 minutos
            min15_indicators = [col for col in features_15m.columns if 
                               any(ind in col for ind in ['Momentum', 'RSI_divergence', 
                                                         'Volume_acceleration', 'BB_squeeze'])]
            min15_features = [f"15m_{col}" for col in min15_indicators]
            
            # Obtener características cross-timeframe
            cross_tf_cols = list(cross_tf_features.columns)
            cross_tf_features = [f"cross_tf_{col}" for col in cross_tf_cols]
            
            # Combinar todas las características
            all_features = hourly_features + daily_features + min15_features + cross_tf_features
            
    return processed_data, all_features

def prepare_data_for_optimization(processed_data: Dict[str, Dict[str, pd.DataFrame]]) -> tuple:
    """
    Prepara los datos en el formato adecuado para la optimización
    
    Args:
        processed_data: Datos procesados
        
    Returns:
        Tupla de (data_1d, data_1h, data_15m)
    """
    data_1d = {}
    data_1h = {}
    data_15m = {}
    
    for symbol in processed_data.keys():
        data_1d[symbol] = processed_data[symbol]["1D"]
        data_1h[symbol] = processed_data[symbol]["1H"]
        data_15m[symbol] = processed_data[symbol]["15M"]
        
    return data_1d, data_1h, data_15m

def run_optimization(data_1d: Dict[str, pd.DataFrame], 
                   data_1h: Dict[str, pd.DataFrame], 
                   data_15m: Dict[str, pd.DataFrame],
                   features: List[str], 
                   config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta la optimización de hiperparámetros
    
    Args:
        data_1d, data_1h, data_15m: Datos de diferentes timeframes
        features: Lista de características
        config: Configuración del sistema
        
    Returns:
        Mejores hiperparámetros encontrados
    """
    logger.info("Iniciando optimización de hiperparámetros...")
    
    # Configuración de optimización
    optimization_config = config.get("optimization", {})
    results_dir = config.get("results_dir", "results")
    output_dir = os.path.join(results_dir, "optimization")
    
    # Crear gestor de optimización
    optimization_manager = OptimizationManager(
        data_1d=data_1d,
        data_1h=data_1h,
        data_15m=data_15m,
        features=features,
        study_name=optimization_config.get("study_name", "ppo_trading_optimization"),
        n_trials=optimization_config.get("n_trials", 50),
        n_startup_trials=optimization_config.get("n_startup_trials", 10),
        n_evaluations=optimization_config.get("n_evaluations", 3),
        n_episodes_per_eval=optimization_config.get("n_episodes_per_eval", 10),
        max_episode_steps=optimization_config.get("max_episode_steps", 5000),
        output_dir=output_dir
    )
    
    # Ejecutar optimización
    study = optimization_manager.run_optimization()
    
    # Obtener mejores hiperparámetros
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Optimización completada. Mejor valor: {best_value:.2f}")
    logger.info(f"Mejores hiperparámetros: {best_params}")
    
    # Entrenar modelo final con los mejores hiperparámetros
    logger.info("Entrenando modelo final con los mejores hiperparámetros...")
    best_agent = optimization_manager.train_best_model(
        n_episodes=optimization_config.get("final_training_episodes", 200)
    )
    
    return best_params

def plot_optimization_results(output_dir: str):
    """
    Visualiza los resultados de la optimización
    
    Args:
        output_dir: Directorio con los resultados de optimización
    """
    try:
        import optuna.visualization as vis
        import plotly.io as pio
        
        # Cargar estudio
        study_path = os.path.join(output_dir, "ppo_trading_optimization.pkl")
        if not os.path.exists(study_path):
            logger.warning(f"No se encontró el archivo del estudio en {study_path}")
            return
            
        import joblib
        study = joblib.load(study_path)
        
        # Crear directorio para visualizaciones
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Guardar visualizaciones
        logger.info("Generando visualizaciones de la optimización...")
        
        # Historial de optimización
        fig = vis.plot_optimization_history(study)
        pio.write_image(fig, os.path.join(vis_dir, "optimization_history.png"))
        
        # Importancia de parámetros
        fig = vis.plot_param_importances(study)
        pio.write_image(fig, os.path.join(vis_dir, "param_importances.png"))
        
        # Coordenadas paralelas
        fig = vis.plot_parallel_coordinate(study)
        pio.write_image(fig, os.path.join(vis_dir, "parallel_coordinate.png"))
        
        # Gráfico de intervalo de confianza
        fig = vis.plot_slice(study)
        pio.write_image(fig, os.path.join(vis_dir, "slice_plot.png"))
        
        logger.info(f"Visualizaciones guardadas en {vis_dir}")
        
    except ImportError as e:
        logger.warning(f"No se pudieron generar visualizaciones: {e}")
        logger.warning("Instala plotly y optuna con 'pip install plotly' para generar visualizaciones")

def main():
    """Función principal para ejecutar la optimización"""
    parser = argparse.ArgumentParser(description="Optimización de hiperparámetros para trading con RL")
    parser.add_argument("--config", type=str, default="src/scripts/config.json", help="Ruta al archivo de configuración")
    parser.add_argument("--trials", type=int, help="Número de trials para la optimización (anula la configuración)")
    args = parser.parse_args()
    
    try:
        # Cargar configuración
        logger.info(f"Cargando configuración desde {args.config}")
        config = load_config(args.config)
        
        # Anular número de trials si se especifica
        if args.trials:
            config["optimization"]["n_trials"] = args.trials
            logger.info(f"Anulando número de trials: {args.trials}")
            
        # Crear directorios necesarios
        os.makedirs(config.get("data_dir", "data"), exist_ok=True)
        os.makedirs(config.get("models_dir", "models"), exist_ok=True)
        os.makedirs(config.get("results_dir", "results"), exist_ok=True)
        
        # Cargar datos
        data = load_data(config)
        
        # Procesar características
        processed_data, features = process_features(data, config)
        
        # Guardar lista de características
        results_dir = config.get("results_dir", "results")
        features_file = os.path.join(results_dir, "features.json")
        with open(features_file, 'w') as f:
            json.dump(features, f, indent=4)
        logger.info(f"Lista de características guardada en {features_file}")
        
        # Preparar datos para optimización
        data_1d, data_1h, data_15m = prepare_data_for_optimization(processed_data)
        
        # Ejecutar optimización
        best_params = run_optimization(data_1d, data_1h, data_15m, features, config)
        
        # Visualizar resultados
        output_dir = os.path.join(config.get("results_dir", "results"), "optimization")
        plot_optimization_results(output_dir)
        
        logger.info("Proceso de optimización completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante la optimización: {e}", exc_info=True)
        raise
        
if __name__ == "__main__":
    main()