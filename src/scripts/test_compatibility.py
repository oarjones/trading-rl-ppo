#!/usr/bin/env python
"""
Script para verificar la compatibilidad entre el entorno TradingEnv y el agente PPO
Utiliza datos sintéticos para realizar una prueba rápida y sencilla.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Importar componentes del sistema
from src.models.trading_env import TradingEnv
from src.models.ppo_agent import PPOAgent
from src.models.feature_engineering import FeatureEngineering

# Importar funciones de verificación (del artefacto - asumimos que se han copiado a este archivo)
# Si las funciones no están en un archivo importable, cópie aquí el contenido del artefacto "Script de Verificación de Dimensiones"
# Para este ejemplo, asumo que se ha creado un archivo verification_utils.py con las funciones
from verification_utils import verify_environment_dimensions, verify_agent_dimensions, perform_compatibility_check

def generate_synthetic_data(symbols, days=100):
    """
    Genera datos sintéticos para pruebas
    
    Args:
        symbols: Lista de símbolos
        days: Número de días de datos a generar
        
    Returns:
        Diccionario con datos por símbolo y timeframe
    """
    data = {}
    
    # Fecha base
    base_date = datetime.now() - timedelta(days=days)
    
    for symbol in symbols:
        data[symbol] = {}
        
        # Generar datos diarios
        dates_1d = [base_date + timedelta(days=i) for i in range(days)]
        data_1d = pd.DataFrame({
            'Open': np.random.normal(100, 10, size=days),
            'High': np.random.normal(105, 10, size=days),
            'Low': np.random.normal(95, 10, size=days),
            'Close': np.random.normal(102, 10, size=days),
            'Volume': np.random.normal(1000000, 200000, size=days)
        }, index=dates_1d)
        
        # Generar datos horarios (8 horas por día)
        hours_per_day = 8  # Solo horas de mercado
        dates_1h = []
        for day in range(days):
            for hour in range(hours_per_day):
                dates_1h.append(base_date + timedelta(days=day, hours=hour))
        
        data_1h = pd.DataFrame({
            'Open': np.random.normal(100, 5, size=days * hours_per_day),
            'High': np.random.normal(102, 5, size=days * hours_per_day),
            'Low': np.random.normal(98, 5, size=days * hours_per_day),
            'Close': np.random.normal(101, 5, size=days * hours_per_day),
            'Volume': np.random.normal(100000, 20000, size=days * hours_per_day)
        }, index=dates_1h)
        
        # Generar datos de 15 minutos (4 por hora)
        mins_per_hour = 4
        dates_15m = []
        for day in range(days):
            for hour in range(hours_per_day):
                for minute in range(mins_per_hour):
                    dates_15m.append(base_date + timedelta(days=day, hours=hour, minutes=minute*15))
        
        data_15m = pd.DataFrame({
            'Open': np.random.normal(100, 3, size=days * hours_per_day * mins_per_hour),
            'High': np.random.normal(101, 3, size=days * hours_per_day * mins_per_hour),
            'Low': np.random.normal(99, 3, size=days * hours_per_day * mins_per_hour),
            'Close': np.random.normal(100.5, 3, size=days * hours_per_day * mins_per_hour),
            'Volume': np.random.normal(25000, 5000, size=days * hours_per_day * mins_per_hour)
        }, index=dates_15m)
        
        # Añadir los datos al diccionario
        data[symbol]['1D'] = data_1d
        data[symbol]['1H'] = data_1h
        data[symbol]['15M'] = data_15m
    
    return data

def main():
    """Función principal"""
    print("=== INICIANDO PRUEBA DE COMPATIBILIDAD ===")
    
    # Configuraciones
    window_size = 30
    commission = 0.001
    initial_balance = 10000
    
    # Generar datos sintéticos
    symbols = ['AAPL', 'INTC']
    print(f"Generando datos sintéticos para {symbols}...")
    data = generate_synthetic_data(symbols)
    
    # Procesar datos con FeatureEngineering
    print("Procesando indicadores técnicos...")
    feature_engineering = FeatureEngineering()
    
    processed_data = {}
    features = []
    
    for symbol in symbols:
        processed_data[symbol] = {}
        
        # Procesar solo datos de 1H para indicadores
        features_1h = feature_engineering.calculate_indicators(data[symbol]['1H'], '1h')
        
        # Mantener los demás datos sin procesar
        processed_data[symbol]['1D'] = data[symbol]['1D']
        processed_data[symbol]['1H'] = features_1h
        processed_data[symbol]['15M'] = data[symbol]['15M']
        
        # Obtener lista de características
        if not features:
            features = list(features_1h.columns)
    
    # Preparar datos para el entorno
    data_1d = {symbol: processed_data[symbol]['1D'] for symbol in symbols}
    data_1h = {symbol: processed_data[symbol]['1H'] for symbol in symbols}
    data_15m = {symbol: processed_data[symbol]['15M'] for symbol in symbols}
    
    # Crear entorno
    print("\nCreando entorno de trading...")
    env = TradingEnv(
        data_1d=data_1d,
        data_1h=data_1h,
        data_15m=data_15m,
        features=features,
        window_size=window_size,
        commission=commission,
        initial_balance=initial_balance
    )
    
    # Crear agente PPO
    print("Creando agente PPO...")
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        entropy_coef=0.01,
        value_coef=0.5,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64]
    )
    
    # Realizar verificación de compatibilidad
    success = perform_compatibility_check(env, agent)
    
    if success:
        print("\n✅ VERIFICACIÓN EXITOSA: El entorno y el agente son compatibles.")
        print("Puedes proceder con el entrenamiento.")
    else:
        print("\n❌ VERIFICACIÓN FALLIDA: Se detectaron problemas de compatibilidad.")
        print("Revisa los mensajes de error anteriores para más detalles.")

if __name__ == "__main__":
    main()