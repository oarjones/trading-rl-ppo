"""
Utilidades para verificar la compatibilidad entre el entorno TradingEnv y el agente PPO.
Copie este código en un archivo llamado verification_utils.py
"""

import gymnasium as gym
import torch
import numpy as np

def verify_environment_dimensions(env):
    """
    Verifica las dimensiones del entorno y muestra información detallada
    
    Args:
        env: Instancia del entorno TradingEnv
    """
    print("\n=== VERIFICACIÓN DE DIMENSIONES DEL ENTORNO ===")
    
    # 1. Verificar espacio de observación
    obs_space = env.observation_space
    print(f"Espacio de observación: {obs_space}")
    
    # 2. Verificar espacio de acción
    action_space = env.action_space
    print(f"Espacio de acción: {action_space}")
    
    # 3. Obtener una observación
    obs, info = env.reset()
    print(f"\nTamaño de la observación 'market_data': {obs['market_data'].shape}")
    print(f"Tamaño de la observación 'account_state': {obs['account_state'].shape}")
    print(f"Tamaño de la observación 'asset_id': {obs['asset_id'].shape}")
    
    # 4. Verificar características
    print(f"\nCaracterísticas utilizadas ({len(env.hourly_features)}):")
    for i, feature in enumerate(env.hourly_features):
        print(f"  {i+1}. {feature}")
    
    # 5. Verificar alineación de datos
    print(f"\nColumnas en datos alineados ({len(env.data_aligned.columns)}):")
    for i, col in enumerate(env.data_aligned.columns):
        print(f"  {i+1}. {col}")
    
    return obs

def verify_agent_dimensions(agent, observation):
    """
    Verifica las dimensiones del agente y su compatibilidad con el entorno
    
    Args:
        agent: Instancia del agente PPO
        observation: Observación de ejemplo del entorno
    """
    print("\n=== VERIFICACIÓN DE DIMENSIONES DEL AGENTE ===")
    
    # 1. Verificar dimensiones de entrada de las redes
    market_data = observation['market_data']
    print(f"Actor - Ventana: {agent.actor.window_size}, Features: {agent.actor._expected_features}")
    print(f"Critic - Ventana: {agent.critic.window_size}, Features: {agent.critic._expected_features}")
    print(f"Observación market_data: {market_data.shape}")
    
    # 2. Probar forward pass
    print("\nProbando forward pass...")
    market_data_tensor = torch.tensor(market_data, dtype=torch.float32).unsqueeze(0)
    asset_id_tensor = torch.tensor(observation['asset_id'], dtype=torch.float32).unsqueeze(0)
    account_state_tensor = torch.tensor(observation['account_state'], dtype=torch.float32).unsqueeze(0)
    
    try:
        # Probar el actor
        action_probs = agent.actor(market_data_tensor, asset_id_tensor, account_state_tensor)
        print(f"Forward pass del actor exitoso. Forma de salida: {action_probs.shape}")
        
        # Probar el crítico
        value = agent.critic(market_data_tensor, asset_id_tensor, account_state_tensor)
        print(f"Forward pass del crítico exitoso. Forma de salida: {value.shape}")
        
        # Probar choose_action
        action, prob, val = agent.choose_action(observation)
        print(f"choose_action exitoso. Acción: {action}, Prob: {prob:.4f}, Valor: {val:.4f}")
        
        return True
    except Exception as e:
        print(f"Error en forward pass: {e}")
        return False

def perform_compatibility_check(env, agent):
    """
    Realiza una verificación completa de compatibilidad entre el entorno y el agente
    
    Args:
        env: Instancia del entorno TradingEnv
        agent: Instancia del agente PPO
    """
    print("\n====== VERIFICACIÓN DE COMPATIBILIDAD ======")
    
    # 1. Verificar dimensiones del entorno
    obs = verify_environment_dimensions(env)
    
    # 2. Verificar dimensiones del agente
    agent_ok = verify_agent_dimensions(agent, obs)
    
    # 3. Probar un episodio completo
    if agent_ok:
        print("\n=== PROBANDO UN EPISODIO COMPLETO ===")
        observation, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        try:
            while not done and steps < 100:  # Limitar a 100 pasos
                action, _, _ = agent.choose_action(observation)
                observation, reward, done, _, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if steps % 20 == 0:
                    print(f"Paso {steps}, Recompensa: {reward:.4f}, Balance: {info['balance']:.2f}")
            
            print(f"\nEpisodio completado. Pasos: {steps}, Recompensa total: {total_reward:.4f}")
            print(f"Balance final: {info['balance']:.2f}")
            return True
        except Exception as e:
            print(f"Error durante el episodio: {e}")
            return False
    else:
        print("\nNo se pudo completar la verificación debido a errores en el agente.")
        return False