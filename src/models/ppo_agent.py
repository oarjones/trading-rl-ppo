import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Union, Any
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from collections import deque

# Configuración de PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOMemory:
    """Memoria para almacenar las experiencias durante el entrenamiento de PPO"""
    
    def __init__(self, batch_size: int):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.asset_ids = []
        self.account_states = []
        
        self.batch_size = batch_size
        
    def store_memory(self, state, action, probs, vals, reward, done):
        """Almacena una experiencia en memoria"""
        self.states.append(state['market_data'])
        self.asset_ids.append(state['asset_id'])
        self.account_states.append(state['account_state'])
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear_memory(self):
        """Limpia toda la memoria"""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.asset_ids = []
        self.account_states = []
        
    def generate_batches(self):
        """Genera batches de experiencias para el entrenamiento"""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_probs = np.array(self.probs)
        vals = np.array(self.vals)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        asset_ids = np.array(self.asset_ids)
        account_states = np.array(self.account_states)
        
        return states, actions, old_probs, vals, rewards, dones, batches, asset_ids, account_states

class ActorNetwork(nn.Module):
    """Red neuronal para el actor (política) del PPO"""
    
    def __init__(
        self, 
        input_dims: Tuple[int, int],  # (window_size, n_features)
        n_assets: int,
        n_actions: int,
        hidden_dims: List[int] = [512, 256, 128]
    ):
        super(ActorNetwork, self).__init__()
        
        # Dimensiones de entrada
        self.window_size = input_dims[0]
        self.n_features = input_dims[1]
        self.n_assets = n_assets
        
        # CORRECCIÓN: Guardar número de características para verificación
        self._expected_features = self.n_features
        
        # Capas convolucionales para procesar secuencias temporales
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.n_features, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Reduce la dimensión temporal a 1
        )
        
        # Red para procesar la identidad del activo
        self.asset_encoder = nn.Sequential(
            nn.Linear(n_assets, 32),
            nn.ReLU()
        )
        
        # Red para procesar el estado de la cuenta
        self.account_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )
        
        # Capas totalmente conectadas para combinar todas las características
        fc_input_dim = 128 + 32 + 32  # Conv output + asset encoder + account encoder
        
        fc_layers = []
        prev_dim = fc_input_dim
        
        for dim in hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.ReLU())
            prev_dim = dim
            
        # Capa de salida
        fc_layers.append(nn.Linear(prev_dim, n_actions))
        fc_layers.append(nn.Softmax(dim=-1))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self, 
                market_data: torch.Tensor, 
                asset_id: torch.Tensor, 
                account_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red del actor
        
        Args:
            market_data: Tensor de forma (batch_size, window_size, n_features)
            asset_id: Tensor de forma (batch_size, n_assets)
            account_state: Tensor de forma (batch_size, 3)
            
        Returns:
            Tensor de forma (batch_size, n_actions) con las probabilidades de cada acción
        """
        # CORRECCIÓN: Verificar y adaptar dimensiones si son diferentes
        current_features = market_data.shape[2]
        if current_features != self._expected_features:
            print(f"ADVERTENCIA: Número de características diferente. Esperado: {self._expected_features}, Actual: {current_features}")
            
            # Dos opciones: o reconstruir las capas convolucionales o adaptar el tensor de entrada
            # Opción 1: Adaptar el tensor (más rápido)
            if current_features < self._expected_features:
                # Rellenar con ceros hasta el número esperado de características
                padding = torch.zeros(market_data.shape[0], market_data.shape[1], 
                                     self._expected_features - current_features, 
                                     device=market_data.device)
                market_data = torch.cat([market_data, padding], dim=2)
            else:
                # Recortar al número esperado de características
                market_data = market_data[:, :, :self._expected_features]
        
        # Procesar datos de mercado con convoluciones
        # Reordenar dimensiones para Conv1d (batch, channels, seq_len)
        x1 = market_data.permute(0, 2, 1)
        x1 = self.conv_layers(x1)
        x1 = x1.view(x1.size(0), -1)  # Aplanar
        
        # Procesar ID del activo
        x2 = self.asset_encoder(asset_id)
        
        # Procesar estado de la cuenta
        x3 = self.account_encoder(account_state)
        
        # Concatenar todas las características
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Pasar por capas totalmente conectadas
        action_probs = self.fc_layers(x)
        
        return action_probs

class CriticNetwork(nn.Module):
    """Red neuronal para el crítico (función de valor) del PPO"""
    
    def __init__(
        self, 
        input_dims: Tuple[int, int],  # (window_size, n_features)
        n_assets: int,
        hidden_dims: List[int] = [512, 256, 128]
    ):
        super(CriticNetwork, self).__init__()
        
        # Dimensiones de entrada
        self.window_size = input_dims[0]
        self.n_features = input_dims[1]
        self.n_assets = n_assets
        
        # CORRECCIÓN: Guardar número de características para verificación
        self._expected_features = self.n_features
        
        # Capas convolucionales para procesar secuencias temporales
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.n_features, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Reduce la dimensión temporal a 1
        )
        
        # Red para procesar la identidad del activo
        self.asset_encoder = nn.Sequential(
            nn.Linear(n_assets, 32),
            nn.ReLU()
        )
        
        # Red para procesar el estado de la cuenta
        self.account_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )
        
        # Capas totalmente conectadas para combinar todas las características
        fc_input_dim = 128 + 32 + 32  # Conv output + asset encoder + account encoder
        
        fc_layers = []
        prev_dim = fc_input_dim
        
        for dim in hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.ReLU())
            prev_dim = dim
            
        # Capa de salida - valor escalar
        fc_layers.append(nn.Linear(prev_dim, 1))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self, 
                market_data: torch.Tensor, 
                asset_id: torch.Tensor, 
                account_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red del crítico
        
        Args:
            market_data: Tensor de forma (batch_size, window_size, n_features)
            asset_id: Tensor de forma (batch_size, n_assets)
            account_state: Tensor de forma (batch_size, 3)
            
        Returns:
            Tensor de forma (batch_size, 1) con el valor estimado
        """
        # CORRECCIÓN: Verificar y adaptar dimensiones si son diferentes
        current_features = market_data.shape[2]
        if current_features != self._expected_features:
            print(f"ADVERTENCIA: Número de características diferente. Esperado: {self._expected_features}, Actual: {current_features}")
            
            # Adaptar el tensor de entrada
            if current_features < self._expected_features:
                # Rellenar con ceros hasta el número esperado de características
                padding = torch.zeros(market_data.shape[0], market_data.shape[1], 
                                     self._expected_features - current_features, 
                                     device=market_data.device)
                market_data = torch.cat([market_data, padding], dim=2)
            else:
                # Recortar al número esperado de características
                market_data = market_data[:, :, :self._expected_features]
        
        # Procesar datos de mercado con convoluciones
        # Reordenar dimensiones para Conv1d (batch, channels, seq_len)
        x1 = market_data.permute(0, 2, 1)
        x1 = self.conv_layers(x1)
        x1 = x1.view(x1.size(0), -1)  # Aplanar
        
        # Procesar ID del activo
        x2 = self.asset_encoder(asset_id)
        
        # Procesar estado de la cuenta
        x3 = self.account_encoder(account_state)
        
        # Concatenar todas las características
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Pasar por capas totalmente conectadas
        value = self.fc_layers(x)
        
        return value


class PPOAgent:
    """Agente PPO para trading"""
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        batch_size: int = 64,
        n_epochs: int = 10,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        actor_hidden_dims: List[int] = [512, 256, 128],
        critic_hidden_dims: List[int] = [512, 256, 128],
        checkpoint_dir: str = 'models'
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Obtener la forma de las observaciones del entorno
        sample_obs = env.observation_space.sample()
        market_data_dims = sample_obs['market_data'].shape
        n_assets = sample_obs['asset_id'].shape[0]
        n_actions = env.action_space.n
        
        # Crear redes de actor y crítico
        self.actor = ActorNetwork(
            input_dims=market_data_dims,
            n_assets=n_assets,
            n_actions=n_actions,
            hidden_dims=actor_hidden_dims
        ).to(device)
        
        self.critic = CriticNetwork(
            input_dims=market_data_dims,
            n_assets=n_assets,
            hidden_dims=critic_hidden_dims
        ).to(device)
        
        # Optimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Memoria para almacenar experiencias
        self.memory = PPOMemory(batch_size)
        
        # Directorio para guardar checkpoints del modelo
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Métricas de entrenamiento
        self.training_metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'total_loss': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'final_balance': []
        }
        
    def remember(self, state, action, probs, vals, reward, done):
        """Almacena una experiencia en memoria"""
        self.memory.store_memory(state, action, probs, vals, reward, done)
        
    def save_models(self, episode: int):
        """Guarda los modelos del actor y crítico"""
        actor_path = os.path.join(self.checkpoint_dir, f'actor_ep{episode}.pth')
        critic_path = os.path.join(self.checkpoint_dir, f'critic_ep{episode}.pth')
        
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
        print(f"Modelos guardados en {self.checkpoint_dir}")
        
    def load_models(self, actor_path: str, critic_path: str):
        """Carga los modelos del actor y crítico"""
        self.actor.load_state_dict(torch.load(actor_path, map_location=device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=device))
        
        print(f"Modelos cargados: {actor_path}, {critic_path}")
        
    def choose_action(self, observation):
        """Selecciona una acción basada en la política actual"""
        market_data = torch.tensor(observation['market_data'], dtype=torch.float).unsqueeze(0).to(device)
        asset_id = torch.tensor(observation['asset_id'], dtype=torch.float).unsqueeze(0).to(device)
        account_state = torch.tensor(observation['account_state'], dtype=torch.float).unsqueeze(0).to(device)
        
        # Obtener distribución de probabilidad sobre acciones
        dist = self.actor(market_data, asset_id, account_state)
        
        # Obtener valor estimado
        value = self.critic(market_data, asset_id, account_state)
        
        # Muestrear acción de la distribución
        dist = Categorical(dist)
        action = dist.sample()
        
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        
        return action, probs, value
    
    def learn(self):
        """Actualiza las redes del actor y crítico usando PPO"""
        # Obtener experiencias de la memoria
        states, actions, old_log_probs, vals, rewards, dones, batches, asset_ids, account_states = \
            self.memory.generate_batches()
        
        # Convertir a tensores
        states = torch.tensor(states, dtype=torch.float).to(device)
        asset_ids = torch.tensor(asset_ids, dtype=torch.float).to(device)
        account_states = torch.tensor(account_states, dtype=torch.float).to(device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float).to(device)
        vals = torch.tensor(vals, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        
        # Calcular ventajas usando GAE (Generalized Advantage Estimation)
        advantages = self._compute_gae(rewards, vals, dones)
        advantages = torch.tensor(advantages, dtype=torch.float).to(device)
        
        # Normalizar ventajas
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calcular returns para el crítico
        returns = advantages + vals
        
        # Actualizar redes durante n_epochs
        for _ in range(self.n_epochs):
            # Para cada batch
            for batch in batches:
                # Obtener distribución de probabilidad sobre acciones
                batch_states = states[batch]
                batch_asset_ids = asset_ids[batch]
                batch_account_states = account_states[batch]
                batch_actions = actions[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = returns[batch]
                
                # Forward pass del actor
                action_probs = self.actor(batch_states, batch_asset_ids, batch_account_states)
                dist = Categorical(action_probs)
                
                # Nuevos log probs
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calcular ratio para PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Valor estimado por el crítico
                values = self.critic(batch_states, batch_asset_ids, batch_account_states).squeeze()
                
                # Función de pérdida del crítico (MSE)
                critic_loss = ((values - batch_returns) ** 2).mean()
                
                # Término de entropía para fomentar la exploración
                entropy = dist.entropy().mean()
                
                # Pérdida total
                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Backpropagation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Guardar métricas de entrenamiento
                self.training_metrics['actor_loss'].append(actor_loss.item())
                self.training_metrics['critic_loss'].append(critic_loss.item())
                self.training_metrics['entropy'].append(entropy.item())
                self.training_metrics['total_loss'].append(total_loss.item())
        
        # Limpiar memoria después de aprender
        self.memory.clear_memory()
        
    def _compute_gae(self, rewards, values, dones):
        """Calcula ventajas usando GAE (Generalized Advantage Estimation)"""
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        last_advantage = 0
        
        # Iterar desde el final hasta el principio
        for t in reversed(range(n_steps - 1)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage * mask
            last_advantage = advantages[t]
            
        return advantages
    
    def train(self, n_episodes: int, max_steps_per_episode: int = 1000, save_freq: int = 10):
        """Entrena al agente durante un número dado de episodios"""
        for episode in range(n_episodes):
            # Reiniciar el entorno
            observation, _ = self.env.reset()
            done = False
            score = 0
            step_count = 0
            
            while not done and step_count < max_steps_per_episode:
                # Elegir acción
                action, prob, val = self.choose_action(observation)
                
                # Ejecutar acción en el entorno
                next_observation, reward, done, _, info = self.env.step(action)
                
                # Almacenar experiencia
                self.remember(observation, action, prob, val, reward, done)
                
                # Actualizar estado actual
                observation = next_observation
                
                # Acumular recompensa
                score += reward
                step_count += 1
                
                # Si hemos acumulado suficientes experiencias, aprender
                if step_count % self.memory.batch_size == 0 and step_count > 0:
                    self.learn()
            
            # Si no hemos aprendido en este episodio, hacerlo ahora
            if len(self.memory.states) > 0:
                self.learn()
                
            # Guardar métricas del episodio
            self.training_metrics['episode_rewards'].append(score)
            self.training_metrics['episode_lengths'].append(step_count)
            self.training_metrics['final_balance'].append(info['balance'])
            
            # Imprimir progreso
            print(f"Episodio {episode+1}/{n_episodes}, Recompensa: {score:.2f}, Pasos: {step_count}, Balance Final: {info['balance']:.2f}")
            
            # Guardar modelos periódicamente
            if (episode + 1) % save_freq == 0:
                self.save_models(episode + 1)
                
        # Guardar modelos finales
        self.save_models(n_episodes)
        
        return self.training_metrics
    
    def evaluate(self, n_episodes: int, render: bool = False):
        """Evalúa el agente sin exploración"""
        total_rewards = []
        final_balances = []
        
        for episode in range(n_episodes):
            # Reiniciar el entorno
            observation, _ = self.env.reset()
            done = False
            score = 0
            step_count = 0
            
            while not done:
                # Elegir mejor acción (sin exploración)
                market_data = torch.tensor(observation['market_data'], dtype=torch.float).unsqueeze(0).to(device)
                asset_id = torch.tensor(observation['asset_id'], dtype=torch.float).unsqueeze(0).to(device)
                account_state = torch.tensor(observation['account_state'], dtype=torch.float).unsqueeze(0).to(device)
                
                # Obtener distribución de probabilidad sobre acciones
                dist = self.actor(market_data, asset_id, account_state)
                
                # Elegir la acción con mayor probabilidad
                action = torch.argmax(dist, dim=1).item()
                
                # Ejecutar acción en el entorno
                next_observation, reward, done, _, info = self.env.step(action)
                
                # Actualizar estado actual
                observation = next_observation
                
                # Acumular recompensa
                score += reward
                step_count += 1
            
            # Renderizar episodio si se solicita
            if render:
                self.env.render()
                
            # Guardar métricas
            total_rewards.append(score)
            final_balances.append(info['balance'])
            
            # Imprimir progreso
            print(f"Evaluación {episode+1}/{n_episodes}, Recompensa: {score:.2f}, Pasos: {step_count}, Balance Final: {info['balance']:.2f}")
            
        # Resultados promedio
        avg_reward = np.mean(total_rewards)
        avg_balance = np.mean(final_balances)
        
        print(f"Evaluación completada. Recompensa promedio: {avg_reward:.2f}, Balance final promedio: {avg_balance:.2f}")
        
        return {
            'rewards': total_rewards,
            'final_balances': final_balances,
            'avg_reward': avg_reward,
            'avg_balance': avg_balance
        }
        
    def plot_training_metrics(self):
        """Visualiza las métricas de entrenamiento"""
        # Crear figura
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Graficar recompensas por episodio
        axs[0, 0].plot(self.training_metrics['episode_rewards'])
        axs[0, 0].set_title('Recompensas por Episodio')
        axs[0, 0].set_xlabel('Episodio')
        axs[0, 0].set_ylabel('Recompensa Total')
        
        # Graficar balance final por episodio
        axs[0, 1].plot(self.training_metrics['final_balance'])
        axs[0, 1].set_title('Balance Final por Episodio')
        axs[0, 1].set_xlabel('Episodio')
        axs[0, 1].set_ylabel('Balance ($)')
        
        # Graficar pérdidas
        axs[1, 0].plot(self.training_metrics['actor_loss'], label='Actor')
        axs[1, 0].plot(self.training_metrics['critic_loss'], label='Crítico')
        axs[1, 0].set_title('Pérdidas durante Entrenamiento')
        axs[1, 0].set_xlabel('Actualización')
        axs[1, 0].set_ylabel('Pérdida')
        axs[1, 0].legend()
        
        # Graficar entropía
        axs[1, 1].plot(self.training_metrics['entropy'])
        axs[1, 1].set_title('Entropía durante Entrenamiento')
        axs[1, 1].set_xlabel('Actualización')
        axs[1, 1].set_ylabel('Entropía')
        
        plt.tight_layout()
        plt.show()

# Ejemplo de uso:
# if __name__ == "__main__":
#     # Crear entorno (ya definido previamente)
    
#     # Inicializar agente PPO
#     agent = PPOAgent(
#         env=env,
#         learning_rate=3e-4,
#         gamma=0.99,
#         gae_lambda=0.95,
#         policy_clip=0.2,
#         batch_size=64,
#         n_epochs=10,
#         entropy_coef=0.01,
#         value_coef=0.5
#     )
    
#     # Entrenar agente
#     training_metrics = agent.train(n_episodes=100, save_freq=20)
    
#     # Visualizar métricas de entrenamiento
#     agent.plot_training_metrics()
    
#     # Evaluar agente
#     eval_results = agent.evaluate(n_episodes=10, render=True)