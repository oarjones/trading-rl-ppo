import os
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, List, Tuple
import gymnasium as gym
import joblib
import json
import time
from datetime import datetime

# Importar las clases definidas anteriormente
# Asume que están en los módulos correspondientes
from src.models.trading_env import TradingEnv
from src.models.feature_engineering import FeatureEngineering
from src.models.ppo_agent import PPOAgent

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationManager:
    """Gestiona la optimización de hiperparámetros con Optuna"""
    
    def __init__(
        self,
        data_1d: Dict[str, pd.DataFrame],
        data_1h: Dict[str, pd.DataFrame],
        data_15m: Dict[str, pd.DataFrame],
        features: List[str],
        study_name: str = "ppo_trading_optimization",
        n_trials: int = 100,
        n_startup_trials: int = 10,
        n_evaluations: int = 5,
        n_episodes_per_eval: int = 10,
        max_episode_steps: int = 1000,
        output_dir: str = "optimization_results"
    ):
        self.data_1d = data_1d
        self.data_1h = data_1h
        self.data_15m = data_15m
        self.features = features
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.n_evaluations = n_evaluations
        self.n_episodes_per_eval = n_episodes_per_eval
        self.max_episode_steps = max_episode_steps
        self.output_dir = output_dir
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Dispositivo para PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Inicializar estudio de Optuna
        self.sampler = TPESampler(n_startup_trials=n_startup_trials)
        self.pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=5)
        
    def _create_env(self, config: Dict[str, Any]) -> TradingEnv:
        """Crea un entorno de trading con la configuración dada"""
        env = TradingEnv(
            data_1d=self.data_1d,
            data_1h=self.data_1h,
            data_15m=self.data_15m,
            features=self.features,
            window_size=config["window_size"],
            commission=config["commission"],
            initial_balance=config["initial_balance"],
            reward_window=config["reward_window"],
            min_price_change=config["min_price_change"]
        )
        return env
    
    def _create_agent(self, env: gym.Env, config: Dict[str, Any]) -> PPOAgent:
        """Crea un agente PPO con la configuración dada"""
        agent = PPOAgent(
            env=env,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            policy_clip=config["policy_clip"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            entropy_coef=config["entropy_coef"],
            value_coef=config["value_coef"],
            actor_hidden_dims=config["actor_hidden_dims"],
            critic_hidden_dims=config["critic_hidden_dims"],
            checkpoint_dir=os.path.join(self.output_dir, "checkpoints")
        )
        return agent
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Muestrea hiperparámetros para un trial de Optuna"""
        # Hiperparámetros del entorno
        env_config = {
            "window_size": trial.suggest_int("window_size", 20, 60),
            "commission": trial.suggest_float("commission", 0.0005, 0.003),
            "initial_balance": trial.suggest_float("initial_balance", 10000, 100000),
            "reward_window": trial.suggest_int("reward_window", 10, 30),
            "min_price_change": trial.suggest_float("min_price_change", 0.02, 0.1)
        }
        
        # Hiperparámetros del agente PPO
        agent_config = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.999),
            "policy_clip": trial.suggest_float("policy_clip", 0.1, 0.3),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "n_epochs": trial.suggest_int("n_epochs", 3, 15),
            "entropy_coef": trial.suggest_float("entropy_coef", 0.001, 0.05),
            "value_coef": trial.suggest_float("value_coef", 0.1, 1.0)
        }
        
        # Hiperparámetros de la arquitectura de red
        n_layers = trial.suggest_int("n_layers", 2, 4)
        actor_hidden_dims = []
        critic_hidden_dims = []
        
        for i in range(n_layers):
            # Dimensiones de capas ocultas para el actor
            actor_dim = trial.suggest_categorical(f"actor_hidden_dim_{i}", [64, 128, 256, 512])
            actor_hidden_dims.append(actor_dim)
            
            # Dimensiones de capas ocultas para el crítico
            critic_dim = trial.suggest_categorical(f"critic_hidden_dim_{i}", [64, 128, 256, 512])
            critic_hidden_dims.append(critic_dim)
        
        agent_config["actor_hidden_dims"] = actor_hidden_dims
        agent_config["critic_hidden_dims"] = critic_hidden_dims
        
        # Combinar todas las configuraciones
        config = {**env_config, **agent_config}
        
        return config
    
    def _evaluate_model(self, agent: PPOAgent, n_episodes: int = 10) -> Dict[str, float]:
        """Evalúa un modelo en múltiples episodios"""
        eval_results = agent.evaluate(n_episodes=n_episodes)
        
        return {
            "mean_reward": eval_results["avg_reward"],
            "mean_balance": eval_results["avg_balance"],
            "max_balance": np.max(eval_results["final_balances"]),
            "min_balance": np.min(eval_results["final_balances"]),
            "std_balance": np.std(eval_results["final_balances"])
        }
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Función objetivo para Optuna"""
        # Muestrear hiperparámetros
        config = self._sample_hyperparameters(trial)
        
        # Crear entorno y agente
        env = self._create_env(config)
        agent = self._create_agent(env, config)
        
        # Lista para almacenar métricas de evaluación
        evaluation_results = []
        
        # Entrenar y evaluar múltiples veces para reducir la varianza
        for eval_idx in range(self.n_evaluations):
            # Entrenar el agente
            logger.info(f"Trial {trial.number}, Evaluación {eval_idx+1}/{self.n_evaluations}: Entrenando agente...")
            training_metrics = agent.train(
                n_episodes=self.n_episodes_per_eval,
                max_steps_per_episode=self.max_episode_steps,
                save_freq=self.n_episodes_per_eval  # Solo guardar al final de cada evaluación
            )
            
            # Evaluar el agente
            logger.info(f"Trial {trial.number}, Evaluación {eval_idx+1}/{self.n_evaluations}: Evaluando agente...")
            eval_metrics = self._evaluate_model(agent, n_episodes=5)
            evaluation_results.append(eval_metrics)
            
            # Reportar valores intermedios para la poda
            mean_reward = eval_metrics["mean_reward"]
            mean_balance = eval_metrics["mean_balance"]
            
            trial.report(mean_balance, eval_idx)
            
            # Comprobar si el trial debe ser podado
            if trial.should_prune():
                logger.info(f"Trial {trial.number} podado.")
                raise optuna.TrialPruned()
        
        # Calcular métricas promedio de todas las evaluaciones
        avg_metrics = {
            key: np.mean([res[key] for res in evaluation_results])
            for key in evaluation_results[0].keys()
        }
        
        # Guardar resultados del trial
        self._save_trial_results(trial.number, config, training_metrics, avg_metrics)
        
        # La métrica objetivo es el balance final promedio
        objective_value = avg_metrics["mean_balance"]
        
        logger.info(f"Trial {trial.number} completado. Valor objetivo: {objective_value:.2f}")
        
        return objective_value
    
    def _save_trial_results(self, trial_number: int, config: Dict[str, Any], 
                          training_metrics: Dict[str, List[float]], 
                          eval_metrics: Dict[str, float]):
        """Guarda los resultados de un trial"""
        # Crear directorio para el trial si no existe
        trial_dir = os.path.join(self.output_dir, f"trial_{trial_number}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Guardar configuración
        with open(os.path.join(trial_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        # Guardar métricas de entrenamiento
        training_df = pd.DataFrame({
            "episode_rewards": training_metrics["episode_rewards"],
            "final_balance": training_metrics["final_balance"]
        })
        training_df.to_csv(os.path.join(trial_dir, "training_metrics.csv"), index=False)
        
        # Guardar métricas de evaluación
        with open(os.path.join(trial_dir, "eval_metrics.json"), "w") as f:
            json.dump(eval_metrics, f, indent=4)
    
    def run_optimization(self) -> optuna.Study:
        """Ejecuta la optimización de hiperparámetros"""
        # Crear o cargar estudio
        study_path = os.path.join(self.output_dir, f"{self.study_name}.db")
        study = optuna.create_study(
            study_name=self.study_name,
            storage=f"sqlite:///{study_path}",
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True
        )
        
        # Ejecutar optimización
        start_time = time.time()
        logger.info(f"Iniciando optimización con {self.n_trials} trials...")
        
        study.optimize(self._objective, n_trials=self.n_trials, timeout=None, 
                     show_progress_bar=True)
        
        # Calcular tiempo total
        total_time = time.time() - start_time
        logger.info(f"Optimización completada en {total_time/3600:.2f} horas.")
        
        # Guardar y visualizar resultados
        self._save_optimization_results(study)
        
        return study
    
    def _save_optimization_results(self, study: optuna.Study):
        """Guarda y visualiza los resultados de la optimización"""
        # Guardar estudio
        joblib.dump(study, os.path.join(self.output_dir, f"{self.study_name}.pkl"))
        
        # Obtener mejores hiperparámetros
        best_params = study.best_params
        best_value = study.best_value
        
        # Guardar mejores hiperparámetros
        with open(os.path.join(self.output_dir, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)
            
        logger.info(f"Mejor valor objetivo: {best_value:.2f}")
        logger.info(f"Mejores hiperparámetros: {best_params}")
        
        # Visualizar importancia de hiperparámetros
        try:
            param_importance = optuna.visualization.plot_param_importances(study)
            param_importance.write_image(os.path.join(self.output_dir, "param_importances.png"))
            
            optimization_history = optuna.visualization.plot_optimization_history(study)
            optimization_history.write_image(os.path.join(self.output_dir, "optimization_history.png"))
            
            parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study)
            parallel_coordinate.write_image(os.path.join(self.output_dir, "parallel_coordinate.png"))
        except Exception as e:
            logger.warning(f"No se pudieron generar las visualizaciones: {e}")
    
    def train_best_model(self, n_episodes: int = 200) -> PPOAgent:
        """Entrena un modelo con los mejores hiperparámetros"""
        # Cargar estudio
        study_path = os.path.join(self.output_dir, f"{self.study_name}.pkl")
        if not os.path.exists(study_path):
            raise FileNotFoundError(f"No se encontró el estudio en {study_path}. Ejecuta run_optimization primero.")
            
        study = joblib.load(study_path)
        best_params = study.best_params
        
        # Reconstruir la configuración completa
        config = {}
        
        # Hiperparámetros del entorno
        env_params = ["window_size", "commission", "initial_balance", "reward_window", "min_price_change"]
        for param in env_params:
            if param in best_params:
                config[param] = best_params[param]
                
        # Hiperparámetros del agente
        agent_params = ["learning_rate", "gamma", "gae_lambda", "policy_clip", 
                        "batch_size", "n_epochs", "entropy_coef", "value_coef"]
        for param in agent_params:
            if param in best_params:
                config[param] = best_params[param]
                
        # Reconstruir arquitectura de la red
        n_layers = best_params.get("n_layers", 3)
        actor_hidden_dims = []
        critic_hidden_dims = []
        
        for i in range(n_layers):
            actor_dim_key = f"actor_hidden_dim_{i}"
            critic_dim_key = f"critic_hidden_dim_{i}"
            
            if actor_dim_key in best_params:
                actor_hidden_dims.append(best_params[actor_dim_key])
                
            if critic_dim_key in best_params:
                critic_hidden_dims.append(best_params[critic_dim_key])
                
        config["actor_hidden_dims"] = actor_hidden_dims
        config["critic_hidden_dims"] = critic_hidden_dims
        
        # Crear directorio para el mejor modelo
        best_model_dir = os.path.join(self.output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        
        # Crear entorno y agente
        env = self._create_env(config)
        agent = self._create_agent(env, config)
        
        # Entrenar el modelo
        logger.info(f"Entrenando el mejor modelo por {n_episodes} episodios...")
        training_metrics = agent.train(
            n_episodes=n_episodes,
            max_steps_per_episode=self.max_episode_steps,
            save_freq=50
        )
        
        # Evaluar el modelo final
        logger.info("Evaluando el modelo final...")
        eval_metrics = self._evaluate_model(agent, n_episodes=20)
        
        # Guardar resultados
        with open(os.path.join(best_model_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
            
        with open(os.path.join(best_model_dir, "eval_metrics.json"), "w") as f:
            json.dump(eval_metrics, f, indent=4)
            
        # Guardar gráficas de entrenamiento
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(training_metrics["episode_rewards"])
        plt.title("Recompensas por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa Total")
        
        plt.subplot(2, 2, 2)
        plt.plot(training_metrics["final_balance"])
        plt.title("Balance Final por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Balance ($)")
        
        plt.subplot(2, 2, 3)
        plt.plot(training_metrics["actor_loss"], label="Actor")
        plt.plot(training_metrics["critic_loss"], label="Crítico")
        plt.title("Pérdidas durante Entrenamiento")
        plt.xlabel("Actualización")
        plt.ylabel("Pérdida")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(training_metrics["entropy"])
        plt.title("Entropía durante Entrenamiento")
        plt.xlabel("Actualización")
        plt.ylabel("Entropía")
        
        plt.tight_layout()
        plt.savefig(os.path.join(best_model_dir, "training_metrics.png"))
        
        logger.info(f"Modelo final guardado en {best_model_dir}")
        logger.info(f"Métricas de evaluación final: {eval_metrics}")
        
        return agent

# Ejemplo de uso:
if __name__ == "__main__":
    # Supongamos que ya tenemos los datos cargados de IBKR
    data_1d, data_1h, data_15m = {...}, {...}, {...}
    
    # Lista de características
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal', 
                'MACD_histogram', 'EMA8', 'EMA21', 'EMA50', 'BB_middle', 'BB_upper', 
                'BB_lower', 'BB_width', 'ATR', 'Stoch_K', 'Stoch_D', 'OBV', 'ADX']
    
    # Inicializar el gestor de optimización
    optimization_manager = OptimizationManager(
        data_1d=data_1d,
        data_1h=data_1h,
        data_15m=data_15m,
        features=features,
        study_name="ppo_trading_optimization",
        n_trials=100,
        n_startup_trials=10,
        n_evaluations=3,
        n_episodes_per_eval=20,
        max_episode_steps=10000,
        output_dir="optimization_results"
    )
    
    # Ejecutar optimización
    study = optimization_manager.run_optimization()
    
    # Entrenar modelo final con los mejores hiperparámetros
    best_agent = optimization_manager.train_best_model(n_episodes=200)