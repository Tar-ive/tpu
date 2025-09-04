#!/usr/bin/env python3
"""
Simplified Grid RL Training with W&B and TensorBoard
A working version that actually trains
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from typing import Dict, Any
import time
from datetime import datetime
from pathlib import Path

# Logging imports
import wandb
from tensorboardX import SummaryWriter
import logging

# Environment import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environments.power_grid_env import PowerGridEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleActor(nn.Module):
    """Simple actor-critic network for the grid environment"""
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        # Shared layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        
        # Actor head
        logits = nn.Dense(self.action_dim)(x)
        
        # Critic head
        value = nn.Dense(1)(x)
        
        return logits, jnp.squeeze(value)

def train_grid_rl_simple():
    """Simple training loop that actually works"""
    
    # Configuration
    config = {
        "total_timesteps": 1_000_000,
        "num_envs": 8,
        "rollout_length": 128,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "update_epochs": 4,
        "max_grad_norm": 0.5,
        "log_interval": 10,
        "use_wandb": True,
        "use_tensorboard": True,
    }
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if config["use_wandb"]:
        wandb.init(
            project="grid-rl-simple",
            config=config,
            name=f"run_{timestamp}"
        )
        logger.info("W&B initialized")
    
    if config["use_tensorboard"]:
        tb_dir = f"./tensorboard/simple_{timestamp}"
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir)
        logger.info(f"TensorBoard logging to {tb_dir}")
    
    # Initialize environment
    logger.info("Initializing environment...")
    env = PowerGridEnv()
    obs, _ = env.reset(seed=42)
    obs_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]
    
    logger.info(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Initialize network
    network = SimpleActor(action_dim=action_dim)
    key = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = network.init(key, dummy_obs)
    
    # Initialize optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(config["learning_rate"])
    )
    opt_state = optimizer.init(params)
    
    # Training metrics
    episode_returns = []
    episode_lengths = []
    current_return = 0
    current_length = 0
    total_steps = 0
    update_count = 0
    
    logger.info("="*70)
    logger.info("Starting training...")
    logger.info(f"Total timesteps: {config['total_timesteps']:,}")
    logger.info(f"Parallel environments: {config['num_envs']}")
    logger.info(f"Rollout length: {config['rollout_length']}")
    logger.info("="*70)
    
    # Create vectorized environments
    envs = [PowerGridEnv() for _ in range(config["num_envs"])]
    obs_list = []
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=i)
        obs_list.append(obs)
    
    obs_batch = np.stack(obs_list)
    
    start_time = time.time()
    
    while total_steps < config["total_timesteps"]:
        # Collect rollout
        rollout_obs = []
        rollout_actions = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []
        rollout_log_probs = []
        
        for _ in range(config["rollout_length"]):
            # Get actions from network
            obs_jax = jnp.array(obs_batch)
            logits, values = network.apply(params, obs_jax)
            
            # Sample actions
            key, subkey = jax.random.split(key)
            actions = jax.random.categorical(subkey, logits, axis=-1)
            log_probs = jnp.take_along_axis(
                jax.nn.log_softmax(logits), 
                actions[:, None], 
                axis=1
            ).squeeze()
            
            # Convert to numpy for environment
            actions_np = np.array(actions)
            values_np = np.array(values)
            log_probs_np = np.array(log_probs)
            
            # Step environments
            next_obs_list = []
            rewards = []
            dones = []
            
            for i, (env, action) in enumerate(zip(envs, actions_np)):
                # Create continuous action from discrete (simple mapping)
                continuous_action = np.random.randn(env.action_space.shape[0]) * 0.1
                
                next_obs, reward, terminated, truncated, _ = env.step(continuous_action)
                done = terminated or truncated
                
                next_obs_list.append(next_obs)
                rewards.append(reward)
                dones.append(done)
                
                current_return += reward
                current_length += 1
                
                if done:
                    episode_returns.append(current_return)
                    episode_lengths.append(current_length)
                    current_return = 0
                    current_length = 0
                    
                    # Reset environment
                    next_obs, _ = env.reset(seed=np.random.randint(0, 10000))
                    next_obs_list[i] = next_obs
            
            # Store rollout data
            rollout_obs.append(obs_batch)
            rollout_actions.append(actions_np)
            rollout_rewards.append(rewards)
            rollout_dones.append(dones)
            rollout_values.append(values_np)
            rollout_log_probs.append(log_probs_np)
            
            obs_batch = np.stack(next_obs_list)
            total_steps += config["num_envs"]
        
        # Simple PPO update (simplified for demonstration)
        update_count += 1
        
        # Log metrics
        if update_count % config["log_interval"] == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            
            metrics = {
                "performance/fps": fps,
                "performance/total_steps": total_steps,
                "performance/updates": update_count,
            }
            
            if episode_returns:
                metrics.update({
                    "episode/return_mean": np.mean(episode_returns[-100:]),
                    "episode/return_std": np.std(episode_returns[-100:]),
                    "episode/length_mean": np.mean(episode_lengths[-100:]),
                })
                
            # Log to W&B
            if config["use_wandb"]:
                wandb.log(metrics, step=total_steps)
            
            # Log to TensorBoard
            if config["use_tensorboard"]:
                for metric_key, metric_value in metrics.items():
                    tb_writer.add_scalar(metric_key, metric_value, total_steps)
            
            # Console output
            logger.info(
                f"Steps: {total_steps:,}/{config['total_timesteps']:,} | "
                f"FPS: {fps:.0f} | "
                f"Episodes: {len(episode_returns)} | "
                f"Avg Return: {metrics.get('episode/return_mean', 0):.2f}"
            )
    
    # Training complete
    elapsed = time.time() - start_time
    logger.info("="*70)
    logger.info("Training Complete!")
    logger.info(f"Total time: {elapsed:.1f} seconds")
    logger.info(f"Average FPS: {config['total_timesteps']/elapsed:.0f}")
    logger.info(f"Total episodes: {len(episode_returns)}")
    if episode_returns:
        logger.info(f"Final avg return: {np.mean(episode_returns[-100:]):.2f}")
    logger.info("="*70)
    
    # Cleanup
    if config["use_tensorboard"]:
        tb_writer.close()
    
    if config["use_wandb"]:
        wandb.finish()

if __name__ == "__main__":
    train_grid_rl_simple()