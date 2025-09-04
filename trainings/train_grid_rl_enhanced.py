#!/usr/bin/env python3
"""
Enhanced Grid RL Training with W&B and TensorBoard
Fixes dimension mismatches and improves monitoring
"""

import os
import sys
import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from typing import NamedTuple, Dict, Any, Tuple
import time
import threading
import queue
import pickle
from datetime import datetime
from pathlib import Path

# Logging imports
import wandb
from tensorboardX import SummaryWriter
import logging

# Environment and agent imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environments.power_grid_env import PowerGridEnv
from agents.multi_agent_grid_rl import MultiAgentGridRL, MultiAgentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GridRLConfig:
    """Enhanced configuration with W&B support"""
    # Training
    total_timesteps: int = 10_000_000
    num_actors: int = 2
    num_learners: int = 2  # Use 2 TPU cores for learning
    
    # Environment
    num_envs_per_actor: int = 64
    trajectory_length: int = 64
    
    # Optimization
    batch_size: int = 128
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    
    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 100
    eval_interval: int = 50
    
    # W&B Configuration
    use_wandb: bool = True
    wandb_project: str = "grid-rl-tpu"
    wandb_entity: str = None  # Set your W&B username/team
    wandb_tags: list = ["tpu-v4", "multi-agent", "power-grid"]
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = "./tensorboard"
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    
    @property
    def total_parallel_envs(self):
        return self.num_actors * self.num_envs_per_actor

class Transition(NamedTuple):
    """Trajectory data"""
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    info: Dict[str, Any]

class EnhancedGridRLTrainer:
    """Enhanced trainer with better logging and fixed dimensions"""
    
    def __init__(self, config: GridRLConfig):
        self.config = config
        self.setup_logging()
        self.setup_environment()
        self.setup_agents()
        
    def setup_logging(self):
        """Initialize W&B and TensorBoard"""
        # Create directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                tags=self.config.wandb_tags,
                config={
                    "total_timesteps": self.config.total_timesteps,
                    "num_actors": self.config.num_actors,
                    "num_learners": self.config.num_learners,
                    "num_envs_per_actor": self.config.num_envs_per_actor,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "trajectory_length": self.config.trajectory_length,
                    "tpu_devices": jax.device_count(),
                    "tpu_type": str(jax.devices()[0]),
                }
            )
            logger.info("W&B initialized successfully")
        
        # Initialize TensorBoard
        if self.config.use_tensorboard:
            Path(self.config.tensorboard_dir).mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(self.config.tensorboard_dir)
            logger.info(f"TensorBoard logging to {self.config.tensorboard_dir}")
    
    def setup_environment(self):
        """Setup power grid environment with correct dimensions"""
        self.env = PowerGridEnv()
        
        # Get actual dimensions from environment
        dummy_obs, _ = self.env.reset(seed=0)
        
        self.obs_dim = dummy_obs.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        logger.info(f"Environment initialized:")
        logger.info(f"  Observation dim: {self.obs_dim}")
        logger.info(f"  Action dim: {self.action_dim}")
        
    def setup_agents(self):
        """Setup multi-agent system with dimension fixes"""
        # Fix agent config to match environment dimensions
        agent_config = MultiAgentConfig()
        
        # Override observation dimensions to match environment
        # Strategic agent gets full observation
        agent_config.strategic_obs_dim = self.obs_dim
        
        # Operational agents get subset (divide observation space)
        agent_config.operational_obs_dim = self.obs_dim // 4  # Each gets quarter
        
        # Safety agent gets critical subset
        agent_config.safety_obs_dim = self.obs_dim // 5
        
        # Action dimensions must sum to environment action space
        # Adjust based on your needs
        agent_config.strategic_action_dim = 32
        agent_config.operational_action_dim = 32  # x4 agents = 128
        agent_config.safety_action_dim = self.action_dim - 32 - (32 * 4)
        
        self.agent_system = MultiAgentGridRL(agent_config)
        
        # Initialize agent parameters
        key = jax.random.PRNGKey(42)
        dummy_obs = jnp.zeros((1, self.obs_dim))
        self.agent_params = self.agent_system.init(key, dummy_obs)
        
        logger.info("Multi-agent system initialized with fixed dimensions")
    
    def create_actor(self, actor_id: int):
        """Create actor with proper error handling"""
        def actor_fn():
            logger.info(f"Actor {actor_id} starting")
            
            # Setup environments
            envs = [PowerGridEnv() for _ in range(self.config.num_envs_per_actor)]
            keys = jax.random.split(jax.random.PRNGKey(actor_id), self.config.num_envs_per_actor)
            
            # Reset all environments
            obs_list = []
            env_infos = []
            for i, env in enumerate(envs):
                obs, info = env.reset(seed=actor_id * 1000 + i)
                obs_list.append(obs)
                env_infos.append(info)
            
            obs_batch = jnp.stack(obs_list)
            
            step = 0
            episode_returns = np.zeros(self.config.num_envs_per_actor)
            episode_lengths = np.zeros(self.config.num_envs_per_actor)
            
            while step < self.config.total_timesteps // self.config.total_parallel_envs:
                # Collect trajectory
                trajectory = []
                
                for t in range(self.config.trajectory_length):
                    # Get actions from agent
                    actions, values, log_probs = self.agent_system.apply(
                        self.agent_params, obs_batch, training=True
                    )
                    
                    # Step environments
                    next_obs_list = []
                    rewards = []
                    dones = []
                    
                    for i, (env, action) in enumerate(zip(envs, actions)):
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        next_obs_list.append(next_obs)
                        rewards.append(reward)
                        done = terminated or truncated
                        dones.append(done)
                        
                        episode_returns[i] += reward
                        episode_lengths[i] += 1
                        
                        if done:
                            # Log episode metrics
                            self.log_metrics({
                                f"actor_{actor_id}/episode_return": episode_returns[i],
                                f"actor_{actor_id}/episode_length": episode_lengths[i],
                            }, step)
                            
                            episode_returns[i] = 0
                            episode_lengths[i] = 0
                            
                            # Reset environment
                            next_obs, _ = env.reset(seed=np.random.randint(0, 10000))
                            next_obs_list[i] = next_obs
                    
                    # Store transition
                    trajectory.append(Transition(
                        obs=obs_batch,
                        action=jnp.stack(actions),
                        reward=jnp.array(rewards),
                        done=jnp.array(dones),
                        value=values,
                        log_prob=log_probs,
                        info={}
                    ))
                    
                    obs_batch = jnp.stack(next_obs_list)
                
                # Send trajectory to learner queue
                # (Implement queue mechanism here)
                
                step += self.config.trajectory_length
                
                if step % (self.config.log_interval * self.config.trajectory_length) == 0:
                    logger.info(f"Actor {actor_id}: Step {step}, "
                              f"Avg return: {np.mean(episode_returns):.2f}")
        
        return actor_fn
    
    def create_learner(self):
        """Create learner with enhanced logging"""
        def learner_fn():
            logger.info("Learner starting")
            
            # Setup optimizer
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optax.adam(self.config.learning_rate)
            )
            
            opt_state = optimizer.init(self.agent_params)
            
            update_step = 0
            
            while update_step < self.config.total_timesteps // self.config.batch_size:
                # Get batch from queue (implement queue mechanism)
                # batch = queue.get()
                
                # Compute loss and gradients
                # loss, grads = jax.value_and_grad(self.compute_loss)(
                #     self.agent_params, batch
                # )
                
                # Update parameters
                # updates, opt_state = optimizer.update(grads, opt_state)
                # self.agent_params = optax.apply_updates(self.agent_params, updates)
                
                # Log metrics
                if update_step % self.config.log_interval == 0:
                    metrics = {
                        "learner/update_step": update_step,
                        # "learner/total_loss": loss,
                        "learner/learning_rate": self.config.learning_rate,
                    }
                    self.log_metrics(metrics, update_step)
                    
                    logger.info(f"Learner: Update {update_step}")
                
                # Checkpoint
                if update_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(update_step)
                
                update_step += 1
        
        return learner_fn
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to both W&B and TensorBoard"""
        # W&B logging
        if self.config.use_wandb:
            wandb.log(metrics, step=step)
        
        # TensorBoard logging
        if self.config.use_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_{step}.pkl"
        
        checkpoint = {
            "step": step,
            "params": self.agent_params,
            "config": self.config,
        }
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Log to W&B
        if self.config.use_wandb:
            wandb.save(str(checkpoint_path))
    
    def train(self):
        """Main training loop"""
        logger.info("="*70)
        logger.info("ENHANCED GRID RL TRAINING")
        logger.info("="*70)
        logger.info(f"TPU devices: {jax.device_count()}")
        logger.info(f"Total parallel environments: {self.config.total_parallel_envs}")
        logger.info(f"W&B enabled: {self.config.use_wandb}")
        logger.info(f"TensorBoard enabled: {self.config.use_tensorboard}")
        logger.info("="*70)
        
        # Create and start actors
        actors = []
        for i in range(self.config.num_actors):
            actor_thread = threading.Thread(target=self.create_actor(i))
            actor_thread.start()
            actors.append(actor_thread)
        
        # Create and start learner
        learner_thread = threading.Thread(target=self.create_learner())
        learner_thread.start()
        
        # Wait for completion
        for actor in actors:
            actor.join()
        learner_thread.join()
        
        # Cleanup
        if self.config.use_tensorboard:
            self.tb_writer.close()
        
        if self.config.use_wandb:
            wandb.finish()
        
        logger.info("Training completed!")

def main():
    """Main entry point"""
    # Setup configuration
    config = GridRLConfig()
    
    # Override with command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-entity", type=str, help="W&B entity/username")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")
    args = parser.parse_args()
    
    if args.wandb_entity:
        config.wandb_entity = args.wandb_entity
    if args.no_wandb:
        config.use_wandb = False
    if args.no_tensorboard:
        config.use_tensorboard = False
    
    # Create trainer and start training
    trainer = EnhancedGridRLTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()