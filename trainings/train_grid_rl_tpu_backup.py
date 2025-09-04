#!/usr/bin/env python3
"""
Main training script for Power Grid RL on TPU
Integrates multi-agent system with Sebulba framework for distributed training
"""

import sys
import os
sys.path.append('/home/tarive/persistent_storage/tpu_rl_workspace')
sys.path.append('/home/tarive/persistent_storage/tpu_rl_workspace/sebulba')

import jax
import jax.numpy as jnp
from jax import pmap, vmap, jit
import numpy as np
import time
from typing import Dict, Any, Tuple, NamedTuple
import optax
from flax.training.train_state import TrainState
import pickle
from datetime import datetime
from functools import partial

# Import Sebulba components (with simplified replacements)
import queue
import threading

# Simplified StoppableComponent
class StoppableComponent(threading.Thread):
    """Simplified version of Sebulba's StoppableComponent"""
    
    def __init__(self, name=None):
        super().__init__(name=name, daemon=True)
        self.should_stop = False
        self.name = name
        
        # Simple logger
        class SimpleLogger:
            def info(self, msg):
                print(f"[{name}] {msg}")
        self.log = SimpleLogger()
    
    def stop(self):
        self.should_stop = True
    
    def run(self):
        self._run()
    
    def _run(self):
        pass


# Simplified ParamsSource
class ParamsSource:
    """Simplified parameter source for actors"""
    
    def __init__(self, initial_params, device):
        self.params = initial_params
        self.device = device
        self._queue = queue.Queue(maxsize=1)
        self._queue.put(initial_params)
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def get(self):
        try:
            params = self._queue.get_nowait()
            self._queue.put(params)  # Put it back immediately
            return params
        except queue.Empty:
            return self.params
    
    def update(self, new_params):
        try:
            self._queue.get_nowait()  # Remove old params
        except queue.Empty:
            pass
        self._queue.put(new_params)
        self.params = new_params

# Import our grid RL components
from grid_rl.environments.power_grid_env import (
    PowerGridEnv, VectorizedPowerGridEnv, make_power_grid_env, GridConfig
)
from grid_rl.agents.multi_agent_grid_rl import (
    MultiAgentConfig, MultiAgentGridRL, create_multi_agent_state,
    compute_gae_hierarchical, multi_agent_loss
)


class GridRLConfig(NamedTuple):
    """Complete configuration for Grid RL training"""
    # Environment
    num_envs_per_actor: int = 64  # Reduced from 256 to avoid memory issues
    num_actors: int = 1  # Reduced from 2 to 1 per TPU core
    trajectory_length: int = 64  # Reduced from 128
    
    # Grid environment
    grid_config: GridConfig = GridConfig()
    
    # Multi-agent
    agent_config: MultiAgentConfig = MultiAgentConfig()
    
    # Training
    total_timesteps: int = 10_000_000
    batch_size: int = 128  # Reduced from 512
    num_epochs: int = 4
    num_minibatches: int = 8
    
    # TPU configuration
    actor_devices: Tuple[int, ...] = (0, 1)  # TPU cores for actors
    learner_devices: Tuple[int, ...] = (2,)  # Single TPU core for learner to simplify
    
    # Logging
    log_frequency: int = 10
    save_frequency: int = 100
    eval_frequency: int = 50


class GridEnvironmentFactory:
    """Factory for creating grid environments with different seeds"""
    
    def __init__(self, config: GridConfig, init_seed: int = 42):
        self.config = config
        self.seed = init_seed
    
    def __call__(self, num_envs: int) -> VectorizedPowerGridEnv:
        """Create vectorized environment"""
        env = make_power_grid_env(
            num_envs=num_envs,
            num_buses=self.config.num_buses,
            num_generators=self.config.num_generators,
            num_loads=self.config.num_loads
        )
        self.seed += num_envs
        return env


class SimplePipeline:
    """Simple pipeline for passing trajectories between actors and learners"""
    
    def __init__(self, max_size: int = 10):
        self._queue = queue.Queue(maxsize=max_size)
        self.running = False
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False
    
    def put(self, item, timeout=None):
        if self.running:
            self._queue.put(item, timeout=timeout)
    
    def get(self, timeout=None):
        if self.running:
            try:
                return self._queue.get(timeout=timeout)
            except queue.Empty:
                return None
        return None


class GridRLActor(StoppableComponent):
    """Actor component for grid RL that collects trajectories"""
    
    def __init__(
        self,
        env_factory: GridEnvironmentFactory,
        actor_device: jax.Device,
        params_source: ParamsSource,
        pipeline: SimplePipeline,
        config: GridRLConfig,
        actor_id: int
    ):
        super().__init__(name=f"GridRLActor-{actor_id}")
        self.env_factory = env_factory
        self.actor_device = actor_device
        self.params_source = params_source
        self.pipeline = pipeline
        self.config = config
        self.actor_id = actor_id
        
        # Create environments
        self.envs = env_factory(config.num_envs_per_actor)
        
        # Initialize RNG
        self.rng = jax.random.PRNGKey(42 + actor_id)
    
    def _run(self):
        """Collect trajectories and send to learner"""
        with jax.default_device(self.actor_device):
            # Reset environments
            obs, _ = self.envs.reset()
            
            # Create action function
            @jit
            def select_actions(params, obs, rng):
                """Select actions for all agents"""
                model = MultiAgentGridRL(self.config.agent_config)
                
                # Split observations for different agents
                # Environment gives 618 dims, we need to map to agent observations
                batch_size = obs.shape[0]
                obs_dim = obs.shape[1]
                
                # Handle dimension mismatch by padding or truncating
                if obs_dim < self.config.agent_config.strategic_obs_dim:
                    # Pad with zeros if observation is smaller
                    padding = self.config.agent_config.strategic_obs_dim - obs_dim
                    obs = jnp.pad(obs, ((0, 0), (0, padding)), mode='constant')
                
                # Extract observations with overlapping windows for comprehensive coverage
                strategic_obs = obs[:, :self.config.agent_config.strategic_obs_dim]
                
                # For operational agents, use middle section with overlap
                op_start = max(0, obs_dim // 4)
                op_end = min(obs_dim, op_start + self.config.agent_config.operational_obs_dim)
                operational_obs = obs[:, op_start:op_end]
                
                # Pad operational obs if needed
                if operational_obs.shape[1] < self.config.agent_config.operational_obs_dim:
                    pad_size = self.config.agent_config.operational_obs_dim - operational_obs.shape[1]
                    operational_obs = jnp.pad(operational_obs, ((0, 0), (0, pad_size)), mode='constant')
                
                # Safety agent gets last portion
                safety_obs = obs[:, -self.config.agent_config.safety_obs_dim:]
                if safety_obs.shape[1] < self.config.agent_config.safety_obs_dim:
                    pad_size = self.config.agent_config.safety_obs_dim - safety_obs.shape[1]
                    safety_obs = jnp.pad(safety_obs, ((0, 0), (0, pad_size)), mode='constant')
                
                # Get outputs from all agents
                outputs = model.apply(
                    {'params': params},
                    strategic_obs,
                    operational_obs,
                    safety_obs
                )
                
                # Sample actions from each agent
                rng, *subkeys = jax.random.split(rng, 4)
                
                # Strategic actions
                strategic_logits = outputs['strategic']['logits']
                strategic_action = jax.random.categorical(subkeys[0], strategic_logits)
                
                # Operational actions (combine multiple agents)
                operational_actions = []
                for i, op_output in enumerate(outputs['operational']):
                    op_action = jax.random.categorical(subkeys[1], op_output['logits'])
                    operational_actions.append(op_action[:, None])  # Add dimension for concatenation
                operational_action = jnp.concatenate(operational_actions, axis=-1)
                
                # Safety actions
                safety_logits = outputs['safety']['logits']
                safety_action = jax.random.categorical(subkeys[2], safety_logits)
                
                # Check for safety override
                override = outputs['safety']['override'] > 0.5
                
                # Combine all actions
                combined_action = jnp.concatenate([
                    strategic_action[:, None],
                    operational_action,
                    safety_action[:, None]
                ], axis=-1)
                
                # Apply safety override if needed
                combined_action = jnp.where(
                    override[:, None],
                    safety_action[:, None],  # Use only safety action if override
                    combined_action
                )
                
                # Adjust action dimensions to match environment expectations
                # Environment expects 145 actions, agents produce more
                env_action_dim = 145
                if combined_action.shape[-1] > env_action_dim:
                    # Truncate to environment action dimension
                    combined_action = combined_action[:, :env_action_dim]
                elif combined_action.shape[-1] < env_action_dim:
                    # Pad with zeros if needed
                    pad_size = env_action_dim - combined_action.shape[-1]
                    combined_action = jnp.pad(combined_action, ((0, 0), (0, pad_size)), mode='constant')
                
                # Also return values and log probs for training
                values = {
                    'strategic': outputs['strategic']['value'],
                    'operational': [op['value'] for op in outputs['operational']],
                    'safety': outputs['safety']['value']
                }
                
                return combined_action, values, outputs, rng
            
            step_count = 0
            episode_returns = np.zeros(self.config.num_envs_per_actor)
            
            while not self.should_stop:
                # Collect trajectory
                trajectory = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'dones': [],
                    'values': [],
                    'log_probs': []
                }
                
                for _ in range(self.config.trajectory_length):
                    # Get current parameters
                    params = self.params_source.get()
                    
                    # Select actions
                    obs_jax = jax.device_put(obs, self.actor_device)
                    actions, values, outputs, self.rng = select_actions(
                        params, obs_jax, self.rng
                    )
                    
                    # Convert actions to numpy for environment
                    actions_np = np.array(actions)
                    
                    # Environment step
                    next_obs, rewards, dones, truncs, infos = self.envs.step(actions_np)
                    
                    # Store trajectory data
                    trajectory['observations'].append(obs)
                    trajectory['actions'].append(actions_np)
                    trajectory['rewards'].append(rewards)
                    trajectory['dones'].append(dones | truncs)
                    trajectory['values'].append(values)
                    
                    # Update episode returns
                    episode_returns += rewards
                    episode_returns[dones | truncs] = 0
                    
                    obs = next_obs
                    step_count += self.config.num_envs_per_actor
                
                # Send trajectory to learner via pipeline
                self.pipeline.put(trajectory)
                
                # Log progress
                if step_count % (self.config.log_frequency * self.config.num_envs_per_actor) == 0:
                    mean_return = np.mean(episode_returns)
                    self.log.info(
                        f"Actor {self.actor_id} - Steps: {step_count:,}, "
                        f"Mean Return: {mean_return:.2f}"
                    )


class GridRLLearner(StoppableComponent):
    """Learner component for grid RL that updates parameters"""
    
    def __init__(
        self,
        pipeline: SimplePipeline,
        learner_devices: list,
        init_state: TrainState,
        config: GridRLConfig,
        params_sources: list
    ):
        super().__init__(name="GridRLLearner")
        self.pipeline = pipeline
        self.learner_devices = learner_devices
        self.state = init_state
        self.config = config
        self.params_sources = params_sources
        
        # Setup pmap for distributed learning
        self.update_fn = pmap(
            self._update_step,
            axis_name='device',
            devices=learner_devices
        )
        
        # Replicate state across devices
        self.state = jax.device_put_replicated(init_state, learner_devices)
    
    @partial(jit, static_argnums=(0,))
    def _update_step(self, state: TrainState, batch: Dict) -> Tuple[TrainState, Dict]:
        """Single update step"""
        
        # Prepare observations
        observations = {
            'strategic': batch['observations'][:, :self.config.agent_config.strategic_obs_dim],
            'operational': batch['observations'][:, 
                          self.config.agent_config.strategic_obs_dim:
                          self.config.agent_config.strategic_obs_dim + 
                          self.config.agent_config.operational_obs_dim],
            'safety': batch['observations'][:, -self.config.agent_config.safety_obs_dim:]
        }
        
        # Prepare actions (need to split combined actions)
        actions = {
            'strategic': batch['actions'][:, 0],
            'operational': [batch['actions'][:, i+1] 
                          for i in range(self.config.agent_config.num_operational_agents)],
            'safety': batch['actions'][:, -1]
        }
        
        # Compute advantages and returns
        advantages, returns = compute_gae_hierarchical(
            batch['rewards'],
            batch['values']['strategic'],  # Use strategic values as baseline
            batch['dones'],
            gamma=self.config.agent_config.gamma,
            gae_lambda=self.config.agent_config.gae_lambda
        )
        
        # Prepare targets
        targets = {
            'strategic': {
                'advantages': advantages,
                'returns': returns,
                'log_probs': batch['log_probs']['strategic']
            },
            'operational': [
                {
                    'advantages': advantages,
                    'returns': returns,
                    'log_probs': batch['log_probs'][f'operational_{i}']
                }
                for i in range(self.config.agent_config.num_operational_agents)
            ],
            'safety': {
                'advantages': advantages,
                'returns': returns,
                'log_probs': batch['log_probs']['safety']
            }
        }
        
        # Compute gradients and update
        (loss, metrics), grads = jax.value_and_grad(
            lambda p: multi_agent_loss(p, observations, actions, targets, self.config.agent_config),
            has_aux=True
        )(state.params)
        
        # Average gradients across devices
        grads = jax.lax.pmean(grads, axis_name='device')
        
        # Update parameters
        state = state.apply_gradients(grads=grads)
        
        # Average metrics across devices
        metrics = jax.lax.pmean(metrics, axis_name='device')
        
        return state, metrics
    
    def _run(self):
        """Main learner loop"""
        update_count = 0
        
        while not self.should_stop:
            # Get batch from pipeline
            batch = self.pipeline.get(timeout=1.0)
            if batch is None:
                continue
            
            # Convert to JAX arrays and stack trajectory data
            # Trajectory data comes as lists of arrays
            def stack_trajectory(x):
                if isinstance(x, list):
                    return jnp.stack(x, axis=0)
                return jnp.array(x)
            
            batch = jax.tree.map(stack_trajectory, batch)
            
            # For single learner device, add a batch dimension at the front
            batch = jax.tree.map(lambda x: x[None, ...], batch)
            
            # Perform multiple epochs of updates
            for _ in range(self.config.num_epochs):
                # Update parameters
                self.state, metrics = self.update_fn(self.state, batch)
                
                # Get updated parameters (from first device)
                new_params = jax.tree.map(lambda x: x[0], self.state.params)
                
                # Update all parameter sources
                for params_source in self.params_sources:
                    params_source.update(new_params)
            
            update_count += 1
            
            # Log metrics
            if update_count % self.config.log_frequency == 0:
                metrics_np = jax.tree.map(lambda x: float(x[0]), metrics)
                self.log.info(
                    f"Update {update_count} - "
                    f"Loss: {metrics_np.get('total_loss', 0):.4f}, "
                    f"Strategic Loss: {metrics_np.get('strategic', {}).get('total_loss', 0):.4f}, "
                    f"Safety Loss: {metrics_np.get('safety', {}).get('total_loss', 0):.4f}"
                )
            
            # Save checkpoint
            if update_count % self.config.save_frequency == 0:
                self._save_checkpoint(update_count)
    
    def _save_checkpoint(self, update_count: int):
        """Save model checkpoint"""
        checkpoint_dir = "/home/tarive/persistent_storage/tpu_rl_workspace/grid_rl/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_{update_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        
        # Get parameters from first device
        params = jax.tree.map(lambda x: x[0], self.state.params)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'params': params,
                'update_count': update_count,
                'config': self.config
            }, f)
        
        self.log.info(f"Saved checkpoint to {checkpoint_path}")


def train_grid_rl():
    """Main training function"""
    print("=" * 60, flush=True)
    print("POWER GRID RL TRAINING ON TPU", flush=True)
    print("=" * 60, flush=True)
    print(f"JAX version: {jax.__version__}", flush=True)
    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}", flush=True)
    print("=" * 60, flush=True)
    
    # Create configuration
    print("Creating configuration...", flush=True)
    config = GridRLConfig()
    
    # Get JAX devices
    print("Getting JAX devices...", flush=True)
    devices = jax.devices()
    actor_devices = [devices[i] for i in config.actor_devices]
    learner_devices = [devices[i] for i in config.learner_devices]
    
    print(f"Actor devices: {actor_devices}", flush=True)
    print(f"Learner devices: {learner_devices}", flush=True)
    
    # Create environment factory
    print("Creating environment factory...", flush=True)
    env_factory = GridEnvironmentFactory(config.grid_config)
    
    # Create pipeline for actor-learner communication
    print("Creating pipeline...", flush=True)
    pipeline = SimplePipeline(max_size=10)
    pipeline.start()
    
    # Initialize multi-agent model
    print("Initializing multi-agent model...", flush=True)
    rng = jax.random.PRNGKey(42)
    init_state = create_multi_agent_state(config.agent_config, rng)
    print("Model initialized!", flush=True)
    
    # Create parameter sources for actors
    params_sources = []
    for device in actor_devices:
        params_source = ParamsSource(init_state.params, device)
        params_source.start()
        params_sources.append(params_source)
    
    # Create actors
    actors = []
    for i, device in enumerate(actor_devices):
        for j in range(config.num_actors):
            actor = GridRLActor(
                env_factory=env_factory,
                actor_device=device,
                params_source=params_sources[i],
                pipeline=pipeline,
                config=config,
                actor_id=i * config.num_actors + j
            )
            actors.append(actor)
    
    # Create learner
    learner = GridRLLearner(
        pipeline=pipeline,
        learner_devices=learner_devices,
        init_state=init_state,
        config=config,
        params_sources=params_sources
    )
    
    # Start all components
    print("\nStarting training components...")
    for actor in actors:
        actor.start()
    learner.start()
    
    print(f"Training with {len(actors)} actors and 1 learner")
    print(f"Total parallel environments: {len(actors) * config.num_envs_per_actor}")
    print("-" * 60)
    
    try:
        # Run training
        start_time = time.time()
        learner.join()  # Wait for learner to finish
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Stop all components
        print("\nStopping training components...")
        for actor in actors:
            actor.stop()
        learner.stop()
        pipeline.stop()
        for params_source in params_sources:
            params_source.stop()
        
        # Wait for clean shutdown
        for actor in actors:
            actor.join(timeout=5)
        learner.join(timeout=5)
        
        elapsed = time.time() - start_time
        print(f"\nTraining time: {elapsed/3600:.2f} hours")
        print("=" * 60)


if __name__ == "__main__":
    train_grid_rl()