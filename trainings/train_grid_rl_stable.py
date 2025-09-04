#!/usr/bin/env python3
"""
Stable Grid RL Training with TPU Optimization
- Proper reward normalization
- Numerical stability checks
- Gradient clipping
- TPU utilization with pmap
"""

import sys
sys.path.append('/home/tarive/persistent_storage/tpu_rl_workspace/grid_rl')

import jax
import jax.numpy as jnp
from jax import random, pmap, jit, vmap, lax
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import time
from typing import NamedTuple, Tuple, Dict, Any
import wandb
from datetime import datetime
from environments.power_grid_env_fixed import PowerGridEnvFixed, GridConfig

print("="*70)
print("STABLE GRID RL TRAINING - TPU OPTIMIZED")
print("="*70)
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
print(f"Backend: {jax.default_backend()}")
print("="*70)


class ActorCritic(nn.Module):
    """Actor-Critic network with numerical stability"""
    action_dim: int = 145  # Grid RL action space
    
    @nn.compact
    def __call__(self, x):
        # Input normalization
        x = jnp.clip(x, -100, 100)  # Prevent extreme inputs
        x = x / 10.0  # Scale down
        
        # Shared layers with layer normalization
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Policy head - output mean and log_std
        policy_mean = nn.Dense(self.action_dim)(x)
        policy_mean = nn.tanh(policy_mean)  # Bound to [-1, 1]
        
        # Separate log_std parameter (learnable but not input-dependent)
        log_std = self.param('log_std', 
                            nn.initializers.constant(-0.5),
                            (self.action_dim,))
        log_std = jnp.clip(log_std, -2.0, 0.5)  # Stability bounds
        
        # Value head
        value = nn.Dense(1)(x)
        value = jnp.squeeze(value, axis=-1)
        value = jnp.clip(value, -100, 100)  # Prevent extreme values
        
        return policy_mean, log_std, value


class Batch(NamedTuple):
    """Batch data structure"""
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray
    log_probs: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray


def check_nan_inf(x, name="tensor"):
    """Check for NaN or Inf values"""
    has_nan = jnp.any(jnp.isnan(x))
    has_inf = jnp.any(jnp.isinf(x))
    return has_nan | has_inf


@jit
def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute GAE with numerical stability"""
    # Clip rewards for stability
    rewards = jnp.clip(rewards, -10, 10)
    values = jnp.clip(values, -100, 100)
    
    def compute_advantages_backwards(carry, t):
        gae, next_value = carry
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        delta = jnp.clip(delta, -10, 10)  # Clip delta for stability
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        gae = jnp.clip(gae, -10, 10)  # Clip GAE
        return (gae, values[t]), gae
    
    _, advantages = lax.scan(
        compute_advantages_backwards,
        (jnp.zeros_like(values[0]), values[-1]),
        jnp.arange(len(rewards) - 1, -1, -1)
    )
    
    advantages = advantages[::-1]
    returns = advantages + values
    
    # Normalize advantages for stability
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    
    return advantages, returns


@pmap
def train_step(state, batch, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """PPO training step with numerical stability"""
    
    def loss_fn(params):
        # Forward pass
        policy_mean, log_std, values = state.apply_fn(params, batch.obs)
        
        # Compute std with stability
        std = jnp.exp(log_std) + 1e-8
        
        # Compute log probs for continuous actions (Gaussian)
        diff = batch.actions - policy_mean
        log_probs = -0.5 * jnp.sum(
            jnp.square(diff / std) + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1
        )
        
        # Check for numerical issues
        log_probs = jnp.where(
            jnp.isfinite(log_probs),
            log_probs,
            -10.0  # Replace non-finite values
        )
        
        # PPO objective with clipping
        ratio = jnp.exp(jnp.clip(log_probs - batch.log_probs, -10, 10))
        ratio = jnp.clip(ratio, 1e-8, 1e8)  # Additional safety
        
        clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -jnp.mean(
            jnp.minimum(ratio * batch.advantages, clipped_ratio * batch.advantages)
        )
        
        # Value loss with clipping
        value_pred_clipped = batch.values + jnp.clip(
            values - batch.values, -clip_eps, clip_eps
        )
        value_losses = jnp.square(values - batch.returns)
        value_losses_clipped = jnp.square(value_pred_clipped - batch.returns)
        value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))
        
        # Entropy bonus for exploration
        entropy = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))
        
        # Total loss with coefficients
        total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        
        # Check for NaN/Inf and return safe loss if found
        if check_nan_inf(total_loss, "loss"):
            total_loss = jnp.array(1000.0)  # Large but finite loss
        
        return total_loss, {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': total_loss,
            'ratio_mean': jnp.mean(ratio),
            'ratio_max': jnp.max(ratio),
            'advantage_mean': jnp.mean(batch.advantages),
            'value_mean': jnp.mean(values),
        }
    
    # Compute loss and gradients with gradient clipping
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Check for NaN/Inf in gradients
    grad_norm = optax.global_norm(grads)
    has_nan_grad = check_nan_inf(grad_norm, "grad_norm")
    
    # Skip update if gradients are bad
    def safe_update(state, grads):
        return state.apply_gradients(grads=grads)
    
    def skip_update(state, grads):
        return state
    
    state = jax.lax.cond(
        has_nan_grad,
        skip_update,
        safe_update,
        state, grads
    )
    
    metrics['grad_norm'] = grad_norm
    metrics['has_nan_grad'] = has_nan_grad
    
    return state, metrics


def create_train_state(rng, obs_shape, action_dim, learning_rate):
    """Create training state with gradient clipping"""
    network = ActorCritic(action_dim=action_dim)
    
    # Initialize with dummy input
    dummy_obs = jnp.zeros((1,) + obs_shape)
    params = network.init(rng, dummy_obs)
    
    # Optimizer with gradient clipping and learning rate schedule
    schedule = optax.linear_schedule(
        init_value=learning_rate,
        end_value=learning_rate * 0.1,
        transition_steps=100000
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),  # Gradient clipping
        optax.adam(schedule, eps=1e-5),
    )
    
    return TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )


def rollout(env, state, rng, num_steps=128):
    """Collect rollout from environment"""
    obs, _ = env.reset()
    
    observations = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    
    for _ in range(num_steps):
        rng, action_rng = random.split(rng)
        
        # Get action from policy
        obs_batch = jnp.expand_dims(obs, 0)
        policy_mean, log_std, value = state.apply_fn(state.params, obs_batch)
        
        # Sample action from Gaussian
        std = jnp.exp(log_std)
        action = policy_mean[0] + std * random.normal(action_rng, policy_mean[0].shape)
        action = jnp.clip(action, -1, 1)
        
        # Compute log prob
        diff = action - policy_mean[0]
        log_prob = -0.5 * jnp.sum(
            jnp.square(diff / std) + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(np.array(action))
        
        # Store transition
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        values.append(value[0])
        log_probs.append(log_prob)
        
        obs = next_obs
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    # Convert to arrays
    observations = jnp.array(observations)
    actions = jnp.array(actions)
    rewards = jnp.array(rewards)
    dones = jnp.array(dones)
    values = jnp.array(values)
    log_probs = jnp.array(log_probs)
    
    # Compute advantages
    advantages, returns = compute_gae(rewards, values, dones)
    
    return Batch(
        obs=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        values=values,
        log_probs=log_probs,
        advantages=advantages,
        returns=returns
    )


def main():
    """Main training loop"""
    # Initialize wandb
    wandb.init(
        project="grid-rl-stable",
        name=f"stable-tpu-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "learning_rate": 3e-4,
            "batch_size": 256,
            "num_epochs": 4,
            "clip_epsilon": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "max_steps": 100000,
            "rollout_length": 128,
            "num_envs": 4,  # One per TPU core
        }
    )
    config = wandb.config
    
    # Create environments (one per TPU core)
    num_devices = jax.device_count()
    envs = [PowerGridEnvFixed(seed=42 + i) for i in range(num_devices)]
    
    # Initialize RNG
    rng = random.PRNGKey(42)
    rng, init_rng = random.split(rng)
    
    # Create train state (replicated across devices)
    obs_shape = envs[0].observation_space.shape
    action_dim = envs[0].action_space.shape[0]
    
    # Create states for each device
    init_rngs = random.split(init_rng, num_devices)
    states = [create_train_state(init_rngs[i], obs_shape, action_dim, config.learning_rate) 
              for i in range(num_devices)]
    
    # Replicate across devices
    states = jax.device_put_replicated(states[0], jax.devices())
    
    # Training loop
    start_time = time.time()
    global_step = 0
    
    print("\n🚀 Starting training loop...")
    print(f"Environments: {num_devices}")
    print(f"Rollout length: {config.rollout_length}")
    print(f"Batch size per device: {config.rollout_length}")
    
    try:
        while global_step < config.max_steps:
            # Collect rollouts from all environments
            rollout_rngs = random.split(rng, num_devices)
            rng = rollout_rngs[0]
            
            batches = []
            for i in range(num_devices):
                batch = rollout(envs[i], 
                              jax.device_get(jax.tree.map(lambda x: x[i], states)),
                              rollout_rngs[i],
                              config.rollout_length)
                batches.append(batch)
            
            # Stack batches for pmap
            batch = jax.tree.map(lambda *xs: jnp.stack(xs), *batches)
            
            # Update for multiple epochs
            for epoch in range(config.num_epochs):
                states, metrics = train_step(states, batch, 
                                            config.clip_epsilon,
                                            config.vf_coef,
                                            config.ent_coef)
                
                # Get metrics from first device
                metrics = jax.device_get(jax.tree.map(lambda x: x[0], metrics))
                
                # Check for NaN
                if metrics['has_nan_grad']:
                    print(f"⚠️  Warning: NaN detected in gradients at step {global_step}")
                
                # Log metrics
                if epoch == 0:  # Log once per rollout
                    fps = config.rollout_length * num_devices / (time.time() - start_time)
                    
                    log_dict = {
                        'loss/total': float(metrics['total_loss']),
                        'loss/policy': float(metrics['policy_loss']),
                        'loss/value': float(metrics['value_loss']),
                        'loss/entropy': float(metrics['entropy']),
                        'train/grad_norm': float(metrics['grad_norm']),
                        'train/ratio_mean': float(metrics['ratio_mean']),
                        'train/ratio_max': float(metrics['ratio_max']),
                        'train/advantage_mean': float(metrics['advantage_mean']),
                        'train/value_mean': float(metrics['value_mean']),
                        'train/fps': fps,
                        'train/global_step': global_step,
                    }
                    
                    # Add environment metrics
                    for i, env in enumerate(envs):
                        if hasattr(env, 'timestep'):
                            log_dict[f'env_{i}/reward'] = float(batch.rewards[i].mean())
                            log_dict[f'env_{i}/episode_length'] = float(env.timestep)
                    
                    wandb.log(log_dict, step=global_step)
                    
                    # Print progress
                    if global_step % 100 == 0:
                        print(f"Step {global_step:6d} | Loss: {metrics['total_loss']:.4f} | "
                              f"FPS: {fps:.1f} | Grad Norm: {metrics['grad_norm']:.4f}")
            
            global_step += config.rollout_length * num_devices
            start_time = time.time()
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()