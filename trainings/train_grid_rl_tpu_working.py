#!/usr/bin/env python3
"""
WORKING TPU Grid RL Training - Actually uses TPU!
Simplified version that demonstrably runs on TPU cores
"""

import jax
import jax.numpy as jnp
from jax import random, pmap, jit, vmap, lax
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import time
from typing import NamedTuple
import wandb
from datetime import datetime

print("="*70)
print("TPU GRID RL TRAINING - WORKING VERSION")
print("="*70)
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
print("="*70)

# Simple Actor-Critic Network
class ActorCritic(nn.Module):
    action_dim: int = 145  # Grid RL action space
    
    @nn.compact
    def __call__(self, x):
        # Scale input
        x = x / 100.0
        
        # Shared layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        
        # Policy head - output mean and log_std
        policy_mean = nn.Dense(self.action_dim)(x)
        policy_mean = nn.tanh(policy_mean)  # Bound to [-1, 1]
        
        # Value head
        value = nn.Dense(1)(x)
        value = jnp.squeeze(value, axis=-1)
        
        return policy_mean, value

class Batch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray
    log_probs: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute GAE advantages"""
    def compute_advantages_backwards(carry, t):
        gae, next_value = carry
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        return (gae, values[t]), gae
    
    _, advantages = lax.scan(
        compute_advantages_backwards,
        (jnp.zeros_like(values[0]), values[-1]),
        jnp.arange(len(rewards) - 1, -1, -1)
    )
    
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns

@pmap
def train_step(state, batch, clip_eps=0.2):
    """PPO training step - runs on each TPU core"""
    
    def loss_fn(params):
        # Forward pass
        policy_mean, values = state.apply_fn(params, batch.obs)
        
        # For simplicity, use fixed std
        log_std = -0.5 * jnp.ones_like(policy_mean)
        std = jnp.exp(log_std)
        
        # Compute log probs (simplified Gaussian)
        log_probs = -0.5 * jnp.sum(
            jnp.square((batch.actions - policy_mean) / std) + 2 * log_std,
            axis=-1
        )
        
        # PPO objective
        ratio = jnp.exp(log_probs - batch.log_probs)
        clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -jnp.mean(
            jnp.minimum(ratio * batch.advantages, clipped_ratio * batch.advantages)
        )
        
        # Value loss
        value_loss = jnp.mean(jnp.square(values - batch.returns))
        
        # Entropy (approximate)
        entropy = jnp.mean(jnp.sum(log_std, axis=-1))
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        return total_loss, {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': total_loss,
        }
    
    # Compute loss and gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, metrics

def create_train_state(rng, obs_shape, action_dim, learning_rate):
    """Create training state"""
    network = ActorCritic(action_dim=action_dim)
    params = network.init(rng, jnp.zeros(obs_shape))
    
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate),
    )
    
    return TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

def generate_fake_batch(rng, batch_size, obs_dim=618, action_dim=145, seq_len=128):
    """Generate fake batch for testing TPU training"""
    keys = random.split(rng, 8)
    
    # Generate fake trajectory data
    obs = random.normal(keys[0], (seq_len, batch_size, obs_dim))
    actions = random.normal(keys[1], (seq_len, batch_size, action_dim)) * 0.1
    rewards = random.normal(keys[2], (seq_len, batch_size)) * 0.1 + 0.01  # Small positive bias
    dones = random.uniform(keys[3], (seq_len, batch_size)) < 0.01  # 1% done rate
    
    # Compute values and advantages
    values = random.normal(keys[4], (seq_len, batch_size))
    advantages, returns = compute_gae(rewards, values, dones)
    
    # Normalize advantages
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    
    # Fake log probs
    log_probs = random.normal(keys[5], (seq_len, batch_size))
    
    # Flatten time dimension
    obs = obs.reshape(-1, obs_dim)
    actions = actions.reshape(-1, action_dim)
    rewards = rewards.reshape(-1)
    dones = dones.reshape(-1)
    values = values.reshape(-1)
    log_probs = log_probs.reshape(-1)
    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)
    
    return Batch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        values=values,
        log_probs=log_probs,
        advantages=advantages,
        returns=returns,
    )

def main():
    """Main training loop that actually uses TPU"""
    
    # Config
    config = {
        'num_updates': 1000,
        'batch_size': 256,  # Per device
        'learning_rate': 3e-4,
        'log_interval': 10,
        'use_wandb': True,
    }
    
    # Initialize W&B
    if config['use_wandb']:
        wandb.init(
            project="grid-rl-tpu-working",
            config=config,
            name=f"tpu_working_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Setup for each TPU core
    print("\nInitializing on TPU cores...")
    rng = random.PRNGKey(42)
    rng, *init_rngs = random.split(rng, jax.device_count() + 1)
    
    # Create train state for each device
    obs_shape = (config['batch_size'], 618)
    train_states = pmap(
        lambda rng: create_train_state(rng, obs_shape, 145, config['learning_rate'])
    )(jnp.array(init_rngs))
    
    print(f"✅ Initialized training state on {jax.device_count()} TPU cores")
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    for update in range(config['num_updates']):
        # Generate fake batch (in real training, this would come from envs)
        rng, batch_rng = random.split(rng)
        batch_single = generate_fake_batch(
            batch_rng, 
            config['batch_size'],
            seq_len=128
        )
        
        # Replicate batch across devices
        batch = jax.tree.map(
            lambda x: jnp.array([x] * jax.device_count()),
            batch_single
        )
        
        # Training step on TPU
        train_states, metrics = train_step(train_states, batch)
        
        # Log metrics
        if update % config['log_interval'] == 0:
            # Get metrics from first device
            metrics_np = jax.tree.map(lambda x: float(x[0]), metrics)
            
            elapsed = time.time() - start_time
            updates_per_sec = (update + 1) / elapsed
            samples_per_sec = updates_per_sec * config['batch_size'] * jax.device_count()
            
            print(f"Update {update:4d} | "
                  f"Loss: {metrics_np['total_loss']:.4f} | "
                  f"Updates/s: {updates_per_sec:.1f} | "
                  f"Samples/s: {samples_per_sec:.0f}")
            
            if config['use_wandb']:
                wandb.log({
                    'performance/updates_per_sec': updates_per_sec,
                    'performance/samples_per_sec': samples_per_sec,
                    'loss/total': metrics_np['total_loss'],
                    'loss/policy': metrics_np['policy_loss'],
                    'loss/value': metrics_np['value_loss'],
                    'metrics/entropy': metrics_np['entropy'],
                }, step=update)
    
    # Final stats
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Average updates/sec: {config['num_updates']/elapsed:.1f}")
    print(f"Average samples/sec: {config['num_updates']*config['batch_size']*jax.device_count()/elapsed:.0f}")
    print("="*70)
    
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    main()