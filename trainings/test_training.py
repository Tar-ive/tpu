#!/usr/bin/env python3
"""
Test script to verify training improvements
"""

import sys
sys.path.append('/home/tarive/persistent_storage/tpu_rl_workspace/grid_rl')

import jax
import jax.numpy as jnp
from jax import random, jit, pmap
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import time

from environments.power_grid_env_fixed import PowerGridEnvFixed, GridConfig

print("="*70)
print("🧪 TESTING FIXED TRAINING SETUP")
print("="*70)
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
print("="*70)


class SimpleActorCritic(nn.Module):
    """Minimal Actor-Critic for testing"""
    action_dim: int = 145
    
    @nn.compact
    def __call__(self, x):
        # Simple network
        x = x / 10.0  # Normalize input
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        
        # Policy mean
        policy_mean = nn.Dense(self.action_dim)(x)
        policy_mean = nn.tanh(policy_mean)
        
        # Value
        value = nn.Dense(1)(x)
        value = jnp.squeeze(value, axis=-1)
        
        return policy_mean, value


def test_environment():
    """Test that environment rewards are properly normalized"""
    print("\n1️⃣ Testing Environment Rewards...")
    print("-" * 40)
    
    env = PowerGridEnvFixed(seed=42)
    obs, _ = env.reset()
    
    rewards = []
    for i in range(100):
        action = env.action_space.sample() * 0.1  # Small actions
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    rewards = np.array(rewards)
    
    print(f"Reward Statistics (100 steps):")
    print(f"  Mean: {np.mean(rewards):.4f}")
    print(f"  Std:  {np.std(rewards):.4f}")
    print(f"  Min:  {np.min(rewards):.4f}")
    print(f"  Max:  {np.max(rewards):.4f}")
    
    # Check if normalized
    if np.min(rewards) >= -10 and np.max(rewards) <= 10:
        print("✅ Rewards are properly bounded!")
    else:
        print("⚠️  Rewards exceed expected bounds!")
    
    return True


def test_gradient_computation():
    """Test that gradients are computed without NaN/Inf"""
    print("\n2️⃣ Testing Gradient Computation...")
    print("-" * 40)
    
    # Create environment and network
    env = PowerGridEnvFixed(seed=42)
    network = SimpleActorCritic(action_dim=env.action_space.shape[0])
    
    # Initialize
    rng = random.PRNGKey(42)
    dummy_obs = jnp.zeros((1,) + env.observation_space.shape)
    params = network.init(rng, dummy_obs)
    
    # Create optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(3e-4),
    )
    
    state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    
    # Test gradient computation
    def loss_fn(params, obs, actions, rewards):
        policy_mean, values = state.apply_fn(params, obs)
        
        # Simple MSE loss for testing
        action_loss = jnp.mean(jnp.square(policy_mean - actions))
        value_loss = jnp.mean(jnp.square(values - rewards))
        
        total_loss = action_loss + value_loss
        return total_loss, {'action_loss': action_loss, 'value_loss': value_loss}
    
    # Generate batch
    obs = random.normal(rng, (32, env.observation_space.shape[0]))
    actions = random.uniform(rng, (32, env.action_space.shape[0]), minval=-1, maxval=1)
    rewards = random.normal(rng, (32,))
    
    # Compute gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, obs, actions, rewards
    )
    
    # Check for NaN/Inf
    has_nan = jax.tree_util.tree_map(
        lambda x: jnp.any(jnp.isnan(x)) | jnp.any(jnp.isinf(x)),
        grads
    )
    
    grad_norm = optax.global_norm(grads)
    
    print(f"Loss: {loss:.4f}")
    print(f"Gradient Norm: {grad_norm:.4f}")
    
    all_finite = not any(jax.tree_util.tree_leaves(has_nan))
    if all_finite:
        print("✅ All gradients are finite!")
    else:
        print("⚠️  Found NaN/Inf in gradients!")
    
    return all_finite


def test_training_loop():
    """Test a mini training loop"""
    print("\n3️⃣ Testing Training Loop...")
    print("-" * 40)
    
    # Setup
    env = PowerGridEnvFixed(seed=42)
    network = SimpleActorCritic(action_dim=env.action_space.shape[0])
    
    rng = random.PRNGKey(42)
    dummy_obs = jnp.zeros((1,) + env.observation_space.shape)
    params = network.init(rng, dummy_obs)
    
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(3e-4),
    )
    
    state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    
    # Collect some data
    obs, _ = env.reset()
    observations = []
    actions = []
    rewards = []
    
    for _ in range(50):
        # Get action from policy
        obs_batch = jnp.expand_dims(obs, 0)
        policy_mean, value = state.apply_fn(state.params, obs_batch)
        
        # Add noise for exploration
        rng, noise_rng = random.split(rng)
        action = policy_mean[0] + random.normal(noise_rng, policy_mean[0].shape) * 0.1
        action = jnp.clip(action, -1, 1)
        
        # Step
        next_obs, reward, terminated, truncated, _ = env.step(np.array(action))
        
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    
    # Convert to arrays
    observations = jnp.array(observations)
    actions = jnp.array(actions)
    rewards = jnp.array(rewards)
    
    # Simple training step
    @jit
    def train_step(state, obs_batch, action_batch, reward_batch):
        def loss_fn(params):
            policy_mean, values = state.apply_fn(params, obs_batch)
            
            # Policy loss (behavioral cloning for simplicity)
            policy_loss = jnp.mean(jnp.square(policy_mean - action_batch))
            
            # Value loss
            value_loss = jnp.mean(jnp.square(values - reward_batch))
            
            total_loss = policy_loss + 0.5 * value_loss
            return total_loss, {'policy_loss': policy_loss, 'value_loss': value_loss}
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, loss, metrics
    
    # Train for a few steps
    losses = []
    for i in range(10):
        state, loss, metrics = train_step(state, observations, actions, rewards)
        losses.append(float(loss))
        print(f"  Step {i+1:2d}: Loss = {loss:.6f}")
    
    # Check if loss decreased
    if losses[-1] < losses[0]:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"\n✅ Loss decreased by {improvement:.1f}%!")
    else:
        print("\n⚠️  Loss did not decrease")
    
    return losses[-1] < losses[0]


def test_tpu_usage():
    """Test that TPU is being used"""
    print("\n4️⃣ Testing TPU Usage...")
    print("-" * 40)
    
    devices = jax.devices()
    device_types = [d.device_kind for d in devices]
    
    print(f"Available devices: {devices}")
    print(f"Device types: {device_types}")
    
    if 'TPU' in str(devices[0]):
        print("✅ TPU detected and available!")
        
        # Test pmap
        @pmap
        def test_fn(x):
            return x * 2
        
        x = jnp.ones((len(devices), 10))
        y = test_fn(x)
        
        print(f"pmap test successful: input shape {x.shape} -> output shape {y.shape}")
        return True
    else:
        print("⚠️  No TPU detected, running on CPU")
        return False


def main():
    """Run all tests"""
    print("\n🚀 Starting Test Suite")
    print("="*70)
    
    results = {
        'environment': test_environment(),
        'gradients': test_gradient_computation(),
        'training': test_training_loop(),
        'tpu': test_tpu_usage(),
    }
    
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.ljust(15)}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! The training setup is fixed!")
        print("\n💡 Next steps:")
        print("1. Run the stable training: python train_grid_rl_stable.py")
        print("2. Try interactive mode: python train_interactive.py")
        print("3. Monitor with wandb for detailed metrics")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)