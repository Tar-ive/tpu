#!/usr/bin/env python3
"""
Test script to verify all components of the Grid RL system
"""

import sys
import os
sys.path.append('/home/tarive/persistent_storage/tpu_rl_workspace')

import jax
import jax.numpy as jnp
import numpy as np


def test_environment():
    """Test power grid environment"""
    print("\n1. Testing Power Grid Environment...")
    print("-" * 40)
    
    from grid_rl.environments.power_grid_env import PowerGridEnv, make_power_grid_env
    
    # Test single environment
    env = PowerGridEnv()
    obs, info = env.reset()
    print(f"✅ Single environment created")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space.shape}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"✅ Environment step successful")
    print(f"   Reward: {reward:.2f}")
    
    # Test vectorized environment
    vec_env = make_power_grid_env(num_envs=4)
    obs, info = vec_env.reset()
    print(f"✅ Vectorized environment created")
    print(f"   Batch observation shape: {obs.shape}")
    
    return True


def test_multi_agent():
    """Test multi-agent architecture"""
    print("\n2. Testing Multi-Agent System...")
    print("-" * 40)
    
    from grid_rl.agents.multi_agent_grid_rl import (
        MultiAgentConfig, MultiAgentGridRL, create_multi_agent_state
    )
    
    config = MultiAgentConfig()
    rng = jax.random.PRNGKey(42)
    
    # Create model
    state = create_multi_agent_state(config, rng)
    print(f"✅ Multi-agent model created")
    
    # Test forward pass
    batch_size = 4
    strategic_obs = jnp.ones((batch_size, config.strategic_obs_dim))
    operational_obs = jnp.ones((batch_size, config.operational_obs_dim))
    safety_obs = jnp.ones((batch_size, config.safety_obs_dim))
    
    outputs = state.apply_fn(
        {'params': state.params},
        strategic_obs,
        operational_obs,
        safety_obs
    )
    
    print(f"✅ Forward pass successful")
    print(f"   Strategic agent output shape: {outputs['strategic']['logits'].shape}")
    print(f"   Number of operational agents: {len(outputs['operational'])}")
    print(f"   Safety override available: {outputs['safety']['override'] is not None}")
    print(f"   Coordination attention shape: {outputs['coordination']['attention'].shape}")
    
    return True


def test_jax_tpu():
    """Test JAX and TPU configuration"""
    print("\n3. Testing JAX/TPU Setup...")
    print("-" * 40)
    
    print(f"✅ JAX version: {jax.__version__}")
    print(f"✅ Devices available: {jax.device_count()}")
    
    devices = jax.devices()
    for i, device in enumerate(devices):
        print(f"   Device {i}: {device.device_kind} - {device}")
    
    # Test simple TPU operation
    @jax.jit
    def simple_matmul(x, y):
        return jnp.matmul(x, y)
    
    x = jnp.ones((1000, 1000))
    y = jnp.ones((1000, 1000))
    
    # Warm-up
    result = simple_matmul(x, y).block_until_ready()
    
    # Time it
    import time
    start = time.time()
    for _ in range(100):
        result = simple_matmul(x, y).block_until_ready()
    elapsed = time.time() - start
    
    tflops = (2 * 1000**3 * 100) / (elapsed * 1e12)
    print(f"✅ Matrix multiplication test: {tflops:.2f} TFLOPS")
    
    return True


def test_training_components():
    """Test training components integration"""
    print("\n4. Testing Training Components...")
    print("-" * 40)
    
    # We can't fully test the distributed components without starting threads,
    # but we can verify imports and basic setup
    
    try:
        from grid_rl.train_grid_rl_tpu import (
            GridRLConfig, GridEnvironmentFactory, GridRLActor, GridRLLearner
        )
        print(f"✅ Training components imported successfully")
        
        config = GridRLConfig()
        print(f"✅ Training configuration created")
        print(f"   Total timesteps: {config.total_timesteps:,}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Trajectory length: {config.trajectory_length}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to import training components: {e}")
        return False


def test_evaluation():
    """Test evaluation suite"""
    print("\n5. Testing Evaluation Suite...")
    print("-" * 40)
    
    from grid_rl.evaluate import EvaluationMetrics, GridRLEvaluator
    
    # Test metrics
    metrics = EvaluationMetrics(
        uptime_percentage=99.5,
        safety_violations=2,
        average_reward=150.0,
        success_rate=0.95
    )
    print(f"✅ Evaluation metrics created")
    print(f"   Uptime: {metrics.uptime_percentage:.1f}%")
    print(f"   Success rate: {metrics.success_rate:.2%}")
    
    # Note: Can't test evaluator without a checkpoint
    print(f"✅ Evaluation framework available")
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("POWER GRID RL SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Environment", test_environment),
        ("Multi-Agent", test_multi_agent),
        ("JAX/TPU", test_jax_tpu),
        ("Training", test_training_components),
        ("Evaluation", test_evaluation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} test failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:15} {status}")
        all_passed = all_passed and success
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready for training.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)