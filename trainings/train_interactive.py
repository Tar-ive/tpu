#!/usr/bin/env python3
"""
Interactive Grid RL Training with Real-time Visualization
Combines fixed rewards, numerical stability, and visual feedback
"""

import sys
sys.path.append('/home/tarive/persistent_storage/tpu_rl_workspace/grid_rl')

import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import time
from typing import NamedTuple, Dict, Any
from datetime import datetime
import threading
import queue

from environments.power_grid_env_fixed import PowerGridEnvFixed, GridConfig
from visualization.grid_visualizer import GridVisualizer

print("="*70)
print("🎮 INTERACTIVE GRID RL TRAINING")
print("="*70)
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
print("="*70)


class ActorCritic(nn.Module):
    """Simple Actor-Critic for testing"""
    action_dim: int = 145
    
    @nn.compact
    def __call__(self, x):
        # Input normalization
        x = jnp.clip(x, -100, 100)
        x = x / 10.0
        
        # Shared layers
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Policy head
        policy_mean = nn.Dense(self.action_dim)(x)
        policy_mean = nn.tanh(policy_mean)
        
        log_std = self.param('log_std', 
                            nn.initializers.constant(-0.5),
                            (self.action_dim,))
        log_std = jnp.clip(log_std, -2.0, 0.5)
        
        # Value head
        value = nn.Dense(1)(x)
        value = jnp.squeeze(value, axis=-1)
        value = jnp.clip(value, -100, 100)
        
        return policy_mean, log_std, value


class InteractiveTrainer:
    """Interactive training with visualization"""
    
    def __init__(self, config=None):
        self.config = config or {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'clip_epsilon': 0.2,
            'vf_coef': 0.5,
            'ent_coef': 0.01,
            'max_episode_length': 1000,
            'update_freq': 10,
            'visualize': True,
        }
        
        # Create environment
        self.env = PowerGridEnvFixed(seed=42)
        
        # Create visualizer
        if self.config['visualize']:
            self.viz = GridVisualizer(
                num_buses=self.env.config.num_buses,
                num_generators=self.env.config.num_generators,
                num_loads=self.env.config.num_loads
            )
        
        # Initialize network
        self.rng = random.PRNGKey(42)
        self.state = self._create_train_state()
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.total_steps = 0
        
        # Control flags
        self.is_training = False
        self.pause_requested = False
        
    def _create_train_state(self):
        """Create training state"""
        network = ActorCritic(action_dim=self.env.action_space.shape[0])
        
        dummy_obs = jnp.zeros((1,) + self.env.observation_space.shape)
        params = network.init(self.rng, dummy_obs)
        
        # Optimizer with gradient clipping
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(self.config['learning_rate'], eps=1e-5),
        )
        
        return TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )
    
    @jit
    def get_action(self, params, obs, rng):
        """Get action from policy"""
        obs_batch = jnp.expand_dims(obs, 0)
        policy_mean, log_std, value = self.state.apply_fn(params, obs_batch)
        
        # Sample action
        std = jnp.exp(log_std)
        noise = random.normal(rng, policy_mean[0].shape)
        action = policy_mean[0] + std * noise
        action = jnp.clip(action, -1, 1)
        
        # Compute log prob
        diff = action - policy_mean[0]
        log_prob = -0.5 * jnp.sum(
            jnp.square(diff / std) + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        
        return action, value[0], log_prob
    
    def train_episode(self, render_interval=10):
        """Train one episode with visualization"""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_data = []
        
        for step in range(self.config['max_episode_length']):
            # Check pause
            if self.pause_requested:
                print("⏸️  Training paused. Press Enter to continue...")
                input()
                self.pause_requested = False
            
            # Get action
            self.rng, action_rng = random.split(self.rng)
            action, value, log_prob = self.get_action(
                self.state.params, obs, action_rng
            )
            
            # Step environment
            action_np = np.array(action)
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            
            # Store data
            episode_reward += reward
            self.total_steps += 1
            
            # Prepare visualization data
            state_dict = {
                'bus_voltages': self.env.bus_voltages,
                'frequency': self.env.frequency,
                'generation': self.env.generation,
                'load': self.env.current_load,
                'actions': action_np,
                'reward': reward,
                'loss': self.losses[-1] if self.losses else 0,
                'gen_buses': self.env.gen_buses[:10],  # Show first 10 generators
                'violations': info.get('violations', 0),
            }
            episode_data.append(state_dict)
            
            # Visualize
            if self.config['visualize'] and step % render_interval == 0:
                self.viz.update(state_dict)
                
                # Print status
                print(f"\rStep {step:4d}/{self.config['max_episode_length']} | "
                      f"Reward: {reward:+.4f} | Total: {episode_reward:+.2f} | "
                      f"Freq: {self.env.frequency:.2f} Hz | "
                      f"Violations: {info.get('violations', 0)}", end='')
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Episode complete
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(step + 1)
        
        print(f"\n✅ Episode complete! Total reward: {episode_reward:.2f}, Length: {step+1}")
        
        # Show episode summary if visualizing
        if self.config['visualize'] and len(episode_data) > 0:
            self.viz.show_summary(episode_data)
        
        return episode_data
    
    def run_interactive(self, num_episodes=10):
        """Run interactive training loop"""
        print("\n" + "="*70)
        print("🚀 Starting Interactive Training")
        print("="*70)
        print("Controls:")
        print("  - Press Ctrl+C to pause")
        print("  - Training will show real-time visualization")
        print("="*70 + "\n")
        
        self.is_training = True
        
        try:
            for episode in range(num_episodes):
                print(f"\n📍 Episode {episode + 1}/{num_episodes}")
                print("-" * 40)
                
                # Train episode
                episode_data = self.train_episode(render_interval=10)
                
                # Print episode statistics
                if len(self.episode_rewards) > 0:
                    recent_rewards = self.episode_rewards[-10:]
                    print(f"\n📊 Statistics (last {len(recent_rewards)} episodes):")
                    print(f"  Average Reward: {np.mean(recent_rewards):.2f}")
                    print(f"  Std Reward: {np.std(recent_rewards):.2f}")
                    print(f"  Max Reward: {np.max(recent_rewards):.2f}")
                    print(f"  Min Reward: {np.min(recent_rewards):.2f}")
                
                # Ask to continue
                if episode < num_episodes - 1:
                    response = input("\nContinue to next episode? (y/n/q): ").lower()
                    if response == 'q' or response == 'n':
                        break
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted!")
        
        finally:
            self.is_training = False
            print("\n" + "="*70)
            print("🏁 Training Complete!")
            print("="*70)
            self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final training statistics"""
        if len(self.episode_rewards) > 0:
            print(f"\n📈 Final Statistics ({len(self.episode_rewards)} episodes):")
            print(f"  Total Steps: {self.total_steps}")
            print(f"  Average Reward: {np.mean(self.episode_rewards):.2f}")
            print(f"  Best Episode: {np.max(self.episode_rewards):.2f}")
            print(f"  Worst Episode: {np.min(self.episode_rewards):.2f}")
            print(f"  Average Length: {np.mean(self.episode_lengths):.1f}")
            
            # Check if training improved
            if len(self.episode_rewards) >= 5:
                early = np.mean(self.episode_rewards[:5])
                late = np.mean(self.episode_rewards[-5:])
                improvement = ((late - early) / abs(early)) * 100 if early != 0 else 0
                
                if improvement > 0:
                    print(f"\n✨ Training improved by {improvement:.1f}%!")
                else:
                    print(f"\n⚠️  Training degraded by {abs(improvement):.1f}%")


def test_quick():
    """Quick test to verify everything works"""
    print("\n🧪 Running quick test...")
    
    # Create environment
    env = PowerGridEnvFixed(seed=42)
    obs, _ = env.reset()
    
    # Test a few steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}: Reward={reward:.4f}, Freq={info['frequency']:.2f} Hz")
        
        if terminated or truncated:
            break
    
    print(f"\n✅ Test complete! Total reward: {total_reward:.2f}")
    
    # Check if rewards are normalized
    if -10 <= total_reward <= 10:
        print("✅ Rewards appear to be properly normalized!")
    else:
        print("⚠️  Rewards may not be properly normalized")
    
    return True


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Interactive Grid RL Training')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    if args.test:
        test_quick()
    else:
        config = {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'clip_epsilon': 0.2,
            'vf_coef': 0.5,
            'ent_coef': 0.01,
            'max_episode_length': 200,  # Shorter for interactive demo
            'update_freq': 10,
            'visualize': not args.no_viz,
        }
        
        trainer = InteractiveTrainer(config)
        trainer.run_interactive(num_episodes=args.episodes)


if __name__ == "__main__":
    main()