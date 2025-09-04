#!/usr/bin/env python3
"""
Evaluation suite for Power Grid RL system
Tests robustness, safety, and performance metrics
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any
import pickle
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
from datetime import datetime

from grid_rl.environments.power_grid_env import PowerGridEnv, GridConfig
from grid_rl.agents.multi_agent_grid_rl import MultiAgentConfig, MultiAgentGridRL


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for grid RL"""
    # Reliability metrics
    uptime_percentage: float = 0.0
    n_minus_1_violations: int = 0
    load_shedding_events: int = 0
    voltage_stability_margin: float = 0.0
    
    # Economic metrics
    total_cost: float = 0.0
    cost_per_mwh: float = 0.0
    market_efficiency: float = 0.0
    
    # Safety metrics
    safety_violations: int = 0
    emergency_actions: int = 0
    constraint_violations: int = 0
    cascading_events_prevented: int = 0
    
    # Environmental metrics
    carbon_emissions: float = 0.0
    renewable_utilization: float = 0.0
    
    # Learning metrics
    average_reward: float = 0.0
    episode_length: float = 0.0
    success_rate: float = 0.0


class GridRLEvaluator:
    """Evaluation framework for grid RL agents"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: MultiAgentConfig = None
    ):
        """Load trained model for evaluation"""
        self.config = config or MultiAgentConfig()
        
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            self.params = checkpoint['params']
        
        # Create model
        self.model = MultiAgentGridRL(self.config)
        
        # Test scenarios
        self.test_scenarios = self._create_test_scenarios()
    
    def _create_test_scenarios(self) -> Dict[str, Dict]:
        """Create comprehensive test scenarios"""
        scenarios = {
            'normal_operation': {
                'description': 'Normal grid conditions',
                'load_factor': 1.0,
                'failure_prob': 0.0,
                'renewable_variability': 0.1
            },
            'peak_demand': {
                'description': 'Peak demand conditions',
                'load_factor': 1.5,
                'failure_prob': 0.0,
                'renewable_variability': 0.1
            },
            'equipment_failure': {
                'description': 'Single generator failure',
                'load_factor': 1.0,
                'failure_prob': 0.1,
                'renewable_variability': 0.1
            },
            'renewable_intermittency': {
                'description': 'High renewable variability',
                'load_factor': 1.0,
                'failure_prob': 0.0,
                'renewable_variability': 0.5
            },
            'cascading_failure': {
                'description': 'Multiple equipment failures',
                'load_factor': 1.2,
                'failure_prob': 0.2,
                'renewable_variability': 0.3
            },
            'extreme_weather': {
                'description': 'Extreme weather event',
                'load_factor': 0.8,
                'failure_prob': 0.3,
                'renewable_variability': 0.7
            }
        }
        return scenarios
    
    def evaluate_scenario(
        self,
        scenario_name: str,
        num_episodes: int = 10,
        render: bool = False
    ) -> EvaluationMetrics:
        """Evaluate agent on specific scenario"""
        scenario = self.test_scenarios[scenario_name]
        metrics = EvaluationMetrics()
        
        # Create environment with scenario parameters
        grid_config = GridConfig()
        env = PowerGridEnv(grid_config)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            
            # Apply scenario conditions
            self._apply_scenario_conditions(env, scenario)
            
            while not (done or truncated):
                # Get action from multi-agent system
                action = self._get_action(obs)
                
                # Environment step
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Update metrics
                self._update_metrics(metrics, info)
                
                if render and episode == 0:
                    env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Compute final metrics
        metrics.average_reward = np.mean(episode_rewards)
        metrics.episode_length = np.mean(episode_lengths)
        metrics.success_rate = np.mean([r > 0 for r in episode_rewards])
        
        return metrics
    
    def _get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from multi-agent system"""
        # Prepare observations for each agent
        strategic_obs = obs[:self.config.strategic_obs_dim]
        operational_obs = obs[self.config.strategic_obs_dim:
                             self.config.strategic_obs_dim + self.config.operational_obs_dim]
        safety_obs = obs[-self.config.safety_obs_dim:]
        
        # Add batch dimension
        strategic_obs = jnp.expand_dims(jnp.array(strategic_obs), 0)
        operational_obs = jnp.expand_dims(jnp.array(operational_obs), 0)
        safety_obs = jnp.expand_dims(jnp.array(safety_obs), 0)
        
        # Get outputs from model
        outputs = self.model.apply(
            {'params': self.params},
            strategic_obs,
            operational_obs,
            safety_obs
        )
        
        # Convert to actions (deterministic for evaluation)
        strategic_action = jnp.argmax(outputs['strategic']['logits'], axis=-1)
        
        operational_actions = []
        for op_output in outputs['operational']:
            op_action = jnp.argmax(op_output['logits'], axis=-1)
            operational_actions.append(op_action)
        
        safety_action = jnp.argmax(outputs['safety']['logits'], axis=-1)
        
        # Check for safety override
        if outputs['safety']['override'][0] > 0.5:
            # Safety override - use only safety action
            action = jnp.zeros((outputs['strategic']['logits'].shape[-1] + 
                              sum(op['logits'].shape[-1] for op in outputs['operational']) +
                              outputs['safety']['logits'].shape[-1]))
            action = action.at[-1].set(safety_action[0])
        else:
            # Combine all actions
            action = jnp.concatenate([
                strategic_action,
                jnp.concatenate(operational_actions),
                safety_action
            ])
        
        return np.array(action[0])
    
    def _apply_scenario_conditions(self, env: PowerGridEnv, scenario: Dict):
        """Apply scenario-specific conditions to environment"""
        # Modify load based on scenario
        env.base_load *= scenario['load_factor']
        
        # Apply equipment failures
        if np.random.random() < scenario['failure_prob']:
            # Randomly fail a generator
            failed_gen = np.random.choice(env.config.num_generators)
            env.gen_max[failed_gen] = 0
    
    def _update_metrics(self, metrics: EvaluationMetrics, info: Dict):
        """Update metrics based on environment info"""
        # Reliability metrics
        if 'violations' in info:
            metrics.constraint_violations += info['violations']
        
        # Economic metrics
        if 'cost' in info:
            metrics.total_cost += info['cost']
        
        # Safety metrics
        if 'frequency' in info:
            freq_deviation = abs(info['frequency'] - 60.0)
            if freq_deviation > 0.5:
                metrics.safety_violations += 1
    
    def run_robustness_tests(self) -> Dict[str, EvaluationMetrics]:
        """Run complete robustness evaluation"""
        results = {}
        
        print("Running robustness tests...")
        print("-" * 60)
        
        for scenario_name in self.test_scenarios:
            print(f"Testing scenario: {scenario_name}")
            metrics = self.evaluate_scenario(scenario_name, num_episodes=10)
            results[scenario_name] = metrics
            
            print(f"  Average Reward: {metrics.average_reward:.2f}")
            print(f"  Success Rate: {metrics.success_rate:.2%}")
            print(f"  Safety Violations: {metrics.safety_violations}")
            print()
        
        return results
    
    def generate_report(self, results: Dict[str, EvaluationMetrics]):
        """Generate evaluation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'scenarios': {}
        }
        
        for scenario_name, metrics in results.items():
            report['scenarios'][scenario_name] = {
                'reliability': {
                    'uptime': metrics.uptime_percentage,
                    'violations': metrics.constraint_violations
                },
                'economics': {
                    'total_cost': metrics.total_cost,
                    'cost_per_mwh': metrics.cost_per_mwh
                },
                'safety': {
                    'violations': metrics.safety_violations,
                    'emergency_actions': metrics.emergency_actions
                },
                'performance': {
                    'average_reward': metrics.average_reward,
                    'success_rate': metrics.success_rate
                }
            }
        
        # Save report
        report_path = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {report_path}")
        return report
    
    def visualize_results(self, results: Dict[str, EvaluationMetrics]):
        """Create visualization of evaluation results"""
        scenarios = list(results.keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Average rewards
        rewards = [results[s].average_reward for s in scenarios]
        axes[0, 0].bar(scenarios, rewards)
        axes[0, 0].set_title('Average Reward by Scenario')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Success rates
        success_rates = [results[s].success_rate for s in scenarios]
        axes[0, 1].bar(scenarios, success_rates)
        axes[0, 1].set_title('Success Rate by Scenario')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Safety violations
        violations = [results[s].safety_violations for s in scenarios]
        axes[1, 0].bar(scenarios, violations, color='red')
        axes[1, 0].set_title('Safety Violations by Scenario')
        axes[1, 0].set_ylabel('Violations')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Total costs
        costs = [results[s].total_cost for s in scenarios]
        axes[1, 1].bar(scenarios, costs, color='green')
        axes[1, 1].set_title('Total Cost by Scenario')
        axes[1, 1].set_ylabel('Cost ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png')
        plt.show()
        
        print("Visualization saved to evaluation_results.png")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Grid RL Agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--scenarios', nargs='+', default=['all'],
                       help='Scenarios to evaluate')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes per scenario')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = GridRLEvaluator(args.checkpoint)
    
    # Run evaluation
    if 'all' in args.scenarios:
        results = evaluator.run_robustness_tests()
    else:
        results = {}
        for scenario in args.scenarios:
            if scenario in evaluator.test_scenarios:
                results[scenario] = evaluator.evaluate_scenario(
                    scenario, 
                    num_episodes=args.episodes
                )
    
    # Generate report
    report = evaluator.generate_report(results)
    
    # Visualize if requested
    if args.visualize:
        evaluator.visualize_results(results)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    # Quick test
    print("Grid RL Evaluation Suite")
    print("=" * 60)
    
    # For testing without checkpoint, create dummy metrics
    metrics = EvaluationMetrics(
        uptime_percentage=99.5,
        safety_violations=2,
        average_reward=150.0,
        success_rate=0.95
    )
    
    print(f"Sample Metrics:")
    print(f"  Uptime: {metrics.uptime_percentage:.1f}%")
    print(f"  Safety Violations: {metrics.safety_violations}")
    print(f"  Average Reward: {metrics.average_reward:.2f}")
    print(f"  Success Rate: {metrics.success_rate:.2%}")