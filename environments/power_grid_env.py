#!/usr/bin/env python3
"""
Power Grid Environment for Sebulba RL Framework
Implements a simplified IEEE 118-bus power grid simulation
Compatible with gym interface and Sebulba's distributed architecture
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
import math


@dataclass
class GridConfig:
    """Configuration for power grid simulation"""
    num_buses: int = 118  # IEEE 118-bus system
    num_generators: int = 54
    num_loads: int = 91
    num_lines: int = 186
    base_mva: float = 100.0
    
    # Grid operational limits
    voltage_min: float = 0.95  # p.u.
    voltage_max: float = 1.05  # p.u.
    frequency_nominal: float = 60.0  # Hz
    frequency_tolerance: float = 0.5  # Hz
    
    # Control parameters
    max_generation_change: float = 0.1  # 10% per step
    max_load_shed: float = 0.2  # 20% maximum load shedding
    
    # Simulation parameters
    dt: float = 0.1  # Time step in seconds
    episode_length: int = 1000
    
    # Reward weights (layered objectives)
    liability_weight: float = 0.5
    cost_weight: float = 0.3
    uptime_weight: float = 0.2
    safety_penalty: float = -1000.0


class PowerGridEnv(gym.Env):
    """
    Power Grid Environment implementing multi-agent control
    State: Grid voltages, power flows, generation, load, equipment status
    Action: Generator dispatch, transmission switching, load control
    """
    
    def __init__(self, config: Optional[GridConfig] = None, seed: int = 42):
        super().__init__()
        self.config = config or GridConfig()
        self.np_random = np.random.RandomState(seed)
        
        # Initialize grid topology (simplified IEEE 118-bus)
        self._init_grid_topology()
        
        # Define observation and action spaces
        self._init_spaces()
        
        # Initialize state variables
        self.reset()
        
    def _init_grid_topology(self):
        """Initialize simplified grid topology and parameters"""
        n = self.config.num_buses
        
        # Bus admittance matrix (simplified)
        self.Y_bus = self._create_admittance_matrix()
        
        # Generator parameters
        self.gen_buses = self.np_random.choice(
            n, self.config.num_generators, replace=False
        )
        self.gen_min = np.ones(self.config.num_generators) * 0.1  # Min generation
        self.gen_max = np.ones(self.config.num_generators) * 2.0  # Max generation
        self.gen_cost = self.np_random.uniform(20, 100, self.config.num_generators)  # $/MWh
        
        # Load parameters
        self.load_buses = self.np_random.choice(
            n, self.config.num_loads, replace=False
        )
        self.base_load = self.np_random.uniform(0.5, 2.0, self.config.num_loads)
        
        # Line parameters
        self.line_limits = np.ones(self.config.num_lines) * 1.5  # Thermal limits
        
    def _create_admittance_matrix(self):
        """Create simplified bus admittance matrix"""
        n = self.config.num_buses
        Y = np.zeros((n, n), dtype=complex)
        
        # Create a connected network (simplified)
        for i in range(n - 1):
            # Connect consecutive buses
            admittance = 1.0 - 0.1j  # Simplified line admittance
            Y[i, i+1] = -admittance
            Y[i+1, i] = -admittance
            Y[i, i] += admittance
            Y[i+1, i+1] += admittance
            
        # Add some additional connections for redundancy
        for _ in range(self.config.num_lines - n + 1):
            i, j = self.np_random.choice(n, 2, replace=False)
            admittance = 1.0 - 0.1j
            Y[i, j] -= admittance
            Y[j, i] -= admittance
            Y[i, i] += admittance
            Y[j, j] += admittance
            
        return Y
    
    def _init_spaces(self):
        """Define observation and action spaces"""
        # State space (per bus): voltage, angle, P, Q
        obs_dim = self.config.num_buses * 4
        obs_dim += self.config.num_generators  # Generator status
        obs_dim += self.config.num_loads  # Load levels
        obs_dim += 1  # System frequency
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: generator dispatch + load control
        act_dim = self.config.num_generators  # Generator setpoints
        act_dim += self.config.num_loads  # Load control (shedding)
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # Initialize bus voltages (all at nominal)
        self.bus_voltages = np.ones(self.config.num_buses)
        self.bus_angles = np.zeros(self.config.num_buses)
        
        # Initialize generation and load
        self.generation = self.gen_min + (self.gen_max - self.gen_min) * 0.5
        self.current_load = self.base_load.copy()
        
        # System frequency
        self.frequency = self.config.frequency_nominal
        
        # Episode tracking
        self.timestep = 0
        self.total_cost = 0.0
        self.violations = 0
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Parse actions
        gen_actions = action[:self.config.num_generators]
        load_actions = action[self.config.num_generators:]
        
        # Apply generator dispatch changes
        gen_changes = gen_actions * self.config.max_generation_change
        self.generation = np.clip(
            self.generation + gen_changes,
            self.gen_min,
            self.gen_max
        )
        
        # Apply load control (shedding if necessary)
        load_factors = 1.0 + load_actions * self.config.max_load_shed
        load_factors = np.clip(load_factors, 0.8, 1.0)  # Allow 20% shedding max
        self.current_load = self.base_load * load_factors
        
        # Run simplified power flow
        self._run_power_flow()
        
        # Check constraints and calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        self.timestep += 1
        terminated = self._check_termination()
        truncated = self.timestep >= self.config.episode_length
        
        # Collect info
        info = {
            'total_generation': np.sum(self.generation),
            'total_load': np.sum(self.current_load),
            'frequency': self.frequency,
            'violations': self.violations,
            'cost': self.total_cost
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _run_power_flow(self):
        """Simplified power flow calculation"""
        # Calculate power imbalance
        total_gen = np.sum(self.generation)
        total_load = np.sum(self.current_load)
        imbalance = total_gen - total_load
        
        # Update frequency based on imbalance
        freq_deviation = imbalance * 0.1  # Simplified frequency response
        self.frequency = self.config.frequency_nominal + freq_deviation
        
        # Update voltages (simplified - normally would solve AC power flow)
        for i, gen_idx in enumerate(self.gen_buses):
            # Generators help maintain voltage
            self.bus_voltages[gen_idx] = 1.0 + 0.01 * (self.generation[i] / self.gen_max[i])
        
        for i, load_idx in enumerate(self.load_buses):
            # Loads cause voltage drop
            self.bus_voltages[load_idx] = 1.0 - 0.01 * (self.current_load[i] / 2.0)
        
        # Clip voltages to limits
        self.bus_voltages = np.clip(
            self.bus_voltages,
            self.config.voltage_min,
            self.config.voltage_max
        )
    
    def _calculate_reward(self) -> float:
        """Calculate layered objective reward function"""
        # Primary: Liability/Safety
        liability_reward = self._compute_liability_reward()
        
        # Check safety violations
        if self._check_safety_violation():
            self.violations += 1
            return self.config.safety_penalty
        
        # Secondary: Cost
        cost_reward = self._compute_cost_reward()
        
        # Tertiary: Uptime
        uptime_reward = self._compute_uptime_reward()
        
        # Weighted combination
        total_reward = (
            self.config.liability_weight * liability_reward +
            self.config.cost_weight * cost_reward +
            self.config.uptime_weight * uptime_reward
        )
        
        return total_reward
    
    def _compute_liability_reward(self) -> float:
        """Reward for meeting power demand and maintaining stability"""
        # Power balance error
        imbalance = abs(np.sum(self.generation) - np.sum(self.current_load))
        power_reward = -imbalance * 10
        
        # Voltage stability
        voltage_violations = np.sum(
            np.maximum(0, self.bus_voltages - self.config.voltage_max) +
            np.maximum(0, self.config.voltage_min - self.bus_voltages)
        )
        voltage_reward = -voltage_violations * 100
        
        # Frequency stability
        freq_deviation = abs(self.frequency - self.config.frequency_nominal)
        freq_reward = -freq_deviation * 100
        
        return power_reward + voltage_reward + freq_reward
    
    def _compute_cost_reward(self) -> float:
        """Reward for minimizing operational costs"""
        # Generation cost
        gen_cost = np.sum(self.generation * self.gen_cost)
        
        # Load shedding cost (high penalty)
        shed_load = np.sum(self.base_load - self.current_load)
        shed_cost = shed_load * 1000  # High penalty for load shedding
        
        total_cost = gen_cost + shed_cost
        self.total_cost += total_cost * self.config.dt
        
        return -total_cost * 0.01  # Scale down
    
    def _compute_uptime_reward(self) -> float:
        """Reward for maximizing system uptime and reliability"""
        # Equipment utilization efficiency
        gen_efficiency = np.mean(
            (self.generation - self.gen_min) / (self.gen_max - self.gen_min)
        )
        
        # Reward moderate utilization (30-90%)
        if 0.3 <= gen_efficiency <= 0.9:
            efficiency_reward = gen_efficiency * 10
        else:
            efficiency_reward = -10
        
        # Reserve margin
        available_capacity = np.sum(self.gen_max)
        current_load = np.sum(self.current_load)
        reserve_margin = (available_capacity - current_load) / current_load
        
        if reserve_margin > 0.15:  # Reward >15% reserve
            reserve_reward = 10
        else:
            reserve_reward = -50
        
        return efficiency_reward + reserve_reward
    
    def _check_safety_violation(self) -> bool:
        """Check if any safety constraints are violated"""
        # Voltage violations
        if np.any(self.bus_voltages < self.config.voltage_min * 0.9):
            return True
        if np.any(self.bus_voltages > self.config.voltage_max * 1.1):
            return True
        
        # Frequency violations
        if abs(self.frequency - self.config.frequency_nominal) > self.config.frequency_tolerance:
            return True
        
        return False
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate due to critical failure"""
        # Cascading failure (multiple violations)
        if self.violations > 5:
            return True
        
        # Blackout (severe voltage collapse)
        if np.mean(self.bus_voltages) < 0.9:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        obs = []
        
        # Bus states (voltage, angle, P, Q simplified)
        obs.extend(self.bus_voltages)
        obs.extend(self.bus_angles)
        obs.extend(np.zeros(self.config.num_buses))  # Simplified P
        obs.extend(np.zeros(self.config.num_buses))  # Simplified Q
        
        # Generator states
        obs.extend(self.generation)
        
        # Load states
        obs.extend(self.current_load)
        
        # System frequency
        obs.append(self.frequency)
        
        return np.array(obs, dtype=np.float32)
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.timestep}")
            print(f"Frequency: {self.frequency:.2f} Hz")
            print(f"Mean Voltage: {np.mean(self.bus_voltages):.3f} p.u.")
            print(f"Total Generation: {np.sum(self.generation):.2f} MW")
            print(f"Total Load: {np.sum(self.current_load):.2f} MW")
            print(f"Violations: {self.violations}")


# Vectorized environment wrapper for Sebulba
class VectorizedPowerGridEnv:
    """Vectorized wrapper for parallel environment execution"""
    
    def __init__(self, num_envs: int, config: Optional[GridConfig] = None, seed: int = 42):
        self.num_envs = num_envs
        self.envs = [
            PowerGridEnv(config, seed + i) 
            for i in range(num_envs)
        ]
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset all environments"""
        observations = []
        infos = []
        
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        
        return np.stack(observations), infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Step all environments"""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            
            # Auto-reset if episode ended
            if terminated or truncated:
                obs, reset_info = env.reset()
                info.update(reset_info)
            
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (
            np.stack(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )
    
    @property
    def observation_space(self):
        return self.envs[0].observation_space
    
    @property
    def action_space(self):
        return self.envs[0].action_space
    
    @property
    def single_action_space(self):
        return self.envs[0].action_space
    
    @property
    def single_observation_space(self):
        return self.envs[0].observation_space


def make_power_grid_env(num_envs: int = 1, **kwargs) -> VectorizedPowerGridEnv:
    """Factory function for creating power grid environments"""
    config = GridConfig(**kwargs) if kwargs else GridConfig()
    return VectorizedPowerGridEnv(num_envs, config)


if __name__ == "__main__":
    # Test environment
    env = PowerGridEnv()
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    
    # Test vectorized environment
    vec_env = make_power_grid_env(num_envs=4)
    obs, _ = vec_env.reset()
    print(f"Vectorized observation shape: {obs.shape}")
    
    # Test step
    actions = np.random.randn(4, vec_env.action_space.shape[0])
    obs, rewards, dones, truncs, infos = vec_env.step(actions)
    print(f"Rewards: {rewards}")