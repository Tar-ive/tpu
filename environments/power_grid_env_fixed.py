#!/usr/bin/env python3
"""
Fixed Power Grid Environment with Normalized Rewards
Implements proper reward scaling and numerical stability
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
    
    # Reward weights (normalized)
    liability_weight: float = 0.5
    cost_weight: float = 0.3
    uptime_weight: float = 0.2
    
    # Reward normalization parameters
    reward_scale: float = 1.0  # Scale factor for final reward
    reward_clip: float = 10.0  # Clip rewards to [-clip, clip]
    use_running_stats: bool = True  # Use running statistics for normalization


class RewardNormalizer:
    """Running statistics for reward normalization"""
    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        
    def update(self, reward):
        """Update running statistics"""
        self.count += 1
        
        # Update mean
        delta = reward - self.mean
        self.mean += (1 - self.alpha) * delta
        
        # Update variance
        self.var = self.alpha * self.var + (1 - self.alpha) * delta * delta
        
    def normalize(self, reward):
        """Normalize reward using running statistics"""
        if self.count < 10:  # Don't normalize until we have enough samples
            return np.clip(reward / 100.0, -1.0, 1.0)
        
        std = np.sqrt(self.var + 1e-8)
        normalized = (reward - self.mean) / std
        return np.clip(normalized, -1.0, 1.0)


class PowerGridEnvFixed(gym.Env):
    """
    Fixed Power Grid Environment with normalized rewards
    """
    
    def __init__(self, config: Optional[GridConfig] = None, seed: int = 42):
        super().__init__()
        self.config = config or GridConfig()
        self.np_random = np.random.RandomState(seed)
        
        # Initialize reward normalizers for each component
        self.liability_normalizer = RewardNormalizer()
        self.cost_normalizer = RewardNormalizer()
        self.uptime_normalizer = RewardNormalizer()
        self.total_normalizer = RewardNormalizer()
        
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
            admittance = self.np_random.uniform(5, 10) - 1j * self.np_random.uniform(1, 3)
            Y[i, i+1] = -admittance
            Y[i+1, i] = -admittance
            Y[i, i] += admittance
            Y[i+1, i+1] += admittance
            
        # Add some random connections for mesh network
        for _ in range(n // 2):
            i, j = self.np_random.choice(n, 2, replace=False)
            if Y[i, j] == 0:
                admittance = self.np_random.uniform(3, 7) - 1j * self.np_random.uniform(0.5, 2)
                Y[i, j] = -admittance
                Y[j, i] = -admittance
                Y[i, i] += admittance
                Y[j, j] += admittance
                
        return Y
    
    def _init_spaces(self):
        """Define observation and action spaces"""
        # Observation space: bus voltages, phases, generation, loads, frequency
        obs_dim = (
            self.config.num_buses * 4 +  # Voltage mag/phase, P/Q injections
            self.config.num_generators +  # Generation levels
            self.config.num_loads +  # Load levels
            1  # System frequency
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: generator dispatch and load control
        action_dim = self.config.num_generators + self.config.num_loads
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state"""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            
        # Initialize bus states
        self.bus_voltages = np.ones(self.config.num_buses)  # p.u.
        self.bus_phases = np.zeros(self.config.num_buses)  # radians
        
        # Initialize generation at mid-range
        self.generation = (self.gen_min + self.gen_max) / 2
        
        # Initialize loads at base values with small variation
        self.current_load = self.base_load * self.np_random.uniform(0.9, 1.1, self.config.num_loads)
        
        # System frequency
        self.frequency = self.config.frequency_nominal
        
        # Episode tracking
        self.timestep = 0
        self.total_cost = 0
        self.violations = 0
        
        # Power flow calculation
        self._simulate_power_flow()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Parse actions
        gen_actions = action[:self.config.num_generators]
        load_actions = action[self.config.num_generators:]
        
        # Apply generation control (bounded)
        gen_change = np.clip(
            gen_actions * self.config.max_generation_change,
            -self.config.max_generation_change,
            self.config.max_generation_change
        )
        self.generation = np.clip(
            self.generation * (1 + gen_change),
            self.gen_min,
            self.gen_max
        )
        
        # Apply load control (only reduction allowed)
        load_factor = np.clip(1 + load_actions * self.config.max_load_shed, 0.8, 1.0)
        self.current_load = self.base_load * load_factor
        
        # Simulate power flow
        self._simulate_power_flow()
        
        # Calculate reward (NORMALIZED)
        reward = self._calculate_reward_normalized()
        
        # Check termination
        self.timestep += 1
        terminated = self.timestep >= self.config.episode_length
        truncated = False
        
        # Add stability bonus for surviving
        if not terminated:
            reward += 0.01  # Small positive reward for stability
        
        info = {
            'timestep': self.timestep,
            'total_cost': self.total_cost,
            'violations': self.violations,
            'raw_generation': np.sum(self.generation),
            'raw_load': np.sum(self.current_load),
            'frequency': self.frequency,
            'voltage_mean': np.mean(self.bus_voltages),
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _simulate_power_flow(self):
        """Simplified power flow calculation"""
        # Calculate power imbalance
        total_gen = np.sum(self.generation)
        total_load = np.sum(self.current_load)
        imbalance = total_gen - total_load
        
        # Update frequency based on imbalance
        freq_change = imbalance * 0.1  # Simplified frequency response
        self.frequency = np.clip(
            self.frequency + freq_change * self.config.dt,
            self.config.frequency_nominal - self.config.frequency_tolerance,
            self.config.frequency_nominal + self.config.frequency_tolerance
        )
        
        # Update voltages (simplified)
        for i in range(self.config.num_buses):
            # Voltage affected by local power balance
            local_gen = self.generation[self.gen_buses == i].sum() if i in self.gen_buses else 0
            local_load = self.current_load[self.load_buses == i].sum() if i in self.load_buses else 0
            
            voltage_change = (local_gen - local_load) * 0.01
            self.bus_voltages[i] = np.clip(
                self.bus_voltages[i] + voltage_change * self.config.dt,
                self.config.voltage_min * 0.9,
                self.config.voltage_max * 1.1
            )
    
    def _get_observation(self):
        """Get current observation"""
        obs = np.concatenate([
            self.bus_voltages,
            self.bus_phases,
            np.zeros(self.config.num_buses),  # P injections (simplified)
            np.zeros(self.config.num_buses),  # Q injections (simplified)
            self.generation,
            self.current_load,
            [self.frequency]
        ])
        
        # Normalize observation
        obs = np.clip(obs / 10.0, -10.0, 10.0)  # Simple normalization
        
        return obs.astype(np.float32)
    
    def _calculate_reward_normalized(self) -> float:
        """Calculate normalized reward with proper scaling"""
        
        # Component rewards (before normalization)
        liability_raw = self._compute_liability_reward()
        cost_raw = self._compute_cost_reward()
        uptime_raw = self._compute_uptime_reward()
        
        # Update running statistics if enabled
        if self.config.use_running_stats:
            self.liability_normalizer.update(liability_raw)
            self.cost_normalizer.update(cost_raw)
            self.uptime_normalizer.update(uptime_raw)
            
            # Normalize each component
            liability_norm = self.liability_normalizer.normalize(liability_raw)
            cost_norm = self.cost_normalizer.normalize(cost_raw)
            uptime_norm = self.uptime_normalizer.normalize(uptime_raw)
        else:
            # Simple normalization
            liability_norm = np.tanh(liability_raw / 100.0)
            cost_norm = np.tanh(cost_raw / 100.0)
            uptime_norm = np.tanh(uptime_raw / 10.0)
        
        # Check for safety violations (normalized penalty)
        if self._check_safety_violation():
            self.violations += 1
            return -1.0  # Normalized safety penalty
        
        # Weighted combination
        total_reward = (
            self.config.liability_weight * liability_norm +
            self.config.cost_weight * cost_norm +
            self.config.uptime_weight * uptime_norm
        )
        
        # Final normalization and clipping
        total_reward = total_reward * self.config.reward_scale
        total_reward = np.clip(total_reward, -self.config.reward_clip, self.config.reward_clip)
        
        # Ensure no NaN or Inf
        if not np.isfinite(total_reward):
            total_reward = -1.0
            
        return float(total_reward)
    
    def _compute_liability_reward(self) -> float:
        """Reward for meeting power demand and maintaining stability"""
        # Power balance error (normalized by total load)
        total_load = np.sum(self.current_load)
        if total_load > 0:
            imbalance = abs(np.sum(self.generation) - total_load) / total_load
            power_reward = -imbalance * 10
        else:
            power_reward = 0
        
        # Voltage stability (normalized by number of buses)
        voltage_violations = np.sum(
            np.maximum(0, self.bus_voltages - self.config.voltage_max) +
            np.maximum(0, self.config.voltage_min - self.bus_voltages)
        )
        voltage_reward = -voltage_violations / self.config.num_buses * 10
        
        # Frequency stability (normalized)
        freq_deviation = abs(self.frequency - self.config.frequency_nominal) / self.config.frequency_tolerance
        freq_reward = -freq_deviation * 10
        
        return power_reward + voltage_reward + freq_reward
    
    def _compute_cost_reward(self) -> float:
        """Reward for minimizing operational costs (normalized)"""
        # Generation cost (normalized by capacity)
        total_capacity = np.sum(self.gen_max)
        if total_capacity > 0:
            gen_cost = np.sum(self.generation * self.gen_cost) / (total_capacity * np.mean(self.gen_cost))
        else:
            gen_cost = 0
        
        # Load shedding cost (normalized)
        total_base_load = np.sum(self.base_load)
        if total_base_load > 0:
            shed_ratio = np.sum(self.base_load - self.current_load) / total_base_load
            shed_cost = shed_ratio * 10  # High penalty for load shedding
        else:
            shed_cost = 0
        
        total_cost = gen_cost + shed_cost
        self.total_cost += total_cost * self.config.dt
        
        return -total_cost
    
    def _compute_uptime_reward(self) -> float:
        """Reward for maximizing system uptime and reliability (normalized)"""
        # Equipment utilization efficiency
        gen_efficiency = np.mean(
            (self.generation - self.gen_min) / (self.gen_max - self.gen_min + 1e-8)
        )
        
        # Reward moderate utilization (30-90%)
        if 0.3 <= gen_efficiency <= 0.9:
            efficiency_reward = gen_efficiency
        else:
            efficiency_reward = -1.0
        
        # Reserve margin (normalized)
        available_capacity = np.sum(self.gen_max)
        current_load = np.sum(self.current_load)
        if current_load > 0:
            reserve_margin = (available_capacity - current_load) / current_load
            
            if reserve_margin > 0.15:  # Reward >15% reserve
                reserve_reward = 1.0
            else:
                reserve_reward = -1.0
        else:
            reserve_reward = 0
        
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
        
        # Line overload (simplified check)
        if np.any(self.generation > self.gen_max * 1.1):
            return True
            
        return False
    
    def render(self):
        """Render the environment (text output for now)"""
        print(f"\n===== Grid State (t={self.timestep}) =====")
        print(f"Frequency: {self.frequency:.2f} Hz")
        print(f"Total Generation: {np.sum(self.generation):.2f} MW")
        print(f"Total Load: {np.sum(self.current_load):.2f} MW")
        print(f"Imbalance: {np.sum(self.generation) - np.sum(self.current_load):.2f} MW")
        print(f"Voltage Range: [{np.min(self.bus_voltages):.3f}, {np.max(self.bus_voltages):.3f}] p.u.")
        print(f"Violations: {self.violations}")
        print(f"Total Cost: ${self.total_cost:.2f}")
        print("=" * 40)