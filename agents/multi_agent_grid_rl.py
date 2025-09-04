#!/usr/bin/env python3
"""
Multi-Agent Reinforcement Learning System for Power Grid Control
Implements Strategic, Operational, and Safety agents with coordination
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from typing import Dict, Tuple, Any, List, NamedTuple
import numpy as np
import chex
from functools import partial


class MultiAgentConfig(NamedTuple):
    """Configuration for multi-agent system"""
    # Agent counts and devices
    num_strategic_agents: int = 1
    num_operational_agents: int = 4
    num_safety_agents: int = 1
    
    # Observation dimensions
    strategic_obs_dim: int = 512  # High-level grid state
    operational_obs_dim: int = 256  # Regional grid state
    safety_obs_dim: int = 128  # Critical safety indicators
    
    # Action dimensions
    strategic_action_dim: int = 32  # High-level directives
    operational_action_dim: int = 64  # Control actions
    safety_action_dim: int = 16  # Override actions
    
    # Network architecture
    strategic_hidden_dims: Tuple[int, ...] = (512, 512, 256)
    operational_hidden_dims: Tuple[int, ...] = (256, 256, 128)
    safety_hidden_dims: Tuple[int, ...] = (128, 128, 64)
    
    # Coordination
    num_attention_heads: int = 8
    attention_dim: int = 256
    
    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    
    # Hierarchical parameters
    strategic_horizon: int = 100  # Long-term planning
    operational_horizon: int = 10  # Medium-term control
    safety_horizon: int = 1  # Immediate response


# Network Architectures

class AttentionCoordination(nn.Module):
    """Attention-based coordination mechanism for multi-agent communication"""
    
    num_agents: int
    num_heads: int = 8
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, agent_states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            agent_states: [batch_size, num_agents, feature_dim]
        Returns:
            coordinated_features: [batch_size, num_agents, hidden_dim]
            attention_weights: [batch_size, num_agents, num_agents]
        """
        batch_size = agent_states.shape[0]
        
        # Multi-head self-attention
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=True
        )
        
        # Add positional encoding for agent ordering
        pos_encoding = self.param(
            'pos_encoding',
            nn.initializers.normal(stddev=0.02),
            (self.num_agents, agent_states.shape[-1])
        )
        
        # Broadcast positional encoding to batch dimension
        agent_states = agent_states + pos_encoding[None, :, :]
        
        # Apply attention
        attended_states = attention(agent_states, agent_states)
        
        # Coordination network
        x = nn.Dense(self.hidden_dim)(attended_states)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        
        # Generate coordination weights
        coord_logits = nn.Dense(self.num_agents)(x)
        coord_weights = nn.softmax(coord_logits, axis=-1)
        
        # Apply coordination using batch matrix multiply
        # coord_weights: [batch_size, num_agents, num_agents]
        # agent_states: [batch_size, num_agents, feature_dim]
        coordinated = jnp.einsum('bna,bad->bnd', coord_weights, agent_states)
        
        return coordinated, coord_weights


class StrategicAgent(nn.Module):
    """Strategic agent for long-term grid planning"""
    
    config: MultiAgentConfig
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns policy logits and value estimate
        """
        # Feature extraction
        for dim in self.config.strategic_hidden_dims:
            x = nn.Dense(
                dim,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))
            )(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x)
        
        # Policy head
        policy_logits = nn.Dense(
            self.config.strategic_action_dim,
            kernel_init=nn.initializers.orthogonal(0.01)
        )(x)
        
        # Value head
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0)
        )(x)
        
        return policy_logits, jnp.squeeze(value)


class OperationalAgent(nn.Module):
    """Operational agent for real-time grid control"""
    
    config: MultiAgentConfig
    strategic_guidance_dim: int = 32
    
    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        strategic_guidance: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns policy logits and value estimate
        Incorporates strategic guidance from higher-level agent
        """
        # Ensure strategic guidance has correct shape
        if strategic_guidance.shape[-1] != self.strategic_guidance_dim:
            strategic_guidance = nn.Dense(self.strategic_guidance_dim)(strategic_guidance)
        
        # Project regional observation to standard dimension if needed
        # Each operational agent gets a slice of the total operational observation space
        expected_obs_dim = self.config.operational_obs_dim // self.config.num_operational_agents
        if x.shape[-1] != expected_obs_dim:
            # If dimension mismatch, project to expected size
            x = nn.Dense(expected_obs_dim)(x)
        
        # Combine local observation with strategic guidance
        x = jnp.concatenate([x, strategic_guidance], axis=-1)
        
        # Feature extraction
        for dim in self.config.operational_hidden_dims:
            x = nn.Dense(
                dim,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))
            )(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x)
        
        # Policy head
        policy_logits = nn.Dense(
            self.config.operational_action_dim,
            kernel_init=nn.initializers.orthogonal(0.01)
        )(x)
        
        # Value head
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0)
        )(x)
        
        return policy_logits, jnp.squeeze(value)


class SafetyAgent(nn.Module):
    """Safety agent for constraint enforcement and emergency response"""
    
    config: MultiAgentConfig
    
    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray,
        safety_critical: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns policy logits, value estimate, and safety override signal
        """
        # Feature extraction with safety-specific processing
        skip = None
        for i, dim in enumerate(self.config.safety_hidden_dims):
            x = nn.Dense(
                dim,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2))
            )(x)
            x = nn.relu(x)
            
            # Skip connections for faster response (matching dimensions)
            if i == 0:
                skip = x
            elif i == len(self.config.safety_hidden_dims) - 1 and skip is not None:
                # Project skip to match current dimension if needed
                if skip.shape[-1] != x.shape[-1]:
                    skip = nn.Dense(x.shape[-1])(skip)
                x = x + skip  # Residual connection
            
            x = nn.LayerNorm()(x)
        
        # Policy head (conservative initialization)
        policy_logits = nn.Dense(
            self.config.safety_action_dim,
            kernel_init=nn.initializers.orthogonal(0.001)  # Very small for safety
        )(x)
        
        # Value head
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0)
        )(x)
        
        # Safety override signal
        override_logit = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(0.1)
        )(x)
        override_prob = nn.sigmoid(override_logit)
        
        return policy_logits, jnp.squeeze(value), jnp.squeeze(override_prob)


class MultiAgentGridRL(nn.Module):
    """Complete multi-agent system for grid control"""
    
    config: MultiAgentConfig
    
    @nn.compact
    def __call__(
        self,
        strategic_obs: jnp.ndarray,
        operational_obs: jnp.ndarray,
        safety_obs: jnp.ndarray
    ) -> Dict[str, Any]:
        """
        Process observations through all agents with coordination
        """
        # Strategic agent processing
        strategic_agent = StrategicAgent(self.config)
        strategic_logits, strategic_value = strategic_agent(strategic_obs)
        
        # Extract strategic guidance for operational agents
        strategic_features = nn.Dense(32)(strategic_obs)
        
        # Operational agents processing (multiple regional agents)
        operational_outputs = []
        for i in range(self.config.num_operational_agents):
            operational_agent = OperationalAgent(self.config)
            # Each operational agent gets a slice of observations
            obs_slice = operational_obs[..., i::self.config.num_operational_agents]
            op_logits, op_value = operational_agent(obs_slice, strategic_features)
            operational_outputs.append((op_logits, op_value))
        
        # Safety agent processing
        safety_agent = SafetyAgent(self.config)
        safety_logits, safety_value, override_prob = safety_agent(safety_obs)
        
        # Coordination mechanism
        num_total_agents = (
            self.config.num_strategic_agents + 
            self.config.num_operational_agents + 
            self.config.num_safety_agents
        )
        
        # Collect all agent states for coordination
        agent_states = []
        agent_states.append(strategic_features)
        for i in range(self.config.num_operational_agents):
            op_features = nn.Dense(32)(
                operational_obs[..., i::self.config.num_operational_agents]
            )
            agent_states.append(op_features)
        safety_features = nn.Dense(32)(safety_obs)
        agent_states.append(safety_features)
        
        # Stack along the first axis to create [num_agents, batch_size, feature_dim]
        # Then transpose to [batch_size, num_agents, feature_dim] for attention
        agent_states = jnp.stack(agent_states, axis=0)
        agent_states = jnp.transpose(agent_states, (1, 0, 2))
        
        # Apply attention-based coordination
        coordinator = AttentionCoordination(
            num_agents=num_total_agents,
            num_heads=self.config.num_attention_heads,
            hidden_dim=self.config.attention_dim
        )
        coordinated_states, attention_weights = coordinator(agent_states)
        
        # Combine outputs
        outputs = {
            'strategic': {
                'logits': strategic_logits,
                'value': strategic_value
            },
            'operational': [
                {'logits': logits, 'value': value}
                for logits, value in operational_outputs
            ],
            'safety': {
                'logits': safety_logits,
                'value': safety_value,
                'override': override_prob
            },
            'coordination': {
                'states': coordinated_states,
                'attention': attention_weights
            }
        }
        
        return outputs


# Training Functions

@jax.jit
def compute_gae_hierarchical(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    horizon: int = 10
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE for hierarchical policies with different horizons"""
    
    def compute_advantages(carry, inputs):
        gae, next_value = carry
        reward, value, done = inputs
        
        # Adjust gamma based on horizon
        adjusted_gamma = gamma ** (1.0 / horizon)
        
        delta = reward + adjusted_gamma * next_value * (1 - done) - value
        gae = delta + adjusted_gamma * gae_lambda * (1 - done) * gae
        
        return (gae, value), gae
    
    _, advantages = jax.lax.scan(
        compute_advantages,
        (jnp.zeros_like(rewards[0]), values[-1]),
        (rewards[:-1], values[:-1], dones[:-1]),
        reverse=True
    )
    
    returns = advantages + values[:-1]
    return advantages, returns


@partial(jax.jit, static_argnums=(4,))
def multi_agent_loss(
    params: Any,
    observations: Dict[str, jnp.ndarray],
    actions: Dict[str, jnp.ndarray],
    targets: Dict[str, jnp.ndarray],
    config: MultiAgentConfig
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute multi-agent loss with coordination"""
    
    # Forward pass through network
    model = MultiAgentGridRL(config)
    outputs = model.apply(
        {'params': params},
        observations['strategic'],
        observations['operational'],
        observations['safety']
    )
    
    total_loss = 0.0
    metrics = {}
    
    # Strategic agent loss
    strategic_loss, strategic_metrics = compute_agent_loss(
        outputs['strategic'],
        actions['strategic'],
        targets['strategic'],
        config,
        agent_type='strategic'
    )
    total_loss += strategic_loss
    metrics['strategic'] = strategic_metrics
    
    # Operational agents loss
    for i, op_output in enumerate(outputs['operational']):
        op_loss, op_metrics = compute_agent_loss(
            op_output,
            actions['operational'][i],
            targets['operational'][i],
            config,
            agent_type='operational'
        )
        total_loss += op_loss
        metrics[f'operational_{i}'] = op_metrics
    
    # Safety agent loss
    safety_loss, safety_metrics = compute_agent_loss(
        outputs['safety'],
        actions['safety'],
        targets['safety'],
        config,
        agent_type='safety'
    )
    # Safety agent has higher weight
    total_loss += safety_loss * 2.0
    metrics['safety'] = safety_metrics
    
    # Coordination loss (encourage consensus)
    coord_loss = compute_coordination_loss(
        outputs['coordination']['attention']
    )
    total_loss += coord_loss * 0.1
    metrics['coordination_loss'] = coord_loss
    
    return total_loss, metrics


def compute_agent_loss(
    agent_output: Dict[str, jnp.ndarray],
    actions: jnp.ndarray,
    targets: Dict[str, jnp.ndarray],
    config: MultiAgentConfig,
    agent_type: str
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute loss for a single agent type"""
    
    logits = agent_output['logits']
    values = agent_output['value']
    
    # Policy loss (PPO clipped objective)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(
        log_probs, actions[..., None], axis=-1
    ).squeeze(-1)
    
    old_log_probs = targets['log_probs']
    advantages = targets['advantages']
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    ratio = jnp.exp(action_log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
    
    policy_loss = -jnp.mean(
        jnp.minimum(ratio * advantages, clipped_ratio * advantages)
    )
    
    # Value loss
    value_targets = targets['returns']
    value_loss = jnp.mean((values - value_targets) ** 2)
    
    # Entropy loss
    entropy = -jnp.mean(jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1))
    
    # Agent-specific loss weighting
    if agent_type == 'strategic':
        # Strategic agents focus more on value accuracy
        total_loss = policy_loss + config.vf_coef * value_loss * 2.0 - config.ent_coef * entropy
    elif agent_type == 'operational':
        # Operational agents balance policy and value
        total_loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy
    else:  # safety
        # Safety agents are more conservative
        total_loss = policy_loss * 0.5 + config.vf_coef * value_loss * 3.0 - config.ent_coef * entropy * 0.5
    
    metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        'total_loss': total_loss
    }
    
    return total_loss, metrics


def compute_coordination_loss(attention_weights: jnp.ndarray) -> jnp.ndarray:
    """Compute loss to encourage agent coordination"""
    # Encourage attention weights to be neither too uniform nor too peaked
    entropy = -jnp.sum(attention_weights * jnp.log(attention_weights + 1e-8), axis=-1)
    target_entropy = jnp.log(attention_weights.shape[-1]) * 0.5  # Half of max entropy
    
    coord_loss = jnp.mean((entropy - target_entropy) ** 2)
    return coord_loss


def create_multi_agent_state(
    config: MultiAgentConfig,
    rng: jax.random.PRNGKey
) -> TrainState:
    """Create initial training state for multi-agent system"""
    
    # Create dummy inputs
    dummy_strategic = jnp.zeros((1, config.strategic_obs_dim))
    dummy_operational = jnp.zeros((1, config.operational_obs_dim))
    dummy_safety = jnp.zeros((1, config.safety_obs_dim))
    
    # Initialize model
    model = MultiAgentGridRL(config)
    params = model.init(
        rng,
        dummy_strategic,
        dummy_operational,
        dummy_safety
    )['params']
    
    # Create optimizer with different learning rates for different agents
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(config.learning_rate)
    )
    
    # Create training state
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    return state


if __name__ == "__main__":
    # Test multi-agent system
    config = MultiAgentConfig()
    rng = jax.random.PRNGKey(42)
    
    # Create training state
    state = create_multi_agent_state(config, rng)
    print("Multi-agent system initialized")
    
    # Test forward pass
    batch_size = 32
    strategic_obs = jnp.ones((batch_size, config.strategic_obs_dim))
    operational_obs = jnp.ones((batch_size, config.operational_obs_dim))
    safety_obs = jnp.ones((batch_size, config.safety_obs_dim))
    
    outputs = state.apply_fn(
        {'params': state.params},
        strategic_obs,
        operational_obs,
        safety_obs
    )
    
    print(f"Strategic logits shape: {outputs['strategic']['logits'].shape}")
    print(f"Operational agents: {len(outputs['operational'])}")
    print(f"Safety override probability: {outputs['safety']['override'].mean():.3f}")
    print(f"Coordination attention shape: {outputs['coordination']['attention'].shape}")