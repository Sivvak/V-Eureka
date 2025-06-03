def get_mock_response(messages, temperature=0.1, iteration=0, response_id=0):
    """
    Returns a mock Gemini response with a valid K-Sim reward class.

    Args:
        messages: List of conversation messages (for context)
        temperature: Temperature parameter (unused in mock)
        iteration: Current iteration number (for variation)
        response_id: Response ID within iteration (for variation)

    Returns:
        dict: Mock response in Gemini format
    """

    # Define different reward class templates for variation
    reward_templates = [
        # Template 0: Basic forward walking reward
        '''Here's a reward class for training a humanoid robot to walk forward stably:

```python
import jax.numpy as jnp
import attrs
import ksim
from ksim.types import Array, Trajectory

@attrs.define(frozen=True, kw_only=True)
class ForwardWalkingReward(ksim.Reward):
    """Reward function for forward walking with stability."""
    
    target_speed: float = attrs.field(default=1.0)
    stability_weight: float = attrs.field(default=0.5)
    reward_scale: float = attrs.field(default=1.0)
    
    def get_reward(self, trajectory: Trajectory) -> Array:
        # Extract base velocity
        base_vel = trajectory.obs["base_linear_velocity_observation"]
        forward_vel = base_vel[..., 0]  # x-axis velocity
        
        # Extract base position for height
        base_pos = trajectory.obs["base_position_observation"]
        height = base_pos[..., 2]
        
        # Forward velocity reward
        speed_reward = jnp.exp(-jnp.abs(forward_vel - self.target_speed))
        
        # Height stability reward (avoid falling)
        height_reward = jnp.where(height > 0.8, 1.0, 0.0)
        
        # Combine rewards
        total_reward = speed_reward + self.stability_weight * height_reward
        scaled_reward = self.reward_scale * total_reward / (1.0 + self.stability_weight)
        return jnp.clip(scaled_reward, 0.0, 1.0)
```''',
        # Template 1: Energy-efficient walking
        '''Here's an energy-efficient walking reward class:

```python
import jax.numpy as jnp
import attrs
import ksim
from ksim.types import Array, Trajectory

@attrs.define(frozen=True, kw_only=True)
class EfficientWalkingReward(ksim.Reward):
    """Energy-efficient walking reward."""
    
    target_velocity: float = attrs.field(default=1.2)
    energy_penalty: float = attrs.field(default=0.1)
    reward_scale: float = attrs.field(default=1.0)
    
    def get_reward(self, trajectory: Trajectory) -> Array:
        # Forward motion
        base_vel = trajectory.obs["base_linear_velocity_observation"]
        forward_vel = base_vel[..., 0]
        
        # Joint velocities for energy calculation
        joint_vel = trajectory.obs["joint_velocity_observation"]
        
        # Velocity reward
        vel_reward = jnp.exp(-jnp.square(forward_vel - self.target_velocity))
        
        # Energy penalty
        energy = jnp.sum(jnp.square(joint_vel), axis=-1)
        energy_reward = jnp.exp(-self.energy_penalty * energy)
        
        combined_reward = vel_reward * energy_reward
        return self.reward_scale * combined_reward
```''',
        # Template 2: Balanced locomotion
        '''Here's a balanced locomotion reward:

```python
import jax.numpy as jnp
import attrs
import ksim
from ksim.types import Array, Trajectory

@attrs.define(frozen=True, kw_only=True)
class BalancedLocomotionReward(ksim.Reward):
    """Balanced locomotion with multiple objectives."""
    
    forward_weight: float = attrs.field(default=2.0)
    balance_weight: float = attrs.field(default=1.0)
    smoothness_weight: float = attrs.field(default=0.5)
    reward_scale: float = attrs.field(default=1.0)
    
    def get_reward(self, trajectory: Trajectory) -> Array:
        # Base motion
        base_vel = trajectory.obs["base_linear_velocity_observation"]
        base_orientation = trajectory.obs["base_orientation_observation"]
        
        # Forward motion reward
        forward_vel = base_vel[..., 0]
        forward_reward = jnp.tanh(forward_vel)
        
        # Balance reward (upright orientation)
        qw, qx, qy, qz = base_orientation[..., 0], base_orientation[..., 1], base_orientation[..., 2], base_orientation[..., 3]
        up_z = 1 - 2 * (qx**2 + qy**2)
        balance_reward = jnp.clip(up_z, 0.0, 1.0)
        
        # Lateral stability
        lateral_vel = jnp.abs(base_vel[..., 1])
        smoothness_reward = jnp.exp(-lateral_vel)
        
        # Weighted combination
        total_reward = (
            self.forward_weight * forward_reward +
            self.balance_weight * balance_reward +
            self.smoothness_weight * smoothness_reward
        )
        
        normalized_reward = total_reward / (self.forward_weight + self.balance_weight + self.smoothness_weight)
        return self.reward_scale * normalized_reward
```''',
    ]

    # Select template based on iteration and response_id for variation
    template_idx = (iteration * 3 + response_id) % len(reward_templates)
    selected_template = reward_templates[template_idx]

    # Create mock response in Gemini format
    mock_response = {"content": selected_template, "finish_reason": "stop"}

    return mock_response
