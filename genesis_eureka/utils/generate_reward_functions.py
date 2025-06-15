from typing import Dict, List
from omegaconf import DictConfig


def generate_reward_functions(model: str, messages: List[Dict[str, str]], chunk_size: int, cfg: DictConfig) -> Dict[str,List[Dict[str,Dict[str,str]]]] :
    # TODO (1) - implement the function. You can change type of `model` or modify any other parameters.
    # TODO (3) - add parameter for input image(s). Implement multimodal reward generation.
    '''
    Function for generating propositions of reward functions for eureka algorithm.
    Input:
        - model: LLM model that generates the response
        - messages: input for the model. 
                    it is either a list of two prompts (initial_user prompt and initial_system prompt)
                    or a list of four messages (initial_user prompt, initial_system prompt and two reflection prompts)
        - chunk_size: number of reward functions to generate
        - cfg: additional configuration
    Output:
        - response: a dictionary of LLM resposes.
                    Key `choices` contains a list of generated reward functions.
                    You can add other keys to track statistics e.g. `number_of_tokens`.
    '''
    # (1) Example messages format
    messages = [{"role": "system", "content": "prompt1"}, {"role": "user", "content": "prompt2"}]
    # (1) Example response format
    response = {}
    response["choices"] = [{"message": {"content": '''
An effective reward function for learning a basic locomotion policy should encourage the agent to follow velocity commands while maintaining stability and moving efficiently. This can be broken down into several components:

1.  **Tracking Command Velocity**: The primary goal. The agent should be rewarded for matching the commanded linear (forward/strafe) and angular (turning) velocities.
2.  **Maintaining Stability**: The agent must stay upright. This means penalizing it for tilting (roll and pitch), for excessive angular velocities in roll and pitch, and for significant vertical motion (jumping/falling). A stable height should also be maintained.
3.  **Smooth and Efficient Motion**: The agent's movements should be smooth to be feasible on real hardware and to conserve energy. This involves penalizing high joint velocities, accelerations, and large, jerky actions.

The following `_reward_eureka` function implements these principles using the available environment variables.

```python
import torch
import math

# This function should be placed inside the Go2Env class.

def _reward_eureka(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Calculates the reward for the Go2 locomotion task.

    This function computes a composite reward signal designed to train a versatile
    locomotion policy. The total reward is a weighted sum of several components:
    - Positive rewards for tracking commanded linear and angular velocities.
    - Negative penalties for instability (e.g., body tilt, high angular velocities).
    - Negative penalties for inefficient or jerky actions (e.g., high joint velocities,
      large action changes).
    """
    # A dictionary to store the raw values of each reward component for logging.
    reward_components = {}

    # --- 1. Primary Task: Velocity Tracking ---
    # Reward for matching the commanded linear velocity in the xy plane.
    # We use an exponential kernel for a smooth reward that is high when error is low.
    lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    reward_components["lin_vel_xy"] = torch.exp(-lin_vel_error / 0.25)

    # Reward for matching the commanded angular (yaw) velocity.
    ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    reward_components["ang_vel_z"] = torch.exp(-ang_vel_error / 0.25)

    # --- 2. Stability Penalties ---
    # Penalize non-level orientation. self.projected_gravity is the gravity vector
    # in the base frame. For a level base, it's [0, 0, -1]. We penalize the x/y components.
    reward_components["orientation"] = -torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # Penalize non-zero linear velocity in the z-axis (prevents jumping/bouncing).
    reward_components["lin_vel_z"] = -torch.square(self.base_lin_vel[:, 2])

    # Penalize angular velocities in roll and pitch (prevents wobbling).
    reward_components["ang_vel_xy"] = -torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    # --- 3. Regularization Penalties for Smoothness and Efficiency ---
    # Penalize large changes in action commands between steps to encourage smoothness.
    reward_components["action_rate"] = -torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    # Penalize high joint velocities to encourage energy efficiency.
    reward_components["dof_vel"] = -torch.sum(torch.square(self.dof_vel), dim=1)
    
    # Penalize deviation from the default joint positions to maintain a reasonable posture.
    reward_components["dof_pos"] = -torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    # --- 4. Combine Rewards with Weights ---
    # The weights determine the importance of each component. These are hyperparameters
    # that may require tuning. The main goal is velocity tracking, so it gets the
    # highest weights. Stability is crucial, so it gets moderate weights. Regularization
    # terms are for fine-tuning behavior and get lower weights.
    
    reward = (
        # Tracking rewards (positive)
        2.0 * reward_components["lin_vel_xy"] +
        1.0 * reward_components["ang_vel_z"] +

        # Stability penalties (negative)
        0.5 * reward_components["orientation"] +
        0.2 * reward_components["lin_vel_z"] +
        0.2 * reward_components["ang_vel_xy"] +

        # Regularization penalties (negative)
        0.01 * reward_components["action_rate"] +
        0.0001 * reward_components["dof_vel"] +
        0.005 * reward_components["dof_pos"]
    )

    return reward, reward_components
```
'''}}] * chunk_size
    
    return response
