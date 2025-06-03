# mypy: disable-error-code="override"
"""Defines simple task for training a standing policy for the default humanoid using an GRU actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import ksim
import xax

from .walking import HumanoidWalkingTask, HumanoidWalkingTaskConfig


@dataclass
class HumanoidStandingTaskConfig(HumanoidWalkingTaskConfig):
    max_steps: int = xax.field(
        value=1000,
        help="The maximum number of steps to train for.",
    )


Config = TypeVar("Config", bound=HumanoidStandingTaskConfig)

###
# TODO - define the reward functions here:
#
#
###


class HumanoidStandingTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        # TODO - update the reward functions here:
        return [
            ksim.BaseHeightRangeReward(
                z_lower=1.1, z_upper=1.5, dropoff=10.0, scale=1.0
            ),
            ksim.StayAliveReward(scale=1.0),
        ]

    def is_training_over(self, state: ksim.State) -> bool:
        return state.num_steps >= self.config.max_steps


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.standing
    # To visualize the environment, use the following command:
    #  python -m examples.standing run_mode=view batch_size=1 num_envs=1
    #   python -m examples.standing run_mode=view
    HumanoidStandingTask.launch(
        HumanoidStandingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=32,
            epochs_per_log_step=1,
            rollout_length_seconds=2.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            iterations=3,
            ls_iterations=5,
        ),
    )
