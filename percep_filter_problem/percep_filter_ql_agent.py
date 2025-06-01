"""Q-learning agent with perceptual filter in Gymnasium"""

from typing import Callable
from copy import deepcopy
import gymnasium as gym

from rl_agent.ql_agent import QLAgent

class PercepFilterQLAgent(QLAgent):
    """
    Q-learning agent with perceptual filter
    """
    def __init__(
        self,
        env: gym.Env,
        obs_filter: Callable[..., str],
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialise Q-learning agent with perceptual filter

        Args:
            env: The training environment
            obs_filter: perceptual filter of agent
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        super().__init__(
            env,
            learning_rate,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
        )

        self.obs_filter = obs_filter
    
    def __deepcopy__(self, memo):
        newone = super().__deepcopy__(memo)
        newone.obs_filter = deepcopy(self.obs_filter, memo)
        return newone

    def get_action(self, obs) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        or a random action with probability epsilon to ensure exploration
        """
        # Filter observation first
        return super().get_action_core(self.obs_filter(obs))
    
    def update(
        self,
        obs,
        action: int,
        reward: int,
        terminated: bool,
        next_obs,
    ):
        """Updates the Q-value of an action."""
        # Filter observations first
        super().update_core(
            obs=self.obs_filter(obs),
            action=action,
            reward=reward,
            terminated=terminated,
            next_obs=self.obs_filter(next_obs),
        )