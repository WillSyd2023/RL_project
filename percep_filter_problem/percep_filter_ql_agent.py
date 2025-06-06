"""Q-learning agent with perceptual filter in Gymnasium"""

from typing import Callable
from copy import deepcopy
import gymnasium as gym

from rl_agent.ql_agent import QLAgent

from percep_filter_problem.percep_filter import filter_four_bits
from percep_filter_problem.percep_filter_env import TwoCupEnv

class PercepFilterQLAgent(QLAgent):
    """
    Q-learning agent with perceptual filter
    """
    def __init__(
        self,
        env: gym.Env = TwoCupEnv(),
        obs_filter: Callable[..., str] = filter_four_bits,
        learning_rate: float = 0.01,
        initial_epsilon: float = 0.1,
        epsilon_decay: float = 0,
        final_epsilon: float = 0.1,
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

    def get_action(self, obs):
        """
        Returns the best action with probability (1 - epsilon)
        or a random action with probability epsilon to ensure exploration

        Args:
            obs: observation; must be consistent in type with obs_filter input
        
        Returns action, which will be integer
        """
        # Filter observation first
        return super()._get_action_core(self.obs_filter(obs))
    
    def update(
        self,
        obs,
        action,
        reward,
        terminated,
        next_obs,
    ):
        """Updates the Q-value of an action.
        
        Args:
            obs: observation; must be consistent in type with obs_filter input
            action: must be integer
            reward: must be integer
            terminated: must be boolean
            next_obs: next observation; same typing as obs
        """
        # Filter observations first
        super()._update_core(
            obs=self.obs_filter(obs),
            action=action,
            reward=reward,
            terminated=terminated,
            next_obs=self.obs_filter(next_obs),
        )