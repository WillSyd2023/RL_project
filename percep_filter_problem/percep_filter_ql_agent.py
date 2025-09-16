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
        obs_filter: Callable[..., str] = filter_four_bits,
        env: gym.Env = TwoCupEnv(),
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
        newone = type(self)(
            obs_filter=self.obs_filter,
            env=deepcopy(self.env),
            learning_rate=self.lr,
            initial_epsilon=self.initial_epsilon,
            epsilon_decay=self.epsilon_decay,
            final_epsilon=self.final_epsilon,
            discount_factor=self.discount_factor,
        )
        newone.epsilon = self.epsilon
        newone.q_values = deepcopy(self.q_values)
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
        obs = self.obs_filter(obs)
        assert isinstance(obs, str), f"Expected obs to be str, got {type(obs).__name__}: {obs}"
        return super()._get_action_core(obs)
    
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
        obs = self.obs_filter(obs)
        next_obs = self.obs_filter(next_obs)
        assert isinstance(obs, str), f"Expected obs to be str, got {type(obs).__name__}: {obs}"
        assert isinstance(action, int) and not isinstance(reward, bool), f"Expected action to be int, got {type(action).__name__}: {action}"
        assert isinstance(reward, int) and not isinstance(reward, bool), f"Expected reward to be int, got {type(reward).__name__}: {reward}"
        assert isinstance(terminated, bool), f"Expected terminated to be bool, got {type(terminated).__name__}: {terminated}"
        assert isinstance(next_obs, str), f"Expected next_obs to be str, got {type(next_obs).__name__}: {next_obs}"
        super()._update_core(
            obs=obs,
            action=action,
            reward=reward,
            terminated=terminated,
            next_obs=next_obs,
        )