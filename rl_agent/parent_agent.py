"""RL Parent Agent in Gymnasium"""

from abc import ABCMeta, abstractmethod
import gymnasium as gym

class Agent(metaclass=ABCMeta):
    """
    Abstract template for an agent class
    """
    def __init__(self, env: gym.Env):
        """
        Initialise agent

        Arg:
            env: the training environment 
        """
        self.env = env
        self.training_error = []

    @abstractmethod
    def get_action_core(self, obs: int) -> int:
        """
        Given observation, get action

        Arg:
            obs: observation, '1' or '0'
        
        Returns action
        """

    @abstractmethod
    def update_core(
        self,
        obs: int,
        action: int,
        reward: int,
        terminated: bool,
        next_obs: int,
    ):
        """Update agent"""

