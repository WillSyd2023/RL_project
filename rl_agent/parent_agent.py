"""RL Parent Agent in Gymnasium"""

from abc import ABCMeta, abstractmethod
import gymnasium as gym

class Agent(metaclass=ABCMeta):
    """
    Abstract template for an agent class
    """
    @abstractmethod
    def __init__(self, env: gym.Env):
        """
        Initialise agent

        Arg:
            env: the training environment 
        """
        self.env = env
        self.training_error = []

    @abstractmethod
    def get_action(self, obs):
        """
        Given observation, get action

        Arg:
            obs: observation
        
        Returns action
        """

    @abstractmethod
    def update(
        self,
        obs,
        action,
        reward,
        terminated,
        next_obs,
    ):
        """Update agent"""

