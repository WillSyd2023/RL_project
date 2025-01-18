"""RL Agents in Gymnasium"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import numpy as np
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
    def get_action(self, obs: int) -> int:
        """
        Given observation, get action

        Arg:
            obs: observation, '1' or '0'
        
        Returns action
        """

    @abstractmethod
    def update(
        self,
        obs: int,
        action: int,
        reward: int,
        terminated: bool,
        next_obs: int
    ):
        """Update agent"""

class QLAgent(Agent):
    """
    Q-learning algorithm agent

    Mostly taken from https://gymnasium.farama.org/introduction/train_agent/
    in 16-17 January 2025
    """
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialise Q-learning RL agent

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        super().__init__(env)
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def get_action(self, obs: int) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        or a random action with probability epsilon to ensure exploration
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: int,
        action: int,
        reward: int,
        terminated: bool,
        next_obs: int
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        """Epsilon decay method"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)