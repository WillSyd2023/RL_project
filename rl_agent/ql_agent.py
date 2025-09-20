"""RL QL Agent in Gymnasium"""

from collections import defaultdict
from copy import deepcopy
import numpy as np
import gymnasium as gym

from rl_agent.parent_agent import Agent

class QLAgent(Agent):
    """
    Q-learning algorithm agent

    Mostly taken from https://gymnasium.farama.org/introduction/train_agent/
    in 16-17 January 2025
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.01,
        initial_epsilon: float = 0.1,
        epsilon_decay: float = 0,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.99,
        seed: int = 0,
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

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.seed=seed
        self.random = np.random.default_rng(seed)

    def __deepcopy__(self, memo):
        newone = type(self)(
            env=deepcopy(self.env),
            learning_rate=self.learning_rate,
            initial_epsilon=self.initial_epsilon,
            epsilon_decay=self.epsilon_decay,
            final_epsilon=self.final_epsilon,
            discount_factor=self.discount_factor,
            seed=self.seed,
        )
        newone.epsilon = self.epsilon
        newone.q_values = deepcopy(self.q_values)
        newone.random = deepcopy(self.random)
        return newone

    def _get_action_core(self, obs: str) -> int:
        # with probability epsilon return a random action to explore the environment
        if self.random.random() < self.epsilon:
            return int(self.env.action_space.sample())

        # with probability (1 - epsilon) act greedily (exploit)
        return int(np.argmax(self.q_values[obs]))

    def get_action(self, obs):
        """
        Returns the best action with probability (1 - epsilon)
        or a random action with probability epsilon to ensure exploration

        Args:
            obs; must be string
        
        Returns action, which will be integer
        """
        assert isinstance(obs, str), f"Expected obs to be str, got {type(obs).__name__}: {obs}"
        action = self._get_action_core(obs)
        return action

    def _update_core(
        self,
        obs: str,
        action: int,
        reward: int,
        terminated: bool,
        next_obs: str,
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.learning_rate * temporal_difference
        )
        self.training_error.append(temporal_difference)
    
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
            obs: observation, must be string
            action: must be integer
            reward: must be integer
            terminated: must be boolean
            next_obs: next observation, must be string
        """
        assert isinstance(obs, str), f"Expected obs to be str, got {type(obs).__name__}: {obs}"
        assert isinstance(action, int) and not isinstance(reward, bool), f"Expected action to be int, got {type(action).__name__}: {action}"
        assert isinstance(reward, int) and not isinstance(reward, bool), f"Expected reward to be int, got {type(reward).__name__}: {reward}"
        assert isinstance(terminated, bool), f"Expected terminated to be bool, got {type(terminated).__name__}: {terminated}"
        assert isinstance(next_obs, str), f"Expected next_obs to be str, got {type(next_obs).__name__}: {next_obs}"
        self._update_core(obs, action, reward, terminated, next_obs)

    def decay_epsilon(self):
        """Epsilon decay method"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)