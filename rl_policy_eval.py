from collections import defaultdict
import numpy as np
import gymnasium as gym

# Self-written code
from rl_env import BitEnv
from rl_agent import QLAgent

class PolicyEvalQL():
    """
    Object for policy evaluation taken from Braun, 2021 on pp. 51 par. 4:

    - Repeat the following independent trials:
        1. Train the agent
        2. After some steps, measure Q-values
        3. Use Q-values greedily on some episodes, get the average reward
        4. Repeat 1 to 3 several times
    - Get the median of average rewards (taken after the same number of steps)
        from every trial
    - Plot medians

    For now, restrict to the Q-learning agent
    """
    def __init__(
        self,
        q_values = None,
        env: gym.Env = BitEnv(),
        learning_rate: float = 0.01,
        initial_epsilon: float = 0.1,
        epsilon_decay: float = 0,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.99,
    ):
        """
        Initialises the default training agent and environment

        Default agent parameters inspired by pp. 51 of Braun, 2021:
        - learning rate is set to 0.01
        - epsilon is constant, set to 0.1
        - discount factor is 0.99
        - Q-values are initialised optimistically by setting all values to 1.0001

        Default environment is just BitEnv with p = 0.5
        - Initialises initial training-agent observation with reset environment

        Arg:
            env: can insert environment here; default as just mentioned
        """
        self.env = env

        self.ql_agent = QLAgent(
            env,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        if q_values is None:
            q_values = defaultdict(lambda: np.ones(env.action_space.n) * 1.0001)
        self.ql_agent.q_values = q_values

        # Initialise initial training-agent observation just in case
        self.train_obs, _ = env.reset()
    
    def train_steps(
        self,
        steps: int = 5_000,
    ):
        """
        Train the training agent to a specified number of steps

        Warning: does not reset environment

        Arg:
            steps: number of steps to play; default is 5_000
        """
        obs = self.train_obs
        agent = self.ql_agent

        for _ in range(steps):
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = self.env.step(action)

            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
        
        self.train_obs = next_obs

    def avg_reward_per_eps(self):
        return