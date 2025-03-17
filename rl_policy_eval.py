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
        env: gym.Env = BitEnv(),
    ):
        """
        Initialises the default training agent and environment

        Default agent parameters inspired by pp. 51 of Braun, 2021:
        - learning rate is set to 0.01
        - epsilon is constant, set to 0.1
        - discount factor is 0.99
        - Q-values are initialised optimistically by setting all values to 1.0001

        Default environment is just BitEnv with p = 0.5

        Arg:
            env: can insert environment here; default as just mentioned
        """
        self.ql_agent = QLAgent(
            env,
            learning_rate=0.01,
            initial_epsilon=0.1,
            epsilon_decay=0,
            final_epsilon=0.1,
            discount_factor=0.99,
        )

        self.ql_agent.q_values = defaultdict(
            lambda: np.ones(env.action_space.n) * 1.0001
        )