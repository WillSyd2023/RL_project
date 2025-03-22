from collections import defaultdict
from tqdm import tqdm
import copy
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

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
        q_values = None,
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
            the other parameters set up the initial training agent
        """
        self.original_env = env
        self.training_env = None

        self.ori_agent = QLAgent(
            env,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

        if q_values is None:
            q_values = defaultdict(lambda: np.ones(env.action_space.n) * 1.0001)
        self.ori_agent.q_values = q_values

        self.train_agent = None

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
        agent = self.train_agent

        for _ in range(steps):
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = self.training_env.step(action)

            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
        
        self.train_obs = next_obs

    def avg_reward_per_eps(
        self,
        q_values,
        time_limit: int = 1_000,
        n_eps: int = 50,
    ):
        """
        Use Q-values greedily on some episodes (with a new agent),
        get the average reward

        Args:
            q_values: to be tested
            time_limit: for a single episode
            n_eps: number of episodes to play

        Returns average reward
        """
        # Copy environment from original and set time limit
        env = copy.deepcopy(self.original_env)
        env = TimeLimit(env, max_episode_steps=time_limit)

        # Initialise agent for testing
        test_agent = QLAgent(
            env,
            learning_rate = 0,
            initial_epsilon = 0,
            epsilon_decay = 0,
            final_epsilon = 0,
            discount_factor = 0
        )
        test_agent.q_values = q_values

        # Record cumulative reward for every single step
        # (n_eps * time_limit)
        total_reward = 0
        for _ in tqdm(range(n_eps)):
            obs, _ = env.reset()
            done = False

            # Play one episode
            while not done:
                action = test_agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                total_reward += reward

                done = terminated or truncated
                obs = next_obs

        return total_reward/(time_limit * n_eps)

    def one_trial(
        self,
        steps_measure: int = 5_000,
        num_measure: int = 50,
        time_limit: int = 1_000,
        n_eps: int = 50,
    ):
        """
        Perform a single independent trial

        Args:
            steps_measure: number of steps to take before measuring q-values
            num_measure: number of times we measure q-values
            time_limit: of a single episode (when testing q-values)
            n_eps: number of episodes (for testing q_values)

        Returns 1-d np.array containing averages from each measuring time
        """
        # Initialise training agent and training environment
        self.train_agent = copy.deepcopy(self.ori_agent)
        self.training_env = self.train_agent.env

        # Train agent and run tests
        # Record results on a list
        averages = np.empty(0)
        for _ in range(num_measure):
            self.train_steps(steps_measure)
            averages = np.append(
                averages,
                self.avg_reward_per_eps(
                    copy.deepcopy(self.train_agent.q_values),
                    time_limit,
                    n_eps,
                ),
            )

        return averages

    def trials(
        self,
        num_trials: int = 10,
        steps_measure: int = 5_000,
        num_measure: int = 50,
        time_limit: int = 1_000,
        n_eps: int = 50,
    ):
        """
        Perform independent trials and return medians of averages from each
        measuring time

        Args:
            num_trials: number of independent trials
            steps_measure: number of steps to take before measuring q-values
            num_measure: number of times we measure q-values
            time_limit: of a single episode (when testing q-values)
            n_eps: number of episodes (for testing q_values)
        
        Returns:
            number of steps taken before measuring
            corresponding medians
        """
        # Get medians and corresponding steps
        trials = np.empty((num_measure, 0))
        steps = np.empty((num_measure, 0))
        for i in range(1, num_trials + 1):
            trials = np.column_stack((
                trials,
                self.one_trial(
                    steps_measure=steps_measure,
                    num_measure=num_measure,
                    time_limit=time_limit,
                    n_eps=n_eps,
                ),
            ))
            steps = np.append(steps, i * steps_measure)
        medians = np.median(trials, axis=1)

        return steps, medians