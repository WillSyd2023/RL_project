"""RL policy evaluation code in Gymnasium"""

from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import TimeLimit

from rl_agent.ql_agent import QLAgent

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
        agent: QLAgent,
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

        Args:
            agent: QL agent, complete with defined environment
            q_values
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        env = agent.env
        if env is None:
            raise ValueError("Need to define agent's environment (inside agent)")
        self.original_env = env

        self.ori_agent = agent
        self.ori_agent.learning_rate = learning_rate
        self.ori_agent.initial_epsilon = initial_epsilon
        self.ori_agent.epsilon_decay = epsilon_decay
        self.ori_agent.final_epsilon = final_epsilon
        self.ori_agent.discount_factor = discount_factor

        if q_values is None:
            q_values = defaultdict(lambda: np.ones(env.action_space.n) * 1.0001)
        self.ori_agent.q_values = q_values

        # Initialised values just in case
        self.training_env = self.original_env
        self.train_agent = self.ori_agent
        self.train_obs, _ = env.reset()
        self.steps = np.arange(
            start=1,
            stop=1,
            step=1,
        )
        self.medians = 0
        self.num_trials = 0

    
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
        time_limit: int = 1_000,
        n_eps: int = 50,
    ):
        """
        Use Q-values greedily on some episodes (with a new agent)
        on a (new) environment, get the average reward

        Args:
            q_values: to be tested
            time_limit: for a single episode
            n_eps: number of episodes to play

        Returns average reward
        """
        # Copy environment from original and set time limit
        env = deepcopy(self.original_env)
        env = TimeLimit(env, max_episode_steps=time_limit)

        # Initialise agent for testing
        test_agent = deepcopy(self.ori_agent)
        test_agent.env = env
        test_agent.learning_rate = 0
        test_agent.initial_epsilon = 0
        test_agent.epsilon_decay = 0
        test_agent.final_epsilon = 0
        test_agent.discount_factor = 0

        test_agent.q_values = deepcopy(self.train_agent.q_values)

        # Record cumulative reward for every single step
        # (n_eps * time_limit)
        total_reward = 0
        for _ in range(n_eps):
            obs, _ = env.reset()
            done = False

            # Play one episode
            while not done:
                action = test_agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                total_reward += reward
                done = terminated or truncated
                obs = next_obs

        return total_reward/n_eps

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
        # Initialise training agent/environment/observation
        self.train_agent = deepcopy(self.ori_agent)
        self.training_env = self.train_agent.env
        self.train_obs, _ = self.training_env.reset()

        # Train agent and run tests
        # Record results on a list
        averages = np.empty(0)
        for _ in range(num_measure):
            self.train_steps(steps_measure)
            averages = np.append(
                averages,
                self.avg_reward_per_eps(
                    time_limit=time_limit,
                    n_eps=n_eps,
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
        Perform independent trials and calculate medians of averages from each
        measuring time

        Args:
            num_trials: number of independent trials
            steps_measure: number of steps to take before measuring q-values
            num_measure: number of times we measure q-values
            time_limit: of a single episode (when testing q-values)
            n_eps: number of episodes (for testing q_values)
        
        Effects - after method, the object stores:
            number of steps taken before measuring
            corresponding medians
            maximum reward for testing
            number of trials
        """
        # Get medians and corresponding steps
        trials = np.empty((num_measure, 0))
        # May put tqdm here
        for _ in tqdm(range(num_trials)):
            trials = np.column_stack((
                trials,
                self.one_trial(
                steps_measure=steps_measure,
                num_measure=num_measure,
                time_limit=time_limit,
                n_eps=n_eps,
                ),
            ))
        medians = np.median(trials, axis=1)

        self.steps = np.arange(
            start=1,
            stop=1 + num_measure,
            step=1,
        ) * steps_measure
        self.medians = medians
        self.num_trials = num_trials

    def visualise(
        self, 
        save: str = "", 
        title: str ="Policy Evaluation",
        max_reward: int = 1_000,
        ):
        """Visualise outcome after trials

        Args:
            save - title of saved picture; no picture saved by default
            title - title for graph; "Policy Evaluation" by default
            max_reward - maximum reward that can be achieved by agent; 1_000 by default
        """
        fig, ax = plt.subplots()
        ax.plot(self.steps, self.medians)

        ax.set(
            xlabel="Steps taken",
            ylabel="Median average reward",
            title=title + ": " + str(self.num_trials) + " trials",
        )
        ax.axhline(
            y=max_reward,
            linestyle="--",
            label="Maximum reward",
        )

        if save != "":
            fig.savefig(save + ".svg")
