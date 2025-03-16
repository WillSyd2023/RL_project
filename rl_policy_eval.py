import gymnasium as gym

def policy_eval_ql(
        env: gym.Env,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        learning_rate: float = 0.01,
        discount_factor: float = 0.95,
    ):
    """Policy-evaluation approach taken from Braun, 2021 on pp. 51 par. 4:

    - Repeat the following independent trials:
        1. Train the agent
        2. After some steps, measure Q-values
        3. Use Q-values greedily on some episodes, get the average reward
        4. Repeat 1 to 3 several times
    - Get the median of average rewards (taken after the same number of steps)
      from every trial
    - Plot medians

    For now, restrict to the Q-learning agent

    Args:
        env: gym environment
        learning_rate: The learning rate
        initial_epsilon: The initial epsilon value
        epsilon_decay: The decay for epsilon
        final_epsilon: The final epsilon value
        discount_factor: The discount factor for computing the Q-value
        n_trials: number of trials
        limit_trial: time limit (in steps) for a trial
        per_trial: after how many steps in a trial should we do steps 2 and 3
        n_eps: number of episodes
        limit_eps: time limit (in steps) for an episode
    """

