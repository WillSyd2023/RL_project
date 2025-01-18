from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from rl_agent import QLAgent

def train_ql(
        env: gym.Env,
        agent: QLAgent,
        time_limit: int = 100_000,
        n_eps: int = 100_000
    ) -> QLAgent:
    """
    Train a Q-learning agent

    Mostly taken from https://gymnasium.farama.org/introduction/train_agent/
    in 18 Jan 2025

    Args:
        env: Gymnasium environment; will get wrapped with:
         - TimeLimit
         - RecordEpisodeStatistics
        agent: Q-learning agent to train
        time_limit: time limit (in steps) for an episode
        n_eps: number of episodes

    Returns trained agent
    """
    # Wrap up environment
    env = TimeLimit(env, max_episode_steps=time_limit)
    env = RecordEpisodeStatistics(env, buffer_length=n_eps)

    for _ in tqdm(range(n_eps)):
        obs, info = env.reset()
        done = False

        # Play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs
        
        agent.decay_epsilon()

    return agent
