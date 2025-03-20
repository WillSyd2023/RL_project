import numpy as np
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from rl_agent import Agent

def train(
        env: gym.Env,
        agent: Agent,
        time_limit: int = 100_000,
        n_eps: int = 1000
    ):
    """
    Train a (Q-learning?) agent

    Mostly taken from https://gymnasium.farama.org/introduction/train_agent/
    in 18 Jan 2025

    Args:
        env: Gymnasium environment; will get wrapped with:
         - TimeLimit
         - RecordEpisodeStatistics
        agent: (Q-learning) agent to train
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

    return (env, agent)

def visualise_train(env: gym.Env, agent: Agent):
    """Visualise training outcomes

    Mostly taken from https://gymnasium.farama.org/introduction/train_agent/
    in 19 Jan 2025

    Args:
        env: post-training gym environment
        agent: post-training agent
    """
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))

    # np.convolve will compute the rolling mean for 100 episodes
    
    axs[0].plot(fftconvolve(env.return_queue, np.ones(100)))
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    
    axs[1].plot(fftconvolve(env.length_queue, np.ones(100)))
    axs[1].set_title("Episode Lengths")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Length")
    
    axs[2].plot(fftconvolve(agent.training_error, np.ones(100)))
    axs[2].set_title("Training Error")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Temporal Difference")
    
    plt.tight_layout()
    plt.show()