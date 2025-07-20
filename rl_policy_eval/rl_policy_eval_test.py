"""Testing policy evaluation object with dummy environment and dummy agent"""

from collections import defaultdict
from copy import deepcopy
from typing import Optional
import numpy as np
import pytest
from gymnasium.spaces import Discrete
from gymnasium import Env
from rl_policy_eval.rl_policy_eval import PolicyEvalQL

class DummyEnv(Env):
    """Dummy environment merely for testing policy evaluation object"""
    def __init__(self):
        self.action_space = Discrete(500_000)
        self.observation_space = Discrete(1)
        self.obs = 0

    def __deepcopy__(self, memo):
        newone = type(self)()
        newone.obs = self.obs
        return newone

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.obs = 0
        return self.obs, {}

    def step(self, action):
        assert action == self.obs or action % 1000 == self.obs
        self.obs += 1
        done = self.obs >= 500_000
        return self.obs, 1, done, False, {}

    def render(self):
        return

@pytest.fixture
def dummy_env():
    return DummyEnv()

class DummyQLAgent:
    """Dummy 'QL agent' merely for testing policy evaluation object"""
    def __init__(self, env):
        self.env = env
        self.q_values = defaultdict(lambda: np.ones(env.action_space.n))
        self.learning_rate = 0
        self.initial_epsilon = 0
        self.epsilon_decay = 0
        self.final_epsilon = 0
        self.discount_factor = 0
        self.total_rewards = 0

    def __deepcopy__(self, memo):
        newone = type(self)(deepcopy(self.env))
        newone.learning_rate = self.learning_rate
        newone.initial_epsilon = self.initial_epsilon
        newone.epsilon_decay = self.epsilon_decay
        newone.final_epsilon = self.final_epsilon
        newone.discount_factor = self.discount_factor
        newone.total_rewards = self.total_rewards
        return newone

    def get_action(self, obs):
        rewards = self.total_rewards
        assert obs == rewards or obs == rewards % 1000
        self.total_rewards += 1
        return rewards

    def update(self, obs, action, reward, done, next_obs):
        assert obs == self.total_rewards - 1 or obs == (self.total_rewards - 1) % 1000
        assert action == self.total_rewards - 1 or action == (self.total_rewards - 1) % 1000
        assert reward == 1
        assert done is False
        assert next_obs == self.total_rewards or next_obs == self.total_rewards % 1000

@pytest.fixture
def dummy_agent(dummy_env):
    return DummyQLAgent(dummy_env)

def test_train_steps(dummy_agent):
    """Test train_steps method for policy evaluation"""
    policy = PolicyEvalQL(agent=dummy_agent)
    policy.train_steps()
    assert dummy_agent.total_rewards == 5_000

def test_avg_reward_per_eps(dummy_agent):
    """Test avg_reward_per_eps method for policy evaluation"""
    policy = PolicyEvalQL(agent=dummy_agent)
    assert int(policy.avg_reward_per_eps()) == 1_000

def test_one_trial(dummy_agent):
    """Test one_trial method for policy evaluation"""
    policy = PolicyEvalQL(agent=dummy_agent)
    array = np.full(50, 1000.0)
    np.testing.assert_array_equal(array, policy.one_trial())

def test_trials(dummy_agent):
    """Test trials method for policy evaluation"""
    policy = PolicyEvalQL(agent=dummy_agent)
    array = np.full(50, 1000.0)
    policy.trials()
    np.testing.assert_array_equal(array, policy.medians)
