"""Testing policy evaluation object with dummy environment and dummy agent"""

from collections import defaultdict
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

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.obs = 0
        return self.obs, {}

    def step(self, action):
        assert action == self.obs
        self.obs += 1
        return self.obs, 1, False, False, {}

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

    def get_action(self, obs):
        assert obs == self.total_rewards
        return self.total_rewards

    def update(self, obs, action, reward, done, next_obs):
        assert obs == self.total_rewards
        assert action == self.total_rewards
        assert reward == 1
        assert done is False
        assert next_obs == self.total_rewards + 1
        self.total_rewards += reward

@pytest.fixture
def dummy_agent(dummy_env):
    return DummyQLAgent(dummy_env)

def test_train_steps(dummy_agent):
    """Test train_steps method for policy evaluation"""
    policy = PolicyEvalQL(agent=dummy_agent)
    policy.train_steps()
    assert dummy_agent.total_rewards == 5_000

