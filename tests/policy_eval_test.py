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
    def __init__(self, env_id: int = 0):
        self.action_space = Discrete(500_000)
        self.observation_space = Discrete(1)
        self.obs = 0
        self.env_id = env_id

    def __deepcopy__(self, memo):
        newone = type(self)(env_id=self.env_id)
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
    return DummyEnv(env_id=23)

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

def test_training_same_as_testing_envs(dummy_agent):
    """Test if we use same envs for testing and training"""
    policy = PolicyEvalQL(agent=dummy_agent)
    assert policy.original_env.env_id == 23
    assert policy.ori_test_env.env_id == 23

    policy.one_trial(num_measure=0)
    assert policy.training_env.env_id == 23

    policy.avg_reward_per_eps(time_limit=-1, n_eps=1)
    assert policy.test_agent.env.env_id == 23

def test_training_vs_testing_envs(dummy_agent):
    """Test if we use different envs for testing and training"""
    policy = PolicyEvalQL(agent=dummy_agent,
        test_env=DummyEnv(env_id=42))
    assert policy.original_env.env_id == 23
    assert policy.ori_test_env.env_id == 42

    policy.one_trial(num_measure=0)
    assert policy.training_env.env_id == 23

    policy.avg_reward_per_eps(time_limit=-1, n_eps=1)
    assert policy.test_agent.env.env_id == 42

def test_test_agent_attribs(dummy_agent):
    """Test attributes of test agent before and after running avg_reward_per_eps"""
    policy = PolicyEvalQL(agent=dummy_agent)
    assert policy.ori_agent.learning_rate == 0.01
    assert policy.ori_agent.initial_epsilon == 0.1
    assert policy.ori_agent.epsilon == 0.1
    assert policy.ori_agent.epsilon_decay == 0
    assert policy.ori_agent.final_epsilon == 0.1
    assert policy.ori_agent.discount_factor == 0.99
    assert isinstance(policy.ori_agent.q_values, defaultdict)
    _ = policy.ori_agent.q_values[0]
    assert policy.ori_agent.q_values[0].shape == (500_000,)
    assert np.all(policy.ori_agent.q_values[0] == 1.0001)

    policy.ori_agent.q_values = defaultdict(int)
    policy.avg_reward_per_eps()

    assert policy.test_agent.learning_rate == 0
    assert policy.test_agent.initial_epsilon == 0
    assert policy.test_agent.epsilon == 0
    assert policy.test_agent.epsilon_decay == 0
    assert policy.test_agent.final_epsilon == 0
    assert policy.test_agent.discount_factor == 0
    assert isinstance(policy.test_agent.q_values, defaultdict)
    _ = policy.test_agent.q_values[0]
    assert policy.test_agent.q_values[0] == 0

# I place it at the back because this is slow
def test_trials(dummy_agent):
    """Test trials method for policy evaluation"""
    policy = PolicyEvalQL(agent=dummy_agent)
    array = np.full(50, 1000.0)
    policy.trials()
    np.testing.assert_array_equal(array, policy.medians)
