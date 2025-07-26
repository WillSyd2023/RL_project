"""Policy evaluation integration with percep-filtered QL agent and TwoCupEnv test"""

import numpy as np
from rl_policy_eval.rl_policy_eval import PolicyEvalQL
from percep_filter_problem.percep_filter_env import TwoCupEnv
from percep_filter_problem.percep_filter_ql_agent import PercepFilterQLAgent
from percep_filter_problem.percep_filter import filter_four_bits, filter_complete

def test_train_steps_percep_filter_problem_4_bits():
    """
    Test train_steps method for policy evaluation
    using TwoCupEnv and QL agent with 4-bit filter
    """
    env = TwoCupEnv()
    agent = PercepFilterQLAgent(
        env = env,
        obs_filter = filter_four_bits,
    )
    policy = PolicyEvalQL(agent=agent)

    before = np.ones(env.action_space.n) * 1.0001

    policy.train_steps(steps=100)

    for _, value in policy.train_agent.q_values.items():
        assert not np.allclose(before, value), "Q-values did not change"

def test_train_steps_percep_filter_problem_complete():
    """
    Test train_steps method for policy evaluation
    using TwoCupEnv and QL agent with complete filter
    """
    env = TwoCupEnv()
    agent = PercepFilterQLAgent(
        env = env,
        obs_filter = filter_complete,
    )
    policy = PolicyEvalQL(agent=agent)

    before = np.ones(env.action_space.n) * 1.0001

    policy.train_steps(steps=100)

    for _, value in policy.train_agent.q_values.items():
        assert not np.allclose(before, value), "Q-values did not change"