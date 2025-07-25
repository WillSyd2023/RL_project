"""Policy evaluation integration with QL agent and MAB test"""

import numpy as np
from rl_policy_eval.rl_policy_eval import PolicyEvalQL
from rl_agent.ql_agent import QLAgent
from nmr_problem.nmr_mab_env import NonMarkovMABEnv
from nmr_problem.rewards import m_reward_1#, m_reward_4, nm_reward_1, nm_reward_4

def test_train_steps_nmr_problem_markovian():
    """
    Test train_steps method for policy evaluation
    using MAB with Markovian reward and QL agent
    """
    env = NonMarkovMABEnv(rewards=[m_reward_1])
    agent = QLAgent(env=env)
    policy = PolicyEvalQL(
    agent=agent,
    initial_epsilon=1.0,
    epsilon_decay=0.00005,
    final_epsilon=0.1,
    )

    before = policy.train_agent.q_values["s"].copy()
    policy.train_steps(steps=100)

    after = policy.train_agent.q_values["s"]
    assert not np.allclose(before, after), "Q-values did not change"
