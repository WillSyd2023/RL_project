"""(Non-)Markovian MAB environment test"""

import pytest
from nmr_problem.nmr_mab_env import NonMarkovMABEnv
from nmr_problem.rewards import m_reward_1, m_reward_4, nm_reward_1, nm_reward_4

nm_env = NonMarkovMABEnv(rewards=[nm_reward_1, nm_reward_4])
@pytest.mark.parametrize("trace_size, action, r", [
    (1, 1, 0),
    (2, 1, 0),
    (3, 1, 0),
    (4, 1, 0),
    (5, 3, 10),
    (6, 3, 0),
    (7, 2, 1),
])
def test_steps_given_nm_rewards(trace_size, action, r):
    """Test step behavior of Non-Markovian environment
    
    Check outputs from sequential behaviour, after each step
    """
    observation, terminated, truncated, reward, _ = nm_env.step(action)

    assert nm_env.trace.size() == trace_size
    assert observation == "s"
    assert terminated is False
    assert truncated is False
    assert reward == r

nm_env_2 = NonMarkovMABEnv(rewards=[m_reward_1, m_reward_4])
@pytest.mark.parametrize("trace_size, action, r", [
    (1, 1, 0),
    (2, 2, 1),
    (3, 3, 100_000),
    (4, 4, 0),
    (5, 5, 0),
])
def test_steps_given_m_rewards(trace_size, action, r):
    """Test step behavior of Markovian environment
    
    Check outputs from sequential behaviour, after each step
    """
    observation, terminated, truncated, reward, _ = nm_env_2.step(action)

    assert nm_env_2.trace.size() == trace_size
    assert observation == "s"
    assert terminated is False
    assert truncated is False
    assert reward == r