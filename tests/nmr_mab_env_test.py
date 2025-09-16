"""(Non-)Markovian MAB environment test"""

from copy import deepcopy
from nmr_problem.nmr_mab_env import NonMarkovMABEnv
from nmr_problem.rewards import m_reward_1, m_reward_4, nm_reward_1, nm_reward_4

def test_reset():
    """
    Test environment reset behaviour
    """
    # Original environment and attributes
    env = NonMarkovMABEnv(rewards=[nm_reward_1, m_reward_4])
    ori_state = env.state
    ori_actions = env.actions
    ori_rewards = env.rewards

    # Reset environment and get attributes
    for action in [0, 3, 4, 2, 1, 4]:
        _, _, _, _, _ = env.step(action)
    state, trace = env.reset()
    actions = env.actions
    rewards = env.rewards

    assert ori_state == state
    assert ori_actions == actions
    assert ori_rewards == rewards
    assert trace.head is None
    assert trace.size() == 0

def test_deepcopy():
    """
    Test deepcopying behaviour
    """
    env = NonMarkovMABEnv(rewards=[nm_reward_1, m_reward_4])
    for action in [0, 3, 4, 2, 1, 4]:
        _, _, _, _, _ = env.step(action)
    env_copy = deepcopy(env)

    # Deepcopy copies actions
    assert env_copy.actions == env.actions

    # Deepcopy does not reset trace
    assert env_copy.trace.head is not None
    assert env_copy.trace.size() == 6

    # Deepcopy copies
    assert env.rewards is not env_copy.rewards
    for i, ori_reward in enumerate(env.rewards):
        assert ori_reward is env_copy.rewards[i]

def test_steps_given_nm_rewards():
    """Test step behavior of Non-Markovian environment

    Check outputs from sequential behaviour, after each step
    """
    nm_env = NonMarkovMABEnv(rewards=[nm_reward_1, nm_reward_4])

    test_cases = [
        (1, 1, 0),
        (2, 1, 0),
        (3, 1, 0),
        (4, 1, 0),
        (5, 3, 1),
        (6, 3, 0),
        (7, 2, 1),
    ]

    for trace_size, action, r in test_cases:
        observation, reward, terminated, truncated, _ = nm_env.step(action)
        assert nm_env.trace.size() == trace_size
        assert observation == "s"
        assert terminated is False
        assert truncated is False
        assert reward == r

def test_steps_given_m_rewards():
    """Test step behavior of Markovian environment

    Check outputs from sequential behaviour, after each step
    """
    m_env = NonMarkovMABEnv(rewards=[m_reward_1, m_reward_4])

    test_cases = [
        (1, 1, 0),
        (2, 2, 1),
        (3, 3, 1),
        (4, 4, 0),
        (5, 5, 0),
    ]

    for trace_size, action, r in test_cases:
        observation, reward, terminated, truncated, _ = m_env.step(action)
        assert m_env.trace.size() == trace_size
        assert observation == "s"
        assert terminated is False
        assert truncated is False
        assert reward == r