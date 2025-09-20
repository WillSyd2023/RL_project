"""Perception filter test"""

from copy import deepcopy
from percep_filter_problem.percep_filter_env import TwoCupEnv
from percep_filter_problem.percep_filter import filter_four_bits, filter_complete

def test_reset_four_bits():
    """4-bit filter should parse possible initial states correctly
    """
    env = TwoCupEnv()
    env.cups = deepcopy(env.init_cups[0])
    filtered_obs = filter_four_bits(env.get_obs())
    assert filtered_obs == "0100"

    env.cups = deepcopy(env.init_cups[1])
    filtered_obs = filter_four_bits(env.get_obs())    
    assert filtered_obs == "1000"

def test_reset_complete():
    """complete filter should parse reset state correctly
    """
    env = TwoCupEnv()
    env.cups = deepcopy(env.init_cups[0])
    filtered_obs = filter_complete(env.get_obs())
    assert filtered_obs == "3,0,1,4,1,1"

    env.cups = deepcopy(env.init_cups[1])
    filtered_obs = filter_complete(env.get_obs())    
    assert filtered_obs == "3,2,1,6,1,1"

def test_left_collision_four_bits():
    """4-bit filter should parse left-collision state correctly
    """
    env = TwoCupEnv()
    obs, _ = env.reset()
    for act in [0, 0, 0, 0]:
        obs, _, _, _, _ = env.step(act)
    filtered_obs = filter_four_bits(obs)
    assert filtered_obs == "0010"

def test_left_collision_complete():
    """Complete filter should parse left-collision state correctly
    """
    env = TwoCupEnv()
    obs = None
    for act in [0, 0, 0, 0]:
        obs, _, _, _, _ = env.step(act)
    filtered_obs = filter_complete(obs)
    assert filtered_obs == "0,0,1,4,1,0"

def test_right_collision_four_bits():
    """4-bit filter should parse right-collision state correctly
    """
    env = TwoCupEnv()
    obs, _ = env.reset()
    for act in [2, 2, 2, 2]:
        obs, _, _, _, _ = env.step(act)
    filtered_obs = filter_four_bits(obs)
    assert filtered_obs == "0001"

def test_right_collision_complete():
    """Complete filter should parse right-collision state correctly
    """
    env = TwoCupEnv()
    obs = None
    for act in [2, 2, 2, 2]:
        obs, _, _, _, _ = env.step(act)
    filtered_obs = filter_complete(obs)
    assert filtered_obs == "6,0,1,4,1,2"
