"""Two-cups environment test"""

import numpy as np
from percep_filter_problem.percep_filter_env import TwoCupEnv

def test_init_two_cup_env():
    """Test initialised Two-Cups environment
    
    Mostly checkaing deepcopying"""
    env = TwoCupEnv()

    assert np.array_equal(env._init_bot_loc, np.array([3], dtype=np.int8))
    assert np.array_equal(env._bot_loc, np.array([3], dtype=np.int8))
    assert env._bot_loc is not env._init_bot_loc

    assert isinstance(env._init_cups[0], tuple)
    assert len(env._init_cups[0]) == 2
    assert isinstance(env._init_cups[0][0], dict)
    for key in env._init_cups[0][0]:
        assert key == "position" or key == "presence"
    assert np.array_equal(env._init_cups[0][0]["position"], np.array([0], dtype=np.int8))
    assert np.array_equal(env._init_cups[0][0]["presence"], 1)
    assert isinstance(env._init_cups[0][1], dict)
    for key in env._init_cups[0][1]:
        assert key == "position" or key == "presence"
    assert np.array_equal(env._init_cups[0][1]["position"], np.array([4], dtype=np.int8))
    assert np.array_equal(env._init_cups[0][1]["presence"], 1)

    assert isinstance(env._cups, tuple)
    assert len(env._cups) == 2
    assert isinstance(env._cups[0], dict)
    for key in env._cups[0]:
        assert key == "position" or key == "presence"
    assert np.array_equal(env._cups[0]["position"], np.array([0], dtype=np.int8))
    assert np.array_equal(env._cups[0]["presence"], 1)
    assert isinstance(env._cups[1], dict)
    for key in env._cups[1]:
        assert key == "position" or key == "presence"
    assert np.array_equal(env._cups[1]["position"], np.array([4], dtype=np.int8))
    assert np.array_equal(env._cups[1]["presence"], 1)

    assert env._cups is not env._init_cups[0]
    assert env._cups[0] is not env._init_cups[0][0]
    assert env._cups[0]["position"] is not env._init_cups[0][0]["position"]
    assert env._cups[1] is not env._init_cups[0][1]
    assert env._cups[1]["position"] is not env._init_cups[0][1]["position"]

