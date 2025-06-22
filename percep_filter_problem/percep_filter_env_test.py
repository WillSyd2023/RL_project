"""Two-cups environment test"""

import pytest
from copy import deepcopy
import numpy as np
from percep_filter_problem.percep_filter_env import TwoCupEnv

def test_init_two_cup_env():
    """Test initialised Two-Cups environment
    
    Mostly checking deepcopying"""
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

    assert env._init_collision == 1
    assert env._collision == 1
    env._collision = 2
    assert env._init_collision == 1

def test_move_left_cups_1():
    """Move bot to the left, then test collision

    Do this with first initial two-cups configuration
    """
    env = TwoCupEnv()
    env._cups = deepcopy(env._init_cups[0])

    for act, bot_loc, cup_1, cup_2, coll, termin, r in [
        (0, 2, 1, 1, 1, False, -1),
        (0, 1, 1, 1, 1, False, -1),
        (0, 0, 1, 1, 1, False, -1),
        (0, 0, 1, 1, 0, False, -1),
        (0, 0, 1, 1, 0, False, -1),
        (2, 1, 1, 1, 1, False, -1),
    ]:
        obs, reward, terminated, truncated, _ = env.step(act)

        assert obs["bot_position"][0] == bot_loc
        assert obs["cups"][0]["presence"] == cup_1
        assert obs["cups"][1]["presence"] == cup_2
        assert obs["collision_happened"] == coll
        assert reward == r
        assert terminated is termin
        assert truncated is False

def test_move_left_cups_2():
    """Move bot to the left, then test collision

    Do this with second initial two-cups configuration
    """
    env = TwoCupEnv()
    env._cups = deepcopy(env._init_cups[1])

    for act, bot_loc, cup_1, cup_2, coll, termin, r in [
        (0, 2, 1, 1, 1, False, -1),
        (0, 1, 1, 1, 1, False, -1),
        (0, 0, 1, 1, 1, False, -1),
        (0, 0, 1, 1, 0, False, -1),
        (0, 0, 1, 1, 0, False, -1),
        (2, 1, 1, 1, 1, False, -1),
    ]:
        obs, reward, terminated, truncated, _ = env.step(act)

        assert obs["bot_position"][0] == bot_loc
        assert obs["cups"][0]["presence"] == cup_1
        assert obs["cups"][1]["presence"] == cup_2
        assert obs["collision_happened"] == coll
        assert reward == r
        assert terminated is termin
        assert truncated is False

def test_move_right_cups_1():
    """Move bot to the right, then test collision

    Do this with first initial two-cups configuration
    """
    env = TwoCupEnv()
    env._cups = deepcopy(env._init_cups[0])

    for act, bot_loc, cup_1, cup_2, coll, termin, r in [
        (2, 4, 1, 1, 1, False, -1),
        (2, 5, 1, 1, 1, False, -1),
        (2, 6, 1, 1, 1, False, -1),
        (2, 6, 1, 1, 2, False, -1),
        (2, 6, 1, 1, 2, False, -1),
        (0, 5, 1, 1, 1, False, -1),
    ]:
        obs, reward, terminated, truncated, _ = env.step(act)

        assert obs["bot_position"][0] == bot_loc
        assert obs["cups"][0]["presence"] == cup_1
        assert obs["cups"][1]["presence"] == cup_2
        assert obs["collision_happened"] == coll
        assert reward == r
        assert terminated is termin
        assert truncated is False

def test_move_right_cups_2():
    """Move bot to the right, then test collision

    Do this with second initial two-cups configuration
    """
    env = TwoCupEnv()
    env._cups = deepcopy(env._init_cups[1])

    for act, bot_loc, cup_1, cup_2, coll, termin, r in [
        (2, 4, 1, 1, 1, False, -1),
        (2, 5, 1, 1, 1, False, -1),
        (2, 6, 1, 1, 1, False, -1),
        (2, 6, 1, 1, 2, False, -1),
        (2, 6, 1, 1, 2, False, -1),
        (0, 5, 1, 1, 1, False, -1),
    ]:
        obs, reward, terminated, truncated, _ = env.step(act)

        assert obs["bot_position"][0] == bot_loc
        assert obs["cups"][0]["presence"] == cup_1
        assert obs["cups"][1]["presence"] == cup_2
        assert obs["collision_happened"] == coll
        assert reward == r
        assert terminated is termin
        assert truncated is False

def test_taking_no_cup():
    """Taking when there is no cup"""
    env = TwoCupEnv()

    for act, bot_loc, cup_1, cup_2, coll, termin, r in [
        (1, 3, 1, 1, 1, False, -1),
    ]:
        obs, reward, terminated, truncated, _ = env.step(act)

        assert obs["bot_position"][0] == bot_loc
        assert obs["cups"][0]["presence"] == cup_1
        assert obs["cups"][1]["presence"] == cup_2
        assert obs["collision_happened"] == coll
        assert reward == r
        assert terminated is termin
        assert truncated is False