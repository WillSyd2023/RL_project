"""Two-cups environment test"""

from copy import deepcopy
import numpy as np
from percep_filter_problem.percep_filter_env import TwoCupEnv

def test_init_two_cup_env():
    """Test initialised Two-Cups environment
    
    Particularly, that deepcopying in __init__ process worked as intended
    """
    env = TwoCupEnv()

    # init_bot_loc vs deepcopy bot_loc
    assert np.array_equal(env.init_bot_loc, np.array([3], dtype=np.int8))
    assert np.array_equal(env.bot_loc, np.array([3], dtype=np.int8))
    assert env.bot_loc is not env.init_bot_loc

    # init_cups[0] vs deepcopy cups
    target = [
        (np.array([0], dtype=np.int8), 1),
        (np.array([4], dtype=np.int8), 1),
    ]

    init_cups = env.init_cups[0]
    assert isinstance(init_cups, tuple)
    assert len(init_cups) == 2
    for i, cup in enumerate(init_cups):
        assert isinstance(cup, dict)
        assert len(cup) == 2
        assert set(cup.keys()) == set(["presence", "position"])
        assert np.array_equal(cup["position"], target[i][0])
        assert cup["presence"] == target[i][1]

    assert isinstance(env.cups, tuple)
    assert len(env.cups) == 2
    for i, cup in enumerate(env.cups):
        assert isinstance(cup, dict)
        assert len(cup) == 2
        assert set(cup.keys()) == set(["presence", "position"])
        assert np.array_equal(cup["position"], target[i][0])
        assert cup["presence"] == target[i][1]

    assert env.cups is not init_cups
    assert env.cups[0] is not init_cups[0]
    assert env.cups[0]["position"] is not init_cups[0]["position"]
    assert env.cups[1] is not env.init_cups[0][1]
    assert env.cups[1]["position"] is not init_cups[1]["position"]

    # collision
    assert env.init_collision == 1
    assert env.collision == 1
    env.collision = 2
    assert env.init_collision == 1

def test_move_left_cups_1():
    """Move bot to the left, then test collision

    Do this with first initial two-cups configuration
    """
    env = TwoCupEnv()
    env.cups = deepcopy(env.init_cups[0])

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
    env.cups = deepcopy(env.init_cups[1])

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
    env.cups = deepcopy(env.init_cups[0])

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
    env.cups = deepcopy(env.init_cups[1])

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
    env.cups = deepcopy(env.init_cups[0])

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

def test_retaking_cup():
    """Test retaking a cup"""
    env = TwoCupEnv()
    env.cups = deepcopy(env.init_cups[0])

    for act, bot_loc, cup_1, cup_2, coll, termin, r in [
        (2, 4, 1, 1, 1, False, -1),
        (1, 4, 1, 0, 1, False, 1),
        (1, 4, 1, 0, 1, False, -1),
    ]:
        obs, reward, terminated, truncated, _ = env.step(act)

        assert obs["bot_position"][0] == bot_loc
        assert obs["cups"][0]["presence"] == cup_1
        assert obs["cups"][1]["presence"] == cup_2
        assert obs["collision_happened"] == coll
        assert reward == r
        assert terminated is termin
        assert truncated is False

def test_terminated_1():
    """Test taking both cups, leading to termination

    Do this with first initial two-cups configuration
    """
    env = TwoCupEnv()
    env.cups = deepcopy(env.init_cups[0])

    for act, bot_loc, cup_1, cup_2, coll, termin, r in [
        (2, 4, 1, 1, 1, False, -1),
        (1, 4, 1, 0, 1, False, 1),
        (0, 3, 1, 0, 1, False, -1),
        (0, 2, 1, 0, 1, False, -1),
        (0, 1, 1, 0, 1, False, -1),
        (0, 0, 1, 0, 1, False, -1),
        (1, 0, 0, 0, 1, True, 1),
    ]:
        obs, reward, terminated, truncated, _ = env.step(act)

        assert obs["bot_position"][0] == bot_loc
        assert obs["cups"][0]["presence"] == cup_1
        assert obs["cups"][1]["presence"] == cup_2
        assert obs["collision_happened"] == coll
        assert reward == r
        assert terminated is termin
        assert truncated is False

def test_terminated_2():
    """Test taking both cups, leading to termination

    Do this with first initial two-cups configuration
    """
    env = TwoCupEnv()
    env.cups = deepcopy(env.init_cups[1])

    for act, bot_loc, cup_1, cup_2, coll, termin, r in [
        (0, 2, 1, 1, 1, False, -1),
        (1, 2, 0, 1, 1, False, 1),
        (2, 3, 0, 1, 1, False, -1),
        (2, 4, 0, 1, 1, False, -1),
        (2, 5, 0, 1, 1, False, -1),
        (2, 6, 0, 1, 1, False, -1),
        (1, 6, 0, 0, 1, True, 1),
    ]:
        obs, reward, terminated, truncated, _ = env.step(act)

        assert obs["bot_position"][0] == bot_loc
        assert obs["cups"][0]["presence"] == cup_1
        assert obs["cups"][1]["presence"] == cup_2
        assert obs["collision_happened"] == coll
        assert reward == r
        assert terminated is termin
        assert truncated is False