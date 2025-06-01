"""Non-Markovian environment with perception filter in Gymnasium"""

from typing import Optional
from copy import deepcopy
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete, Tuple

class TwoCupEnv(gym.Env):
    """Environment to simulate two-cup problem

    - Action space is Discrete(3):
        - 0: move left
        - 1: taken cup
        - 2: move right

    - Observation space:
        - bot_position: 1D, from 0 to 6 inclusive; initialised as 3
        - cup1 and cup2, each has:
            - position:
                - 1D, from 0 to 6 inclusive
                - there are 2 possible initial positions
            - presence: binary on whether cup is still there or taken
        - collision_happened:
            - 0: collide left
            - 1: no collision
            - 2: collide right
    """
    def __init__(self):
        # Action space
        self.action_space = Discrete(3)

        # Observation space
        self._lo = 0
        lo = self._lo
        self._hi = 6
        hi = self._hi
        cup = Dict({
            "position": Box(low=lo, high=hi, shape=(1,), dtype=np.int8),
            "presence": Discrete(2)
        })
        self.observation_space = Dict({
            "bot_position": Box(low=lo, high=hi, shape=(1,), dtype=np.int8),
            "cups": Tuple((cup, cup)),
            "collision_happened": Discrete(3)
        })

        # Initial bot location (always the same)
        # And bot location class attribute
        self._init_bot_loc = np.array([3], dtype=np.int8)
        self._bot_loc = deepcopy(self._init_bot_loc)

        # Two options for initial cup attributes
        # And cup class attribute
        self._init_cups = {
            0: (
                {
                    "position": np.array([0], dtype=np.int8),
                    "presence": 1
                },
                {
                    "position": np.array([4], dtype=np.int8),
                    "presence": 1
                }
            ),
            1: (
                {
                    "position": np.array([2], dtype=np.int8),
                    "presence": 1
                },
                {
                    "position": np.array([6], dtype=np.int8),
                    "presence": 1
                }
            )
        }
        self._cups = deepcopy(self._init_cups[0])

        # Initial collision value (always the same)
        # And collision attribute
        self._init_collision = 1
        self._collision = self._init_collision

    def __deepcopy__(self, memo):
        newone = type(self)()
        return newone

    def _get_obs(self):
        return {
            "bot_position": deepcopy(self._bot_loc),
            "cups": deepcopy(self._cups),
            "collision_happened": self._collision,
        }
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Set bot's location
        self._bot_loc = deepcopy(self._init_bot_loc)

        # Initialise cups randomly
        randint = self.np_random.integers(0, 2, dtype=int)
        self._cups = deepcopy(self._init_cups[randint])

        # Set collision attribute
        self._collision = self._init_collision

        return self._get_obs()

    def step(self, action):
        self._collision = 1

        reward = 0
        terminated = False
        truncated = False

        # If the action is to try to take a cup
        if action == 1:
            cups = self._cups

            # If cup is indeed taken, then give reward
            # (and update cup presence)
            for cup in cups:
                if (cup["presence"] == 1 and
                    np.array_equal(self._bot_loc, cup["position"])):
                    cup["presence"] = 0
                    reward = 1
                    break

            # terminate if all cups are taken
            if (cups[0]["presence"] == 0 and
                cups[1]["presence"] == 0):
                terminated = True

            observation = self._get_obs()
            return observation, reward, terminated, truncated
        
        # If the action is to move left/right
        move = action - 1
        next_loc = self._bot_loc + move

        # Check for collision
        if np.all(next_loc < self._lo) or np.all(next_loc > self._hi):
            self._collision = action
        else:
            self._bot_loc += move
        
        observation = self._get_obs()
        return observation, reward, terminated, truncated
