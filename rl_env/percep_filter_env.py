"""Non-Markovian environment with perception filter in Gymnasium"""

from typing import Optional
from copy import deepcopy
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete

class TwoCupEnv(gym.Env):
    """Environment to simulate two-cup problem

    - Action space is Discrete(3):
        - 0: move left
        - 1: move right
        - 2: take cup

    - Observation space:
        - bot_position: 1D, from 0 to 6 inclusive; initialised as 3
        - cup1 and cup2, each has:
            - position:
                - 1D, from 0 to 6 inclusive
                - there are 2 possible initial positions
            - presence: binary on whether cup is still there or taken
        - collision_happened:
            binary on whether previous action caused collision on bot
    """
    def __init__(self):
        # Action space
        self.action_space = Discrete(3)

        # Observation space
        self.observation_space = Dict({
            "bot_position": Box(low=0, high=6, shape=(1,), dtype=np.int8),
            "cup1": Dict({
                "position": Box(low=0, high=6, shape=(1,), dtype=np.int8),
                "presence": Discrete(2)
            }),
            "cup2": Dict({
                "position": Box(low=0, high=6, shape=(1,), dtype=np.int8),
                "presence": Discrete(2)
            }),
            "collision_happened": Discrete(2)
        })

        # Initial bot location (always the same)
        # And bot location class attribute
        self._init_bot_loc = np.array([3], dtype=np.int8)
        self._bot_loc = deepcopy(self._init_bot_loc)

        # Two options for initial cup attributes
        # And cup class attribute
        self._init_cups = {
            0: {
                "cup1": {
                    "position": np.array([0], dtype=np.int8),
                    "presence": 1
                },
                "cup2": {
                    "position": np.array([4], dtype=np.int8),
                    "presence": 1
                }
            },
            1: {
                "cup1": {
                    "position": np.array([2], dtype=np.int8),
                    "presence": 1
                },
                "cup2": {
                    "position": np.array([6], dtype=np.int8),
                    "presence": 1
                }
            }
        }
        self._cups = deepcopy(self._init_cups[0])

        # Initial collision value (always the same)
        # And collision attribute
        self._init_collision = 0
        self._collision = self._init_collision

    def _get_obs(self):
        cups = deepcopy(self._cups)

        return {
            "bot_position":  self._bot_loc,
            "cup1": cups["cup1"],
            "cup2": cups["cup2"],
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