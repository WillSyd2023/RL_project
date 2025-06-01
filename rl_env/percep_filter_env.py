"""Non-Markovian environment with perception filter in Gymnasium"""

from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import numpy as np

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
        self._init_bot_loc = np.array([3], dtype=np.int8)

        # Two options for initial cup attributes
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

        # Initial collision value (always the same)
        self._init_collision = 0
