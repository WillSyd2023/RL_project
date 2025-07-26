"""Perceptual filters for PercepFilterQLAgent"""

from typing import Dict, Tuple, Union
import numpy as np

def filter_four_bits(
    obs:
        Dict[str,
            Union[
                np.ndarray,
                Tuple[
                    Dict[str, Union[np.ndarray, int]],
                    Dict[str, Union[np.ndarray, int]],
                ],
                int,
            ],
        ]
    ) -> str:
    """Filter obs. to 4 bits (hence making process non-Markovian):
    1. Is there a cup in the immediate left of bot
    2. Is there a cup in the immediate right of bot
    3. Was there just previously collision on the left
    4. Was there just previously collision on the right
    """
    bits = ""

    # 1. Check if there is a cup in the immediate left of bot
    pos = obs["bot_position"] - 1
    left = "0"
    for cup in obs["cups"]:
        if (cup["presence"] == 1 and
            np.array_equal(pos, cup["position"])):
            left = "1"
    bits += left

    # 2. Check if there is a cup in the immediate right of bot
    pos = obs["bot_position"] + 1
    right = "0"
    for cup in obs["cups"]:
        if (cup["presence"] == 1 and
            np.array_equal(pos, cup["position"])):
            right = "1"
    bits += right

    # 3. Record collision
    if obs["collision_happened"] == 0:
        bits += "10"
    elif obs["collision_happened"] == 1:
        bits += "00"
    else:
        bits += "01"

    return bits

def filter_complete(
    obs:
        Dict[str,
            Union[
                np.ndarray,
                Tuple[
                    Dict[str, Union[np.ndarray, int]],
                    Dict[str, Union[np.ndarray, int]],
                ],
                int,
            ],
        ]
    ) -> str:
    """Filter obs. to bits representing complete env. information (hence making process Markovian):
    1. current bot position
    2. leftmost cup initialised position
    3. is leftmost cup present now?
    4. rightmost cup initialised position
    5. is rightmost cup present now?
    6. Was there just previously collision? 0 for left collision, 1 for no collision, 2 for right collision
    """
    state = ""
    state += str(obs["bot_position"][0]) + ","
    for cup in obs["cups"]:
        state += str(cup["position"][0]) + ","
        state += str(cup["presence"]) + ","
    state += str(obs["collision_happened"])
    return state
