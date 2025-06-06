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