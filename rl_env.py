"""RL Environment Classes in Gymnasium"""

from collections.abc import Callable
import numpy as np
from scipy.optimize import minimize
import gymnasium as gym

class BitEnv(gym.Env):
    """Bit environment

    - Observation/state space contains only '0' or '1'
    - Sample from state space with probability 'p' for '1'
    - 'p' can be constant or function with parameter 'n' (current agent step)
    - Returns award of +1 if sample matches agent's guess; -1 if mismatch
    """

    def __init__(self, p: float | Callable[[int], float]=0.5):
        """Initialise bit environment.

        Only arg. is 'p':
        - constant number or function with parameter 'n' (current agent step)
        - must always be 0 <= p <= 1; otherwise, exception returned
        """
        # Check if probability is correct
        # If p is function of n
        if callable(p):
            # Do (imperfect) numerical check for non-constant probability
            # Check minimum value possible for p
            sol = minimize(lambda x: p(x[0]), [1], bounds=[(0, np.inf)])
            x = round(sol.x[0])
            if p(x) < 0:
                raise ValueError(
                    "Function 'p'=" + str(p(x)) + " when x=" + str(x)
                )
            # Check maximun value possible for p
            sol = minimize(lambda x: -p(x[0]), [1], bounds=[(0, np.inf)])
            x = round(sol.x[0])
            if p(x) > 1:
                raise ValueError(
                    "Function 'p'=" + str(p(x)) + " when x=" + str(x)
                )

            # Take note of type of p
            self._p_type = "function of n"
            
        # If p is constant
        else:
            if p < 0 or p > 1:
                raise ValueError("Constant 'p' out of range")

            # Take note of type of p
            self._p_type = "constant"

        # All good
        self._p = p
        
        # Action and observation spaces, which only contain '1' and '0'
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)

    def _get_info(self):
        return {"p type": self._p_type}