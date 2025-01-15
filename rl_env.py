"""RL Environment Classes in Gymnasium"""

from collections.abc import Callable
from typing import Optional
import numpy as np
from scipy.optimize import minimize
import gymnasium as gym

class BitEnv(gym.Env):
    """Bit environment

    - Observation is either '0' or '1'
    - Action from agent is array where first element is either '0' or '1' and
      second element is number of steps taken so far
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
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]), high=np.array([1, np.inf]), dtype=np.uint64
        )
        self.observation_space = gym.spaces.Discrete(2)

        self._obs = 0

    def _sample_obs(self, n: int = 0):
        if callable(p):
            p = self._p(n)
        else:
            p = self._p
        if self.np_random.random() < p:
            self._obs = 1
        else:
            self._obs = 0

    def _get_info(self):
        return {"p type": self._p_type}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Initialise seed and sample obs.
        super().reset(seed=seed)
        self._sample_obs()

        return self._obs, self._get_info()

    # def sample(self, action):
        #