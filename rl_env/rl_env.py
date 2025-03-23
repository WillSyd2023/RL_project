"""RL Environment Classes in Gymnasium"""

from collections.abc import Callable
from typing import Optional
import copy
import numpy as np
from scipy.optimize import minimize
import gymnasium as gym

class BitEnv(gym.Env):
    """Bit environment

    - Observation is either '0' or '1'
    - Action from agent is either 'guess 0' or 'guess 1'
    - Sample from state space with probability 'p' for '1'
    - 'p' can be constant or function with parameter 'n' (current agent step)
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
            try:
                # Check minimum value possible for p
                sol = minimize(lambda x: p(x[0]), [1], bounds=[(0, np.inf)])
                x = round(sol.x[0])
                if p(x) < 0:
                    raise ValueError()
                # Check maximum value possible for p
                sol = minimize(lambda x: -p(x[0]), [1], bounds=[(0, np.inf)])
                x = round(sol.x[0])
                if p(x) > 1:
                    raise ValueError()
            except TypeError as e:
                raise TypeError("Function 'p' probably returns nothing") from e
            except ZeroDivisionError as e:
                raise ZeroDivisionError(
                    "Function 'p' may cause zero division"
                ) from e
            except ValueError as e:
                raise ValueError(
                    "Function 'p'=" + str(p(x)) + " when x=" + str(x)
                ) from e

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
        
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)

        # Number of steps taken by agent so far
        self._steps = 0

        self._obs = 0

    def __deepcopy__(self, memo):
        newone = type(self)(self._p)
        return newone

    def _sample_obs(self):
        if callable(self._p):
            p = self._p(self._steps)
        else:
            p = self._p
        if self.np_random.random() < p:
            self._obs = 1
        else:
            self._obs = 0

    def get_obs(self):
        return self._obs

    def _get_info(self):
        return {"p type": self._p_type}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset function

        Sample observation with default step 0
        """
        # Initialise seed and sample obs.
        super().reset(seed=seed)
        self._steps = 0
        self._sample_obs()

        return self._obs, self._get_info()

    def step(self, action):
        """Step function

        - Sample '1'/'0' from environment as specified
        - Returns award of +1 if sample matches agent's guess; -1 if mismatch
        - No termination criteria
        - No truncation criteria by default (use TimeLimit wrapper)
        """
        # Update current number of steps
        self._steps += 1

        # Sample observation, providing current number of steps
        self._sample_obs()

        # See if agent guess matches sampled bit
        reward = 1 if self._obs == action else -1

        terminated = False
        truncated = False

        return self._obs, reward, terminated, truncated, self._get_info()

    def render(self):
        """Render function

        I don't think there is any rendering requirement at the moment
        """
        return

class DualEnv(gym.Env):
    """Dual BitEnv environment

    'What if there are two environments, one that always produces 0,
    another that always produces 1, but we expose the agent to each of these
    for T steps in some random order'

    - Observation is either '0' or '1'
    - Action from agent is either 'guess 0' or 'guess 1'
    - Sample randomly from one of two given BitEnv's
    """
    def __init__(self, env1: BitEnv = BitEnv(p=0), env2: BitEnv = BitEnv(p=1)):
        """Initialise bit environment.

        Args are the two environments:
        - env1: default always produce 0
        - env2: default always produce 1
        """
        self._env1 = env1
        self._env2 = env2

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)

        # Number of steps taken by agent so far
        self._steps = 0

        self._obs = 0

    def __deepcopy__(self, memo):
        newone = type(self)(
            env1=copy.deepcopy(self._env1),
            env2=copy.deepcopy(self._env2),
        )
        return newone

    def _get_info(self):
        return {"env1": self._env1, "env2": self._env2}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset function

        Samples observation randomly from one of two given BitEnv's
        """
        # Initialise seed
        super().reset(seed=seed)

        # Reset the two environments and sample from one of them randomly
        self._env1.reset()
        self._env2.reset()
        if self.np_random.integers(1, high=3) == 1:
            self._obs = self._env1.get_obs()
        else:
            self._obs = self._env2.get_obs()

        return self._obs, self._get_info