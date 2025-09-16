"""Bit environment in Gymnasium
"""

from typing import Optional
import gymnasium as gym

class BitEnv(gym.Env):
    """Bit environment

    - Observation is either '0' or '1'
    - Action from agent is either 'guess 0' or 'guess 1'
    - Sample from state space with probability 'p' for '1'
    - 'p' is constant
    """
    def __init__(self, p: float=0.5):
        if p < 0 or p > 1:
            raise ValueError("Constant 'p' out of range")
        self._p = p

        # Take note of type of p
        self._p_type = "constant"

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)

        self._obs = 0

    def __deepcopy__(self, memo):
        newone = type(self)(self._p)
        return newone

    def _get_info(self):
        return {"p type": self._p_type}

    def _sample_obs(self):
        p = self._p
        if self.np_random.random() <= p:
            self._obs = 1
        else:
            self._obs = 0

    def get_obs(self):
        """Return currently-stored obs."""
        return str(self._obs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset function

        Sample observation with default step 0
        """
        # Initialise seed and sample obs.
        super().reset(seed=seed)
        self._sample_obs()

        return str(self._obs), self._get_info()

    def step(self, action):
        """Step function

        - Sample '1'/'0' from environment as specified
        - Returns award of +1 if sample matches agent's guess; -1 if mismatch
        - No termination criteria
        - No truncation criteria
        """
        # Sample observation, providing current number of steps
        self._sample_obs()

        # See if agent guess matches sampled bit
        reward = 1 if self._obs == action else -1

        terminated = False
        truncated = False

        return str(self._obs), reward, terminated, truncated, self._get_info()

    def render(self):
        """Render function

        I don't think there is any rendering requirement at the moment
        """
        return
