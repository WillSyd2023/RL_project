"""Non-Markovian-reward MAB in Gymnasium

Also, implementation of trace via linked list
"""

from typing import Set, List, Callable, Optional
from copy import deepcopy
import gymnasium as gym
from gymnasium.spaces import Discrete, Text

class TraceNode:
    """Node of trace linked list"""
    def __init__(self, node_id: int, pi: Set[str] = None):
        self.node_id = node_id
        if pi is None:
            pi = set()
        self.pi = pi
        self.next = None
        self.prev = None

class TraceList:
    """Trace linked list"""
    def __init__(self):
        self.next_id = 0
        self.head = None
    
    def add_node(self, pi: Set[str] = None):
        """Add next node as head"""
        new_node = TraceNode(
            node_id=self.next_id, pi=pi)

        self.next_id += 1

        back = self.head
        self.head = new_node
        self.head.next = back
        if back is not None:
            back.prev = self.head

    def size(self):
        """Size of list"""
        return self.next_id

class NonMarkovMABEnv(gym.Env):
    """Environment for simulating MAB with non-Markovian rewards

    Warning: deepcopying will wipe out trace list memory

    Args:
    - actions: number of actions possible; default is 5
    - rewards: non-Markovian rewards which takes TraceList and returns
        reward value; default is None
    """
    def __init__(self, actions: int = 5,
        rewards: List[Callable[TraceList, int]] = None):
        # Action spaces
        self.actions = actions
        self.action_space = Discrete(actions)

        # Observation space
        # Since this is MAB, then state is always "0"
        self.observation_space = Text(max_length=1, charset="s")
        self.state = "s"

        self.rewards = rewards

        self.trace = TraceList()
    
    def __deepcopy__(self, memo):
        newone = type(self)(
            actions=self.actions,
            rewards=deepcopy(self.rewards),
        )
        return newone
    
    def _get_info(self):
        return self.trace

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset simply resets the trace
        
        Returns state (which is always 's') and trace as TraceList object
        """
        super().reset(seed=seed)

        self.trace = TraceList()

        return self.state, self._get_info()

    def step(self, action):
        observation = self.state
        terminated = False
        truncated = False

        # Figure out reward
        reward = 0
        if self.rewards is not None:
            for call in self.rewards:
                reward += call(self.trace)
        
        return observation, terminated, truncated, reward, self._get_info()

    def render(self):
        """Render function

        I don't think there is any rendering requirement at the moment
        """
        return
    