"""Non-Markovian rewards"""

from nonmarkov_reward_problem.trace import TraceList

def reward_1(trace: TraceList) -> int:
    """Reward for using arm 1 four consecutive steps, followed by arm 3"""
    latest_actions = ["3", "1", "1", "1", "1"]
    reward = 1
    node = trace.head
    for action in latest_actions:
        if action in node.pi:
            node = node.next
        else:
            reward = 0
            break

    return reward

def reward_4(trace: TraceList) -> int:
    """Reward for playing arm 3 twice in a row, then arm 2"""
    latest_actions = ["2", "3", "3"]
    reward = 1
    node = trace.head
    for action in latest_actions:
        if action in node.pi:
            node = node.next
        else:
            reward = 0
            break

    return reward