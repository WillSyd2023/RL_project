"""(Non-)Markovian rewards"""

from nmr_problem.nmr_trace import TraceList

def m_reward_1(trace: TraceList) -> int:
    """Reward for using arm 3"""
    if trace.size() < 1:
        return 0

    node = trace.head
    
    if "3" in node.pi:
        return 1
    else:
        return 0

def m_reward_4(trace: TraceList) -> int:
    """Reward for using arm 2"""
    if trace.size() < 1:
        return 0

    node = trace.head
    
    if "2" in node.pi:
        return 1
    else:
        return 0

def nm_reward_1(trace: TraceList) -> int:
    """Reward for using arm 1 four consecutive steps, followed by arm 3"""
    if trace.size() < 5:
        return 0

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

def nm_reward_4(trace: TraceList) -> int:
    """Reward for playing arm 3 twice in a row, then arm 2"""
    if trace.size() < 3:
        return 0

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