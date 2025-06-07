"""Non-Markovian-reward MAB in Gymnasium

Also, implementation of trace via linked list
"""

class TraceNode:
    """Node of trace linked list"""
    def __init__(self, node_id: int, pi: set = None):
        self.node_id = node_id
        if pi is None:
            pi = set()
        self.pi = pi
        self.next = None

class TraceList:
    """Trace linked list"""
    def __init__(self):
        self.next_id = 0
        self.head = None
    
    def add_node(self, pi: set = None):
        """Add next node as head"""
        new_node = TraceNode(
            node_id=self.next_id, pi=pi)

        self.next_id += 1

        back = self.head
        self.head = new_node
        self.head.next = back

