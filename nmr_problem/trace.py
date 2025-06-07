"""Implementation of trace"""

from typing import Set

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