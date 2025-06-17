import pytest
from nmr_problem.trace import TraceNode, TraceList

@pytest.fixture
def init_list():
    """Initialise trace list"""
    return TraceList()

def test_initial_list(init_list):
    """Check initial trace list"""
    assert init_list.next_id == 0
    assert init_list.head is None

def test_filled_list(init_list):
    """Check trace list with some filled items"""
    init_list.add_node()
    first_node = init_list.head
    init_list.add_node(pi=set(["1", "0"]))
    second_node = init_list.head

    assert init_list.next_id == 2
    assert init_list.size() == 2
    assert init_list.head is not None
    
    node = init_list.head
    assert node.node_id == 1
    assert node.prev is None
    assert node.next == first_node
    assert node.pi == set(["0", "1"])

    node = node.next
    assert node.node_id == 0
    assert node.prev == second_node
    assert node.next is None
    assert len(node.pi) == 0