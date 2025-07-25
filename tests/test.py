"""For tests that don't fit anywhere else"""

from collections import defaultdict
from copy import deepcopy
import numpy as np

def test_deepcopy_default_dicts():
    """Testing deepcopying default dicts"""
    dict1 = defaultdict(lambda: np.ones(10))
    dict1["s"]
    dict2 = deepcopy(dict1)
    assert dict1 is not dict2
    for k, v in dict1.items():
        assert np.array_equal(v, dict2[k])
