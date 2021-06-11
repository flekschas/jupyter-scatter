from .utils import sorting_to_dict

def test_sorting_to_dict():
    d = sorting_to_dict([1, 3, 2, 0])
    assert d == { 0: 3, 1: 0, 2: 2, 3: 1 }
