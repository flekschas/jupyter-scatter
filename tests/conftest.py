import pytest
from importlib.util import find_spec

geoindex_available = find_spec('geoindex_rs') is not None


def pytest_collection_modifyitems(config, items):
    if geoindex_available:
        return
    skip = pytest.mark.skip(reason='geoindex-rs not available')
    for item in items:
        if 'test_label_placement' in item.nodeid:
            item.add_marker(skip)
