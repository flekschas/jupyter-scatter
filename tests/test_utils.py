import pytest

from jscatter.utils import (
    any_not,
    create_labeling,
    sorting_to_dict,
    to_scale_type,
    to_uint8,
    uri_validator,
)


@pytest.mark.parametrize(
    'sorting,expected_output',
    [
        ([1, 3, 2, 0], {0: 3, 1: 0, 2: 2, 3: 1}),
        ([], {}),
    ],
)
def test_sorting_to_dict(sorting, expected_output):
    assert sorting_to_dict(sorting) == expected_output


@pytest.mark.parametrize(
    'partial_labeling,column,expected',
    [
        # Test cases for dict input
        (
            {'minValue': 0, 'maxValue': 10, 'variable': 'test'},
            None,
            {'minValue': 0, 'maxValue': 10, 'variable': 'test'},
        ),
        (
            {'minValue': 0, 'maxValue': 10},
            'temperature',
            {'minValue': 0, 'maxValue': 10, 'variable': 'temperature'},
        ),
        # Test cases for list input
        ([0, 10, 'test'], None, {'minValue': 0, 'maxValue': 10, 'variable': 'test'}),
        ([0, 10], 'humidity', {'minValue': 0, 'maxValue': 10, 'variable': 'humidity'}),
        # Test empty or incomplete inputs
        ({}, 'empty', {'variable': 'empty'}),
        ([], None, {}),
    ],
)
def test_create_labeling(partial_labeling, column, expected):
    assert create_labeling(partial_labeling, column) == expected


@pytest.mark.parametrize(
    'input_value,expected_output',
    [
        (0, 0),
        (1, 255),
        (0.5, 127),
        (-0.1, 0),
        (1.1, 255),
    ],
)
def test_to_uint8(input_value, expected_output):
    assert to_uint8(input_value) == expected_output


@pytest.mark.parametrize(
    'list_input,value,expected_result',
    [
        ([None, None, 1], None, True),
        ([None, None, None], None, False),
        ([0, 0, 0], None, True),
    ],
)
def test_any_not(list_input, value, expected_result):
    assert any_not(list_input, value) == expected_result


@pytest.mark.parametrize(
    'uri_input,expected_validity',
    [
        ('http://example.com/foo', True),
        ('http://example.com', False),  # Must have a path
        ('not a uri', False),
    ],
)
def test_uri_validator(uri_input, expected_validity):
    assert uri_validator(uri_input) == expected_validity


@pytest.mark.parametrize(
    'norm_input,expected_scale_type',
    [
        (None, 'categorical'),
        ('some_other_norm', 'categorical'),
    ],
)
def test_to_scale_type_default_and_unsupported(norm_input, expected_scale_type):
    assert to_scale_type(norm_input) == expected_scale_type
