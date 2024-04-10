import pytest
from matplotlib.colors import Normalize

from jscatter.encodings import Components, Encodings, create_legend


def test_encodings_initial_state():
    enc = Encodings()
    assert len(enc.data) == 0, 'Initial data dictionary should be empty'
    assert (
        enc.max == 2
    ), 'Max should reflect the difference between total and reserved components'


def test_encodings_set_single_channel():
    enc = Encodings()
    enc.set('color', 'test')
    assert len(enc.data) == 1, 'Data should contain one entry after setting a channel'
    assert (
        len(enc.visual) == 1
    ), 'Visual should contain one entry after setting a channel'
    assert 'color' in enc.visual, 'Color channel should exist in visual mappings'


def test_encodings_set_same_data_different_channels():
    enc = Encodings()
    enc.set('color', 'test')
    enc.set('opacity', 'test')
    assert (
        len(enc.data) == 1
    ), 'Data should not duplicate entries for the same dimension'
    assert (
        len(enc.visual) == 2
    ), 'Visual should contain two entries after setting another channel'


def test_encodings_set_different_data():
    enc = Encodings()
    enc.set('color', 'test')
    enc.set('size', 'test2')
    assert (
        len(enc.data) == 2
    ), 'Data should contain two distinct entries for different dimensions'
    assert (
        len(enc.visual) == 2
    ), 'Visual should contain two entries for different channels'


def test_encodings_overwrite_channel_with_new_data():
    enc = Encodings()
    enc.set('opacity', 'test')
    enc.set('opacity', 'test2')
    assert len(enc.data) == 1, 'Data should remain with unique dimensions'
    assert (
        len(enc.visual) == 1
    ), 'Visual should update existing channel with new dimension'
    assert (
        enc.visual['opacity'].dimension == 'test2'
    ), 'Opacity channel should update to new dimension'


def test_encodings_reach_max_capacity():
    enc = Encodings()
    enc.set('opacity', 'test')
    enc.set('color', 'test2')
    with pytest.raises(AssertionError):
        enc.set('size', 'test3')  # Exceeding max capacity


def test_encodings_delete_channel():
    enc = Encodings()
    enc.set('color', 'test')
    color_component = enc.get('color').index
    enc.delete('color')
    assert (
        'color' not in enc.visual
    ), 'Color channel should be removed from visual mappings'
    enc.set('opacity', 'test3')
    assert (
        enc.visual['opacity'].dimension == 'test3'
    ), 'Opacity channel should be set to new dimension after deletion'
    assert (
        enc.get('opacity').index == color_component
    ), 'Component index should be reused after deletion'


def test_encodings_update_channel_to_new_data():
    enc = Encodings()
    enc.set('opacity', 'test')
    enc.set('opacity', 'test4')  # Update existing channel to new data
    assert (
        enc.visual['opacity'].dimension == 'test4'
    ), 'Opacity channel should update to new dimension'
    assert len(enc.data) == 1, 'Data should contain unique dimension entries'
    assert len(enc.visual) == 1, 'Visual should contain unique channel entries'


def test_components_add_and_delete():
    components = Components(total=4, reserved=2)
    assert components.size == 2, 'Reserved components should be counted.'
    components.add('test_encoding')
    assert components.size == 3, 'Adding an encoding should increase size.'
    components.delete('test_encoding')
    assert components.size == 2, 'Deleting an encoding should decrease size.'


def test_encodings_set_and_delete():
    encodings = Encodings()
    encodings.set('color', 'temperature')
    assert 'color' in encodings.visual, 'Color channel should be set.'
    encodings.delete('color')
    assert 'color' not in encodings.visual, 'Color channel should be deleted.'


def test_create_legend_categorical():
    encoding = [0.1, 0.5, 0.9]
    norm = None
    categories = {'cold': 0, 'warm': 1, 'hot': 2}
    legend = create_legend(encoding, norm, categories)
    assert legend['categorical'] is True, 'Legend should be categorical.'
    assert len(legend['values']) == 3, 'Legend values should match category count.'


def test_create_legend_continuous():
    encoding = [0, 0.5, 1]
    norm = Normalize(vmin=0, vmax=1)
    categories = None
    legend = create_legend(encoding, norm, categories, linspace_num=3)
    assert legend['categorical'] is False, 'Legend should be continuous.'
    assert len(legend['values']) == 3, 'Legend values should match linspace_num.'


@pytest.mark.parametrize(
    'setup_channels, check_channel, dimension, expected',
    [
        (['color'], 'color', 'temperature', True),
        (['size', 'opacity'], 'opacity', 'humidity', False),
    ],
)
def test_encodings_is_unique(setup_channels, check_channel, dimension, expected):
    encodings = Encodings()
    for channel in setup_channels:
        encodings.set(channel, dimension)
    assert (
        encodings.is_unique(check_channel) == expected
    ), f"Uniqueness check failed for {check_channel} with dimension '{dimension}'."
