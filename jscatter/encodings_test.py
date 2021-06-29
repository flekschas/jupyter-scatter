from functools import reduce
from .encodings import Encodings

def test_encodings():
    enc = Encodings()

    assert len(enc.data) == 0
    assert enc.max == 2

    enc.set('color', 'test')
    assert len(enc.data) == 1
    assert len(enc.visual) == 1
    assert enc.data[enc.visual['color']].component == 2

    enc.set('opacity', 'test')
    # The data component should remain at one since we only encode
    # the same data twice visually
    assert len(enc.data) == 1
    assert len(enc.visual) == 2
    assert enc.data[enc.visual['opacity']].component == 2

    enc.set('size', 'test2')
    assert len(enc.data) == 2
    assert len(enc.visual) == 3
    assert enc.data[enc.visual['size']].component == 3

    enc.set('opacity', 'test2')
    assert len(enc.data) == 2
    assert len(enc.visual) == 3
    assert enc.data[enc.visual['opacity']].component == 3

    try:
        enc.set('opacity', 'test3')
    except AssertionError:
        pass

    x = reduce(
        lambda acc, i: acc + int(enc.components._components[i].used),
        enc.components._components,
        0
    )
    print(x)
    print([enc.components._components[c].used for c in enc.components._components])
    print(enc.components.size)
    print(enc.components.full)

    # Nothing should have changed
    assert len(enc.data) == 2
    assert len(enc.visual) == 3
    assert enc.visual['opacity'] == 'test2'
    assert enc.data[enc.visual['opacity']].component == 3

    color_component = enc.get('color').component
    assert color_component == 2
    enc.delete('color')

    enc.set('opacity', 'test3')
    assert len(enc.data) == 2
    assert len(enc.visual) == 2
    assert enc.visual['opacity'] == 'test3'
    assert enc.data[enc.visual['opacity']].component == color_component

    enc.set('opacity', 'test4')
    assert len(enc.data) == 2
    assert len(enc.visual) == 2
    assert enc.visual['opacity'] == 'test4'
