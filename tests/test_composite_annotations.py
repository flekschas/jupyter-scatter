import matplotlib
matplotlib.use('Agg')

import seaborn as sns

from jscatter.annotations import Line
from jscatter.composite_annotations import Contour
from jscatter.jscatter import Scatter

def test_contour():
    c = Contour()

    assert c.by is None
    assert c.line_color is None
    assert c.line_width is None
    assert c.line_opacity_by_level is False

    scatter = Scatter(
        data=sns.load_dataset("geyser"),
        x='waiting',
        y='duration',
        color_by='kind',
        annotations=[Contour()],
    )

    assert len(scatter._annotations) > 0
    assert all([isinstance(a, Line) for a in scatter._annotations])

    num_lines = len(scatter._annotations)

    c2 = Contour(by='kind')
    scatter.annotations([c2])

    assert len(scatter._annotations) > 0
    assert len(scatter._annotations) != num_lines

