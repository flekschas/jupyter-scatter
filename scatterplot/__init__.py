from .__version__ import __version__
from .scatterplot import display


def _jupyter_nbextension_paths():
    return [
        {
            "section": "notebook",
            "src": "static",
            "dest": "jupyter-scatterplot",
            "require": "jupyter-scatterplot/extension",
        }
    ]
