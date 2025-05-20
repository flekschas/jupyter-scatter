from importlib.util import find_spec
from typing import Optional


class DependencyError(ImportError):
    """Error raised when an optional dependency is not installed."""

    pass


def check_optional_dependency(
    feature_name: str, package_name: str, extra_name: Optional[str] = None
) -> None:
    """
    Check if an optional dependency is installed.

    Parameters
    ----------
    feature_name : str
        Name of the feature requiring this dependency.
    package_name : str
        Name of the Python package to check for.
    extra_name : str, optional
        Name of the extra that provides this dependency.

    Raises
    ------
    DependencyError
        If the dependency is not installed.
    """
    if find_spec(package_name) is None:
        extra = extra_name or package_name
        raise DependencyError(
            f"The '{feature_name}' feature requires the '{package_name}' package. "
            f'Please install it with: pip install "jupyter-scatter[{extra}]" '
            f'or pip install "jupyter-scatter[all]"'
        )


def check_annotation_extras_dependencies() -> None:
    """Check if label extras dependencies are installed."""
    check_optional_dependency('contour annotation', 'seaborn', 'annotation-extras')


def check_label_extras_dependencies() -> None:
    """Check if label extras dependencies are installed."""
    check_optional_dependency('progress display', 'tqdm', 'label-extras')
    check_optional_dependency('largest_cluster positioning', 'hdbscan', 'label-extras')
