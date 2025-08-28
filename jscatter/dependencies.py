from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from typing import Any, Optional


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


@dataclass
class MissingPackage:
    """
    This is used as a stand-in object when importing dependencies that were not
    installed, so as to keep the dependency profile slimmer. It duck-talks like
    a module. If any function, class or object is queried out of it, it raises
    an exception. Developers should be sure to guard such accesses with
    properly positioned check_*_dependencies function calls.
    """

    name: str
    extra: str

    def __getattr__(self, name: str) -> Any:
        raise DependencyError(
            f'Attemping to query member {name} from module {self.name}. '
            'However, this module was not installed as an optional dependency '
            'to Jupyter Scatter. Please install it with either '
            f'`pip install "jupyter-scatter[{self.extra}]"` '
            'or `pip install "jupyter-scatter[all]".'
        )


class Action(Enum):
    instantiate_class = 'instantiate class'
    call_function = 'call function'


@dataclass
class MissingCallable:
    action: Action
    name: str
    module: str
    extra: str

    def __call__(self, *args, **kwargs):
        raise DependencyError(
            f'Attempting to {self.action.value} {self.name} from '
            f'module {self.module}. '
            'However, this module was not installed as an optional dependency '
            'to Jupyter Scatter. Please install it with either '
            f'`pip install "jupyter-scatter[{self.extra}]"` '
            'or `pip install "jupyter-scatter[all]".'
        )

    @classmethod
    def class_(cls, name: str, module: str, extra: str) -> 'MissingCallable':
        return cls(
            action=Action.instantiate_class,
            name=name,
            module=module,
            extra=extra,
        )

    @classmethod
    def function(cls, name: str, module: str, extra: str) -> 'MissingCallable':
        return cls(
            action=Action.call_function,
            name=name,
            module=module,
            extra=extra,
        )
