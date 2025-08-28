import pytest
import re

from jscatter.dependencies import (
    DependencyError,
    MissingCallable,
    MissingPackage,
)


@pytest.fixture
def missing_module():
    return MissingPackage(name='missing', extra='youneeditnow')


def check_pip_install_extra(value):
    assert re.search(r'pip install "jupyter-scatter\[youneeditnow\]"', value)


def test_missing_package(missing_module):
    with pytest.raises(DependencyError) as err:
        missing_module.func()
    assert re.search(r'member func from module missing\.', str(err.value))
    check_pip_install_extra(str(err.value))


@pytest.fixture
def missing_function():
    return MissingCallable.function('func', 'missing', 'youneeditnow')


def test_missing_function(missing_function):
    with pytest.raises(DependencyError) as err:
        missing_function()
    assert re.match(
        r'Attempting to call function func from module missing.', str(err.value)
    )
    check_pip_install_extra(str(err.value))


@pytest.fixture
def missing_class():
    return MissingCallable.class_('Class', 'missing', 'youneeditnow')


def test_missing_class(missing_class):
    with pytest.raises(DependencyError) as err:
        missing_class()
    assert re.match(
        r'Attempting to instantiate class Class from module missing.', str(err.value)
    )
    check_pip_install_extra(str(err.value))
