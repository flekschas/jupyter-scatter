from typing import (
    Any,
    List,
    Sequence,
    Type,
    TypeVar,
)

try:
    from typing import TypeGuard
except ImportError:
    from typing_extensions import TypeGuard

T = TypeVar('T')


def is_list_of(lst: Sequence[Any], cls: Type[T]) -> TypeGuard[List[T]]:
    """Typeguard for checking if all items in a list are instances of a specific class."""
    return all(isinstance(item, cls) for item in lst)
