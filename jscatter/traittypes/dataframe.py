import inspect

from traitlets import TraitError, Undefined

from .empty import empty
from .scitype import SciType


class PandasType(SciType):
    """A Pandas DataFrame or Series trait type."""

    info_text = 'a Pandas DataFrame or Series'

    klass = None

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        if value is None or value is Undefined:
            return super(PandasType, self).validate(obj, value)
        try:
            value = self.klass(value)
        except (ValueError, TypeError) as e:
            raise TraitError(e)
        return super(PandasType, self).validate(obj, value)

    def set(self, obj, value):
        new_value = self._validate(obj, value)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value
        if (
            (old_value is None and new_value is not None)
            or (old_value is Undefined and new_value is not Undefined)
            or
            # Fix: when `old_value` and `new_value` are `None`,
            # `old_value.equals` breaks. To prevent that from happening we check
            # if both values are of type `self.klass`
            (
                isinstance(old_value, self.klass)
                and isinstance(new_value, self.klass)
                and not old_value.equals(new_value)
            )
        ):
            obj._notify_trait(self.name, old_value, new_value)

    def __init__(self, default_value=empty, allow_none=False, klass=None, **kwargs):
        if klass is None:
            klass = self.klass
        if (klass is not None) and inspect.isclass(klass):
            self.klass = klass
        else:
            raise TraitError('The klass attribute must be a class not: %r' % klass)
        if default_value is empty:
            default_value = klass()
        elif default_value is not None and default_value is not Undefined:
            default_value = klass(default_value)
        super(PandasType, self).__init__(
            default_value=default_value, allow_none=allow_none, **kwargs
        )

    def make_dynamic_default(self):
        if self.default_value is None or self.default_value is Undefined:
            return self.default_value
        else:
            return self.default_value.copy()


class DataFrame(PandasType):
    """A Pandas DataFrame trait type."""

    info_text = 'a Pandas DataFrame'

    def __init__(self, default_value=empty, allow_none=False, **kwargs):
        if 'klass' not in kwargs and self.klass is None:
            import pandas as pd

            kwargs['klass'] = pd.DataFrame
        super(DataFrame, self).__init__(
            default_value=default_value, allow_none=allow_none, **kwargs
        )
