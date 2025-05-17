import warnings

import numpy as np
from traitlets import TraitError, Undefined

from .empty import empty
from .scitype import SciType


class Array(SciType):
    """A numpy array trait type."""

    info_text = 'a numpy array'
    dtype = None

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        if value is None or value is Undefined:
            return super(Array, self).validate(obj, value)
        try:
            r = np.asarray(value, dtype=self.dtype)
            if isinstance(value, np.ndarray) and r is not value:
                warnings.warn(
                    'Given trait value dtype "%s" does not match required type "%s". '
                    'A coerced copy has been created.'
                    % (np.dtype(value.dtype).name, np.dtype(self.dtype).name)
                )
            value = r
        except (ValueError, TypeError) as e:
            raise TraitError(e)
        return super(Array, self).validate(obj, value)

    def set(self, obj, value):
        new_value = self._validate(obj, value)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value
        if not np.array_equal(old_value, new_value):
            obj._notify_trait(self.name, old_value, new_value)

    def __init__(self, default_value=empty, allow_none=False, dtype=None, **kwargs):
        self.dtype = dtype
        if default_value is empty:
            default_value = np.array(0, dtype=self.dtype)
        elif default_value is not None and default_value is not Undefined:
            default_value = np.asarray(default_value, dtype=self.dtype)
        super(Array, self).__init__(
            default_value=default_value, allow_none=allow_none, **kwargs
        )

    def make_dynamic_default(self):
        if self.default_value is None or self.default_value is Undefined:
            return self.default_value
        else:
            return np.copy(self.default_value)
