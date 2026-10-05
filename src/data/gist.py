# Deprecated shim: `src.data.gist` moved to `ssa.data.gist`. Aliases the real module so state
# (e.g. Global_data) is shared. Will be removed once callers migrate.
import sys
import warnings

import ssa.data.gist as _m

warnings.warn("src.data.gist is deprecated; use ssa.data.gist", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _m
