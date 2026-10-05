# Deprecated shim: `gist_utils` moved to `ssa.utils`. Aliases the real module so state
# (e.g. Global_data) is shared. Will be removed once callers migrate.
import sys
import warnings

import ssa.utils as _m

warnings.warn("gist_utils is deprecated; use ssa.utils", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _m
