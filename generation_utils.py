# Deprecated shim: `generation_utils` moved to `ssa.generation.mixin`. Aliases the real module so state
# (e.g. Global_data) is shared. Will be removed once callers migrate.
import sys
import warnings

import ssa.generation.mixin as _m

warnings.warn("generation_utils is deprecated; use ssa.generation.mixin", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _m
