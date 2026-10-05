# Deprecated shim: `gist_caching` moved to `ssa.generation.caching`. Aliases the real module so state
# (e.g. Global_data) is shared. Will be removed once callers migrate.
import sys
import warnings

import ssa.generation.caching as _m

warnings.warn("gist_caching is deprecated; use ssa.generation.caching", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _m
