# Deprecated shim: `gist_llama` moved to `ssa.models.llama`. Aliases the real module so state
# (e.g. Global_data) is shared. Will be removed once callers migrate.
import sys
import warnings

import ssa.models.llama as _m

warnings.warn("gist_llama is deprecated; use ssa.models.llama", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _m
