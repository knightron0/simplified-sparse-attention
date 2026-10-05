# Deprecated shim: `gist_qwen2` moved to `ssa.models.qwen2`. Aliases the real module so state
# (e.g. Global_data) is shared. Will be removed once callers migrate.
import sys
import warnings

import ssa.models.qwen2 as _m

warnings.warn("gist_qwen2 is deprecated; use ssa.models.qwen2", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _m
