# Deprecated shim: `src.data.gist_input_preprocessor` moved to `ssa.data.gist_input_preprocessor`. Aliases the real module so state
# (e.g. Global_data) is shared. Will be removed once callers migrate.
import sys
import warnings

import ssa.data.gist_input_preprocessor as _m

warnings.warn("src.data.gist_input_preprocessor is deprecated; use ssa.data.gist_input_preprocessor", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _m
