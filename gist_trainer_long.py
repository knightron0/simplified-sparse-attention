# Deprecated entry point: use `python -m ssa.training.long`. Will be removed once callers migrate.
import runpy

runpy.run_module("ssa.training.long", run_name="__main__")
