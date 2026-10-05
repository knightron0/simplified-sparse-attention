# Deprecated entry point: use `python -m ssa.training.sft`. Will be removed once callers migrate.
import runpy

runpy.run_module("ssa.training.sft", run_name="__main__")
