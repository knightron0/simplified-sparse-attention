# Deprecated entry point: use `python -m ssa.training.sft_link`. Will be removed once callers migrate.
import runpy

runpy.run_module("ssa.training.sft_link", run_name="__main__")
