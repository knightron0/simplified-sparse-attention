# Deprecated entry point: use `python -m ssa.training.cpt`. Will be removed once callers migrate.
import runpy

runpy.run_module("ssa.training.cpt", run_name="__main__")
