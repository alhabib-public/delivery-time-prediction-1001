"""Pipeline layer: the generic phase runner an experiment module calls once."""

from .regression import ExecutionResult, Experiment, execute

__all__ = ["ExecutionResult", "Experiment", "execute"]
