from .h5_sequence import H5GoalDataset

import importlib as _importlib


def __getattr__(name):
    """Lazy-import d4rl-dependent symbols so that H5GoalDataset can be used
    without mujoco_py / d4rl installed."""
    if name == "load_environment":
        from .d4rl import load_environment

        return load_environment
    # Everything else lives in .sequence (SequenceDataset, GoalDataset, etc.)
    _seq = _importlib.import_module(".sequence", __name__)
    val = getattr(_seq, name, None)
    if val is not None:
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
