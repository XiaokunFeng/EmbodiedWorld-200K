"""
EmbodiedWorld-200K evaluation toolkit.

Implements the five paper metrics (TM, DirAcc, nDTW, SR, NE) and the
batch CLI that consumes a flat-list eval JSON dump produced by your
inference loop. See ``eval.py`` for the entry point and ``metrics.py``
for the underlying algorithms.
"""

from .metrics import (
    DEFAULT_ROT_DEG,
    DEFAULT_TRANS_UNIT,
    TAUS,
    aggregate,
    aggregate_by_bucket,
    eval_one,
    keyset_f1,
    load_json_str,
    reconstruct_traj,
    traj_metrics,
)

__all__ = [
    "TAUS",
    "DEFAULT_ROT_DEG",
    "DEFAULT_TRANS_UNIT",
    "load_json_str",
    "eval_one",
    "aggregate",
    "aggregate_by_bucket",
    "reconstruct_traj",
    "traj_metrics",
    "keyset_f1",
]

__version__ = "0.1.0"
