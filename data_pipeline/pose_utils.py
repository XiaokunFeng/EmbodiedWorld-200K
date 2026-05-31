"""
Camera-pose I/O and coordinate-frame utilities used across the
EmbodiedWorld-200K data pipeline.

Each game clip in the EmbodiedWorld-200K dataset is paired with a 6-DoF
camera-pose trajectory recovered by VIPE (Huang et al., 2025). The pose
is stored as a JSON file with the schema::

    {
        "pose":     [[4x4 c2w], ...],          # one matrix per frame, world frame
        "metadata": {...}                       # arbitrary, optional
    }

This module provides three things:

1. ``read_pose_from_json``     — load the raw c2w list from disk;
2. ``ensure_4x4`` /
   ``relative_c2w``            — cast 3x4 → 4x4 and re-anchor a trajectory
                                  to its first frame;
3. ``TRANSFORM_MATRIX``        — a fixed coordinate swap that aligns the
                                  recovered camera frame with the conventional
                                  axis layout used throughout the paper:
                                  x = left/right, y = forward/backward,
                                  z = up/down.

Down-stream modules (``trajectory_segmentation`` and
``action_quantization``) consume the *transformed* trajectory; the
transform is fixed and should not be touched unless your pose source
uses a different camera convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------
#
# After applying TRANSFORM_MATRIX @ c2w to every (relativised) pose, the
# camera positions live in a frame where:
#   x-axis  = left (-) / right (+)
#   y-axis  = back (-) / forward (+)
#   z-axis  = down (-) / up (+)
#
# This is the convention used throughout the paper (Sec. 3.2 / App. A).
TRANSFORM_MATRIX: np.ndarray = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def read_pose_from_json(
    pose_file_path: Union[str, Path],
) -> Tuple[List[np.ndarray], Optional[dict]]:
    """Load a per-frame camera-to-world (c2w) trajectory from a JSON file.

    Args:
        pose_file_path: Path to a JSON file with ``{"pose": [...], "metadata": ...}``.

    Returns:
        ``(c2ws, metadata)`` where ``c2ws`` is a list of np.ndarray (each 3x4
        or 4x4 in shape) and ``metadata`` is the raw metadata dict (or ``None``).
    """
    pose_file_path = Path(pose_file_path)
    with pose_file_path.open("r", encoding="utf-8") as f:
        pose_json = json.load(f)
    extrinsics = pose_json.get("pose", [])
    c2ws = [np.asarray(pose, dtype=float) for pose in extrinsics]
    metadata = pose_json.get("metadata", None)
    return c2ws, metadata


def ensure_4x4(c2ws: List[np.ndarray]) -> List[np.ndarray]:
    """Promote any 3x4 c2w in the list to a homogeneous 4x4 matrix."""
    out: List[np.ndarray] = []
    for c2w in c2ws:
        c2w = np.asarray(c2w, dtype=float)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0.0, 0.0, 0.0, 1.0]])
        out.append(c2w)
    return out


# ---------------------------------------------------------------------------
# Coordinate operations
# ---------------------------------------------------------------------------

def relative_c2w(abs_c2ws: List[np.ndarray]) -> List[np.ndarray]:
    """Re-express a c2w trajectory so that the first frame becomes the origin.

    The first pose is mapped to the identity; every subsequent pose is
    expressed in the first-frame's coordinate system.
    """
    if not abs_c2ws:
        return []
    first_c2w = np.asarray(abs_c2ws[0], dtype=float)
    first_w2c = np.linalg.inv(first_c2w)
    rel: List[np.ndarray] = [np.eye(4)]
    for c2w in abs_c2ws[1:]:
        rel.append(first_w2c @ np.asarray(c2w, dtype=float))
    return rel


def transform_c2ws(c2ws: List[np.ndarray]) -> List[np.ndarray]:
    """Apply ``TRANSFORM_MATRIX`` to every pose.

    Equivalent to ``[TRANSFORM_MATRIX @ c2w for c2w in c2ws]``; provided as
    a function for clarity at call sites.
    """
    return [TRANSFORM_MATRIX @ np.asarray(c2w, dtype=float) for c2w in c2ws]


def extract_positions(pose_file_path: Union[str, Path]) -> np.ndarray:
    """Convenience: load a pose JSON and return the full (N, 3) position array
    in the *transformed*, first-frame-anchored coordinate system.

    Returns an empty ``(0, 3)`` array if the file has no poses.
    """
    c2ws, _ = read_pose_from_json(pose_file_path)
    if not c2ws:
        return np.zeros((0, 3))
    c2ws = ensure_4x4(c2ws)
    rel = relative_c2w(c2ws)
    transformed = transform_c2ws(rel)
    positions = np.array([c2w[:3, 3] for c2w in transformed])
    return positions


def extract_segment_c2ws_local(
    full_c2ws_4x4: List[np.ndarray], start: int, end: int,
) -> List[np.ndarray]:
    """Extract the ``[start, end]`` slice of a full c2w trajectory and
    re-anchor it to its own first frame (then apply ``TRANSFORM_MATRIX``).

    This is the canonical input format expected by both the segmentation
    and the action-quantization modules.

    Args:
        full_c2ws_4x4: Output of ``ensure_4x4`` on the *world*-frame c2w list.
        start, end:    Inclusive frame indices.

    Returns:
        List of 4x4 ndarray, length ``end - start + 1``. If the slice has
        fewer than two frames, it is returned unchanged (still in world
        frame); callers should guard against that case.
    """
    sub = full_c2ws_4x4[start : end + 1]
    if len(sub) < 2:
        return sub
    rel = relative_c2w(sub)
    return transform_c2ws(rel)


__all__ = [
    "TRANSFORM_MATRIX",
    "read_pose_from_json",
    "ensure_4x4",
    "relative_c2w",
    "transform_c2ws",
    "extract_positions",
    "extract_segment_c2ws_local",
]
