"""
Magnitude-based variable-length W/A/S/D action quantization.

Implements **Step 2** of the EmbodiedWorld-200K annotation pipeline,
described in Sec. 3.2 (Step 2) and App. A.2 of the paper. Given an
already-segmented navigation-coherent segment (NCS), this module
discretizes its continuous camera-pose motion into two variable-length
W/A/S/D streams:

  * a translation stream    — W=forward, S=backward, A=left, D=right
  * a rotation    stream    — W=pitch up, S=pitch down, A=yaw left, D=yaw right

with N denoting "no-op" on either stream. Composite tokens (e.g. ``WD``,
``SA``) are emitted when both axes contribute substantially within the
same bin.

Algorithmic outline (paper App. A.2)
------------------------------------

1. **Unit calibration.** A translation token corresponds to a fixed
   ``trans_unit`` (default ``0.05`` in the dataset's pose-coordinate
   scale) and a rotation token corresponds to ``rot_unit_deg`` (default
   ``5.0°``). These two constants are calibrated *once* over the entire
   corpus so that the resulting sequence length ``T`` lies in a
   training-friendly range.

2. **Mixed-motion accumulator.** For every adjacent frame pair ``i→i+1``
   we form a single scalar budget::

        m[i] = ||p[i+1] - p[i]|| / trans_unit
             + sqrt(Δyaw² + Δpitch²) / rot_unit_deg

   We scan the segment in temporal order and close a new bin every time
   the running sum reaches 1.0; any tail under 1.0 is folded into the
   last bin.

3. **Per-bin token emission.** Inside every bin we compute the *net*
   ``(Δx, Δy, Δyaw, Δpitch)`` and pick a translation/rotation token via
   thresholded comparisons against ``trans_unit`` / ``rot_unit_deg``.
   Several jitter-suppression rules (described inline) prevent noisy
   sub-units from being mislabelled as direction tokens.

The two output streams have the *same* length (one token per bin), which
makes the design directly compatible with the JSON output schema used by
the EmbodiedWorld-200K release::

    {
        "translation": ["W", "WA", "W", "N", ...],
        "rotation":    ["N", "A",  "A", "N", ...]
    }
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Default hyperparameters (paper App. A.2)
# ---------------------------------------------------------------------------
DEFAULT_TRANS_UNIT: float = 0.05            # 1 translation token  ≈ 0.05 pose-coord units
DEFAULT_ROT_UNIT_DEG: float = 5.0           # 1 rotation token     ≈ 5° (yaw or pitch)
DEFAULT_STATIC_TRANS_THRESH: float = 0.02   # whole-segment static (translation)
DEFAULT_STATIC_ROT_THRESH_DEG: float = 2.5  # whole-segment static (rotation)


def discretize_segment_by_magnitude(
    c2ws_segment: List[np.ndarray],
    *,
    trans_unit: float = DEFAULT_TRANS_UNIT,
    rot_unit_deg: float = DEFAULT_ROT_UNIT_DEG,
    static_trans_thresh: float = DEFAULT_STATIC_TRANS_THRESH,
    static_rot_thresh_deg: float = DEFAULT_STATIC_ROT_THRESH_DEG,
) -> Dict:
    """Discretize one NCS into a pair of variable-length W/A/S/D sequences.

    Args:
        c2ws_segment:           List of 4x4 c2w matrices for the segment, **already
                                relativised to the segment's first frame and
                                transformed by ``pose_utils.TRANSFORM_MATRIX``**.
                                See ``pose_utils.extract_segment_c2ws_local``.
        trans_unit:             Physical magnitude of one translation token
                                (default 0.05 pose-coord units).
        rot_unit_deg:           Physical magnitude of one rotation token
                                (default 5.0 degrees).
        static_trans_thresh:    If the segment's positional spread is below
                                this value AND the rotation spread is below
                                ``static_rot_thresh_deg``, a single ``["N"]``
                                token is emitted on each stream.
        static_rot_thresh_deg:  See above.

    Returns:
        A dict with::

            {
                "translation": ["W", "WA", ...],
                "rotation":    ["N",  "A", ...],
                "meta": {
                    "trans_unit":         float,
                    "rot_unit_deg":       float,
                    "num_action_tokens":  int,
                    "L_trans":            float,    # total path length in pose units
                    "L_rot_deg":          float,    # total accumulated rotation in degrees
                    "boundaries":         [int],    # bin boundaries (frame indices into c2ws_segment)
                }
            }
    """
    n = len(c2ws_segment)
    if n < 2:
        return _make_static_result(
            trans_unit=trans_unit, rot_unit_deg=rot_unit_deg,
            L_trans=0.0, L_rot_deg=0.0, boundaries=[0],
        )

    positions = np.array([c[:3, 3] for c in c2ws_segment])
    forwards = np.array([c[:3, 2] for c in c2ws_segment])
    yaws = np.arctan2(forwards[:, 0], forwards[:, 1])
    pitches = np.arcsin(np.clip(forwards[:, 2], -1.0, 1.0))

    # Per-step increments
    dxyz = np.diff(positions, axis=0)
    dtrans = np.linalg.norm(dxyz, axis=1)
    dyaw = np.diff(yaws)
    dyaw = np.where(dyaw > math.pi, dyaw - 2 * math.pi, dyaw)
    dyaw = np.where(dyaw < -math.pi, dyaw + 2 * math.pi, dyaw)
    dpitch = np.diff(pitches)
    drot_deg = np.degrees(np.sqrt(dyaw ** 2 + dpitch ** 2))

    L_trans = float(dtrans.sum())
    L_rot_deg = float(drot_deg.sum())

    # Whole-segment static check
    pos_range = positions.max(axis=0) - positions.min(axis=0)
    static_trans = float(pos_range.max()) < static_trans_thresh
    static_rot = L_rot_deg < static_rot_thresh_deg
    if static_trans and static_rot:
        return _make_static_result(
            trans_unit=trans_unit, rot_unit_deg=rot_unit_deg,
            L_trans=L_trans, L_rot_deg=L_rot_deg,
            boundaries=[0, n - 1],
        )

    # Mixed-motion accumulator
    m_step = dtrans / trans_unit + drot_deg / rot_unit_deg  # length n-1
    boundaries: List[int] = [0]
    acc = 0.0
    for i in range(n - 1):
        acc += float(m_step[i])
        if acc >= 1.0:
            boundaries.append(i + 1)
            acc = 0.0
    # Tail < 1.0: fold into the last bin
    if boundaries[-1] != n - 1:
        if len(boundaries) > 1:
            boundaries[-1] = n - 1
        else:
            boundaries.append(n - 1)

    trans_seq: List[str] = []
    rot_seq: List[str] = []

    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        # Translation: y = forward/back, x = left/right
        dy_bin = float(positions[e, 1] - positions[s, 1])
        dx_bin = float(positions[e, 0] - positions[s, 0])
        path_xyz_bin = float(dtrans[s:e].sum())
        # Rotation: yaw (xy-plane heading), pitch (vertical look)
        dyaw_bin = float(yaws[e] - yaws[s])
        if dyaw_bin > math.pi:
            dyaw_bin -= 2 * math.pi
        elif dyaw_bin < -math.pi:
            dyaw_bin += 2 * math.pi
        dpitch_bin = float(pitches[e] - pitches[s])
        rot_deg_bin = float(drot_deg[s:e].sum())

        trans_seq.append(_bin_trans_token(
            dx_bin, dy_bin, path_xyz_bin, trans_unit=trans_unit,
        ))
        rot_seq.append(_bin_rot_token(
            dyaw_bin, dpitch_bin,
            rot_deg_in_bin=rot_deg_bin, path_xyz_bin=path_xyz_bin,
            trans_unit=trans_unit, rot_unit_deg=rot_unit_deg,
        ))

    return {
        "translation": trans_seq,
        "rotation": rot_seq,
        "meta": {
            "trans_unit": trans_unit,
            "rot_unit_deg": rot_unit_deg,
            "num_action_tokens": len(trans_seq),
            "L_trans": L_trans,
            "L_rot_deg": L_rot_deg,
            "boundaries": [int(b) for b in boundaries],
        },
    }


# ---------------------------------------------------------------------------
# Bin-level quantization
# ---------------------------------------------------------------------------
#
# Background: each bin has accumulated motion ≈ 1 unit, but that unit
# may be a half-translation + half-rotation mix. Naively thresholding a
# single axis at "half a unit" would either (a) flag clean translation
# bins as N (when rotation drove most of the unit), or (b) emit spurious
# rotation tokens on bins that were filled by slow translation drift.
# The constants below are tuned to make the trade-off symmetric.
# ---------------------------------------------------------------------------

_SECONDARY_AXIS_RATIO_TRANS: float = 0.3
_SECONDARY_AXIS_RATIO_ROT: float = 0.5
_JITTER_STRAIGHT_RATIO: float = 0.4
_MIN_ROT_DOMINANCE: float = 0.65


def _bin_trans_token(
    dx: float, dy: float, path_xyz: float, *, trans_unit: float,
) -> str:
    """Pick the W/A/S/D translation token for one bin."""
    abs_trans_noise = trans_unit * 0.1
    if path_xyz < abs_trans_noise:
        return "N"
    ax, ay = abs(dx), abs(dy)
    major = max(ax, ay)
    if major < abs_trans_noise * 0.5:
        return "N"
    tok = ""
    if ay >= ax:  # y dominant -> forward/backward
        tok += "W" if dy > 0 else "S"
        if ax >= major * _SECONDARY_AXIS_RATIO_TRANS:
            tok += "D" if dx > 0 else "A"
    else:         # x dominant -> left/right
        if ay >= major * _SECONDARY_AXIS_RATIO_TRANS:
            tok += "W" if dy > 0 else "S"
        tok += "D" if dx > 0 else "A"
    return tok or "N"


def _bin_rot_token(
    dyaw: float,
    dpitch: float,
    *,
    rot_deg_in_bin: float,
    path_xyz_bin: float,
    trans_unit: float,
    rot_unit_deg: float,
) -> str:
    """Pick the W/A/S/D rotation token for one bin (stricter than translation)."""
    abs_rot_noise_deg = rot_unit_deg * 0.4
    min_major_rot_deg = rot_unit_deg * 0.3

    if rot_deg_in_bin < abs_rot_noise_deg:
        return "N"

    rot_contrib = rot_deg_in_bin / rot_unit_deg
    trans_contrib = path_xyz_bin / trans_unit
    total_contrib = rot_contrib + trans_contrib
    if total_contrib > 1e-6:
        if (rot_contrib / total_contrib) < _MIN_ROT_DOMINANCE:
            return "N"

    net_deg = math.degrees(math.sqrt(dyaw * dyaw + dpitch * dpitch))
    if rot_deg_in_bin > 1e-6 and (net_deg / rot_deg_in_bin) < _JITTER_STRAIGHT_RATIO:
        return "N"

    ayaw_deg = math.degrees(abs(dyaw))
    apitch_deg = math.degrees(abs(dpitch))
    major_deg = max(ayaw_deg, apitch_deg)
    if major_deg < min_major_rot_deg:
        return "N"

    tok = ""
    if apitch_deg >= ayaw_deg:  # pitch dominant -> up/down
        tok += "W" if dpitch > 0 else "S"
        if ayaw_deg >= major_deg * _SECONDARY_AXIS_RATIO_ROT:
            tok += "D" if dyaw > 0 else "A"
    else:                        # yaw dominant -> left/right
        if apitch_deg >= major_deg * _SECONDARY_AXIS_RATIO_ROT:
            tok += "W" if dpitch > 0 else "S"
        tok += "D" if dyaw > 0 else "A"
    return tok or "N"


def _make_static_result(
    *, trans_unit: float, rot_unit_deg: float,
    L_trans: float, L_rot_deg: float, boundaries: List[int],
) -> Dict:
    return {
        "translation": ["N"],
        "rotation": ["N"],
        "meta": {
            "trans_unit": trans_unit,
            "rot_unit_deg": rot_unit_deg,
            "num_action_tokens": 1,
            "L_trans": L_trans,
            "L_rot_deg": L_rot_deg,
            "boundaries": boundaries,
        },
    }


__all__ = [
    "discretize_segment_by_magnitude",
    "DEFAULT_TRANS_UNIT",
    "DEFAULT_ROT_UNIT_DEG",
    "DEFAULT_STATIC_TRANS_THRESH",
    "DEFAULT_STATIC_ROT_THRESH_DEG",
]


# ---------------------------------------------------------------------------
# CLI: read a pose JSON, segment, and emit (NCS, W/A/S/D streams) per segment.
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    import json
    import sys

    from .pose_utils import (
        ensure_4x4, extract_segment_c2ws_local, read_pose_from_json,
    )
    from .trajectory_segmentation import segment_trajectory

    p = argparse.ArgumentParser(
        description=(
            "Magnitude-based variable-length W/A/S/D quantization. "
            "Loads a pose JSON, segments it (NCS), and discretizes every "
            "segment with the variable-length scheme described in App. A.2."
        ),
    )
    p.add_argument("--pose_json", required=True)
    p.add_argument("--out", default=None, help="Output JSON path (stdout if omitted).")
    p.add_argument("--trans_unit", type=float, default=DEFAULT_TRANS_UNIT)
    p.add_argument("--rot_unit_deg", type=float, default=DEFAULT_ROT_UNIT_DEG)
    args = p.parse_args()

    segs = segment_trajectory(args.pose_json)
    c2ws_raw, _ = read_pose_from_json(args.pose_json)
    c2ws = ensure_4x4(c2ws_raw)

    out_segs = []
    for seg in segs:
        local = extract_segment_c2ws_local(c2ws, seg["start_frame"], seg["end_frame"])
        q = discretize_segment_by_magnitude(
            local,
            trans_unit=args.trans_unit,
            rot_unit_deg=args.rot_unit_deg,
        )
        out_segs.append({**seg, "action": q})

    payload = {
        "pose_json": args.pose_json,
        "num_segments": len(out_segs),
        "segments": out_segs,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[quantize] {len(out_segs)} segment(s) written to {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    _cli()
