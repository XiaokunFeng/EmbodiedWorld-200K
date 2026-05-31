"""
Navigation-Coherent Segmentation (NCS).

Implements **Step 1** of the EmbodiedWorld-200K annotation pipeline,
described in Sec. 3.2 / App. A.1 of the paper. Given a 6-DoF camera-pose
trajectory (typically recovered by VIPE), this module produces a list of
**navigation-coherent segments**: contiguous frame intervals during
which the camera moves toward a stable destination with a single
dominant principal direction.

Algorithm (matches App. A.1 verbatim):

  1.  Re-anchor the full c2w trajectory to its first frame and apply the
      coordinate transform from ``pose_utils.TRANSFORM_MATRIX`` (so that
      ``+y`` is forward, ``+x`` is right, ``+z`` is up).
  2.  Compute per-frame displacement vectors and smooth them with a
      sliding-window average of size ``smooth_window`` (default 5).
  3.  Maintain a running estimate of the cumulative principal direction
      from the segment start to the current frame. A cut is triggered
      whenever the instantaneous (smoothed) motion deviates from the
      principal direction by more than ``angle_threshold_deg`` (default
      90°) for at least ``min_consecutive`` (default 3) consecutive
      frames.
  4.  Filter quality:
        - drop segments with ``displacement / path_length <
          displacement_ratio_thresh`` (default 0.3) *and* curvature
          variance ``> curvature_var_thresh`` (default 1.5);
        - drop segments shorter than ``min_segment_len`` (default 60).
  5.  For every surviving segment, return ``(start_frame, end_frame,
      main_direction, direction_description, ...)``.

This module deliberately performs **only** segmentation; the
W/A/S/D action quantization is in ``action_quantization.py`` so that
downstream code can swap in different action-space designs without
re-segmenting.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .pose_utils import extract_positions


# ---------------------------------------------------------------------------
# Default hyperparameters (paper App. A.1)
# ---------------------------------------------------------------------------
DEFAULT_MIN_SEGMENT_LEN: int = 60
DEFAULT_DISPLACEMENT_RATIO_THRESH: float = 0.3
DEFAULT_CURVATURE_VAR_THRESH: float = 1.5
DEFAULT_ANGLE_THRESHOLD_DEG: float = 90.0
DEFAULT_MIN_CONSECUTIVE_CHANGE: int = 3
DEFAULT_SMOOTH_WINDOW: int = 5


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_deltas(positions: np.ndarray, window: int = DEFAULT_SMOOTH_WINDOW) -> np.ndarray:
    """Compute per-frame displacement vectors with a sliding-window average.

    Args:
        positions: (N, 3) per-frame 3-D positions.
        window:    Smoothing window size (>=1). ``1`` means no smoothing.

    Returns:
        (N - 1, 3) smoothed displacement vectors.
    """
    if len(positions) < 2:
        return np.zeros((0, 3))

    deltas = np.diff(positions, axis=0)  # (N-1, 3)
    if window <= 1 or len(deltas) < window:
        return deltas

    kernel = np.ones(window) / window
    smoothed = np.zeros_like(deltas)
    for dim in range(3):
        smoothed[:, dim] = np.convolve(deltas[:, dim], kernel, mode="same")
    return smoothed


# ---------------------------------------------------------------------------
# Direction-change detection
# ---------------------------------------------------------------------------

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the angle (in radians) between two 3-D vectors.

    Returns ``0.0`` if either vector is degenerate.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def find_direction_split_points(
    positions: np.ndarray,
    smoothed_deltas: np.ndarray,
    angle_threshold_deg: float = DEFAULT_ANGLE_THRESHOLD_DEG,
    min_consecutive: int = DEFAULT_MIN_CONSECUTIVE_CHANGE,
) -> List[int]:
    """Find frame indices at which the trajectory's principal direction
    abruptly changes.

    A split point is emitted at the *first* frame of a deviation streak
    whose length reaches ``min_consecutive``.
    """
    if len(smoothed_deltas) < 2:
        return []

    angle_threshold_rad = np.radians(angle_threshold_deg)
    split_points: List[int] = []

    segment_start = 0
    consecutive_count = 0
    change_start_idx = -1

    for i in range(len(smoothed_deltas)):
        cumulative_disp = positions[i + 1] - positions[segment_start]
        cum_norm = np.linalg.norm(cumulative_disp)
        if cum_norm < 1e-10:
            consecutive_count = 0
            continue
        main_direction = cumulative_disp / cum_norm

        delta_norm = np.linalg.norm(smoothed_deltas[i])
        if delta_norm < 1e-10:
            consecutive_count = 0
            continue

        angle = angle_between_vectors(main_direction, smoothed_deltas[i])
        if angle > angle_threshold_rad:
            if consecutive_count == 0:
                change_start_idx = i
            consecutive_count += 1
            if consecutive_count >= min_consecutive:
                split_frame = change_start_idx + 1
                if split_frame > segment_start:
                    split_points.append(split_frame)
                    segment_start = split_frame
                consecutive_count = 0
        else:
            consecutive_count = 0

    return split_points


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

def compute_displacement_ratio(positions: np.ndarray, start: int, end: int) -> float:
    """Net displacement / accumulated path length for ``positions[start:end+1]``.

    1.0 means a perfectly straight segment; values close to 0 indicate
    back-and-forth motion ("shaky").
    """
    seg = positions[start : end + 1]
    if len(seg) < 2:
        return 0.0
    straight = float(np.linalg.norm(seg[-1] - seg[0]))
    path = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    if path < 1e-10:
        return 0.0
    return straight / path


def compute_curvature_variance(
    smoothed_deltas: np.ndarray, start: int, end: int,
) -> float:
    """Variance (radians²) of the per-step turning angle inside ``[start, end]``."""
    seg_deltas = smoothed_deltas[start:end]
    if len(seg_deltas) < 2:
        return 0.0
    angles = [
        angle_between_vectors(seg_deltas[i], seg_deltas[i + 1])
        for i in range(len(seg_deltas) - 1)
    ]
    if not angles:
        return 0.0
    return float(np.var(angles))


def is_chaotic(
    positions: np.ndarray,
    smoothed_deltas: np.ndarray,
    start: int,
    end: int,
    disp_ratio_thresh: float = DEFAULT_DISPLACEMENT_RATIO_THRESH,
    curv_var_thresh: float = DEFAULT_CURVATURE_VAR_THRESH,
) -> Tuple[bool, float, float]:
    """Decide whether a candidate segment is "chaotic" (drop it).

    A segment is chaotic iff *both* the displacement ratio is below the
    threshold *and* the curvature variance is above the threshold.
    """
    disp = compute_displacement_ratio(positions, start, end)
    curv = compute_curvature_variance(smoothed_deltas, start, end)
    chaotic = (disp < disp_ratio_thresh) and (curv > curv_var_thresh)
    return chaotic, disp, curv


# ---------------------------------------------------------------------------
# Segment summary
# ---------------------------------------------------------------------------

def compute_main_direction(positions: np.ndarray, start: int, end: int) -> np.ndarray:
    """Unit vector from the first to the last position of the segment.

    Returns the zero vector when the segment is effectively static.
    """
    disp = positions[end] - positions[start]
    n = float(np.linalg.norm(disp))
    if n < 1e-10:
        return np.zeros(3)
    return disp / n


def direction_to_description(direction: np.ndarray) -> str:
    """Convert a 3-D principal direction into a hyphen-joined human-readable
    label such as ``"left-forward"`` or ``"right-backward-up"``.

    The axis layout (after ``TRANSFORM_MATRIX``) is::
        x -> left(-) / right(+)
        y -> back(-) / forward(+)
        z -> down(-) / up(+)

    Components with absolute value below 0.15 are dropped, which suppresses
    spurious labels caused by numerical jitter. ``"stationary"`` is
    returned when no component survives the threshold.
    """
    x, y, z = float(direction[0]), float(direction[1]), float(direction[2])
    parts: List[str] = []
    if abs(x) > 0.15:
        parts.append("right" if x > 0 else "left")
    if abs(y) > 0.15:
        parts.append("forward" if y > 0 else "backward")
    if abs(z) > 0.15:
        parts.append("up" if z > 0 else "down")
    return "-".join(parts) if parts else "stationary"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def segment_trajectory(
    pose_file_path: str,
    *,
    min_segment_len: int = DEFAULT_MIN_SEGMENT_LEN,
    angle_threshold_deg: float = DEFAULT_ANGLE_THRESHOLD_DEG,
    min_consecutive: int = DEFAULT_MIN_CONSECUTIVE_CHANGE,
    disp_ratio_thresh: float = DEFAULT_DISPLACEMENT_RATIO_THRESH,
    curv_var_thresh: float = DEFAULT_CURVATURE_VAR_THRESH,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> List[Dict]:
    """Segment one camera-pose trajectory into navigation-coherent segments.

    Args:
        pose_file_path: Path to the pose JSON (see ``pose_utils.read_pose_from_json``).
        Other args:     See module-level docstring; defaults match the paper.

    Returns:
        A list of dicts, one per surviving segment. Each dict has keys::

            {
                "start_frame":           int,    # inclusive
                "end_frame":             int,    # inclusive
                "num_frames":            int,
                "main_direction":        [x, y, z],
                "direction_description": str,    # e.g. "left-forward"
                "displacement_ratio":    float,
                "curvature_variance":    float,
                "is_chaotic":            False,  # always False for surviving NCSs
            }

        ``start_frame`` / ``end_frame`` are indices into the **pose** trajectory.
        Callers that load video frames separately must remap indices when
        the pose-fps differs from the video-fps (see ``run_pipeline.py``).
    """
    positions = extract_positions(pose_file_path)
    if len(positions) < min_segment_len:
        return []

    smoothed = smooth_deltas(positions, window=smooth_window)

    split_points = find_direction_split_points(
        positions, smoothed,
        angle_threshold_deg=angle_threshold_deg,
        min_consecutive=min_consecutive,
    )

    boundaries = [0] + split_points + [len(positions) - 1]
    candidates: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s + 1 >= min_segment_len:
            candidates.append((s, e))

    valid_segments: List[Dict] = []
    for s, e in candidates:
        chaotic, disp, curv = is_chaotic(
            positions, smoothed, s, e,
            disp_ratio_thresh=disp_ratio_thresh,
            curv_var_thresh=curv_var_thresh,
        )
        if chaotic:
            continue
        main_dir = compute_main_direction(positions, s, e)
        valid_segments.append({
            "start_frame": int(s),
            "end_frame": int(e),
            "num_frames": int(e - s + 1),
            "main_direction": main_dir.tolist(),
            "direction_description": direction_to_description(main_dir),
            "displacement_ratio": round(float(disp), 4),
            "curvature_variance": round(float(curv), 6),
            "is_chaotic": False,
        })
    return valid_segments


__all__ = [
    "segment_trajectory",
    "smooth_deltas",
    "find_direction_split_points",
    "compute_displacement_ratio",
    "compute_curvature_variance",
    "is_chaotic",
    "compute_main_direction",
    "direction_to_description",
    "DEFAULT_MIN_SEGMENT_LEN",
    "DEFAULT_DISPLACEMENT_RATIO_THRESH",
    "DEFAULT_CURVATURE_VAR_THRESH",
    "DEFAULT_ANGLE_THRESHOLD_DEG",
    "DEFAULT_MIN_CONSECUTIVE_CHANGE",
    "DEFAULT_SMOOTH_WINDOW",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(
        description=(
            "Navigation-Coherent Segmentation (NCS). Loads a pose JSON, "
            "segments the trajectory, and prints the resulting NCSs as JSON."
        ),
    )
    p.add_argument("--pose_json", required=True, help="Path to the camera-pose JSON file.")
    p.add_argument("--out", default=None, help="Optional output JSON path (stdout if omitted).")
    p.add_argument("--min_segment_len", type=int, default=DEFAULT_MIN_SEGMENT_LEN)
    p.add_argument("--angle_threshold_deg", type=float, default=DEFAULT_ANGLE_THRESHOLD_DEG)
    p.add_argument("--min_consecutive", type=int, default=DEFAULT_MIN_CONSECUTIVE_CHANGE)
    p.add_argument("--disp_ratio_thresh", type=float, default=DEFAULT_DISPLACEMENT_RATIO_THRESH)
    p.add_argument("--curv_var_thresh", type=float, default=DEFAULT_CURVATURE_VAR_THRESH)
    p.add_argument("--smooth_window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    args = p.parse_args()

    segs = segment_trajectory(
        args.pose_json,
        min_segment_len=args.min_segment_len,
        angle_threshold_deg=args.angle_threshold_deg,
        min_consecutive=args.min_consecutive,
        disp_ratio_thresh=args.disp_ratio_thresh,
        curv_var_thresh=args.curv_var_thresh,
        smooth_window=args.smooth_window,
    )
    payload = {"pose_json": args.pose_json, "num_segments": len(segs), "segments": segs}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[ncs] {len(segs)} segment(s) written to {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    _cli()
