"""
EmbodiedWorld-200K data construction pipeline.

This package implements the three annotation steps described in
Sec. 3.2 / App. A of the paper:

  Step 1  -- Navigation-Coherent Segmentation        (``trajectory_segmentation``)
  Step 2  -- Variable-length W/A/S/D quantization    (``action_quantization``)
  Step 3  -- Instruction annotation                  (``instruction_annotation``)
              * Detailed Movement Instruction
              * Direction-consistency post-check
              * High-Level Goal-Navigation Caption

Camera-pose I/O and the canonical coordinate transform live in
``pose_utils``; the all-in-one CLI orchestrating Steps 1+2 (+ optional
Step 3 if a vLLM-served VLM is available) lives in ``run_pipeline``.
"""

from .pose_utils import (
    TRANSFORM_MATRIX,
    ensure_4x4,
    extract_positions,
    extract_segment_c2ws_local,
    read_pose_from_json,
    relative_c2w,
    transform_c2ws,
)
from .trajectory_segmentation import (
    direction_to_description,
    segment_trajectory,
)
from .action_quantization import discretize_segment_by_magnitude

__all__ = [
    # pose
    "TRANSFORM_MATRIX",
    "read_pose_from_json",
    "ensure_4x4",
    "relative_c2w",
    "transform_c2ws",
    "extract_positions",
    "extract_segment_c2ws_local",
    # segmentation
    "segment_trajectory",
    "direction_to_description",
    # quantization
    "discretize_segment_by_magnitude",
]

__version__ = "0.1.0"
