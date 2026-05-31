"""
End-to-end driver for the EmbodiedWorld-200K data pipeline.

Given a directory of *raw sample manifests* — each a JSON file pointing
to one gameplay clip and its accompanying camera-pose JSON — this
script runs Steps 1+2 of the pipeline:

  * Step 1 -- Navigation-Coherent Segmentation       (CPU only, fast)
  * Step 2 -- Variable-length W/A/S/D quantization   (CPU only, fast)

and writes one *labeled* sample JSON per input. By default Step 3
(VLM-based instruction annotation) is **skipped** because it requires a
multi-GPU vLLM deployment; pass ``--run_step3_detailed`` and/or
``--run_step3_goal`` to enable it.

Each input manifest must minimally contain::

    {
        "video_path":            "/abs/path/to/video.mp4",
        "camera_pose_json_path": "/abs/path/to/pose.json",
        "data_source":           "<arbitrary tag>",
        ...                      # any extra fields are preserved
    }

The output JSON inherits all input fields and adds::

    {
        "segments": {
            "total_pose_frames": int,
            "num_valid_segments": int,
            "segment_list": [
                {
                    "start_frame":           int,
                    "end_frame":             int,
                    "num_frames":            int,
                    "main_direction":        [x, y, z],
                    "direction_description": "left-forward",
                    "displacement_ratio":    float,
                    "curvature_variance":    float,
                    "is_chaotic":            False,
                    "translation":           ["W", "WA", ...],   # Step 2
                    "rotation":              ["N",  "A", ...],   # Step 2
                    "discretize_meta": {
                        "trans_unit":         0.05,
                        "rot_unit_deg":       5.0,
                        "num_action_tokens":  int,
                        "L_trans":            float,
                        "L_rot_deg":          float,
                        "boundaries":         [int],
                    }
                },
                ...
            ]
        }
    }

If ``--run_step3_detailed`` is set, every segment additionally gets a
``sub_traj_label`` with the parsed Detailed Movement Instruction;
``--run_step3_goal`` further appends ``l2_label`` for segments that pass
the 4-step grounding-and-verification loop.

Examples
--------

CPU-only Steps 1+2 over a directory::

    python -m data_pipeline.run_pipeline \\
        --input_dir  examples/raw_samples/ \\
        --output_dir out/labeled/

Add Detailed Movement Instruction (requires GPU + vLLM)::

    python -m data_pipeline.run_pipeline \\
        --input_dir  examples/raw_samples/ \\
        --output_dir out/labeled/ \\
        --run_step3_detailed \\
        --vlm_model_path  Qwen/Qwen3.5-27B \\
        --gpu_nums 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from .action_quantization import (
    DEFAULT_ROT_UNIT_DEG,
    DEFAULT_TRANS_UNIT,
    discretize_segment_by_magnitude,
)
from .pose_utils import (
    ensure_4x4,
    extract_segment_c2ws_local,
    read_pose_from_json,
)
from .trajectory_segmentation import segment_trajectory


# ---------------------------------------------------------------------------
# Steps 1+2: pose-only, CPU-only, fast
# ---------------------------------------------------------------------------

def run_segmentation_and_quantization(
    sample: Dict,
    *,
    trans_unit: float = DEFAULT_TRANS_UNIT,
    rot_unit_deg: float = DEFAULT_ROT_UNIT_DEG,
    seg_kwargs: Dict | None = None,
) -> Dict:
    """Mutate ``sample`` in-place, adding the ``segments`` block.

    Returns the same ``sample`` dict for chaining.
    """
    pose_path = sample.get("camera_pose_json_path", "")
    if not pose_path or not os.path.isfile(pose_path):
        sample["segments"] = {
            "total_pose_frames": 0,
            "num_valid_segments": 0,
            "segment_list": [],
            "_error": f"camera_pose_json_path missing or not found: {pose_path}",
        }
        return sample

    seg_kwargs = seg_kwargs or {}
    segments = segment_trajectory(pose_path, **seg_kwargs)

    # Re-load the full c2w trajectory once for quantization
    c2ws_raw, _ = read_pose_from_json(pose_path)
    c2ws = ensure_4x4(c2ws_raw)
    total_frames = len(c2ws)

    for seg in segments:
        local = extract_segment_c2ws_local(c2ws, seg["start_frame"], seg["end_frame"])
        q = discretize_segment_by_magnitude(
            local, trans_unit=trans_unit, rot_unit_deg=rot_unit_deg,
        )
        seg["translation"] = q["translation"]
        seg["rotation"] = q["rotation"]
        seg["discretize_meta"] = q["meta"]

    sample["segments"] = {
        "total_pose_frames": total_frames,
        "num_valid_segments": len(segments),
        "segment_list": segments,
    }
    return sample


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _maybe_run_step3_detailed(
    samples_with_paths: List,
    *,
    model_path: str,
    gpu_nums: int,
    sample_fps: int,
) -> None:
    """Run Detailed Movement Instruction annotation on the labeled samples
    in-place. Importing inside the function so users without the heavy
    ML stack can still run Steps 1+2."""
    from .instruction_annotation.detailed_movement import (
        DetailedMovementCaptioner,
        annotate_segments_for_sample,
    )
    cap = DetailedMovementCaptioner(model_path=model_path, gpu_nums=gpu_nums)
    for sample, _ in samples_with_paths:
        try:
            annotate_segments_for_sample(sample, cap, sample_fps=sample_fps)
        except Exception as exc:  # noqa: BLE001
            sample.setdefault("_step3_errors", []).append(str(exc))


def _maybe_run_step3_goal(
    samples_with_paths: List,
    *,
    model_path: str,
    gpu_nums: int,
    output_key: str,
) -> None:
    """Run High-Level Goal-Navigation caption annotation on the labeled
    samples in-place."""
    from .instruction_annotation.goal_navigation import (
        GoalNavigationCaptioner,
        annotate_one_segment,
    )
    cap = GoalNavigationCaptioner(model_path=model_path, gpu_nums=gpu_nums)
    for sample, _ in samples_with_paths:
        seg_info = sample.get("segments", {})
        seg_list = seg_info.get("segment_list", [])
        total_pose_frames = int(seg_info.get("total_pose_frames", 0) or 0)
        video_path = sample.get("video_path", "")
        if not video_path or not seg_list or not os.path.exists(video_path):
            continue
        for idx, seg in enumerate(seg_list):
            if output_key in seg:
                continue
            try:
                res = annotate_one_segment(seg, idx, video_path, total_pose_frames, cap)
            except Exception as exc:  # noqa: BLE001
                seg[output_key] = {"status": "error", "reason": str(exc)}
                continue
            if res is not None:
                seg[output_key] = res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "End-to-end EmbodiedWorld-200K data pipeline. "
            "By default runs CPU-only Steps 1+2 (segmentation + quantization). "
            "Pass --run_step3_* flags to additionally run VLM-based annotation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input_dir", required=True,
                    help="Directory of raw sample JSON manifests.")
    ap.add_argument("--output_dir", required=True,
                    help="Output directory for labeled sample JSONs.")

    # Step 1+2 knobs
    ap.add_argument("--trans_unit", type=float, default=DEFAULT_TRANS_UNIT,
                    help="Per-token translation magnitude (default 0.05 pose units).")
    ap.add_argument("--rot_unit_deg", type=float, default=DEFAULT_ROT_UNIT_DEG,
                    help="Per-token rotation magnitude in degrees (default 5.0).")

    # Step 3 (optional)
    ap.add_argument("--run_step3_detailed", action="store_true",
                    help="Also run Detailed Movement Instruction annotation (needs vLLM + GPU).")
    ap.add_argument("--run_step3_goal", action="store_true",
                    help="Also run Goal-Navigation Caption annotation (needs vLLM + GPU; "
                         "requires --run_step3_detailed first or pre-existing sub_traj_label).")
    ap.add_argument("--vlm_model_path", default="Qwen/Qwen3.5-27B",
                    help="HF Hub id or local path of the VLM used for Step 3.")
    ap.add_argument("--gpu_nums", type=int, default=1,
                    help="Tensor-parallel size for vLLM (Step 3 only).")
    ap.add_argument("--sample_fps", type=int, default=2,
                    help="Frames-per-second sampled from each video segment for Step 3a.")
    ap.add_argument("--goal_output_key", default="l2_label",
                    help="Where Step 3b writes its caption record into each segment.")

    # Sharding
    ap.add_argument("--chunk_id", type=int, default=-1,
                    help="Chunk id for data-parallel sharding (-1 = no sharding).")
    ap.add_argument("--chunk_nums", type=int, default=1)

    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.is_dir():
        sys.exit(f"[error] --input_dir not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f.name for f in in_dir.iterdir() if f.is_file() and f.suffix == ".json")
    if args.chunk_id >= 0:
        per = max(1, len(files) // max(1, args.chunk_nums))
        s = args.chunk_id * per
        e = len(files) if args.chunk_id == args.chunk_nums - 1 else (args.chunk_id + 1) * per
        files = files[s:e]
        print(f"[chunk] {args.chunk_id}/{args.chunk_nums}: {len(files)} files")

    # ---- Steps 1+2 (CPU) ----
    print(f"[step1+2] processing {len(files)} files (CPU; segmentation + quantization)")
    samples_with_paths: List = []
    for fname in files:
        with (in_dir / fname).open("r", encoding="utf-8") as f:
            sample = json.load(f)
        run_segmentation_and_quantization(
            sample,
            trans_unit=args.trans_unit,
            rot_unit_deg=args.rot_unit_deg,
        )
        samples_with_paths.append((sample, out_dir / fname))

    # ---- Step 3 (optional) ----
    if args.run_step3_detailed:
        print(f"[step3a] Detailed Movement Instruction "
              f"(model={args.vlm_model_path}, gpu_nums={args.gpu_nums})")
        _maybe_run_step3_detailed(
            samples_with_paths,
            model_path=args.vlm_model_path,
            gpu_nums=args.gpu_nums,
            sample_fps=args.sample_fps,
        )

    if args.run_step3_goal:
        print(f"[step3b] Goal-Navigation Caption "
              f"(model={args.vlm_model_path}, gpu_nums={args.gpu_nums})")
        _maybe_run_step3_goal(
            samples_with_paths,
            model_path=args.vlm_model_path,
            gpu_nums=args.gpu_nums,
            output_key=args.goal_output_key,
        )

    # ---- Save ----
    for sample, out_path in samples_with_paths:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(samples_with_paths)} labeled JSONs to {out_dir}")


if __name__ == "__main__":
    main()
