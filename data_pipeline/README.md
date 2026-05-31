# EmbodiedWorld-200K Data Construction Pipeline

This directory contains the reference implementation of the
EmbodiedWorld-200K **data construction pipeline**. Given raw gameplay clips and their
6-DoF camera-pose trajectories, the pipeline produces structured
`(o₀, ℓ, a₁:T)` triplets ready for EWA training.

## Pipeline overview

```
raw  (video, camera-pose trajectory)
 │
 ▼  Step 1   trajectory_segmentation.py
NCS list  [{start_frame, end_frame, main_direction, ...}, ...]
 │
 ▼  Step 2   action_quantization.py
Variable-length W/A/S/D streams + per-bin metadata
 │
 ▼  Step 3a  instruction_annotation/detailed_movement.py
Detailed Movement Instruction (always-produced track)
 │
 ▼  Step 3b  instruction_annotation/direction_consistency.py
Direction-consistency post-check (filters bad annotations)
 │
 ▼  Step 3c  instruction_annotation/goal_navigation.py
High-Level Goal-Navigation Caption (admitted only when target is unique)
 │
 ▼
labeled JSON ready for EWA training
```

The first two steps are **CPU-only** and run in seconds per clip; Step 3
calls a frontier vision-language model via [vLLM](https://github.com/vllm-project/vllm)
and is what dominates the wall-clock cost of the dataset construction.

## Quick start

### Steps 1+2 only (CPU)

```bash
# from the repo root
python -m data_pipeline.run_pipeline \
    --input_dir  /path/to/raw_samples/ \
    --output_dir /path/to/labeled_out/
```

Each input JSON in `--input_dir` must point to one video and one
camera-pose JSON; see [`examples/example_input.json`](examples/example_input.json)
for the minimal schema. The output mirrors filenames and adds a
`segments` block to every sample.

### Steps 1+2+3 (requires a vLLM-served VLM)

```bash
python -m data_pipeline.run_pipeline \
    --input_dir  /path/to/raw_samples/ \
    --output_dir /path/to/labeled_out/ \
    --run_step3_detailed \
    --run_step3_goal \
    --vlm_model_path Qwen/Qwen3.5-27B \
    --gpu_nums 4
```

Heavy ML dependencies (`torch`, `transformers`, `vllm`, `decord`,
`qwen_vl_utils`) are imported lazily inside the Step 3 modules, so
omitting Step 3 lets you keep an extremely lean Python environment.

## Module reference

### `pose_utils.py`
Pose I/O and the canonical coordinate transform. The transform aligns
the recovered camera frame so that `+x = right`, `+y = forward`,
`+z = up`, which is the convention used throughout the paper.

### `trajectory_segmentation.py` &mdash; Step 1 (NCS)
Segments a 6-DoF trajectory into navigation-coherent segments:

| Hyperparameter         | Default | Paper section |
|------------------------|--------:|---------------|
| Smoothing window       | 5       | App. A.1      |
| Direction-cut angle θ  | 90°     | App. A.1      |
| Min consecutive frames | 3       | App. A.1      |
| Min segment length     | 60      | App. A.1      |
| Displacement-ratio ρ   | 0.30    | App. A.1      |
| Curvature-var σ²       | 1.50    | App. A.1      |

The CLI version (`python -m data_pipeline.trajectory_segmentation
--pose_json ...`) emits raw NCSs as JSON to stdout, useful for sanity
checks.

### `action_quantization.py` &mdash; Step 2 (variable-length W/A/S/D)
Implements the magnitude-based variable-length scheme from App. A.2:

| Constant                | Default     |
|-------------------------|------------:|
| `trans_unit`            | 0.05 pose-units |
| `rot_unit_deg`          | 5.0°        |
| `static_trans_thresh`   | 0.02        |
| `static_rot_thresh_deg` | 2.5°        |

Each surviving NCS is partitioned into bins along a single mixed-motion
budget `m[i] = ‖Δp‖/trans_unit + sqrt(Δyaw²+Δpitch²)/rot_unit_deg`; one
W/A/S/D token per bin is emitted on each stream so the two streams have
identical length.

### `instruction_annotation/`
Three vLLM-backed annotators. Each module exposes a Python
class for in-process use plus an independent CLI for batch
processing:

| Module                          | Track                                  | Default model |
|---------------------------------|----------------------------------------|---------------|
| `detailed_movement.py`          | Detailed Movement Instruction          | Qwen3.5-27B   |
| `direction_consistency.py`      | Post-hoc direction-consistency check   | Qwen3.5-27B   |
| `goal_navigation.py`            | High-Level Goal-Navigation Caption (4-step) | Qwen3.5-27B |

All prompt templates live in `instruction_annotation/prompts/` and are
loaded verbatim at runtime. Replace the model with any VLM exposing the
same chat-template + grounding format if you prefer a smaller / open
alternative.

## Output schema

After `run_pipeline.py` completes (any combination of steps), each
sample JSON is augmented with:

```json
{
  "video_path": "...",
  "camera_pose_json_path": "...",
  "segments": {
    "total_pose_frames": 720,
    "num_valid_segments": 3,
    "segment_list": [
      {
        "start_frame": 0,
        "end_frame":   145,
        "num_frames":  146,
        "main_direction":        [0.18, 0.97, 0.05],
        "direction_description": "left-forward",
        "displacement_ratio":    0.86,
        "curvature_variance":    0.07,
        "is_chaotic":            false,
        "translation":           ["W", "WA", "W", "N", "..."],
        "rotation":              ["N", "A",  "A", "N", "..."],
        "discretize_meta": {
          "trans_unit":          0.05,
          "rot_unit_deg":        5.0,
          "num_action_tokens":   12,
          "L_trans":             0.43,
          "L_rot_deg":           6.5,
          "boundaries":          [0, 23, 45, 60, 78, 96, 110, 122, 130, 137, 142, 146]
        },
        "sub_traj_label": {
          "annotatable":      true,
          "perspective":      "First-person",
          "task_description": "The character advances toward the front-left, ...",
          "move_target":      "the tent entrance to the front-left",
          "move_type":        "moving toward the front-left",
          "direction_consist": true,
          "direction_consist_post_check": { "is_consistent": true, "...": "..." }
        },
        "l2_label": {
          "status":      "ok",
          "l2_caption":  "Head to the tent entrance.",
          "clean_target": "the tent entrance",
          "first_bbox":  [120, 230, 280, 410],
          "last_bbox":   [ 90, 200, 320, 470],
          "verify":      { "is_same_target": true, "reason": "..." }
        }
      }
    ]
  }
}
```

Fields beyond `segments.segment_list[*].translation` and
`...rotation` are only present when the corresponding step is enabled.

## Frame-index remapping

When the pose-fps differs from the video-fps (e.g. pose at 60fps,
video at 25fps), Step 3 modules automatically remap segment indices
via `video_idx = pose_idx × (video_total / pose_total)`. The pose-side
`start_frame` / `end_frame` are kept in the output unchanged, so
downstream code remains coordinate-system-agnostic.
