# EmbodiedWorld-200K

Reference code for **EmbodiedWorld-200K**, the large-scale open-world
embodied-planning dataset. This repository releases the **data
construction pipeline** and the **evaluation
toolkit**  so that:

1. **Anyone can reproduce or extend the dataset.** The pipeline turns
   raw gameplay clips and 6-DoF camera-pose trajectories into the
   canonical `(o₀, ℓ, a₁:T)` triplet format, with all hyperparameters
   matching the paper.
2. **Anyone can score new methods on EmbodiedWorld-200K under the
   same protocol** that we used to report the numbers in our tables.

The dataset itself, baseline checkpoints, and the trained EWA model are
hosted separately on the project page:
<https://xiaokunfeng.github.io/EmbodiedWorld-200K/>


## Installation

The code targets **Python ≥ 3.10** and is intentionally lean. The
"core" dependencies (NumPy, Pillow) are needed to run Steps 1+2 of the
pipeline and the entire evaluation toolkit on CPU; the Step 3 VLM
annotation modules need an additional GPU stack.

```bash
# 1. Clone and enter the release directory
git clone <THIS_REPO_URL>
cd code_release

# 2. Core deps (Steps 1+2 + evaluation, CPU-only is fine)
pip install -r requirements.txt

# 3. (Optional) heavy stack for Step 3 VLM annotation
pip install torch transformers vllm decord qwen-vl-utils
```

`vllm` requires a recent CUDA-capable GPU. We tested with
`Qwen3.5-27B`, `vllm>=0.6`, and `transformers>=4.45`.

## Quick start

### A. Build the dataset (Steps 1+2, CPU)

Given a directory of raw sample manifests (one JSON per gameplay clip,
each pointing to a video and its 6-DoF camera-pose JSON; see
[`data_pipeline/examples/example_input.json`](data_pipeline/examples/example_input.json)):

```bash
python -m data_pipeline.run_pipeline \
    --input_dir  /path/to/raw_samples/ \
    --output_dir /path/to/labeled_out/
```

Each output JSON inherits all input fields and adds a `segments` block
containing every navigation-coherent segment together with its
variable-length W/A/S/D action streams. Defaults match the paper:
`trans_unit=0.05`, `rot_unit_deg=5.0`, `min_segment_len=60`,
`angle_threshold_deg=90`.

### B. Add VLM-based instruction annotation (Step 3, GPU + vLLM)

```bash
python -m data_pipeline.run_pipeline \
    --input_dir  /path/to/raw_samples/ \
    --output_dir /path/to/labeled_out/ \
    --run_step3_detailed \
    --run_step3_goal \
    --vlm_model_path Qwen/Qwen3.5-27B \
    --gpu_nums 4
```

You can also call each step independently:
`data_pipeline/instruction_annotation/{detailed_movement.py,
direction_consistency.py, goal_navigation.py}` each ship with their own
CLI, useful for chunked / multi-machine deployment.

### C. Evaluate predictions

Given a flat-list eval JSON dumped by your inference loop (see
[`evaluation/examples/example_eval_input.json`](evaluation/examples/example_eval_input.json)):

```bash
python -m evaluation.eval --eval-json my_eval_dump.json
```

The console output reports the five paper metrics (**TM**, **DirAcc**,
**nDTW**, **SR**, **NE**) plus complementary diagnostics, with a
per-`move_type_bucket` breakdown when the eval JSON carries that meta
field. Use `--csv per_sample.csv` to dump per-sample numbers and
`--report-json` to write an aggregate JSON summary next to the input.



## Package vs. script use

Every module in `data_pipeline/` and `evaluation/` is also importable
as a Python package:

```python
from data_pipeline import segment_trajectory, discretize_segment_by_magnitude
from evaluation     import eval_one, aggregate, load_json_str
```

This makes it trivial to plug the algorithms into your own training
loop or to swap individual hyperparameters without touching the CLI.



## Acknowledgements

The pipeline builds on top of the following community efforts: the
[OGameData](https://huggingface.co/datasets/...) gameplay-video
repository, the [VIPE](https://github.com/...) 6-DoF pose estimator,
and the [Qwen3.5](https://huggingface.co/Qwen) family of vision-language
models. Please cite these works when using their components.
