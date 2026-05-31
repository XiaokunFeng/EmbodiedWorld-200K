# EmbodiedWorld-200K Evaluation Toolkit

Reference implementation of the five paper metrics described in
Sec. 3.4 / App. B for the variable-length W/A/S/D action format used
by EmbodiedWorld-200K.

## Metrics

| Metric                       | Direction | Family       | What it measures |
|------------------------------|----------:|--------------|------------------|
| **TM** (Token Match)         | ↑         | Token-level  | Mean per-step keyset-F1 between predicted and GT tokens, treating each token as a subset of `{W, A, S, D}` and `N` as the empty set. Computed independently on translation and rotation streams, then averaged. |
| **DirAcc** (Direction Acc.)  | ↑         | Token-level  | Per-axis global-direction agreement averaged over four axes (translation forward/back, translation left/right, rotation pitch, rotation yaw). |
| **nDTW** (normalised DTW)    | ↑         | Trajectory   | Normalised dynamic time warping on reconstructed 3-D trajectories, averaged over a distance-tolerance coefficient `τ ∈ {0.5, 1.0, 2.0}`. |
| **SR** (Success Rate)        | ↑         | Trajectory   | Fraction of episodes whose terminal-xy distance ≤ τ, averaged over the same three τ values. |
| **NE** (Navigation Error)    | ↓         | Trajectory   | Terminal-xy displacement, in translation units. Continuous companion to SR. |

The toolkit also reports a number of complementary diagnostics — JSON
parse rate, vocabulary validity, length quality (`len_match_gt_rate`,
`len_abs_err_mean`, `len_ratio_mean`), per-axis macro-F1, oracle
success rate, SPL, SDTW, soft-SR — for finer-grained ablation analysis.

## Shared pre-processing

Both your prediction and the ground truth may have **different
lengths**. To make over-shooting and under-shooting comparable, every
token-level metric right-pads both sequences with the no-op token
``"N"`` to ``L = max(|gt|, |pred|)``. Samples whose ``pred`` field
fails JSON parsing are replaced by the single-token sequence ``["N"]``
on each stream and scored through the same pipeline; **no samples are
dropped** from the validity stats.

Trajectory reconstruction uses the dataset-calibrated physical units
``trans_unit = 0.05`` and ``rot_unit_deg = 5.0`` (Sec. 3.2 / App. A.2),
so the absolute scale of NE / SR thresholds is consistent across runs.

## Input record schema

The CLI expects a flat JSON list whose elements look like this (see
[`examples/example_eval_input.json`](examples/example_eval_input.json)
for a working sample):

```json
{
  "image":    "...png",                                          // optional, used as sample id
  "instruct": "...",                                             // optional
  "gt":       "{\"translation\":[...],\"rotation\":[...]}",
  "pred":     "{\"translation\":[...],\"rotation\":[...]}",
  "meta":     {"move_type_bucket": "forward", ...}               // optional
}
```

The ``"pred"`` field may also be named ``"lora_pred"`` for backwards
compatibility with older inference scripts.

## Quick start

```bash
# from the repo root
python -m evaluation.eval --eval-json evaluation/examples/example_eval_input.json
```

You should see a console report with five sections (Validity, Token-
level, Token Macro-F1 / TM, Global Direction / DirAcc, Trajectory-level)
and a per-bucket table broken down by ``meta.move_type_bucket``.

### Common flags

```bash
# Per-sample CSV for offline analysis
python -m evaluation.eval \
    --eval-json my_eval_dump.json \
    --csv per_sample.csv

# JSON summary written next to the input file (my_eval_dump.metrics.json)
python -m evaluation.eval \
    --eval-json my_eval_dump.json \
    --report-json

# Skip the per-bucket table (useful when meta.move_type_bucket is missing)
python -m evaluation.eval \
    --eval-json my_eval_dump.json \
    --no-bucket
```

## Programmatic use

```python
from evaluation import eval_one, aggregate, load_json_str

# Score one sample
gt = load_json_str('{"translation":["W","W","WA"],"rotation":["N","A","N"]}')
pr = load_json_str('{"translation":["W","WA","W"],"rotation":["N","A","N"]}')
record, axis_counts = eval_one(gt, pr)
print(record["keyset_f1_overall"], record["ne_xy"], record["sr@1.0"])

# Aggregate a list of records
valid, token_stats, mf1, dir_rates, traj_stats = aggregate(
    [record], [axis_counts],
)
```

The same five paper metrics map onto the keys returned by ``aggregate``
as follows:

| Paper metric | Aggregator key                          |
|--------------|-----------------------------------------|
| **TM**       | ``mf1["keyset_f1_overall"]``            |
| **DirAcc**   | mean of `dir_rates["dir_trans_fb_ok"]`, `dir_rates["dir_trans_lr_ok"]`, `dir_rates["dir_rot_yaw_ok"]`, `dir_rates["dir_rot_pitch_ok"]` |
| **nDTW**     | mean of `traj_stats["sdtw@0.5"]`, `traj_stats["sdtw@1.0"]`, `traj_stats["sdtw@2.0"]` (use the τ-averaged form for the headline number) |
| **SR**       | mean of `traj_stats["sr@0.5"]`, `traj_stats["sr@1.0"]`, `traj_stats["sr@2.0"]` |
| **NE**       | ``traj_stats["ne_xy"]``                 |

> The companion ``eval.py`` CLI prints all five values directly so you
> rarely have to derive them by hand.
