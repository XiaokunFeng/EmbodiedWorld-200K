"""
Command-line entry point for the EmbodiedWorld-200K evaluation toolkit.

Reads a *flat-list* JSON of evaluation records produced by your
inference loop, computes the five paper metrics (TM, DirAcc, nDTW, SR,
NE) plus a battery of complementary diagnostics, and prints a human-
readable report. Optionally also writes:

  * a per-sample CSV (``--csv``) for offline analysis,
  * a JSON summary (``--report-json``) next to the input file,
  * a per-bucket breakdown (default on; disable with ``--no-bucket``).

Input record schema
-------------------

Each record in the input JSON list is::

    {
      "image":    "...png",                                    # optional, used as sample id
      "instruct": "...",                                       # optional
      "gt":       "{\\"translation\\":[...],\\"rotation\\":[...]}",
      "pred":     "{\\"translation\\":[...],\\"rotation\\":[...]}",
      "meta":     {"move_type_bucket": "forward", ...}         # optional
    }

The ``"pred"`` field may also be named ``"lora_pred"`` for backwards
compatibility with older inference scripts.

Examples
--------

Basic::

    python -m evaluation.eval --eval-json my_eval_dump.json

With per-sample CSV and a JSON summary::

    python -m evaluation.eval \\
        --eval-json my_eval_dump.json \\
        --csv per_sample.csv \\
        --report-json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

from . import metrics
from .metrics import (
    TAUS,
    aggregate,
    aggregate_by_bucket,
    eval_one,
    load_json_str,
)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(n: int, valid: Dict, token_stats: Dict, mf1: Dict,
                 dir_rates: Dict, traj_stats: Dict) -> None:
    print(f"\n=== Results (n={n}) ===\n")
    print("[Validity / Length quality]")
    for k in ("parse_rate", "trans_rot_len_match_rate", "vocab_ok_rate",
              "len_match_gt_rate", "len_abs_err_mean", "len_ratio_mean",
              "len_gt_mean", "len_pred_mean"):
        print(f"  {k:28s} {valid.get(k, 0.0):.4f}")
    print("\n[Token-level]")
    for k, v in token_stats.items():
        print(f"  {k:28s} {v:.4f}")
    print("\n[Token Macro-F1 / TM]")
    for k, v in mf1.items():
        print(f"  {k:28s} {v:.4f}")
    print("\n[Global Direction (DirAcc components)]")
    for k, v in dir_rates.items():
        print(f"  {k:28s} {v:.4f}")
    print("\n[Trajectory-level]")
    for k, v in traj_stats.items():
        print(f"  {k:28s} {v:.4f}")
    print()


def print_bucket_table(by_bucket: Dict[str, Dict]) -> None:
    if not by_bucket:
        return
    print("\n=== Per-bucket (move_type_bucket) summary ===\n")
    cols = ("bucket", "n", "em_overall", "keyset_f1_overall",
            "dir_all_ok", "ne_xy", "ndtw", "sr@1.0")
    print("  " + "  ".join(f"{c:>20s}" for c in cols))
    for b, stats in by_bucket.items():
        row = [
            b, stats["n"],
            stats["token_level"]["em_overall"],
            stats["macro_f1"]["keyset_f1_overall"],
            stats["global_direction"]["dir_all_ok"],
            stats["trajectory"]["ne_xy"],
            stats["trajectory"]["ndtw"],
            stats["trajectory"]["sr@1.0"],
        ]
        cells: List[str] = []
        for v in row:
            if isinstance(v, int):
                cells.append(f"{v:>20d}")
            elif isinstance(v, float):
                cells.append(f"{v:>20.4f}")
            else:
                cells.append(f"{str(v):>20s}")
        print("  " + "  ".join(cells))
    print()


def write_csv(path: str, records: List[Dict]) -> None:
    cols = [
        "eval_idx", "sample_id", "bucket",
        "parse_ok", "trans_rot_len_match", "vocab_ok",
        "len_gt", "len_pred", "len_match_gt", "len_abs_err", "len_ratio",
        "em_trans", "em_rot", "em_overall",
        "keyset_f1_trans", "keyset_f1_rot", "keyset_f1_overall",
        "axis_acc_trans_fb", "axis_acc_trans_lr",
        "axis_acc_rot_pitch", "axis_acc_rot_yaw",
        "dir_trans_fb_ok", "dir_trans_lr_ok",
        "dir_rot_yaw_ok", "dir_rot_pitch_ok", "dir_all_ok",
        "tl_gt", "tl_pred", "ne_xy", "ne_xyz", "ndtw",
        "heading_yaw_deg", "heading_pitch_deg",
    ]
    for tau in TAUS:
        cols += [f"sr@{tau}", f"osr@{tau}", f"spl@{tau}",
                 f"sdtw@{tau}", f"soft_sr@{tau}"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow([r.get(c) for c in cols])
    print(f"[csv] -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_records(data: List[Dict]):
    """Process a list of eval records and return ``(records, axis_counts_list)``.

    ``records`` is a list of per-sample dicts (the same format
    ``write_csv`` consumes); ``axis_counts_list`` is the parallel list of
    confusion-matrix dicts used by ``aggregate``.
    """
    records: List[Dict] = []
    axis_counts_list: List[Optional[Dict]] = []
    for i, item in enumerate(data):
        gt_obj = load_json_str(item.get("gt"))
        pred_raw = item.get("pred")
        if pred_raw is None:
            pred_raw = item.get("lora_pred")
        pred_obj = load_json_str(pred_raw)

        rec, cnts = eval_one(gt_obj, pred_obj)
        rec["eval_idx"] = i
        img = item.get("image")
        if isinstance(img, list):
            img = img[0] if img else ""
        rec["sample_id"] = (
            (img or "").split("/")[-1].replace(".png", "") or f"sample_{i}"
        )
        meta = item.get("meta") or {}
        rec["bucket"] = meta.get("move_type_bucket") or "unknown"
        records.append(rec)
        axis_counts_list.append(cnts)
    return records, axis_counts_list


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "EmbodiedWorld-200K evaluation: TM, DirAcc, nDTW, SR, NE "
            "(plus complementary diagnostics) on a flat-list eval JSON."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--eval-json", required=True,
                    help="Flat-list eval JSON with image/instruct/gt/pred[/meta] fields.")
    ap.add_argument("--csv", default=None,
                    help="Optional path to dump a per-sample CSV.")
    ap.add_argument("--report-json", action="store_true",
                    help="Also write aggregate stats to <eval-json>.metrics.json.")
    ap.add_argument("--no-bucket", action="store_true",
                    help="Skip the per-move-type-bucket breakdown.")
    # Backwards-compat alias for older invocations
    ap.add_argument("--eval_json", dest="eval_json_legacy", default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    eval_json = args.eval_json_legacy or args.eval_json
    if not eval_json:
        ap.error("--eval-json is required")

    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[load] {len(data)} entries from {eval_json}")

    records, axis_counts_list = evaluate_records(data)
    usable = [r for r in records if "em_overall" in r]
    usable_cnts = [
        c for r, c in zip(records, axis_counts_list) if "em_overall" in r
    ]
    if len(usable) < len(records):
        print(f"[warn] {len(records) - len(usable)} samples were unscoreable "
              "(parse/gt failed); they remain in validity stats only.")

    valid, token_stats, mf1, dir_rates, traj_stats = aggregate(
        usable, usable_cnts,
    )
    print_report(len(usable), valid, token_stats, mf1, dir_rates, traj_stats)

    by_bucket = None
    if not args.no_bucket:
        by_bucket = aggregate_by_bucket(records, axis_counts_list)
        print_bucket_table(by_bucket)

    if args.csv:
        write_csv(args.csv, records)

    if args.report_json:
        summary = {
            "eval_json": os.path.abspath(eval_json),
            "n_total": len(records),
            "n_evaluated": len(usable),
            "validity": valid,
            "token_level": token_stats,
            "macro_f1": mf1,
            "global_direction": dir_rates,
            "trajectory": traj_stats,
        }
        if by_bucket is not None:
            summary["by_bucket"] = by_bucket
        out_path = eval_json
        if out_path.endswith(".json"):
            out_path = out_path[: -len(".json")]
        out_path = out_path + ".metrics.json"
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        os.replace(tmp, out_path)
        print(f"[report-json] -> {out_path}")


if __name__ == "__main__":
    main()
