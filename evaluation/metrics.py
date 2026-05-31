"""
Quantitative metrics for EmbodiedWorld-200K.

Implements the five evaluation metrics described in Sec. 3.4 / App. B
of the paper, applied to predictions whose action-sequence length is
**not fixed** (the released dataset uses the variable-length W/A/S/D
scheme from ``action_quantization.py``).

Metrics
-------

Token-level (per-step):

  * **TM    (Token Match, ↑)** -- mean per-step keyset-F1 between
    predicted and ground-truth tokens, treating each token as a subset
    of ``{W, A, S, D}`` and ``N`` as the empty set. Computed
    independently on the translation and rotation streams and then
    averaged.

  * **DirAcc (Direction Accuracy, ↑)** -- per-axis global-direction
    agreement averaged over four axes (translation forward/backward
    and left/right; rotation pitch and yaw).

Trajectory-level (full-sequence):

  * **nDTW  (normalized DTW, ↑)** -- normalised dynamic time warping on
    reconstructed 3-D trajectories, averaged over a distance-tolerance
    coefficient ``τ ∈ {0.5, 1.0, 2.0}``.

  * **SR    (Success Rate, ↑)** -- fraction of episodes whose
    terminal-xy distance ≤ τ, averaged over the same three τ values.

  * **NE    (Navigation Error, ↓)** -- terminal-xy displacement, in
    translation units (continuous counterpart to SR).

Shared pre-processing
---------------------

* Predictions and ground truth may have **different lengths**.
  Token-level metrics right-pad both sequences with the no-op token
  ``"N"`` to ``L = max(|gt|, |pred|)`` so over-shooting and
  under-shooting are penalised symmetrically.

* Samples whose ``pred`` field fails JSON parsing are replaced by the
  single-token sequence ``["N"]`` on each stream and scored through the
  same pipeline; **no samples are dropped** from the evaluation pool.

* Trajectory reconstruction uses the dataset-calibrated physical units
  ``trans_unit = 0.05`` and ``rot_unit_deg = 5.0`` (Sec. 3.2 / App. A.2).

Input record schema
-------------------

The companion ``eval.py`` CLI consumes a flat JSON list whose entries
look like::

    {
      "image":    "...png",
      "instruct": "...",
      "gt":       "{\\"translation\\":[...],\\"rotation\\":[...]}",
      "pred":     "{\\"translation\\":[...],\\"rotation\\":[...]}",
      "meta":     {"move_type_bucket": "forward", ...}     # optional
    }

The ``"pred"`` field may equivalently be named ``"lora_pred"`` for
backwards compatibility.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Action-space constants
# ---------------------------------------------------------------------------
ALLOWED = set("WASD")
CONFLICT = [frozenset({"W", "S"}), frozenset({"A", "D"})]
TOKEN_RE = re.compile(r"^(N|[WASD]{1,2})$")

#: τ thresholds for trajectory-level SR / SPL / SDTW / soft-SR.
TAUS: Tuple[float, ...] = (0.5, 1.0, 2.0)

#: No-op pad token used to align sequences of unequal length.
PAD: str = "N"

#: Default per-token physical scale (matches App. A.2).
DEFAULT_ROT_DEG: float = 5.0
DEFAULT_TRANS_UNIT: float = 0.05


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def load_json_str(raw) -> Optional[Dict]:
    """Parse a ``pred`` / ``lora_pred`` / ``gt`` field into a dict.

    Accepts ``str``, ``dict`` or single-element ``list``. Returns ``None``
    when parsing ultimately fails.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*"translation"[^{}]*\}', s, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def norm(t) -> str:
    """Uppercase + strip a single token; ``None`` becomes ``""``."""
    return str(t).strip().upper() if t is not None else ""


def is_valid_token(t: str) -> bool:
    """Whether ``t`` is a syntactically valid W/A/S/D token (or ``N``)."""
    if not TOKEN_RE.match(t):
        return False
    if t == "N":
        return True
    letters = set(t)
    if not letters.issubset(ALLOWED):
        return False
    for pair in CONFLICT:
        if pair.issubset(letters):
            return False
    return True


def seq_all_valid(seq: List[str]) -> bool:
    return all(is_valid_token(t) for t in seq) if seq else False


def pad_to(seq: List[str], n: int, pad: str = PAD) -> List[str]:
    """Right-pad ``seq`` with ``pad`` up to length ``n`` (truncate if longer)."""
    if len(seq) >= n:
        return list(seq[:n])
    return list(seq) + [pad] * (n - len(seq))


# ---------------------------------------------------------------------------
# Token-level metrics
# ---------------------------------------------------------------------------

def exact_match(gt: List[str], pred: List[str]) -> float:
    """Per-step exact-match accuracy after right-padding to ``max(|gt|, |pred|)``."""
    n = max(len(gt), len(pred))
    if n == 0:
        return 0.0
    g = pad_to(gt, n)
    p = pad_to(pred, n)
    return sum(1 for i in range(n) if g[i] == p[i]) / n


def keyset_f1_step(g: str, p: str) -> float:
    """Per-step set-F1 over the letters in the token; ``N`` ⇔ empty set."""
    gs = set() if g == "N" else set(g)
    ps = set() if p == "N" else set(p)
    if not gs and not ps:
        return 1.0
    if not gs or not ps:
        return 0.0
    inter = len(gs & ps)
    prec = inter / len(ps)
    rec = inter / len(gs)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def keyset_f1(gt: List[str], pred: List[str]) -> float:
    """**TM** -- mean per-step keyset-F1 over the aligned (padded) sequences."""
    n = max(len(gt), len(pred))
    if n == 0:
        return 0.0
    g = pad_to(gt, n)
    p = pad_to(pred, n)
    return sum(keyset_f1_step(g[i], p[i]) for i in range(n)) / n


# Per-axis projections ------------------------------------------------------

def proj_ws(t: str) -> str:
    """Project a token onto the W/S axis (forward/backward or pitch)."""
    if "W" in t and "S" not in t:
        return "W"
    if "S" in t and "W" not in t:
        return "S"
    return "N"


def proj_ad(t: str) -> str:
    """Project a token onto the A/D axis (left/right or yaw)."""
    if "D" in t and "A" not in t:
        return "D"
    if "A" in t and "D" not in t:
        return "A"
    return "N"


def axis_counts(gt: List[str], pred: List[str], proj) -> Dict[Tuple[str, str], int]:
    """Confusion matrix of per-step projections, used by ``axis_acc`` and ``macro_f1``."""
    n = max(len(gt), len(pred))
    g = pad_to(gt, n)
    p = pad_to(pred, n)
    c: Dict[Tuple[str, str], int] = defaultdict(int)
    for i in range(n):
        c[(proj(g[i]), proj(p[i]))] += 1
    return c


def axis_acc(c: Dict[Tuple[str, str], int]) -> float:
    total = sum(c.values())
    if total == 0:
        return 0.0
    return sum(v for (g, p), v in c.items() if g == p) / total


def macro_f1(c: Dict[Tuple[str, str], int], labels: Tuple[str, ...]) -> float:
    f1s = []
    for lab in labels:
        tp = c.get((lab, lab), 0)
        fp = sum(c.get((g, lab), 0) for g in labels if g != lab)
        fn = sum(c.get((lab, p), 0) for p in labels if p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


# ---------------------------------------------------------------------------
# Global direction
# ---------------------------------------------------------------------------

def global_dir(seq: List[str], pos_k: str, neg_k: str) -> str:
    """Categorical label in ``{positive, negative, neutral}`` for one axis."""
    pos = sum(1 for t in seq if pos_k in t)
    neg = sum(1 for t in seq if neg_k in t)
    if pos > neg:
        return "positive"
    if pos < neg:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Trajectory reconstruction + VLN-style metrics
# ---------------------------------------------------------------------------

def reconstruct_traj(
    trans_seq: List[str], rot_seq: List[str],
    *,
    rot_deg: float = DEFAULT_ROT_DEG,
    unit: float = DEFAULT_TRANS_UNIT,
) -> List[Dict[str, float]]:
    """Rebuild the (L+1)-pose 3-D trajectory from W/A/S/D streams.

    The ``rot_deg`` / ``unit`` defaults match the per-token physical
    scale used to construct EmbodiedWorld-200K. They affect only the
    absolute scale of NE / SR thresholds; they do *not* change the
    relative ranking of methods.
    """
    rot_step = math.radians(rot_deg)
    n = min(len(trans_seq), len(rot_seq))
    if n == 0:
        return [{"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0}]
    x = y = z = yaw = pitch = 0.0
    pts: List[Dict[str, float]] = []
    for i in range(n):
        pts.append({"x": x, "y": y, "z": z, "yaw": yaw, "pitch": pitch})
        r = rot_seq[i]
        t = trans_seq[i]
        if "D" in r: yaw += rot_step
        if "A" in r: yaw -= rot_step
        if "W" in r: pitch += rot_step
        if "S" in r: pitch -= rot_step
        pitch = max(-math.pi / 2, min(math.pi / 2, pitch))
        if "W" in t: y += unit
        if "S" in t: y -= unit
        if "D" in t: x += unit
        if "A" in t: x -= unit
    pts.append({"x": x, "y": y, "z": z, "yaw": yaw, "pitch": pitch})
    return pts


def _dist3(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt(
        (a["x"] - b["x"]) ** 2
        + (a["y"] - b["y"]) ** 2
        + (a["z"] - b["z"]) ** 2
    )


def path_len(pts: List[Dict[str, float]]) -> float:
    return sum(_dist3(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def calc_ndtw(pred_pts: List[Dict], gt_pts: List[Dict], tau: float) -> float:
    """nDTW(τ) on reconstructed 3-D trajectories."""
    if not pred_pts or not gt_pts:
        return 0.0
    m, n = len(pred_pts), len(gt_pts)
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = _dist3(pred_pts[i - 1], gt_pts[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return math.exp(-dp[m][n] / (max(n, 1) * max(tau, 1e-6)))


def angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % (2 * math.pi)
    if d > math.pi:
        d = 2 * math.pi - d
    return math.degrees(d)


def traj_metrics(
    gt_t: List[str], gt_r: List[str], pred_t: List[str], pred_r: List[str],
    *,
    rot_deg: float = DEFAULT_ROT_DEG,
    unit: float = DEFAULT_TRANS_UNIT,
) -> Dict[str, float]:
    """Compute all trajectory-level metrics for one sample."""
    gt_pts = reconstruct_traj(gt_t, gt_r, rot_deg=rot_deg, unit=unit)
    pr_pts = reconstruct_traj(pred_t, pred_r, rot_deg=rot_deg, unit=unit)

    tl_gt = path_len(gt_pts)
    tl_pred = path_len(pr_pts)
    ge = gt_pts[-1]
    pe = pr_pts[-1]
    ne_xy = math.sqrt((ge["x"] - pe["x"]) ** 2 + (ge["y"] - pe["y"]) ** 2)
    ne_xyz = math.sqrt(
        (ge["x"] - pe["x"]) ** 2
        + (ge["y"] - pe["y"]) ** 2
        + (ge["z"] - pe["z"]) ** 2
    )
    oracle_best = min(_dist3(p, ge) for p in pr_pts)
    nd = calc_ndtw(pr_pts, gt_pts, tau=1.0)

    out: Dict[str, float] = {
        "tl_gt": tl_gt,
        "tl_pred": tl_pred,
        "ne_xy": ne_xy,
        "ne_xyz": ne_xyz,
        "ndtw": nd,
        "heading_yaw_deg": angle_diff_deg(ge["yaw"], pe["yaw"]),
        "heading_pitch_deg": angle_diff_deg(ge["pitch"], pe["pitch"]),
    }
    for tau in TAUS:
        success = ne_xy <= tau
        oracle_ok = oracle_best <= tau
        out[f"sr@{tau}"] = 1.0 if success else 0.0
        out[f"osr@{tau}"] = 1.0 if oracle_ok else 0.0
        out[f"spl@{tau}"] = (tl_gt / max(tl_gt, tl_pred)) if (
            success and max(tl_gt, tl_pred) > 0) else 0.0
        out[f"sdtw@{tau}"] = calc_ndtw(pr_pts, gt_pts, tau=tau) if success else 0.0
        out[f"soft_sr@{tau}"] = math.exp(-ne_xy / max(tau, 1e-6))
    return out


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def eval_one(
    gt_obj: Optional[Dict],
    pred_obj: Optional[Dict],
) -> Tuple[Dict, Optional[Dict]]:
    """Compute every metric for one (gt, pred) pair.

    Returns ``(record, axis_counts)`` where ``axis_counts`` is the dict of
    per-axis confusion matrices (used by the aggregator) or ``None`` if
    the sample is unscoreable (parse failure or empty gt).
    """
    parse_ok = pred_obj is not None
    pred_t: List[str] = []
    pred_r: List[str] = []
    gt_t: List[str] = []
    gt_r: List[str] = []
    if gt_obj is not None:
        gt_t = [norm(x) for x in gt_obj.get("translation", [])]
        gt_r = [norm(x) for x in gt_obj.get("rotation", [])]
    if pred_obj is not None:
        pred_t = [norm(x) for x in pred_obj.get("translation", [])]
        pred_r = [norm(x) for x in pred_obj.get("rotation", [])]

    trans_rot_len_match = parse_ok and len(pred_t) == len(pred_r)
    vocab_ok = parse_ok and seq_all_valid(pred_t) and seq_all_valid(pred_r)

    rec: Dict = {
        "parse_ok": parse_ok,
        "trans_rot_len_match": trans_rot_len_match,
        "vocab_ok": vocab_ok,
        "len_gt": len(gt_t),
        "len_pred": len(pred_t),
    }
    if not parse_ok or not gt_t:
        return rec, None

    # Length-quality
    rec["len_match_gt"] = (len(pred_t) == len(gt_t))
    rec["len_abs_err"] = abs(len(pred_t) - len(gt_t))
    rec["len_ratio"] = (len(pred_t) / len(gt_t)) if len(gt_t) > 0 else 0.0

    # Token-level
    rec["em_trans"] = exact_match(gt_t, pred_t)
    rec["em_rot"] = exact_match(gt_r, pred_r)
    rec["em_overall"] = (rec["em_trans"] + rec["em_rot"]) / 2
    rec["keyset_f1_trans"] = keyset_f1(gt_t, pred_t)         # TM (translation)
    rec["keyset_f1_rot"] = keyset_f1(gt_r, pred_r)           # TM (rotation)
    rec["keyset_f1_overall"] = (
        rec["keyset_f1_trans"] + rec["keyset_f1_rot"]) / 2   # TM
    c_fb = axis_counts(gt_t, pred_t, proj_ws)
    c_lr = axis_counts(gt_t, pred_t, proj_ad)
    c_pitch = axis_counts(gt_r, pred_r, proj_ws)
    c_yaw = axis_counts(gt_r, pred_r, proj_ad)
    rec["axis_acc_trans_fb"] = axis_acc(c_fb)
    rec["axis_acc_trans_lr"] = axis_acc(c_lr)
    rec["axis_acc_rot_pitch"] = axis_acc(c_pitch)
    rec["axis_acc_rot_yaw"] = axis_acc(c_yaw)
    counts = {"trans_fb": c_fb, "trans_lr": c_lr,
              "rot_pitch": c_pitch, "rot_yaw": c_yaw}

    # Global direction (component of DirAcc)
    gt_fb = global_dir(gt_t, "W", "S")
    pr_fb = global_dir(pred_t, "W", "S")
    gt_lr = global_dir(gt_t, "D", "A")
    pr_lr = global_dir(pred_t, "D", "A")
    gt_yaw = global_dir(gt_r, "D", "A")
    pr_yaw = global_dir(pred_r, "D", "A")
    gt_pitch = global_dir(gt_r, "W", "S")
    pr_pitch = global_dir(pred_r, "W", "S")
    rec["dir_trans_fb_ok"] = gt_fb == pr_fb
    rec["dir_trans_lr_ok"] = gt_lr == pr_lr
    rec["dir_rot_yaw_ok"] = gt_yaw == pr_yaw
    rec["dir_rot_pitch_ok"] = gt_pitch == pr_pitch
    rec["dir_all_ok"] = all([
        rec["dir_trans_fb_ok"], rec["dir_trans_lr_ok"],
        rec["dir_rot_yaw_ok"], rec["dir_rot_pitch_ok"],
    ])

    # Trajectory-level
    rec.update(traj_metrics(gt_t, gt_r, pred_t, pred_r))
    return rec, counts


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _rate(xs):
    xs = list(xs)
    return sum(1 for x in xs if x) / len(xs) if xs else 0.0


def aggregate(records: List[Dict], axis_counts_list: List[Optional[Dict]]):
    """Macro-aggregate per-sample records into corpus-level summary stats."""
    token_keys = [
        "em_trans", "em_rot", "em_overall",
        "axis_acc_trans_fb", "axis_acc_trans_lr",
        "axis_acc_rot_pitch", "axis_acc_rot_yaw",
    ]
    token_stats = {k: _mean([r.get(k) for r in records]) for k in token_keys}

    keyset_keys = ["keyset_f1_trans", "keyset_f1_rot", "keyset_f1_overall"]
    merged = {
        "trans_fb": defaultdict(int), "trans_lr": defaultdict(int),
        "rot_pitch": defaultdict(int), "rot_yaw": defaultdict(int),
    }
    for c in axis_counts_list:
        if c is None:
            continue
        for axis, cnt in c.items():
            for k, v in cnt.items():
                merged[axis][k] += v
    WS = ("W", "S", "N")
    AD = ("A", "D", "N")
    mf1 = {k: _mean([r.get(k) for r in records]) for k in keyset_keys}
    mf1.update({
        "macro_f1_trans_fb": macro_f1(merged["trans_fb"], WS),
        "macro_f1_trans_lr": macro_f1(merged["trans_lr"], AD),
        "macro_f1_rot_pitch": macro_f1(merged["rot_pitch"], WS),
        "macro_f1_rot_yaw": macro_f1(merged["rot_yaw"], AD),
    })

    dir_keys = [
        "dir_trans_fb_ok", "dir_trans_lr_ok",
        "dir_rot_yaw_ok", "dir_rot_pitch_ok", "dir_all_ok",
    ]
    dir_rates = {
        k: _rate([r.get(k) for r in records if r.get(k) is not None])
        for k in dir_keys
    }
    valid = {
        "parse_rate": _rate([r["parse_ok"] for r in records]),
        "trans_rot_len_match_rate": _rate([r["trans_rot_len_match"] for r in records]),
        "vocab_ok_rate": _rate([r["vocab_ok"] for r in records]),
        "len_match_gt_rate": _rate([
            r.get("len_match_gt") for r in records if "len_match_gt" in r
        ]),
        "len_abs_err_mean": _mean([r.get("len_abs_err") for r in records]),
        "len_ratio_mean": _mean([r.get("len_ratio") for r in records]),
        "len_gt_mean": _mean([r.get("len_gt") for r in records]),
        "len_pred_mean": _mean([r.get("len_pred") for r in records]),
    }
    traj_keys = [
        "tl_gt", "tl_pred", "ne_xy", "ne_xyz", "ndtw",
        "heading_yaw_deg", "heading_pitch_deg",
    ]
    for tau in TAUS:
        traj_keys += [f"sr@{tau}", f"osr@{tau}", f"spl@{tau}",
                      f"sdtw@{tau}", f"soft_sr@{tau}"]
    traj_stats = {k: _mean([r.get(k) for r in records]) for k in traj_keys}

    return valid, token_stats, mf1, dir_rates, traj_stats


def aggregate_by_bucket(
    records: List[Dict], axis_counts_list: List[Optional[Dict]],
) -> Dict[str, Dict]:
    """Group by ``meta.move_type_bucket`` and aggregate each group separately."""
    by_bucket_recs: Dict[str, List[Dict]] = defaultdict(list)
    by_bucket_cnts: Dict[str, List[Optional[Dict]]] = defaultdict(list)
    for r, c in zip(records, axis_counts_list):
        b = r.get("bucket") or "unknown"
        by_bucket_recs[b].append(r)
        by_bucket_cnts[b].append(c)
    out: Dict[str, Dict] = {}
    for b in sorted(by_bucket_recs.keys()):
        recs = [r for r in by_bucket_recs[b] if "em_overall" in r]
        cnts = [
            c for r, c in zip(by_bucket_recs[b], by_bucket_cnts[b])
            if "em_overall" in r
        ]
        if not recs:
            continue
        v, t, m, d, tr = aggregate(recs, cnts)
        out[b] = {
            "n": len(recs),
            "n_total": len(by_bucket_recs[b]),
            "validity": v,
            "token_level": t,
            "macro_f1": m,
            "global_direction": d,
            "trajectory": tr,
        }
    return out


__all__ = [
    # constants
    "ALLOWED", "CONFLICT", "PAD", "TAUS", "TOKEN_RE",
    "DEFAULT_ROT_DEG", "DEFAULT_TRANS_UNIT",
    # parsing
    "load_json_str", "norm", "is_valid_token", "seq_all_valid", "pad_to",
    # token metrics
    "exact_match", "keyset_f1_step", "keyset_f1",
    "proj_ws", "proj_ad", "axis_counts", "axis_acc", "macro_f1",
    # global direction
    "global_dir",
    # trajectory metrics
    "reconstruct_traj", "path_len", "calc_ndtw",
    "angle_diff_deg", "traj_metrics",
    # entry points
    "eval_one", "aggregate", "aggregate_by_bucket",
]
