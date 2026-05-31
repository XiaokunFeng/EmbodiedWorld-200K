"""
High-Level Goal-Navigation Caption.

Implements the "High-Level Goal-Navigation Caption" track of Step 3 in
the EmbodiedWorld-200K annotation pipeline (Sec. 3.2 / App. A.3 of the
paper). Compared with the always-produced *Detailed Movement
Instruction*, this track yields a **direction-free** caption that only
specifies the navigation destination, e.g.::

    "Head to the line of trees."
    "Navigate to the wooden gate structure."

Because such a caption requires the navigation target to be **uniquely
identifiable** within the segment, we wrap caption generation in a
4-step grounding-and-verification loop:

  1. **Strip directional words** from the ``move_target`` field of the
     Detailed Movement Instruction to obtain a candidate target phrase.
  2. **Ground on the initial frame** with an open-vocabulary detector
     (Qwen3.5-27B with grounding prompt). Keep only single-bbox results.
  3. **Ground on the final frame** of the NCS, again keeping only
     single-bbox results.
  4. **LLM-based same-entity verification** between the two boxes.

Only NCSs that pass all four checks are admitted to this track and
prompted with ``goal_caption.txt`` to produce the short direction-free
caption.

The default VLM is the same ``Qwen3.5-27B`` used elsewhere in the
pipeline. All three prompts live in ``./prompts/``::

  goal_grounding.txt        — open-vocabulary detection on a single frame
  same_target_verify.txt    — same-entity check on two annotated frames
  goal_caption.txt          — final caption generation
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw

# Lazy ML deps; the heavy imports (torch, transformers, vllm, decord) are
# inside the class methods.

from .detailed_movement import (
    _prepare_inputs_for_vllm,
    parse_model_output,
)

DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-27B"
PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_GROUNDING_PROMPT = PROMPT_DIR / "goal_grounding.txt"
DEFAULT_VERIFY_PROMPT = PROMPT_DIR / "same_target_verify.txt"
DEFAULT_CAPTION_PROMPT = PROMPT_DIR / "goal_caption.txt"

DEFAULT_OUTPUT_KEY = "l2_label"


# ---------------------------------------------------------------------------
# Direction-word stripping
# ---------------------------------------------------------------------------

# Patterns to strip from ``move_target``; ordered with the longest phrases
# first so they win over shorter overlapping patterns. Compiled
# case-insensitively.
_DIRECTION_PATTERNS = [
    r"\b(?:directly\s+)?(?:ahead\s+and\s+to\s+the\s+(?:left|right))\b",
    r"\b(?:to\s+the\s+)?(?:front[- ]?(?:left|right))\b",
    r"\b(?:to\s+the\s+)?(?:rear[- ]?(?:left|right))\b",
    r"\b(?:to\s+the\s+)?(?:back[- ]?(?:left|right))\b",
    r"\bahead\s+(?:on\s+the\s+)?(?:left|right)\b",
    r"\b(?:on\s+the\s+)?(?:far\s+)?(?:left|right)(?:\s+side)?\b",
    r"\b(?:to\s+the\s+)?(?:left|right)(?:\s+side)?\b",
    r"\b(?:in\s+)?(?:the\s+)?(?:forward|front)\s+(?:direction|area)\b",
    r"\bdirectly\s+(?:ahead|forward|in\s+front)\b",
    r"\bstraight\s+ahead\b",
    r"\bahead\b",
    r"\bforward\b",
    r"\bbackward\b",
    r"\bin\s+front(?:\s+of)?\b",
    r"\bbehind\b",
    r"\b(?:to\s+the\s+)?(?:upper|lower)\s*[-]?\s*(?:left|right)\b",
    r"\b(?:on|to)\s+the\s+(?:left|right)\b",
    r"\b(?:left|right)\s*[-]?\s*(?:hand\s+)?side\b",
]
_DIRECTION_REGEXES = [re.compile(p, re.IGNORECASE) for p in _DIRECTION_PATTERNS]


def strip_direction_from_target(move_target: str) -> str:
    """Remove direction modifiers from a ``move_target`` phrase.

    Example::

        "the entrance of the large building ahead and to the left"
        -> "the entrance of the large building"

    Returns the original string if all words would be stripped.
    """
    if not move_target:
        return move_target
    out = move_target
    for rx in _DIRECTION_REGEXES:
        out = rx.sub("", out)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"[,\s]+$", "", out)
    out = re.sub(r"\s+(?:at|in|on|to|from|near|of)\s*$", "", out).strip()
    return out or move_target


# ---------------------------------------------------------------------------
# Frame extraction (single frame)
# ---------------------------------------------------------------------------

def extract_single_frame(video_path: str, frame_idx: int) -> Tuple[Image.Image, float, int]:
    """Read one frame from a video as a PIL Image.

    Returns ``(image, native_fps, total_frames)``. ``frame_idx`` is clamped
    to ``[0, total_frames - 1]``.
    """
    from decord import VideoReader, cpu  # lazy

    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    total = len(vr)
    frame_idx = max(0, min(frame_idx, total - 1))
    arr = vr[frame_idx].asnumpy()
    return Image.fromarray(arr), float(native_fps), int(total)


# ---------------------------------------------------------------------------
# bbox utils
# ---------------------------------------------------------------------------

def normalize_bbox(
    bbox, img_width: int, img_height: int,
) -> Optional[List[int]]:
    """Convert a Qwen3-VL grounding bbox into pixel coordinates.

    Qwen3-VL/Qwen3.5 outputs bbox coordinates as floats in ``[0, 1000]``
    (per-mille relative). This helper handles:

      * the normal per-mille case (rescale to pixels);
      * already-pixel coordinates (any side > 1000);
      * nested ``[[x1,y1,x2,y2]]`` lists, string-typed coords, etc.

    Returns ``None`` when the input cannot be parsed into 4 floats.
    """
    if isinstance(bbox, (list, tuple)) and len(bbox) == 1 and isinstance(bbox[0], (list, tuple)):
        bbox = bbox[0]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        coords = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return None
    x1, y1, x2, y2 = coords
    if all(0 <= v <= 1000 for v in coords):
        x1 = x1 / 1000 * img_width
        y1 = y1 / 1000 * img_height
        x2 = x2 / 1000 * img_width
        y2 = y2 / 1000 * img_height
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def parse_grounding_result(
    raw_result, img_width: int, img_height: int,
) -> List[List[int]]:
    """Extract a list of pixel-space bboxes from a (possibly malformed)
    grounding model output. See ``normalize_bbox`` for input handling."""
    if raw_result is None:
        return []
    bboxes: List[List[int]] = []

    # Native Qwen3-VL grounding format: list of {"bbox_2d": [...], "label": ...}.
    if isinstance(raw_result, list):
        for item in raw_result:
            if isinstance(item, dict) and "bbox_2d" in item:
                bbox = normalize_bbox(item["bbox_2d"], img_width, img_height)
                if bbox is not None:
                    bboxes.append(bbox)
        return bboxes

    if isinstance(raw_result, dict) and raw_result.get("parse_success") is False:
        return []
    if isinstance(raw_result, dict) and "bbox_2d" in raw_result:
        bbox = normalize_bbox(raw_result["bbox_2d"], img_width, img_height)
        if bbox is not None:
            bboxes.append(bbox)
        return bboxes
    if isinstance(raw_result, dict) and "bboxes" in raw_result:
        if not raw_result.get("found", False):
            return []
        for bb in raw_result.get("bboxes", []):
            bbox = normalize_bbox(bb, img_width, img_height)
            if bbox is not None:
                bboxes.append(bbox)
        return bboxes
    return []


def draw_bbox_on_image(
    image: Image.Image, bbox: List[int], *, color: str = "red", width: int = 3,
) -> Image.Image:
    """Return a copy of ``image`` with a colored rectangle drawn around ``bbox``."""
    img = image.copy()
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return img
    try:
        rect = tuple(int(round(v)) for v in bbox)
    except (ValueError, TypeError):
        return img
    ImageDraw.Draw(img).rectangle(rect, outline=color, width=width)
    return img


# ---------------------------------------------------------------------------
# Captioner
# ---------------------------------------------------------------------------

class GoalNavigationCaptioner:
    """Three-in-one captioner for the Goal-Navigation track.

    Exposes three methods, one per VLM call:

      * ``grounding(image, target)``         - locate the target in one frame.
      * ``verify_same_target(img1, img2)``   - same-entity check on two
                                                images carrying red bboxes.
      * ``generate_l2_caption(image, target)`` - emit the final
                                                  direction-free caption.

    Only one ``vllm.LLM`` instance is created; sampling parameters are
    swapped per-call (low temperature for grounding, default for the
    other two).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        grounding_prompt_path: Optional[Union[str, Path]] = None,
        verify_prompt_path: Optional[Union[str, Path]] = None,
        caption_prompt_path: Optional[Union[str, Path]] = None,
        gpu_nums: int = 1,
        gpu_memory_util: float = 0.85,
        max_model_len: int = 32768,
        enable_thinking: bool = False,
    ) -> None:
        from transformers import AutoProcessor   # lazy
        from vllm import LLM, SamplingParams     # lazy

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self.model_path = model_path
        self.enable_thinking = enable_thinking

        seed = random.randint(0, 10_000)
        self.model = LLM(
            model=self.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_util,
            enforce_eager=False,
            tensor_parallel_size=gpu_nums,
            seed=seed,
            max_model_len=max_model_len,
        )

        # Default sampling params (verification + caption).
        self.sampling_params = SamplingParams(
            temperature=0.7, top_p=0.8, top_k=20,
            max_tokens=1024, presence_penalty=1.5,
        )
        # Lower temperature for grounding (precise coords).
        self.grounding_sampling_params = SamplingParams(
            temperature=0.1, top_p=0.9, top_k=20,
            max_tokens=512, presence_penalty=0.0,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # Load prompts
        gp = Path(grounding_prompt_path) if grounding_prompt_path else DEFAULT_GROUNDING_PROMPT
        vp = Path(verify_prompt_path) if verify_prompt_path else DEFAULT_VERIFY_PROMPT
        cp = Path(caption_prompt_path) if caption_prompt_path else DEFAULT_CAPTION_PROMPT
        self._grounding_template = gp.read_text(encoding="utf-8")
        self._verify_template = vp.read_text(encoding="utf-8")
        self._caption_template = cp.read_text(encoding="utf-8")
        self.prompt_paths = {"grounding": str(gp), "verify": str(vp), "caption": str(cp)}

    # ---- low-level vLLM helpers ----

    def _run_single_image(self, image: Image.Image, prompt_text: str, *, sampling_params=None):
        sp = sampling_params or self.sampling_params
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        inputs = _prepare_inputs_for_vllm(messages, self.processor, enable_thinking=self.enable_thinking)
        outputs = self.model.generate(inputs, sampling_params=sp)
        return parse_model_output(outputs[0].outputs[0].text) or {
            "raw_response": outputs[0].outputs[0].text, "parse_success": False,
        }

    def _run_two_image(self, img1: Image.Image, img2: Image.Image, prompt_text: str, *, sampling_params=None):
        sp = sampling_params or self.sampling_params
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Image 1 (first frame):"},
            {"type": "image", "image": img1},
            {"type": "text", "text": "Image 2 (last frame):"},
            {"type": "image", "image": img2},
            {"type": "text", "text": prompt_text},
        ]}]
        inputs = _prepare_inputs_for_vllm(messages, self.processor, enable_thinking=self.enable_thinking)
        outputs = self.model.generate(inputs, sampling_params=sp)
        return parse_model_output(outputs[0].outputs[0].text) or {
            "raw_response": outputs[0].outputs[0].text, "parse_success": False,
        }

    # ---- public API ----

    def grounding(self, image: Image.Image, target_description: str):
        """Run open-vocabulary grounding on a single frame and return the
        raw model output (parsed JSON list/dict). Use ``parse_grounding_result``
        to extract pixel-space bboxes."""
        prompt = self._grounding_template.format(target_description=target_description)
        return self._run_single_image(image, prompt, sampling_params=self.grounding_sampling_params)

    def verify_same_target(self, img1_with_bbox: Image.Image, img2_with_bbox: Image.Image) -> Dict:
        """Same-entity check on two images, each carrying a red bbox."""
        return self._run_two_image(img1_with_bbox, img2_with_bbox, self._verify_template)

    def generate_l2_caption(self, first_frame: Image.Image, target_description: str) -> Dict:
        """Generate the direction-free Goal-Navigation caption."""
        prompt = self._caption_template.format(target_description=target_description)
        return self._run_single_image(first_frame, prompt)


# ---------------------------------------------------------------------------
# Per-segment driver
# ---------------------------------------------------------------------------

def annotate_one_segment(
    seg: Dict,
    seg_idx: int,
    video_path: str,
    total_pose_frames: int,
    captioner: GoalNavigationCaptioner,
) -> Optional[Dict]:
    """Run the 4-step grounding-and-verification loop for one segment.

    Returns:
        * Caption record on success, e.g.::

              {
                  "status": "ok",
                  "l2_caption": "...",
                  "target_visual_description": "...",
                  "clean_target": "...",
                  "first_bbox": [x1, y1, x2, y2],
                  "last_bbox":  [x1, y1, x2, y2],
                  "verify": {...},
              }

        * Skip record on filtering, e.g.
          ``{"status": "skip", "reason": "target_not_in_first_frame", ...}``;
        * ``None`` if the segment is unannotatable (no L1 ``move_target``).
    """
    sub = seg.get("sub_traj_label", {}) or {}
    if not sub.get("annotatable", False):
        return None
    move_target = sub.get("move_target")
    if not move_target or str(move_target).lower() == "null":
        return None

    clean_target = strip_direction_from_target(move_target)

    # --- Frame index remap (pose -> video) ---
    from decord import VideoReader, cpu  # lazy
    pose_s = int(seg["start_frame"])
    pose_e = int(seg["end_frame"])
    vr = VideoReader(video_path, ctx=cpu(0))
    total_video = len(vr)
    del vr
    ratio = total_video / total_pose_frames if total_pose_frames > 0 else 1.0
    v_s = max(0, min(int(round(pose_s * ratio)), total_video - 1))
    v_e = max(v_s, min(int(round(pose_e * ratio)), total_video - 1))

    first_frame, _, _ = extract_single_frame(video_path, v_s)
    last_frame, _, _ = extract_single_frame(video_path, v_e)

    # Step D: ground on first frame
    raw_first = captioner.grounding(first_frame, clean_target)
    first_bboxes = parse_grounding_result(raw_first, first_frame.width, first_frame.height)
    if len(first_bboxes) == 0:
        return {"status": "skip", "reason": "target_not_in_first_frame",
                "grounding_first_raw": raw_first, "clean_target": clean_target}
    if len(first_bboxes) > 1:
        return {"status": "skip", "reason": "multiple_targets_in_first_frame",
                "grounding_first_raw": raw_first, "num_bboxes": len(first_bboxes),
                "clean_target": clean_target}
    first_bbox = first_bboxes[0]

    # Step E: ground on last frame
    raw_last = captioner.grounding(last_frame, clean_target)
    last_bboxes = parse_grounding_result(raw_last, last_frame.width, last_frame.height)
    if len(last_bboxes) == 0:
        return {"status": "skip", "reason": "target_not_in_last_frame",
                "grounding_last_raw": raw_last, "clean_target": clean_target,
                "first_bbox": first_bbox}
    if len(last_bboxes) > 1:
        return {"status": "skip", "reason": "multiple_targets_in_last_frame",
                "grounding_last_raw": raw_last, "num_bboxes": len(last_bboxes),
                "clean_target": clean_target, "first_bbox": first_bbox}
    last_bbox = last_bboxes[0]

    # Step F: same-entity verification
    img1 = draw_bbox_on_image(first_frame, first_bbox, color="red", width=3)
    img2 = draw_bbox_on_image(last_frame, last_bbox, color="red", width=3)
    verify = captioner.verify_same_target(img1, img2)
    if not isinstance(verify, dict) or not verify.get("is_same_target"):
        return {"status": "skip", "reason": "different_target_in_two_frames",
                "verify": verify, "clean_target": clean_target,
                "first_bbox": first_bbox, "last_bbox": last_bbox}

    # Step G: generate the final caption
    cap = captioner.generate_l2_caption(first_frame, clean_target)
    if not isinstance(cap, dict) or "l2_caption" not in cap:
        return {"status": "skip", "reason": "caption_parse_failed",
                "caption_raw": cap, "clean_target": clean_target,
                "first_bbox": first_bbox, "last_bbox": last_bbox,
                "verify": verify}
    return {
        "status": "ok",
        "l2_caption": cap["l2_caption"],
        "target_visual_description": cap.get("target_visual_description"),
        "clean_target": clean_target,
        "first_bbox": first_bbox,
        "last_bbox": last_bbox,
        "verify": verify,
    }


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Run High-Level Goal-Navigation caption annotation over a "
            "directory of sample JSONs that already carry Detailed Movement "
            "Instructions in segments[*].sub_traj_label."
        ),
    )
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--gpu_nums", type=int, default=1)
    p.add_argument("--output_key", default=DEFAULT_OUTPUT_KEY)
    p.add_argument("--chunk_id", type=int, default=-1)
    p.add_argument("--chunk_nums", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    captioner = GoalNavigationCaptioner(model_path=args.model_path, gpu_nums=args.gpu_nums)

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".json"))
    if args.chunk_id >= 0:
        per = len(files) // max(1, args.chunk_nums)
        s = args.chunk_id * per
        e = len(files) if args.chunk_id == args.chunk_nums - 1 else (args.chunk_id + 1) * per
        files = files[s:e]

    for fname in files:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        video_path = data.get("video_path", "")
        seg_info = data.get("segments", {})
        seg_list = seg_info.get("segment_list", [])
        total_pose_frames = int(seg_info.get("total_pose_frames", 0) or 0)
        if not video_path or not os.path.exists(video_path) or not seg_list:
            continue
        for idx, seg in enumerate(seg_list):
            if args.output_key in seg:
                continue
            try:
                res = annotate_one_segment(
                    seg, idx, video_path, total_pose_frames, captioner,
                )
            except Exception as exc:  # noqa: BLE001
                seg[args.output_key] = {"status": "error", "reason": str(exc)}
                continue
            if res is not None:
                seg[args.output_key] = res
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [ok] {fname}")


if __name__ == "__main__":
    _cli()
