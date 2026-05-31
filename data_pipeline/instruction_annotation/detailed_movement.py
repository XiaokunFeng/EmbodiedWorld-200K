"""
Detailed Movement Instruction annotation.

For every navigation-coherent segment (NCS) produced by Steps 1+2 of
the pipeline (see ``trajectory_segmentation`` and
``action_quantization``), this module prompts a frontier vision-language
model to emit a **fine-grained natural-language description** of the
embodied-navigation task implicitly carried by the segment.

The annotation track corresponds to the "Detailed Movement Instruction"
described in Sec. 3.2 (Step 3) and App. A.3 of the paper. The full
prompt is reproduced at ``./prompts/detailed_movement.txt`` and is
loaded verbatim at run-time; the only template variable is
``{{MOVEMENT_DIRECTION}}`` (e.g. ``"left-forward"``), supplied per-NCS
from ``segment["direction_description"]``.

The output is a JSON record::

    {
        "annotatable":        bool,
        "perspective":        "First-person" | "Third-person" | "Other" | null,
        "task_description":   str | null,
        "move_target":        str | null,
        "move_type":          str | null,
        "direction_consist":  bool | null
    }

stored in ``segment["sub_traj_label"]``.

Implementation notes
--------------------
* We use ``vllm`` with tensor parallelism, default model
  ``Qwen3.5-27B``. ``apply_chat_template(..., enable_thinking=False)``
  disables the model's thinking mode for faster, deterministic JSON output.
* Per-segment frames are sampled uniformly at ``2 fps`` from the video
  clip via ``decord``. We additionally remap ``[start_frame, end_frame]``
  from pose-fps coordinates to video-fps coordinates when the two differ.
* All heavy ML dependencies (``torch``, ``transformers``, ``vllm``,
  ``decord``, ``qwen_vl_utils``) are imported lazily inside class
  methods so that the rest of the pipeline can be used in a CPU-only
  environment without installing them.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# NOTE: heavy ML deps (torch / transformers / vllm / decord) are imported
# lazily inside class methods so the rest of the pipeline can run on
# CPU-only hosts (e.g. for trajectory segmentation and evaluation).

DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "detailed_movement.txt"
DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-27B"   # HuggingFace Hub id; override via CLI
DEFAULT_SAMPLE_FPS = 2
DEFAULT_OUTPUT_FIELD = "sub_traj_label"


# ---------------------------------------------------------------------------
# vLLM input + output helpers
# ---------------------------------------------------------------------------

def _prepare_inputs_for_vllm(messages, processor, *, enable_thinking: bool = False):
    """Wrap a chat-style ``messages`` list into the dict expected by
    ``vllm.LLM.generate``.

    ``enable_thinking=False`` disables Qwen3.5's chain-of-thought, which
    yields tighter, more deterministic JSON output.
    """
    from qwen_vl_utils import process_vision_info  # lazy

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def parse_model_output(raw_text: str) -> Optional[Dict]:
    """Multi-fallback JSON parser for the VLM's free-form text output.

    Handles: stray ``</think>`` markers, ```` ```json ... ``` `` wrappers,
    Python-style ``True/False/None`` literals, and trailing prose; returns
    ``None`` if all attempts fail.
    """
    text = raw_text.split("\n</think>\n")[-1].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fixed = cleaned
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", fixed)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames_by_range(
    video_path: str,
    start_frame: int,
    end_frame: int,
    *,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
) -> tuple:
    """Extract uniformly-sampled frames in ``[start_frame, end_frame]``.

    Returns ``(frame_list, native_fps)`` where ``frame_list`` is a list
    of PIL images. Frame indices are clamped to the video's range.
    """
    from decord import VideoReader, cpu  # lazy

    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    total_frames = len(vr)

    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total_frames - 1)
    num_segment_frames = max(1, end_frame - start_frame + 1)
    segment_duration = num_segment_frames / max(native_fps, 1e-6)
    num_samples = max(1, int(segment_duration * sample_fps))

    indices = np.linspace(start_frame, end_frame, num_samples, dtype=int).tolist()
    frames_np = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f) for f in frames_np], native_fps


# ---------------------------------------------------------------------------
# Captioner
# ---------------------------------------------------------------------------

class DetailedMovementCaptioner:
    """vLLM-served VLM wrapper for Detailed Movement Instruction annotation.

    Args:
        model_path:       HuggingFace Hub id or local path to the VLM.
                          Default ``"Qwen/Qwen3.5-27B"``; any Qwen3-style
                          chat-template VLM that exposes ``apply_chat_template``
                          should work.
        prompt_path:      Path to the prompt template file. Defaults to
                          ``./prompts/detailed_movement.txt``.
        gpu_nums:         Tensor-parallel size for vLLM.
        gpu_memory_util:  vLLM ``gpu_memory_utilization`` knob.
        enable_thinking:  Whether to keep Qwen3.5's thinking mode on; the
                          paper used ``False`` (faster, deterministic).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        prompt_path: Optional[str] = None,
        *,
        gpu_nums: int = 1,
        gpu_memory_util: float = 0.85,
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
        )

        # Sampling params for Qwen3.5's non-thinking mode (cf. Qwen docs).
        self.sampling_params = SamplingParams(
            temperature=0.7, top_p=0.8, top_k=20,
            max_tokens=1024, presence_penalty=1.5,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        prompt_path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
        with prompt_path.open("r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        self.prompt_path = str(prompt_path)

    # -----------------------------------------------------------------------

    def annotate(
        self, frame_list: List[Image.Image], direction_description: str,
    ) -> Dict:
        """Run inference for one NCS.

        Args:
            frame_list:           Pre-extracted frames sampled at ``sample_fps``.
            direction_description: e.g. ``"left-forward"``; substitutes
                                  ``{{MOVEMENT_DIRECTION}}`` in the prompt.

        Returns:
            The parsed JSON record described in the module docstring, or
            a fallback ``{"raw_response": ..., "parse_success": False}``
            on parse failure.
        """
        prompt_text = self.prompt_template.replace(
            "{{MOVEMENT_DIRECTION}}", direction_description,
        )
        messages = [
            {"role": "user", "content": [
                {
                    "type": "video",
                    "video": frame_list,
                    "total_pixels": 20480 * 16 * 16,
                    "min_pixels": 64 * 32 * 32,
                    "max_frames": 2048,
                    "min_frames": 30,
                },
                {"type": "text", "text": prompt_text},
            ]},
        ]
        inputs = _prepare_inputs_for_vllm(
            messages, self.processor, enable_thinking=self.enable_thinking,
        )
        outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
        output_text = outputs[0].outputs[0].text

        parsed = parse_model_output(output_text)
        if parsed is not None:
            return parsed
        return {"raw_response": output_text, "parse_success": False}


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------

def annotate_segments_for_sample(
    sample_data: Dict,
    captioner: DetailedMovementCaptioner,
    *,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    output_field: str = DEFAULT_OUTPUT_FIELD,
    skip_existing: bool = True,
) -> Dict:
    """Annotate every segment in a sample dict in-place.

    Expected ``sample_data`` schema (keys read by this function)::

        {
            "video_path":   "/abs/path/to/video.mp4",
            "segments": {
                "total_pose_frames": int,        # optional, used for fps remap
                "segment_list": [
                    {
                        "start_frame": int,      # pose-fps index
                        "end_frame":   int,
                        "direction_description": "left-forward",
                        ...
                    },
                    ...
                ]
            }
        }

    For every segment, a new key ``output_field`` is added with the
    parsed annotation record. When ``skip_existing=True`` segments that
    already carry that key are left untouched.
    """
    from decord import VideoReader, cpu  # lazy

    video_path = sample_data.get("video_path", "")
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"video_path missing or not found: {video_path}")

    seg_info = sample_data.get("segments", {})
    seg_list = seg_info.get("segment_list", [])
    if not seg_list:
        return sample_data

    # Frame-index remap (pose-fps -> video-fps) when needed
    total_pose_frames = int(seg_info.get("total_pose_frames", 0) or 0)
    vr = VideoReader(video_path, ctx=cpu(0))
    total_video_frames = len(vr)
    del vr
    if total_pose_frames > 0 and total_video_frames > 0:
        ratio = total_video_frames / total_pose_frames
    else:
        ratio = 1.0

    for seg in seg_list:
        if skip_existing and output_field in seg:
            continue
        try:
            pose_s = int(seg["start_frame"])
            pose_e = int(seg["end_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        v_start = max(0, min(int(round(pose_s * ratio)), total_video_frames - 1))
        v_end = max(v_start, min(int(round(pose_e * ratio)), total_video_frames - 1))
        direction = seg.get("direction_description", "unknown")
        try:
            frames, _ = extract_frames_by_range(
                video_path, v_start, v_end, sample_fps=sample_fps,
            )
            seg[output_field] = captioner.annotate(frames, direction)
        except Exception as exc:  # noqa: BLE001 — keep going on per-seg errors
            seg[output_field] = {"raw_response": str(exc), "parse_success": False}
    return sample_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Run Detailed Movement Instruction annotation over a directory of "
            "sample JSONs. Each input JSON must have a `video_path` and a "
            "`segments.segment_list` populated by Steps 1+2 of the pipeline."
        ),
    )
    p.add_argument("--input_dir", required=True,
                   help="Directory of sample JSONs (each with video_path + segments).")
    p.add_argument("--output_dir", required=True,
                   help="Output directory; mirrors filenames from input_dir.")
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--prompt_path", default=str(DEFAULT_PROMPT_PATH))
    p.add_argument("--gpu_nums", type=int, default=1)
    p.add_argument("--sample_fps", type=int, default=DEFAULT_SAMPLE_FPS)
    p.add_argument("--chunk_id", type=int, default=-1,
                   help="Chunk index for data-parallel sharding (-1 = no sharding).")
    p.add_argument("--chunk_nums", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    captioner = DetailedMovementCaptioner(
        model_path=args.model_path,
        prompt_path=args.prompt_path,
        gpu_nums=args.gpu_nums,
    )

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".json"))
    if args.chunk_id >= 0:
        per = len(files) // max(1, args.chunk_nums)
        s = args.chunk_id * per
        e = len(files) if args.chunk_id == args.chunk_nums - 1 else (args.chunk_id + 1) * per
        files = files[s:e]
        print(f"[chunk] {args.chunk_id}/{args.chunk_nums}: {len(files)} files")

    for fname in files:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        with open(in_path, "r", encoding="utf-8") as f:
            sample = json.load(f)
        try:
            annotate_segments_for_sample(sample, captioner, sample_fps=args.sample_fps)
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {fname}: {exc}")
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        print(f"  [ok] {fname}")


if __name__ == "__main__":
    _cli()
