"""
Instruction annotation modules.

Implements **Step 3** of the EmbodiedWorld-200K annotation pipeline
(Sec. 3.2 / App. A.3). Each NCS produced by Steps 1+2 receives a
natural-language task instruction from a frontier VLM:

  * ``detailed_movement``        — Detailed Movement Instruction (always-produced track).
  * ``direction_consistency``    — Post-hoc consistency check between the
                                    instruction's free-text move-type and
                                    the underlying action sequence.
  * ``goal_navigation``          — High-Level Goal-Navigation Caption
                                    (target-grounding + same-entity verification
                                    + caption generation).

All three modules use ``vllm`` for inference and ``Qwen3.5-27B`` (with
``enable_thinking=False``) as the default VLM in the released
EmbodiedWorld-200K. The corresponding prompt templates live in
``./prompts/``.
"""

__all__ = []
