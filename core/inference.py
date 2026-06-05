"""Local HuggingFace inference backend (Phase 1 — single GPU sandbox)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.schemas import ContractParams, EvaluationResult, EvaluationThinking

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = """\
You are a strict, deterministic visual contract arbitrator.

CONTRACT
  Target object  : {target_object}
  Required state : {required_state}
  Failure signals: {negative_indicators}

INSTRUCTIONS
Examine the image carefully and produce a single JSON object with this exact structure:

{{
  "thinking": {{
    "observations":      [ "<neutral factual observations about the image>" ],
    "positive_evidence": [ "<specific visual details supporting PASS>" ],
    "negative_evidence": [ "<specific visual details supporting FAIL>" ],
    "reasoning": "<step-by-step chain of thought weighing the evidence above>"
  }},
  "passed":     <true | false>,
  "confidence": <float 0.0–1.0>,
  "rationale":  "<one concise sentence summarising the final verdict>"
}}

Rules:
- Output ONLY the raw JSON — no markdown fences, no prose before or after.
- Every array must have at least one element; write "none observed" if genuinely empty.
- Confidence must reflect genuine uncertainty: do not round to 0.0 or 1.0 unless certain.\
"""


def _build_system_prompt(params: ContractParams) -> str:
    return _SYSTEM_TEMPLATE.format(
        target_object=params.target_object,
        required_state=params.required_state,
        negative_indicators=", ".join(params.negative_indicators) or "none specified",
    )


class VisualContractOracle:
    """Wraps Qwen2.5-VL for local single-GPU evaluation."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        # Lazy import so the module loads even without GPU/torch installed
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Local GPU backend requires the 'local-gpu' extras: "
                "poetry install --extras local-gpu"
            ) from e

        import torch

        self._process_vision_info = process_vision_info
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        logger.info("VisualContractOracle loaded model %s", model_id)

    def evaluate_evidence(
        self, image_source: str | Path, params: ContractParams
    ) -> EvaluationResult:
        """Run a single contract assertion against an image file path or URL."""
        system_prompt = _build_system_prompt(params)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_source)},
                    {"type": "text", "text": "Run compliance assertion."},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        # Slice off the input tokens — generate() returns the full sequence by default
        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return _parse_result(raw)


def _parse_result(raw: str) -> EvaluationResult:
    """Extract the JSON payload from model output, tolerating surrounding text."""
    # Strip native <think>…</think> tokens emitted by reasoning models (QwQ, Qwen3-thinking)
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning("Model output had no JSON object: %r", raw[:200])
        return EvaluationResult(
            passed=False,
            confidence=0.0,
            rationale=f"Parse failure — raw output: {raw[:300]}",
        )
    try:
        data = json.loads(raw[start:end])
        thinking: EvaluationThinking | None = None
        if "thinking" in data and isinstance(data["thinking"], dict):
            t = data["thinking"]
            thinking = EvaluationThinking(
                observations=_coerce_list(t.get("observations")),
                positive_evidence=_coerce_list(t.get("positive_evidence")),
                negative_evidence=_coerce_list(t.get("negative_evidence")),
                reasoning=str(t.get("reasoning", "")),
            )
        return EvaluationResult(
            passed=bool(data["passed"]),
            confidence=float(data["confidence"]),
            rationale=str(data["rationale"]),
            thinking=thinking,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to parse model JSON: %s — raw: %r", exc, raw[:200])
        return EvaluationResult(
            passed=False,
            confidence=0.0,
            rationale=f"Parse failure ({exc}): {raw[:300]}",
        )


def _coerce_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return ["none observed"]
