"""Local HuggingFace inference backend (Phase 1 — single GPU sandbox)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.schemas import ContractParams, EvaluationResult

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = (
    "You are a strict, deterministic visual contract arbitrator. "
    "Your task is to inspect the provided image and verify whether the target object "
    "'{target_object}' satisfies the expected condition: '{required_state}'. "
    "Also scan explicitly for these failure indicators: {negative_indicators}. "
    "Reply ONLY with a valid JSON object containing exactly three keys: "
    '"passed" (boolean), "confidence" (float 0.0-1.0), "rationale" (string). '
    "No markdown, no explanation outside the JSON."
)


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

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        raw = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return _parse_result(raw)


def _parse_result(raw: str) -> EvaluationResult:
    """Extract the JSON payload from model output, tolerating surrounding text."""
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
        return EvaluationResult(
            passed=bool(data["passed"]),
            confidence=float(data["confidence"]),
            rationale=str(data["rationale"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Failed to parse model JSON: %s — raw: %r", exc, raw[:200])
        return EvaluationResult(
            passed=False,
            confidence=0.0,
            rationale=f"Parse failure ({exc}): {raw[:300]}",
        )
