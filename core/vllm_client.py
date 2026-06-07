"""Production vLLM async client — Phase 3.

Connects to a running vLLM OpenAI-compatible server and dispatches
contract assertions using dynamic multi-LoRA routing.
"""

from __future__ import annotations

import json
import logging

import openai

from core.config import settings
from core.inference import _build_system_prompt, _parse_result
from core.schemas import ContractParams, EvaluationResult, EvaluationThinking

logger = logging.getLogger(__name__)


class VLLMOrchestrator:
    """Async client for a vLLM multi-LoRA server.

    Pass `lora_id` to route through a specific fine-tuned adapter;
    omit it to use the base model (zero-shot unified parameter mode).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
    ):
        self._base_url = base_url or settings.vllm_base_url
        self._api_key = api_key or settings.vllm_api_key
        self._default_model = default_model or settings.vllm_default_model
        self._client = openai.AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )

    async def run_evaluation(
        self,
        image_source: str,
        params: ContractParams,
        lora_id: str | None = None,
    ) -> EvaluationResult:
        """Send an assertion request to the vLLM server.

        If `lora_id` is provided, vLLM dynamically maps the adapter weights
        for that request (S-LoRA / Punica kernels handle batching transparently).
        """
        model = lora_id or self._default_model
        system_prompt = _build_system_prompt(params)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_source},
                    },
                    {"type": "text", "text": "Run compliance assertion."},
                ],
            },
        ]

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,  # deterministic for contract evaluation
                # enable_thinking surfaces native <think> tokens on reasoning models
                # (QwQ, Qwen3-thinking); _parse_result strips them before JSON parsing
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            raw = response.choices[0].message.content or ""
            logger.debug("vLLM raw response [model=%s]: %s", model, raw[:500])
            return _parse_result(raw)
        except openai.APIConnectionError as exc:
            logger.error("vLLM server unreachable at %s: %s", self._base_url, exc)
            return EvaluationResult(
                passed=False,
                confidence=0.0,
                rationale=f"Inference backend unavailable: {exc}",
            )
        except openai.APIStatusError as exc:
            logger.error("vLLM API error (status %s): %s", exc.status_code, exc.message)
            return EvaluationResult(
                passed=False,
                confidence=0.0,
                rationale=f"Inference error {exc.status_code}: {exc.message}",
            )

    async def call_raw(self, image_source: str, system_prompt: str) -> str:
        """Call the model with a fully custom system prompt. Returns raw model text output."""
        try:
            response = await self._client.chat.completions.create(
                model=self._default_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_source}},
                            {"type": "text", "text": "Analyze this image."},
                        ],
                    },
                ],
                max_tokens=1024,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("vLLM call_raw error: %s", exc)
            return ""

    async def run_juror_panel(
        self,
        image_source: str,
        params: ContractParams,
        lora_id: str | None = None,
    ) -> EvaluationResult:
        """Three-juror consensus evaluation (positive + adversarial + base).

        Juror 1: domain LoRA (positive assertion)
        Juror 2: adversarial LoRA (negative assertion — looks for failure signals)
        Juror 3: base model zero-shot (unbiased tiebreaker)
        """
        import asyncio

        adversarial_params = ContractParams(
            target_object=params.target_object,
            required_state=f"NOT ({params.required_state}) — signs of failure or incomplete work",
            negative_indicators=params.negative_indicators,
            strictness_coefficient=params.strictness_coefficient,
        )

        juror1, juror2, juror3 = await asyncio.gather(
            self.run_evaluation(image_source, params, lora_id=lora_id),
            self.run_evaluation(image_source, adversarial_params, lora_id=lora_id),
            self.run_evaluation(image_source, params, lora_id=None),
        )

        # Weighted consensus: positive 0.5 / base 0.3 / adversarial (inverted) 0.2
        positive_score = juror1.confidence if juror1.passed else 1.0 - juror1.confidence
        adversarial_score = juror2.confidence if not juror2.passed else 1.0 - juror2.confidence
        base_score = juror3.confidence if juror3.passed else 1.0 - juror3.confidence

        consensus_score = 0.5 * positive_score + 0.3 * base_score + 0.2 * adversarial_score
        passed = consensus_score >= params.strictness_coefficient

        rationale = (
            f"Juror panel: specialist={juror1.confidence:.2f}({'PASS' if juror1.passed else 'FAIL'}), "
            f"adversarial={juror2.confidence:.2f}({'FAIL' if juror2.passed else 'PASS'}), "
            f"base={juror3.confidence:.2f}({'PASS' if juror3.passed else 'FAIL'}). "
            f"Consensus={consensus_score:.2f}. "
            f"Primary rationale: {juror1.rationale}"
        )

        # Merge thinking blocks from all three jurors into a single composite trace
        merged_thinking: EvaluationThinking | None = None
        if any(j.thinking for j in (juror1, juror2, juror3)):
            def _safe(j, field):
                return getattr(j.thinking, field, []) if j.thinking else []

            merged_thinking = EvaluationThinking(
                observations=_safe(juror1, "observations"),
                positive_evidence=_safe(juror1, "positive_evidence"),
                negative_evidence=_safe(juror2, "positive_evidence"),  # adversarial PASS = failure evidence
                reasoning=(
                    f"[Specialist] {getattr(juror1.thinking, 'reasoning', '') if juror1.thinking else juror1.rationale} | "
                    f"[Adversarial] {getattr(juror2.thinking, 'reasoning', '') if juror2.thinking else juror2.rationale} | "
                    f"[Base] {getattr(juror3.thinking, 'reasoning', '') if juror3.thinking else juror3.rationale}"
                ),
            )

        return EvaluationResult(
            passed=passed,
            confidence=round(consensus_score, 4),
            rationale=rationale,
            thinking=merged_thinking,
        )
