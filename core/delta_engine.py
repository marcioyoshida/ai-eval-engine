"""Delta Engine — analyzes an S0 image against a contract target and produces:
  - gap_analysis   : what is wrong / missing in S0
  - sf_description : precise visual description of the required final state SF
  - tasks          : ordered action plan to reach SF from S0
  - sf_image_ref   : stub (future: Flux/SD-generated synthetic SF image)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_DELTA_SYSTEM_TEMPLATE = """\
You are a visual gap analysis and task planning engine.

OBJECT     : {target_object}
TARGET (SF): {required_state}
FAILURE SIGNALS TO LOOK FOR: {negative_indicators}

You are given the CURRENT STATE image (S0).

Produce a single JSON object with exactly these keys:

{{
  "gap_analysis"  : "<concise description of what is wrong, incomplete, or absent in S0>",
  "sf_description": "<precise visual description of the fully achieved target state SF — what it must look like when done>",
  "tasks": [
    {{"step": 1, "action": "<imperative verb phrase>", "detail": "<specific, measurable instructions including materials, tools and acceptance criteria>"}},
    {{"step": 2, "action": "...", "detail": "..."}},
    ...
  ]
}}

Rules:
- tasks must be ordered chronologically — first physical action first
- Be concrete: name materials, measurements, tools, and observable pass criteria
- sf_description must be detailed enough to use as a visual evaluation contract
- Output ONLY the raw JSON — no markdown fences, no prose outside the JSON\
"""


def _build_delta_prompt(target_object: str, required_state: str, negative_indicators: list[str]) -> str:
    return _DELTA_SYSTEM_TEMPLATE.format(
        target_object=target_object,
        required_state=required_state,
        negative_indicators=", ".join(negative_indicators) or "none specified",
    )


def _parse_delta_output(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning("Delta engine: no JSON in model output: %r", raw[:200])
        return _fallback()
    try:
        data = json.loads(raw[start:end])
        tasks = data.get("tasks", [])
        if isinstance(tasks, list):
            tasks = [
                {"step": t.get("step", i + 1), "action": str(t.get("action", "")), "detail": str(t.get("detail", ""))}
                for i, t in enumerate(tasks) if isinstance(t, dict)
            ]
        return {
            "gap_analysis": str(data.get("gap_analysis", "")),
            "sf_description": str(data.get("sf_description", "")),
            "tasks": tasks,
        }
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Delta engine parse error: %s", exc)
        return _fallback()


def _fallback() -> dict:
    return {
        "gap_analysis": "Analysis unavailable — model output could not be parsed.",
        "sf_description": "",
        "tasks": [],
    }


def analyze_s0(
    s0_path: str | Path,
    target_object: str,
    required_state: str,
    negative_indicators: list[str],
) -> dict:
    """Run S0 → SF gap analysis using the configured local inference backend.

    Returns dict with gap_analysis, sf_description, tasks.
    SF image generation is stubbed — would call Flux.1/SDXL in production.
    """
    from core.config import settings

    system_prompt = _build_delta_prompt(target_object, required_state, negative_indicators)

    if settings.inference_backend == "vllm":
        import asyncio
        from core.vllm_client import VLLMOrchestrator
        orchestrator = VLLMOrchestrator()
        raw = asyncio.get_event_loop().run_until_complete(
            orchestrator.call_raw(str(s0_path), system_prompt)
        )
    else:
        from core.inference import VisualContractOracle
        oracle = VisualContractOracle(settings.local_model_id)
        raw = oracle.call_raw(s0_path, system_prompt)

    result = _parse_delta_output(raw)
    logger.info(
        "Delta analysis complete: %d tasks generated for object=%r",
        len(result["tasks"]), target_object,
    )
    return result
