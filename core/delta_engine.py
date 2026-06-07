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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.inference import VisualContractOracle

logger = logging.getLogger(__name__)

# Module-level singleton — avoids reloading the 7B model on every delta request.
_oracle: "VisualContractOracle | None" = None


def _get_oracle() -> "VisualContractOracle":
    global _oracle
    if _oracle is None:
        from core.config import settings
        from core.inference import VisualContractOracle
        _oracle = VisualContractOracle(settings.local_model_id)
    return _oracle

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


async def _generate_sf_image(
    sf_description: str,
    delta_id: str,
    images_dir: Path,
) -> str | None:
    """Generate synthetic SF image from sf_description via local Flux.1.

    Returns the filename (relative to images_dir) or None on failure.
    """
    import asyncio

    try:
        from core.image_gen import _get_generator
        from core.config import settings

        gen = _get_generator()
        filename = f"sf_{delta_id}.png"
        output_path = images_dir / filename

        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            None,
            lambda: gen.generate(
                prompt=sf_description,
                width=settings.flux_image_width,
                height=settings.flux_image_height,
            ),
        )
        image.save(output_path)
        logger.info("SF image saved to %s", output_path)
        return filename
    except Exception as exc:
        logger.error("SF image generation failed: %s", exc, exc_info=True)
        return None


async def analyze_s0(
    s0_path: str | Path,
    target_object: str,
    required_state: str,
    negative_indicators: list[str],
    delta_id: str | None = None,
    images_dir: Path | None = None,
) -> dict:
    """Run S0 → SF gap analysis using the configured inference backend.

    Returns dict with gap_analysis, sf_description, tasks, and sf_image_ref
    (filename of the generated SF image, or None if generation is disabled/failed).
    """
    import asyncio
    from core.config import settings

    # Always use an absolute path so qwen_vl_utils can locate the file regardless of cwd.
    abs_path = Path(s0_path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"S0 image not found: {abs_path}")

    system_prompt = _build_delta_prompt(target_object, required_state, negative_indicators)

    if settings.inference_backend == "vllm":
        from core.vllm_client import VLLMOrchestrator
        orchestrator = VLLMOrchestrator()
        raw = await orchestrator.call_raw(str(abs_path), system_prompt)
    else:
        oracle = _get_oracle()
        # call_raw is synchronous/CPU-bound; run in executor to keep the event loop free.
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, oracle.call_raw, str(abs_path), system_prompt)

    logger.debug("Delta raw output (%d chars): %r", len(raw), raw[:400])
    result = _parse_delta_output(raw)
    logger.info(
        "Delta analysis complete: %d tasks generated for object=%r",
        len(result["tasks"]), target_object,
    )

    result["sf_image_ref"] = None
    if settings.generate_sf_image and delta_id and result.get("sf_description"):
        _images_dir = (images_dir or Path("data/images")).resolve()
        _images_dir.mkdir(parents=True, exist_ok=True)
        result["sf_image_ref"] = await _generate_sf_image(
            result["sf_description"], delta_id, _images_dir
        )

    return result
