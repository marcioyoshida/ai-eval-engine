"""Synthetic data bootstrapping pipeline stub — Phase 4.

Full pipeline when implemented:
  Step 1  Contract Analysis  → LLM expands raw assertion into permutation matrix
  Step 2  Image Generation   → Flux.1 / SDXL + ControlNet in-painting
  Step 3  Auto-labelling     → Frontier VLM writes SFT conversation JSON
  Step 4  Quality Gate       → CLIP embedding filter drops low-similarity outputs
  Step 5  Dataset Export     → Hugging Face datasets JSONL for PEFT LoRA training
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# Minimum CLIP semantic similarity to keep a generated image
CLIP_THRESHOLD = 0.28

# Target dataset size per contract
TARGET_POSITIVE = 250
TARGET_NEGATIVE = 250


def expand_contract_to_prompt_matrix(contract_id: str) -> dict[str, list[str]]:
    """Query the DB for a contract and expand it into positive/negative prompt lists.

    Stub — in production this calls an LLM (e.g. Llama-3-70B) to generate
    rich, varied scene descriptions covering lighting conditions, angles,
    surface textures, and geographic/seasonal variation.
    """
    from db.models import ContractDefinition
    from db.session import AsyncSessionLocal
    from sqlalchemy import select

    async def _fetch():
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ContractDefinition).where(ContractDefinition.id == contract_id)
            )
            return result.scalar_one_or_none()

    contract = asyncio.get_event_loop().run_until_complete(_fetch())
    if not contract:
        raise ValueError(f"Contract {contract_id} not found")

    positive_prompts = _generate_positive_variants(
        contract.target_object, contract.required_state
    )
    negative_prompts = _generate_negative_variants(
        contract.target_object, contract.negative_indicators or []
    )

    return {"positive": positive_prompts, "negative": negative_prompts}


def _generate_positive_variants(target_object: str, required_state: str) -> list[str]:
    """Stub — would call an LLM to produce diverse success-condition prompts."""
    base = f"Photorealistic image of {target_object} in state: {required_state}"
    lighting_variants = ["natural daylight", "overcast diffuse light", "harsh overhead artificial light"]
    angle_variants = ["close-up macro", "standard eye-level angle", "wide-angle overview"]
    return [f"{base}, {l}, {a}" for l in lighting_variants for a in angle_variants]


def _generate_negative_variants(target_object: str, negative_indicators: list[str]) -> list[str]:
    """Stub — would call an LLM to produce diverse failure-condition prompts."""
    prompts = []
    for indicator in negative_indicators:
        prompts.append(
            f"Photorealistic image of {target_object} showing clear signs of: {indicator}, "
            "natural daylight, standard eye-level angle"
        )
    if not prompts:
        prompts.append(
            f"Photorealistic image of {target_object} in a failed, incomplete, or damaged state"
        )
    return prompts


def clip_quality_filter(image_path: str, prompt: str) -> bool:
    """Return True if the image is semantically aligned with the prompt.

    Stub — production implementation loads a CLIP model and computes
    cosine similarity between image and text embeddings.
    """
    logger.debug("CLIP filter stub: %s — would compute similarity vs %r", image_path, prompt[:80])
    return True  # stub passes everything


def build_sft_record(image_path: str, passed: bool, rationale: str, assertion: str) -> dict:
    """Format a single image+label pair as an SFT conversation record (JSONL line)."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Evaluate assertion: {assertion}"},
                    {"type": "image", "image_url": image_path},
                ],
            },
            {
                "role": "assistant",
                "content": {
                    "assertion_passed": passed,
                    "confidence_score": 0.95 if passed else 0.05,
                    "rationale": rationale,
                },
            },
        ]
    }
