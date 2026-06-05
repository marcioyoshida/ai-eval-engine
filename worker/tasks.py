"""Celery background workers — Phase 4.

Tasks:
  process_flagged_evaluation  — logs low-confidence evaluations for human review
  trigger_lora_retrain        — fires when a LoRA's success rate drops below threshold
  bootstrap_contract_data     — synthetic data generation for new contract types
"""

from __future__ import annotations

import asyncio
import logging

from celery import Celery

from core.config import settings

logger = logging.getLogger(__name__)

app = Celery("ai_eval_engine", broker=settings.redis_url, backend=settings.redis_url)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)


def _run_async(coro):
    """Run an async coroutine from a synchronous Celery task."""
    return asyncio.get_event_loop().run_until_complete(coro)


@app.task(bind=True, max_retries=3)
def process_flagged_evaluation(self, evaluation_id: str):
    """Persist a low-confidence evaluation to the FlaggedQueue and optionally
    trigger a LoRA retrain check for the associated contract."""
    try:
        from db.crud import get_recent_success_rate
        from db.models import EvaluationRecord
        from db.session import AsyncSessionLocal
        from sqlalchemy import select

        async def _process():
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(EvaluationRecord).where(EvaluationRecord.id == evaluation_id)
                )
                record = result.scalar_one_or_none()
                if not record:
                    logger.warning("process_flagged_evaluation: record %s not found", evaluation_id)
                    return

                logger.info(
                    "Flagged evaluation %s — contract=%s confidence=%.3f routed_to_human=%s",
                    evaluation_id,
                    record.contract_id,
                    record.confidence,
                    record.routed_to_human,
                )

                # Check if the LoRA success rate has dropped below threshold
                success_rate = await get_recent_success_rate(
                    session, record.contract_id, settings.lora_retrain_window
                )
                if success_rate < settings.lora_retrain_threshold:
                    logger.warning(
                        "Contract %s success rate %.2f below threshold %.2f — queueing retrain",
                        record.contract_id,
                        success_rate,
                        settings.lora_retrain_threshold,
                    )
                    trigger_lora_retrain.delay(record.contract_id)

        _run_async(_process())
    except Exception as exc:
        logger.error("process_flagged_evaluation failed: %s", exc)
        raise self.retry(exc=exc, countdown=30)


@app.task(bind=True, max_retries=2)
def trigger_lora_retrain(self, contract_id: str):
    """Trigger automated LoRA retraining for a specific contract.

    In production this would:
    1. Pull the latest human-annotated records from FlaggedQueue
    2. Merge with existing synthetic dataset
    3. Submit a training job to your GPU cluster / SageMaker
    4. On completion, call POST /v1/load_lora_adapter on the vLLM server

    This stub logs the intent and queues a synthetic data refresh.
    """
    logger.info("trigger_lora_retrain: contract_id=%s", contract_id)
    # Bootstrap synthetic data to complement real-world flagged samples
    bootstrap_contract_data.delay(contract_id)


@app.task(bind=True, max_retries=2)
def bootstrap_contract_data(self, contract_id: str):
    """Stub for synthetic data bootstrapping (Phase 4 active-learning loop).

    Full implementation would:
    1. Fetch the ContractDefinition from DB
    2. Call expand_contract_to_prompt_matrix() to build positive/negative prompt variants
    3. Submit prompts to a Stable Diffusion / Flux.1 endpoint
    4. Run CLIP-based quality filter on generated images
    5. Call a frontier VLM (GPT-4o / Claude 3.5) to auto-label each image
    6. Store labelled pairs as SFT JSONL ready for PEFT LoRA training
    """
    logger.info("bootstrap_contract_data stub: contract_id=%s", contract_id)
    from core.synthetic_bootstrap import expand_contract_to_prompt_matrix

    try:
        prompt_matrix = expand_contract_to_prompt_matrix(contract_id)
        logger.info(
            "Prompt matrix expanded: %d positive / %d negative prompts",
            len(prompt_matrix["positive"]),
            len(prompt_matrix["negative"]),
        )
        # TODO: submit to image generation pipeline
    except Exception as exc:
        logger.error("bootstrap_contract_data failed for %s: %s", contract_id, exc)
        raise self.retry(exc=exc, countdown=60)
