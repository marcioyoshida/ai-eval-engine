"""POST /evaluate — accepts an image upload or URL and a contract_id."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.schemas import ContractParams, EvaluationResult
from db.crud import flag_for_review, get_contract_by_id, save_evaluation
from db.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level singletons — initialised lazily on first request
_local_oracle = None
_vllm_orchestrator = None


def _get_local_oracle():
    global _local_oracle
    if _local_oracle is None:
        from core.inference import VisualContractOracle
        _local_oracle = VisualContractOracle(settings.local_model_id)
    return _local_oracle


def _get_vllm_orchestrator():
    global _vllm_orchestrator
    if _vllm_orchestrator is None:
        from core.vllm_client import VLLMOrchestrator
        _vllm_orchestrator = VLLMOrchestrator()
    return _vllm_orchestrator


class EvaluationResponse(BaseModel):
    evaluation_id: str
    contract_id: str
    passed: bool
    confidence: float
    rationale: str
    threshold_applied: float
    routed_to_human: bool


@router.post("", response_model=EvaluationResponse)
async def evaluate(
    contract_id: str = Form(...),
    image_url: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    session: AsyncSession = Depends(get_session),
):
    contract = await get_contract_by_id(session, contract_id)
    if not contract or not contract.active:
        raise HTTPException(status_code=404, detail="Contract not found")

    image_ref, tmp_path = await _resolve_image(image_url, file)

    params = ContractParams(
        target_object=contract.target_object,
        required_state=contract.required_state,
        negative_indicators=contract.negative_indicators or [],
        strictness_coefficient=contract.strictness_coefficient,
    )

    try:
        result = await _run_inference(params, image_ref, tmp_path, contract.lora_id)
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

    # Apply strictness threshold
    if result.confidence >= params.strictness_coefficient:
        result.passed = result.passed  # already set by model
    else:
        result.passed = False
    result.threshold_applied = params.strictness_coefficient

    # Route to human review if confidence falls in the ambiguous band
    in_review_band = (
        settings.human_review_lower <= result.confidence <= settings.human_review_upper
    )
    result.routed_to_human = in_review_band

    record = await save_evaluation(
        session,
        {
            "contract_id": contract_id,
            "image_ref": image_ref,
            "passed": result.passed,
            "confidence": result.confidence,
            "rationale": result.rationale,
            "routed_to_human": result.routed_to_human,
        },
    )

    if in_review_band:
        await flag_for_review(
            session,
            {
                "evaluation_id": record.id,
                "contract_id": contract_id,
                "image_ref": image_ref,
                "confidence": result.confidence,
            },
        )
        # Kick off async active-learning task
        try:
            from worker.tasks import process_flagged_evaluation
            process_flagged_evaluation.delay(record.id)
        except Exception:
            logger.warning("Celery worker unavailable — flagged record %s not queued", record.id)

    return EvaluationResponse(
        evaluation_id=record.id,
        contract_id=contract_id,
        passed=result.passed,
        confidence=result.confidence,
        rationale=result.rationale,
        threshold_applied=result.threshold_applied,
        routed_to_human=result.routed_to_human,
    )


async def _resolve_image(
    image_url: str | None, file: UploadFile | None
) -> tuple[str, Path | None]:
    if file:
        suffix = Path(file.filename or "upload").suffix or ".jpg"
        tmp = Path(tempfile.mktemp(suffix=suffix))
        tmp.write_bytes(await file.read())
        return str(tmp), tmp
    if image_url:
        return image_url, None
    raise HTTPException(status_code=422, detail="Provide either 'file' or 'image_url'")


async def _run_inference(
    params: ContractParams,
    image_ref: str,
    tmp_path: Path | None,
    lora_id: str | None,
) -> EvaluationResult:
    if settings.inference_backend == "vllm":
        orchestrator = _get_vllm_orchestrator()
        return await orchestrator.run_evaluation(image_ref, params, lora_id=lora_id)
    else:
        oracle = _get_local_oracle()
        source = tmp_path if tmp_path else image_ref
        return oracle.evaluate_evidence(source, params)
