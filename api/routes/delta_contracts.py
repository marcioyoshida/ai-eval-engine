"""POST /delta-contracts — create a contract with S0/SF image pair and VLM task plan."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.crud import create_contract, create_delta_contract, get_delta_contract, update_delta_contract
from db.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter()

_IMAGES_DIR = Path("data/images")


class TaskItem(BaseModel):
    step: int
    action: str
    detail: str


class DeltaContractResponse(BaseModel):
    id: str
    contract_id: str
    domain: str
    name: str
    target_object: str
    required_state: str
    s0_image_url: str
    sf_image_url: str | None
    gap_analysis: str | None
    sf_description: str | None
    tasks: list[TaskItem]
    generation_status: str


@router.post("", response_model=DeltaContractResponse, status_code=201)
async def create_delta(
    background_tasks: BackgroundTasks,
    domain: str = Form(...),
    name: str = Form(...),
    target_object: str = Form(...),
    required_state: str = Form(...),
    negative_indicators: str = Form(default=""),
    strictness_coefficient: float = Form(default=0.80),
    lora_id: str | None = Form(default=None),
    s0_file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Save S0 image — use absolute path so background task can locate it from any cwd.
    delta_id = str(uuid.uuid4())
    suffix = Path(s0_file.filename or "upload").suffix or ".jpg"
    s0_filename = f"s0_{delta_id}{suffix}"
    s0_path = _IMAGES_DIR.resolve() / s0_filename
    s0_path.write_bytes(await s0_file.read())

    indicators = [i.strip() for i in negative_indicators.replace("\n", ",").split(",") if i.strip()]

    # Create the base ContractDefinition first
    contract = await create_contract(
        session,
        {
            "domain": domain,
            "name": name,
            "target_object": target_object,
            "required_state": required_state,
            "negative_indicators": indicators,
            "strictness_coefficient": strictness_coefficient,
            "lora_id": lora_id or None,
        },
    )

    # Create the DeltaContract record (analysis runs in background)
    delta = await create_delta_contract(
        session,
        {
            "id": delta_id,
            "contract_id": contract.id,
            "s0_image_ref": s0_filename,
            "generation_status": "pending",
        },
    )

    # Run VLM analysis in background so the HTTP response is immediate
    background_tasks.add_task(
        _run_analysis,
        delta_id=delta_id,
        s0_path=str(s0_path),
        target_object=target_object,
        required_state=required_state,
        negative_indicators=indicators,
    )

    return DeltaContractResponse(
        id=delta.id,
        contract_id=contract.id,
        domain=domain,
        name=name,
        target_object=target_object,
        required_state=required_state,
        s0_image_url=f"/images/{s0_filename}",
        sf_image_url=None,
        gap_analysis=None,
        sf_description=None,
        tasks=[],
        generation_status="pending",
    )


@router.get("/{delta_id}", response_model=DeltaContractResponse)
async def get_delta(delta_id: str, session: AsyncSession = Depends(get_session)):
    from db.crud import get_contract_by_id

    delta = await get_delta_contract(session, delta_id)
    if not delta:
        raise HTTPException(status_code=404, detail="Delta contract not found")

    contract = await get_contract_by_id(session, delta.contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Base contract not found")

    tasks = [TaskItem(**t) for t in (delta.tasks or [])]

    return DeltaContractResponse(
        id=delta.id,
        contract_id=delta.contract_id,
        domain=contract.domain,
        name=contract.name,
        target_object=contract.target_object,
        required_state=contract.required_state,
        s0_image_url=f"/images/{delta.s0_image_ref}",
        sf_image_url=f"/images/{delta.sf_image_ref}" if delta.sf_image_ref else None,
        gap_analysis=delta.gap_analysis,
        sf_description=delta.sf_description,
        tasks=tasks,
        generation_status=delta.generation_status,
    )


async def _run_analysis(
    delta_id: str,
    s0_path: str,
    target_object: str,
    required_state: str,
    negative_indicators: list[str],
) -> None:
    """Background task: call the delta engine and persist results."""
    from db.session import AsyncSessionLocal

    try:
        from core.delta_engine import analyze_s0
        result = await analyze_s0(s0_path, target_object, required_state, negative_indicators)
        updates = {
            "gap_analysis": result["gap_analysis"],
            "sf_description": result["sf_description"],
            "tasks": result["tasks"],
            "generation_status": "complete",
        }
    except Exception as exc:
        logger.error("Delta analysis failed for %s: %s", delta_id, exc, exc_info=True)
        updates = {"generation_status": "failed"}

    async with AsyncSessionLocal() as session:
        await update_delta_contract(session, delta_id, updates)
