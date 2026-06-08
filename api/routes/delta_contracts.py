"""Delta Contracts — create S0/SF image pair with VLM-derived task plan and contract fields."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.crud import (
    create_contract,
    create_delta_contract,
    get_all_delta_contracts,
    get_contract_by_id,
    get_delta_contract,
    update_contract,
    update_delta_contract,
)
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
    negative_indicators: list[str]
    s0_image_url: str
    sf_image_url: str | None
    gap_analysis: str | None
    sf_description: str | None
    tasks: list[TaskItem]
    generation_status: str


class DeltaSummary(BaseModel):
    id: str
    domain: str
    gap_analysis: str | None
    sf_description: str | None


@router.get("", response_model=list[DeltaSummary])
async def list_deltas(session: AsyncSession = Depends(get_session)):
    """Return completed delta contracts — used to populate UI autocomplete suggestions."""
    rows = await get_all_delta_contracts(session)
    return [
        DeltaSummary(
            id=delta.id,
            domain=contract.domain,
            gap_analysis=delta.gap_analysis,
            sf_description=delta.sf_description,
        )
        for delta, contract in rows
    ]


@router.post("", response_model=DeltaContractResponse, status_code=201)
async def create_delta(
    background_tasks: BackgroundTasks,
    domain: str = Form(...),
    name: str = Form(...),
    target_object: str = Form(...),
    # required_state and negative_indicators are NOT collected from the user.
    # The VLM derives them from the gap analysis after S0 is analysed.
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

    # Create the base ContractDefinition with placeholder state — filled in after analysis.
    contract = await create_contract(
        session,
        {
            "domain": domain,
            "name": name,
            "target_object": target_object,
            "required_state": "",
            "negative_indicators": [],
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

    background_tasks.add_task(
        _run_analysis,
        delta_id=delta_id,
        contract_id=contract.id,
        s0_path=str(s0_path),
        images_dir=_IMAGES_DIR.resolve(),
        target_object=target_object,
    )

    return DeltaContractResponse(
        id=delta.id,
        contract_id=contract.id,
        domain=domain,
        name=name,
        target_object=target_object,
        required_state="",
        negative_indicators=[],
        s0_image_url=f"/images/{s0_filename}",
        sf_image_url=None,
        gap_analysis=None,
        sf_description=None,
        tasks=[],
        generation_status="pending",
    )


@router.get("/{delta_id}", response_model=DeltaContractResponse)
async def get_delta(delta_id: str, session: AsyncSession = Depends(get_session)):
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
        negative_indicators=contract.negative_indicators or [],
        s0_image_url=f"/images/{delta.s0_image_ref}",
        sf_image_url=f"/images/{delta.sf_image_ref}" if delta.sf_image_ref else None,
        gap_analysis=delta.gap_analysis,
        sf_description=delta.sf_description,
        tasks=tasks,
        generation_status=delta.generation_status,
    )


async def _run_analysis(
    delta_id: str,
    contract_id: str,
    s0_path: str,
    images_dir: Path,
    target_object: str,
) -> None:
    """Background task: VLM gap analysis → derive contract fields → persist everything."""
    from db.session import AsyncSessionLocal

    try:
        from core.delta_engine import analyze_s0

        # Pass empty strings for required_state / negative_indicators — the VLM
        # will ignore them and derive them from the image + target_object alone.
        result = await analyze_s0(
            s0_path,
            target_object,
            required_state="",
            negative_indicators=[],
            delta_id=delta_id,
            images_dir=images_dir,
        )

        delta_updates = {
            "gap_analysis": result["gap_analysis"],
            "sf_description": result["sf_description"],
            "tasks": result["tasks"],
            "sf_image_ref": result.get("sf_image_ref"),
            "generation_status": "complete",
        }
        contract_updates = {
            "required_state": result.get("derived_required_state", ""),
            "negative_indicators": result.get("derived_failure_signals", []),
        }

    except Exception as exc:
        logger.error("Delta analysis failed for %s: %s", delta_id, exc, exc_info=True)
        delta_updates = {"generation_status": "failed"}
        contract_updates = {}

    async with AsyncSessionLocal() as session:
        await update_delta_contract(session, delta_id, delta_updates)

    if contract_updates:
        async with AsyncSessionLocal() as session:
            await update_contract(session, contract_id, contract_updates)
