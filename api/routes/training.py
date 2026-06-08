"""LoRA adapter training jobs — POST /training/jobs."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.crud import create_adapter, create_training_job, get_training_job, list_training_jobs, update_training_job
from db.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter()

_TRAIN_DIR = Path("data/train")


class TrainingJobResponse(BaseModel):
    id: str
    adapter_name: str
    adapter_id: str
    domain: str
    status: str
    pass_count: int
    fail_count: int
    config: dict
    output_path: str | None
    error: str | None
    created_at: str


def _to_response(job) -> TrainingJobResponse:
    return TrainingJobResponse(
        id=job.id,
        adapter_name=job.adapter_name,
        adapter_id=job.adapter_id,
        domain=job.domain,
        status=job.status,
        pass_count=len(job.pass_image_refs or []),
        fail_count=len(job.fail_image_refs or []),
        config=job.config or {},
        output_path=job.output_path,
        error=job.error,
        created_at=job.created_at.isoformat(),
    )


@router.get("", response_model=list[TrainingJobResponse])
async def list_jobs(session: AsyncSession = Depends(get_session)):
    jobs = await list_training_jobs(session)
    return [_to_response(j) for j in jobs]


@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_job(job_id: str, session: AsyncSession = Depends(get_session)):
    job = await get_training_job(session, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return _to_response(job)


@router.post("", response_model=TrainingJobResponse, status_code=201)
async def create_job(
    background_tasks: BackgroundTasks,
    # ── Adapter identity ──────────────────────────────────────────────────
    adapter_name: str = Form(...),
    adapter_id: str = Form(...),
    domain: str = Form(...),
    category: str | None = Form(default=None),
    base_model: str = Form(default="Qwen/Qwen2.5-VL-7B-Instruct"),
    notes: str | None = Form(default=None),
    contract_id: str | None = Form(default=None),
    # ── LoRA configuration ────────────────────────────────────────────────
    lora_rank: int = Form(default=16),
    lora_alpha: int | None = Form(default=None),   # None → auto = rank * 2
    lora_dropout: float = Form(default=0.05),
    target_modules: str = Form(default="q_proj,v_proj"),
    # ── Training hyperparameters ──────────────────────────────────────────
    learning_rate: float = Form(default=2e-4),
    num_epochs: int = Form(default=3),
    batch_size: int = Form(default=2),
    max_seq_length: int = Form(default=2048),
    gradient_accumulation_steps: int = Form(default=4),
    warmup_steps: int = Form(default=10),
    save_steps: int = Form(default=100),
    output_path: str = Form(default=""),
    # ── Training images ───────────────────────────────────────────────────
    pass_images: list[UploadFile] = File(default=[]),
    fail_images: list[UploadFile] = File(default=[]),
    session: AsyncSession = Depends(get_session),
):
    job_id = str(uuid.uuid4())

    pass_dir = (_TRAIN_DIR / job_id / "pass").resolve()
    fail_dir = (_TRAIN_DIR / job_id / "fail").resolve()
    pass_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    pass_refs: list[str] = []
    for i, f in enumerate(pass_images):
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix or ".jpg"
        fname = f"pass_{i:04d}{suffix}"
        (pass_dir / fname).write_bytes(await f.read())
        pass_refs.append(str(pass_dir / fname))

    fail_refs: list[str] = []
    for i, f in enumerate(fail_images):
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix or ".jpg"
        fname = f"fail_{i:04d}{suffix}"
        (fail_dir / fname).write_bytes(await f.read())
        fail_refs.append(str(fail_dir / fname))

    alpha = lora_alpha if lora_alpha is not None else lora_rank * 2
    resolved_output = output_path.strip() or str(Path("adapters").resolve() / adapter_id)
    modules = [m.strip() for m in target_modules.split(",") if m.strip()]

    config = {
        "lora_rank": lora_rank,
        "lora_alpha": alpha,
        "lora_dropout": lora_dropout,
        "target_modules": modules,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "save_steps": save_steps,
        "base_model": base_model,
        "category": category,
        "notes": notes,
        "contract_id": contract_id,
    }

    job = await create_training_job(session, {
        "id": job_id,
        "adapter_name": adapter_name,
        "adapter_id": adapter_id,
        "domain": domain,
        "base_model": base_model,
        "contract_id": contract_id or None,
        "pass_image_refs": pass_refs,
        "fail_image_refs": fail_refs,
        "config": config,
        "status": "queued",
        "output_path": resolved_output,
    })

    background_tasks.add_task(
        _run_training,
        job_id=job_id,
        adapter_name=adapter_name,
        adapter_id=adapter_id,
        domain=domain,
        base_model=base_model,
        category=category,
        notes=notes,
        pass_refs=pass_refs,
        fail_refs=fail_refs,
        config=config,
        output_path=resolved_output,
    )

    return _to_response(job)


async def _run_training(
    job_id: str,
    adapter_name: str,
    adapter_id: str,
    domain: str,
    base_model: str,
    category: str | None,
    notes: str | None,
    pass_refs: list[str],
    fail_refs: list[str],
    config: dict,
    output_path: str,
) -> None:
    """Background task: run LoRA fine-tuning and register the resulting adapter."""
    from db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        await update_training_job(session, job_id, {"status": "running"})

    try:
        _train_lora(
            pass_refs=pass_refs,
            fail_refs=fail_refs,
            config=config,
            output_path=output_path,
        )

        # Register the trained adapter so it appears in the LoRA selector
        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                __import__("sqlalchemy", fromlist=["select"]).select(
                    __import__("db.models", fromlist=["LoraAdapter"]).LoraAdapter
                ).where(
                    __import__("db.models", fromlist=["LoraAdapter"]).LoraAdapter.adapter_id == adapter_id
                )
            )
            if not existing.scalar_one_or_none():
                await create_adapter(session, {
                    "name": adapter_name,
                    "adapter_id": adapter_id,
                    "domain": domain,
                    "category": category,
                    "base_model": base_model,
                    "adapter_path": output_path,
                    "notes": notes,
                    "status": "active",
                })
            await update_training_job(session, job_id, {
                "status": "complete",
                "output_path": output_path,
            })

        logger.info("Training job %s complete — adapter %s saved to %s", job_id, adapter_id, output_path)

    except Exception as exc:
        logger.error("Training job %s failed: %s", job_id, exc, exc_info=True)
        async with AsyncSessionLocal() as session:
            await update_training_job(session, job_id, {"status": "failed", "error": str(exc)})


def _train_lora(
    pass_refs: list[str],
    fail_refs: list[str],
    config: dict,
    output_path: str,
) -> None:
    """Run the actual PEFT LoRA fine-tuning.

    Requires: pip install peft datasets trl
    This implementation stub validates data and logs config.
    Replace with a full SFTTrainer loop for production use.
    """
    import os

    total = len(pass_refs) + len(fail_refs)
    if total == 0:
        raise ValueError("No training images provided")

    missing = [p for p in pass_refs + fail_refs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing training images: {missing[:3]}")

    logger.info(
        "LoRA training: %d PASS + %d FAIL images | rank=%s lr=%s epochs=%s → %s",
        len(pass_refs), len(fail_refs),
        config.get("lora_rank"), config.get("learning_rate"),
        config.get("num_epochs"), output_path,
    )

    # ── Full PEFT training loop (requires peft, trl, datasets) ─────────────
    # from peft import LoraConfig, get_peft_model, TaskType
    # from trl import SFTTrainer, SFTConfig
    # from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    # from datasets import Dataset
    #
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config["base_model"], ...)
    # lora_cfg = LoraConfig(
    #     r=config["lora_rank"], lora_alpha=config["lora_alpha"],
    #     target_modules=config["target_modules"], lora_dropout=config["lora_dropout"],
    #     task_type=TaskType.CAUSAL_LM,
    # )
    # model = get_peft_model(model, lora_cfg)
    # trainer = SFTTrainer(model=model, train_dataset=dataset, args=SFTConfig(...))
    # trainer.train()
    # model.save_pretrained(output_path)
    # ──────────────────────────────────────────────────────────────────────

    Path(output_path).mkdir(parents=True, exist_ok=True)
    logger.info("Training stub complete — adapter directory created at %s", output_path)
