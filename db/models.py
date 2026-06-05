import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ContractDefinition(Base):
    __tablename__ = "contract_definitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    domain: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(256))
    target_object: Mapped[str] = mapped_column(String(256))
    required_state: Mapped[str] = mapped_column(Text)
    negative_indicators: Mapped[list] = mapped_column(JSON, default=list)
    strictness_coefficient: Mapped[float] = mapped_column(Float, default=0.80)
    # Optional LoRA adapter to use when inference_backend=vllm
    lora_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class EvaluationRecord(Base):
    __tablename__ = "evaluation_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    contract_id: Mapped[str] = mapped_column(String(36), index=True)
    image_ref: Mapped[str] = mapped_column(Text)
    passed: Mapped[bool] = mapped_column(Boolean)
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(Text)
    routed_to_human: Mapped[bool] = mapped_column(Boolean, default=False)
    # human override; null until reviewed
    human_verdict: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class LoraAdapter(Base):
    __tablename__ = "lora_adapters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # Human-readable display name
    name: Mapped[str] = mapped_column(String(256))
    # Slug used as the vLLM model name and stored in ContractDefinition.lora_id
    adapter_id: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    domain: Mapped[str] = mapped_column(String(128), index=True)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    base_model: Mapped[str] = mapped_column(String(256), default="Qwen/Qwen2.5-VL-7B-Instruct")
    adapter_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # active | training | deprecated
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class FlaggedQueue(Base):
    __tablename__ = "flagged_queue"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    evaluation_id: Mapped[str] = mapped_column(String(36), index=True)
    contract_id: Mapped[str] = mapped_column(String(36), index=True)
    image_ref: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
