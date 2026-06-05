from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ContractDefinition, EvaluationRecord, FlaggedQueue


async def create_contract(session: AsyncSession, data: dict) -> ContractDefinition:
    contract = ContractDefinition(**data)
    session.add(contract)
    await session.commit()
    await session.refresh(contract)
    return contract


async def get_contracts_by_domain(session: AsyncSession, domain: str) -> list[ContractDefinition]:
    result = await session.execute(
        select(ContractDefinition).where(
            ContractDefinition.domain == domain, ContractDefinition.active == True  # noqa: E712
        )
    )
    return list(result.scalars().all())


async def get_contract_by_id(session: AsyncSession, contract_id: str) -> ContractDefinition | None:
    result = await session.execute(
        select(ContractDefinition).where(ContractDefinition.id == contract_id)
    )
    return result.scalar_one_or_none()


async def save_evaluation(session: AsyncSession, data: dict) -> EvaluationRecord:
    record = EvaluationRecord(**data)
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return record


async def flag_for_review(session: AsyncSession, data: dict) -> FlaggedQueue:
    item = FlaggedQueue(**data)
    session.add(item)
    await session.commit()
    await session.refresh(item)
    return item


async def get_recent_success_rate(session: AsyncSession, contract_id: str, window: int) -> float:
    """Returns the True Positive + True Negative rate over the last `window` human-reviewed records."""
    result = await session.execute(
        select(EvaluationRecord)
        .where(
            EvaluationRecord.contract_id == contract_id,
            EvaluationRecord.human_verdict.is_not(None),
        )
        .order_by(EvaluationRecord.created_at.desc())
        .limit(window)
    )
    records = list(result.scalars().all())
    if not records:
        return 1.0
    correct = sum(1 for r in records if r.passed == r.human_verdict)
    return correct / len(records)
