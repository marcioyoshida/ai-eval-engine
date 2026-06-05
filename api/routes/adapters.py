from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.crud import create_adapter, get_adapter_by_id, get_all_adapters
from db.session import get_session

router = APIRouter()


class AdapterCreateRequest(BaseModel):
    name: str
    adapter_id: str
    domain: str
    category: str | None = None
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    adapter_path: str | None = None
    notes: str | None = None
    status: str = "active"


class AdapterResponse(BaseModel):
    id: str
    name: str
    adapter_id: str
    domain: str
    category: str | None
    base_model: str
    adapter_path: str | None
    notes: str | None
    status: str

    model_config = {"from_attributes": True}


@router.get("", response_model=list[AdapterResponse])
async def list_adapters(session: AsyncSession = Depends(get_session)):
    return await get_all_adapters(session)


@router.post("", response_model=AdapterResponse, status_code=201)
async def register_adapter(
    body: AdapterCreateRequest, session: AsyncSession = Depends(get_session)
):
    existing = await get_adapter_by_id(session, body.adapter_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Adapter with adapter_id '{body.adapter_id}' already exists.",
        )
    return await create_adapter(session, body.model_dump())
