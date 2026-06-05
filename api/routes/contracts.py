from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from db.crud import create_contract, get_all_contracts, get_contracts_by_domain
from db.session import get_session

router = APIRouter()


class ContractCreateRequest(BaseModel):
    domain: str
    name: str
    target_object: str
    required_state: str
    negative_indicators: list[str] = Field(default_factory=list)
    strictness_coefficient: float = Field(default=0.80, ge=0.0, le=1.0)
    lora_id: str | None = None


class ContractResponse(BaseModel):
    id: str
    domain: str
    name: str
    target_object: str
    required_state: str
    negative_indicators: list[str]
    strictness_coefficient: float
    lora_id: str | None

    model_config = {"from_attributes": True}


@router.post("", response_model=ContractResponse, status_code=201)
async def register_contract(
    body: ContractCreateRequest, session: AsyncSession = Depends(get_session)
):
    contract = await create_contract(session, body.model_dump())
    return contract


@router.get("", response_model=list[ContractResponse])
async def list_all_contracts(session: AsyncSession = Depends(get_session)):
    return await get_all_contracts(session)


@router.get("/{domain}", response_model=list[ContractResponse])
async def list_contracts(domain: str, session: AsyncSession = Depends(get_session)):
    contracts = await get_contracts_by_domain(session, domain)
    if not contracts:
        raise HTTPException(status_code=404, detail=f"No active contracts for domain '{domain}'")
    return contracts
