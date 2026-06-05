"""Integration tests for the evaluation pipeline (no GPU required)."""

import pytest
from httpx import AsyncClient, ASGITransport

from main import app


@pytest.fixture(autouse=True)
async def setup_db():
    from db.session import init_db
    await init_db()


@pytest.mark.asyncio
async def test_register_and_list_contract():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "domain": "municipal_infrastructure",
            "name": "Pothole Repair Check",
            "target_object": "pothole",
            "required_state": "fully filled with dark asphalt, level with surrounding road",
            "negative_indicators": ["loose gravel", "exposed base layer", "water pooling"],
            "strictness_coefficient": 0.85,
        }
        response = await client.post("/contracts", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["domain"] == "municipal_infrastructure"
        assert data["id"]

        list_response = await client.get("/contracts/municipal_infrastructure")
        assert list_response.status_code == 200
        assert len(list_response.json()) >= 1


@pytest.mark.asyncio
async def test_list_contracts_unknown_domain_returns_404():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/contracts/nonexistent_domain_xyz")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_evaluate_missing_image_returns_422():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # First register a contract
        contract = await client.post(
            "/contracts",
            json={
                "domain": "insurance",
                "name": "Car Dent Fix",
                "target_object": "car bumper",
                "required_state": "smooth surface, no visible dent",
                "negative_indicators": ["dent", "crease", "paint chip"],
            },
        )
        contract_id = contract.json()["id"]

        # Attempt evaluation without image
        response = await client.post("/evaluate", data={"contract_id": contract_id})
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
