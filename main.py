from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from api.routes import contracts, evaluation
from db.session import init_db

_UI_PATH = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="AI Evaluation Engine",
    description="Visual contract assertion pipeline — multi-tenant VLM + LoRA serving",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(contracts.router, prefix="/contracts", tags=["contracts"])
app.include_router(evaluation.router, prefix="/evaluate", tags=["evaluation"])


@app.get("/", include_in_schema=False)
async def ui():
    return FileResponse(_UI_PATH)


@app.get("/health")
async def health():
    return {"status": "ok"}
