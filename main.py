from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from api.routes import adapters, contracts, delta_contracts, evaluation, training
from db.session import init_db

_UI_PATH = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("data/images").mkdir(parents=True, exist_ok=True)
    await init_db()
    yield


app = FastAPI(
    title="AI Evaluation Engine",
    description="Visual contract assertion pipeline — multi-tenant VLM + LoRA serving",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(adapters.router,        prefix="/adapters",         tags=["adapters"])
app.include_router(contracts.router,       prefix="/contracts",        tags=["contracts"])
app.include_router(delta_contracts.router, prefix="/delta-contracts",  tags=["delta-contracts"])
app.include_router(evaluation.router,      prefix="/evaluate",         tags=["evaluation"])
app.include_router(training.router,        prefix="/training/jobs",    tags=["training"])


@app.get("/", include_in_schema=False)
async def ui():
    return FileResponse(_UI_PATH)


@app.get("/images/{filename}", include_in_schema=False)
async def serve_image(filename: str):
    path = Path("data/images") / filename
    if not path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
