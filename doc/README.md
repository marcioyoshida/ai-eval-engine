# AI Evaluation Engine — Documentation

Visual contract assertion pipeline: ingest images, evaluate them against plain-English contracts, and return structured verdicts. Multi-tenant, VLM-powered, with active learning and LoRA fine-tuning.

---

## Documents

| File | Contents |
|------|----------|
| [architecture.md](architecture.md) | System components, database schema, ER diagram, deployment topology — all in Mermaid |
| [workflow.md](workflow.md) | Sequence diagrams for every major flow: evaluation, delta contract analysis, LoRA training, active learning loop, juror panel |
| [layers.md](layers.md) | Deep-dive into each layer: Presentation → API → Core → Persistence → Models → Worker |
| [api-reference.md](api-reference.md) | All endpoints with request/response schemas |
| [deployment.md](deployment.md) | Local dev, CUDA Docker, macOS native, Kubernetes, memory requirements |

---

## Project at a Glance

```mermaid
graph LR
    IMG[Evidence Image]
    CONTRACT[Visual Contract\n"required state / failure signals"]
    VLM["Qwen2.5-VL\nor vLLM + S-LoRA"]
    VERDICT["Verdict\npassed | confidence | rationale"]

    IMG --> VLM
    CONTRACT --> VLM
    VLM --> VERDICT
```

### Key Capabilities

- **Contract Registry** — define visual assertions in plain English per domain
- **Evaluation** — submit images, get structured pass/fail verdicts with reasoning
- **Delta Contracts** — upload a before-state image, receive an AI-generated gap analysis, ordered task plan, and optional synthetic target-state image (FLUX.1)
- **Active Learning** — low-confidence evaluations route to human review; accuracy drops trigger automatic LoRA retraining
- **LoRA Training** — upload PASS/FAIL example sets via UI; backend fine-tunes domain adapters with PEFT
- **Multi-backend** — local HuggingFace inference (MPS/CUDA/CPU) or production vLLM with S-LoRA multi-tenancy

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.115+, Pydantic v2, uvicorn |
| Database | SQLAlchemy 2.0 async, aiosqlite (SQLite), asyncpg (PostgreSQL) |
| VLM (local) | Qwen2.5-VL-7B-Instruct via `transformers` |
| VLM (production) | vLLM OpenAI-compatible server + S-LoRA |
| Image generation | FLUX.1-schnell via `diffusers` |
| Background tasks | Celery 5.4 + Redis |
| Fine-tuning | PEFT LoRA + TRL SFTTrainer |
| Device support | Apple MPS, NVIDIA CUDA, CPU |
| UI | Vanilla JS + CSS custom properties (no framework) |

---

## Repository Layout

```
ai-eval-engine/
├── main.py                   # App entry point
├── core/
│   ├── config.py             # Settings (env-driven)
│   ├── inference.py          # Local Qwen2.5-VL oracle
│   ├── vllm_client.py        # vLLM client + juror panel
│   ├── delta_engine.py       # S0→SF gap analysis
│   ├── image_gen.py          # FLUX.1 text-to-image
│   └── synthetic_bootstrap.py
├── db/
│   ├── models.py             # ORM models (6 tables)
│   ├── crud.py               # Async CRUD helpers
│   └── session.py            # Engine + session factory
├── api/routes/
│   ├── contracts.py
│   ├── delta_contracts.py
│   ├── evaluation.py
│   ├── adapters.py
│   └── training.py
├── worker/tasks.py           # Celery background tasks
├── static/index.html         # SPA UI
├── deploy/
│   ├── cuda/                 # NVIDIA Docker
│   └── macos/                # macOS Docker / native
└── doc/                      # ← you are here
```
