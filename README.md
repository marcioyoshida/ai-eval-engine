# AI Evaluation Engine

Visual contract assertion pipeline — multi-tenant VLM + LoRA serving.

Ingests raw images or streamed video frames, evaluates them against a plain-English contract
("Is the pothole fixed?", "Is the car dent repaired?", "Is the house cleaned?"), and returns a
structured verdict with a confidence score. Designed for industries that need automated,
auditable physical-world quality assurance: municipal contractors, insurance carriers, and
household service platforms.

---

## Architecture overview

```
[Ingestion] ──► [Dynamic Orchestrator] ──► [Multi-LoRA Inference]
                        │                           │
                        ▼                           ▼
               [Contract Registry]           [Juror Engine]
                        │                           │
                        ▼                           ▼
               [LoRA Training CI/CD] ◄── [Active Learning Loop]
```

The system has two switchable inference backends controlled by `INFERENCE_BACKEND`:

| Mode | When to use | How it works |
|------|-------------|--------------|
| `local` | Single-GPU pilot / dev | Loads Qwen2.5-VL-7B directly via HuggingFace `transformers` |
| `vllm` | Production / multi-tenant | Calls a running vLLM OpenAI-compatible server with dynamic S-LoRA swapping |

---

## Pilot file structure

```
ai-eval-engine/
├── main.py                         # FastAPI entrypoint + lifespan DB init
├── pyproject.toml                  # Poetry: fastapi, sqlalchemy, transformers, openai, celery
├── pytest.ini
├── .env.example                    # All runtime switches documented
│
├── core/
│   ├── config.py                   # Settings loaded from env vars
│   ├── schemas.py                  # ContractParams + EvaluationResult (Pydantic v2)
│   ├── inference.py                # VisualContractOracle — local Qwen2.5-VL-7B (Phase 1)
│   ├── vllm_client.py              # VLLMOrchestrator — async, dynamic LoRA routing (Phase 3)
│   │                               #   includes run_juror_panel() — 3-juror consensus
│   └── synthetic_bootstrap.py     # Prompt expansion + CLIP filter + SFT record stubs (Phase 4)
│
├── db/
│   ├── models.py                   # ContractDefinition, EvaluationRecord, FlaggedQueue
│   ├── session.py                  # Async SQLAlchemy engine + init_db()
│   └── crud.py                     # create/get contracts, save evaluations, success rate calc
│
├── api/
│   └── routes/
│       ├── contracts.py            # POST /contracts, GET /contracts/{domain}
│       └── evaluation.py          # POST /evaluate — file or URL + contract_id
│
├── worker/
│   └── tasks.py                    # Celery: process_flagged_evaluation
│                                   #         trigger_lora_retrain
│                                   #         bootstrap_contract_data
│
├── deploy/
│   ├── Dockerfile                  # CUDA 12.4 base, Poetry, GPU-ready
│   ├── docker-compose.yml          # api + worker + redis; vllm behind --profile production
│   └── k8s-deployment.yaml        # K8s Deployment + LoadBalancer Service
│
└── tests/
    ├── test_inference_parsing.py   # JSON parse edge cases (no GPU required)
    └── test_evaluation_pipeline.py # API integration tests (no GPU required)
```

---

## How to run the pilot locally

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Redis (only needed to run background workers)
- NVIDIA GPU with 16 GB+ VRAM for `local` inference mode (RTX 3090/4090 or A100)
- No GPU needed to run the API + tests in `vllm` mode against an external server

### 1. Install dependencies

```bash
# CPU-only (API, tests, worker — no model inference)
poetry install

# With local GPU inference support
poetry install --extras local-gpu
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set INFERENCE_BACKEND=local for single-GPU pilot
```

### 3. Start the API

```bash
poetry run uvicorn main:app --reload
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 4. Register a contract

```bash
curl -X POST http://localhost:8000/contracts \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "municipal_infrastructure",
    "name": "Pothole Repair Check",
    "target_object": "pothole",
    "required_state": "fully filled with dark asphalt, level with surrounding road",
    "negative_indicators": ["loose gravel", "exposed base layer", "water pooling"],
    "strictness_coefficient": 0.85
  }'
```

### 5. Submit an evaluation

```bash
# Using a local image file
curl -X POST http://localhost:8000/evaluate \
  -F "contract_id=<id_from_step_4>" \
  -F "file=@test_pothole.jpg"

# Using a public image URL
curl -X POST http://localhost:8000/evaluate \
  -F "contract_id=<id_from_step_4>" \
  -F "image_url=https://example.com/pothole_before.jpg"
```

**Example response:**

```json
{
  "evaluation_id": "a3f1c2d4-...",
  "contract_id": "b7e9a1f0-...",
  "passed": false,
  "confidence": 0.89,
  "rationale": "The hole has been filled but failed compliance due to loose gravel along the edges and uneven leveling.",
  "threshold_applied": 0.85,
  "routed_to_human": false
}
```

### 6. Run tests (no GPU required)

```bash
poetry run pytest
```

---

## Switching to production vLLM mode

Start a vLLM server with your LoRA adapters:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --enable-lora \
  --max-loras 16 \
  --max-cpu-loras 128 \
  --max-lora-rank 64 \
  --lora-modules \
    insurance_car=/path/to/adapters/insurance_car_v1 \
    muni_road=/path/to/adapters/muni_road_v1 \
    house_clean=/path/to/adapters/house_clean_v1
```

Then set in `.env`:

```
INFERENCE_BACKEND=vllm
VLLM_BASE_URL=http://localhost:8001/v1
```

Contracts with a `lora_id` field set will automatically route through the matching adapter.
Contracts without a `lora_id` fall back to zero-shot base model inference.

---

## Docker Compose (local stack)

```bash
cd deploy
docker compose up        # api + worker + redis
docker compose --profile production up  # adds the vLLM server
```

---

## Confidence thresholds

| Confidence range | Action |
|-----------------|--------|
| `>= strictness_coefficient` | Auto-resolved — `passed: true/false` returned immediately |
| `0.45 – 0.65` | Routed to human review queue + Celery task queued for active learning |
| `< strictness_coefficient` | Auto-failed |

When a contract's human-reviewed success rate drops below `LORA_RETRAIN_THRESHOLD` (default 95%) over the last `LORA_RETRAIN_WINDOW` (default 100) transactions, the system automatically queues a LoRA retraining job.

---

## Domain examples

| Domain | Contract assertion | `target_object` |
|--------|--------------------|-----------------|
| Municipal | "Is the pothole filled and level?" | `pothole` |
| Insurance | "Is the car dent repaired?" | `car bumper` |
| Household | "Is the kitchen countertop wiped down?" | `kitchen countertop` |
