# System Architecture

## High-Level Overview

```mermaid
graph TB
    subgraph Client["Client Layer"]
        UI[Web UI<br/>static/index.html]
        CLI[curl / API client]
    end

    subgraph API["API Layer — FastAPI"]
        R_C["/contracts<br/>Contract Registry"]
        R_D["/delta-contracts<br/>Delta Engine"]
        R_E["/evaluate<br/>Evaluation"]
        R_A["/adapters<br/>LoRA Registry"]
        R_T["/training/jobs<br/>LoRA Training"]
    end

    subgraph Core["Core Layer"]
        INF[inference.py<br/>VisualContractOracle]
        DEL[delta_engine.py<br/>DeltaEngine]
        IMG[image_gen.py<br/>FluxImageGenerator]
        VLLM[vllm_client.py<br/>vLLMClient + JurorPanel]
        CFG[config.py<br/>Settings]
    end

    subgraph Models["Model Layer"]
        QWEN["Qwen2.5-VL-7B<br/>(local HuggingFace)"]
        FLUX["FLUX.1-schnell<br/>(diffusers)"]
        VLLM_SRV["vLLM Server<br/>(S-LoRA / multi-tenant)"]
    end

    subgraph DB["Persistence Layer"]
        SQLITE[(SQLite<br/>eval_engine.db)]
        REDIS[(Redis<br/>task queue)]
    end

    subgraph Worker["Worker Layer"]
        CELERY[Celery Worker<br/>worker/tasks.py]
    end

    UI --> R_C & R_D & R_E & R_A & R_T
    CLI --> R_C & R_D & R_E & R_A & R_T

    R_C --> DB
    R_A --> DB
    R_T --> Core
    R_D --> DEL
    R_E --> INF
    R_E --> VLLM

    DEL --> INF
    DEL --> IMG

    INF --> QWEN
    IMG --> FLUX
    VLLM --> VLLM_SRV

    R_E --> REDIS
    CELERY --> REDIS
    CELERY --> DB

    Core --> SQLITE
    API --> SQLITE
```

---

## Component Responsibilities

| Component | File | Role |
|-----------|------|------|
| **Web UI** | `static/index.html` | SPA — Register Contract, Delta Contract, Evaluate, Train LoRA tabs |
| **FastAPI App** | `main.py` | Router registration, lifespan DB init, static file serving |
| **Config** | `core/config.py` | Pydantic BaseSettings — all tunables from `.env` |
| **Local Oracle** | `core/inference.py` | HuggingFace Qwen2.5-VL inference, JSON parsing, confidence scoring |
| **vLLM Client** | `core/vllm_client.py` | OpenAI-compatible async client, S-LoRA routing, juror panel |
| **Delta Engine** | `core/delta_engine.py` | S0→SF gap analysis, task decomposition, contract field derivation |
| **Image Gen** | `core/image_gen.py` | FLUX.1 text-to-image with VLM memory swap |
| **DB Models** | `db/models.py` | SQLAlchemy ORM — 6 tables |
| **CRUD** | `db/crud.py` | Async DB queries & mutations |
| **Session** | `db/session.py` | Async engine, `get_session` dependency, `init_db` |
| **Worker** | `worker/tasks.py` | Celery tasks — active learning, retrain triggers |

---

## Database Schema

```mermaid
erDiagram
    ContractDefinition {
        string id PK
        string domain
        string name
        string target_object
        text required_state
        json negative_indicators
        float strictness_coefficient
        string lora_id FK
        datetime created_at
        bool active
    }

    EvaluationRecord {
        string id PK
        string contract_id FK
        text image_ref
        bool passed
        float confidence
        text rationale
        bool routed_to_human
        bool human_verdict
        datetime created_at
    }

    DeltaContract {
        string id PK
        string contract_id FK
        text s0_image_ref
        text sf_image_ref
        text gap_analysis
        text sf_description
        json tasks
        string generation_status
        datetime created_at
    }

    LoraAdapter {
        string id PK
        string name
        string adapter_id UK
        string domain
        string category
        string base_model
        text adapter_path
        text notes
        string status
        datetime created_at
    }

    LoraTrainingJob {
        string id PK
        string adapter_name
        string adapter_id
        string domain
        string base_model
        string contract_id FK
        json pass_image_refs
        json fail_image_refs
        json config
        string status
        text output_path
        text error
        datetime created_at
        datetime updated_at
    }

    FlaggedQueue {
        string id PK
        string evaluation_id FK
        string contract_id FK
        text image_ref
        float confidence
        bool reviewed
        datetime created_at
    }

    ContractDefinition ||--o{ EvaluationRecord : "evaluated by"
    ContractDefinition ||--o{ DeltaContract : "analyzed by"
    ContractDefinition ||--o{ LoraTrainingJob : "trains for"
    ContractDefinition ||--o{ FlaggedQueue : "flagged from"
    LoraAdapter ||--o{ ContractDefinition : "used by"
```

---

## Inference Backend Selection

```mermaid
flowchart LR
    ENV{INFERENCE_BACKEND}
    ENV -->|local| LOCAL["VisualContractOracle\ncore/inference.py\nHuggingFace transformers"]
    ENV -->|vllm| PROD["vLLMClient\ncore/vllm_client.py\nOpenAI-compatible API"]

    LOCAL --> DEV_DEV["Device Detection"]
    DEV_DEV -->|MPS available| MPS["Apple Silicon GPU\nfloat16"]
    DEV_DEV -->|CUDA available| CUDA["NVIDIA GPU\nfloat16"]
    DEV_DEV -->|fallback| CPU["CPU\nfloat32"]

    PROD --> SLORA["S-LoRA Routing\ndomain → adapter_id"]
    SLORA --> JUROR["Juror Panel\n3-perspective consensus"]
```

---

## Deployment Topology

```mermaid
graph LR
    subgraph CUDA["deploy/cuda/ — NVIDIA Linux"]
        API_C[FastAPI :8000]
        VLLM_C[vLLM Server :8001]
        REDIS_C[Redis :6379]
        API_C <--> VLLM_C
        API_C <--> REDIS_C
    end

    subgraph MACOS["deploy/macos/ — Apple Silicon (native)"]
        API_M[FastAPI :8000<br/>MPS device]
        REDIS_M[Redis :6379]
        API_M <--> REDIS_M
    end

    subgraph K8S["k8s-deployment.yaml — Kubernetes"]
        DEP[Deployment]
        SVC[Service]
        DEP --> SVC
    end
```
