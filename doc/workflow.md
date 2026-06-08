# Workflows

## 1. Image Evaluation

The core pipeline: submit an image against a registered contract and receive a structured verdict.

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as POST /evaluate
    participant DB as SQLite
    participant INF as VisualContractOracle
    participant VLM as Qwen2.5-VL / vLLM
    participant Q as Redis / Celery

    User->>UI: Upload image + select contract
    UI->>API: POST /evaluate (multipart: file, contract_id)
    API->>DB: get_contract_by_id(contract_id)
    DB-->>API: ContractDefinition
    API->>INF: evaluate_evidence(image, contract)
    INF->>VLM: vision inference (system prompt + image)
    VLM-->>INF: raw text (JSON + optional <think> block)
    INF->>INF: parse JSON → EvaluationResult
    INF-->>API: {passed, confidence, rationale}
    API->>DB: save_evaluation(record)

    alt confidence in [0.45, 0.65]
        API->>DB: flag_for_review(record)
        API->>Q: process_flagged_evaluation.delay(id)
    end

    API-->>UI: verdict + confidence + rationale
    UI-->>User: Pass / Fail card with reasoning
```

### Confidence Bands

| Range | Action |
|-------|--------|
| `< lower` (0.45) | Auto-FAIL, no human review |
| `[0.45, 0.65]` | Route to human review queue |
| `> upper` (0.65) | Auto-PASS if `confidence >= strictness_coefficient` |

The `strictness_coefficient` (per contract, default 0.80) is the final pass threshold — a contract with `strictness=0.90` requires higher confidence to auto-pass.

---

## 2. Delta Contract — S0 → SF Analysis

Analyzes the current state of an object, identifies gaps, and produces a task plan plus an optional synthetic target-state image.

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as POST /delta-contracts
    participant BG as Background Task
    participant DEL as delta_engine.py
    participant VLM as VisualContractOracle
    participant FLUX as FluxImageGenerator
    participant DB as SQLite

    User->>UI: Upload S0 image + domain + contract name
    UI->>API: POST /delta-contracts (multipart)
    API->>DB: create_contract(required_state="", negative_indicators=[])
    API->>DB: create_delta_contract(status=pending)
    API-->>UI: {delta_id, status: pending}
    API->>BG: background_task(_run_analysis)

    BG->>DEL: analyze_s0(s0_path, target_object, domain)
    DEL->>VLM: call_raw(image, gap_analysis_prompt)
    VLM-->>DEL: gap_analysis + sf_description + tasks[]

    DEL->>DEL: _derive_failure_signals(gap_analysis)
    Note over DEL: Split on sentence boundaries → up to 8 signals

    opt GENERATE_SF_IMAGE=true
        DEL->>DEL: offload VLM oracle to CPU
        DEL->>FLUX: generate(sf_description, 1024×1024)
        FLUX-->>DEL: PIL Image
        DEL->>DEL: save sf_image to data/images/
        DEL->>DEL: restore VLM oracle to device
    end

    DEL-->>BG: {gap_analysis, sf_description, tasks, derived_required_state, derived_failure_signals, sf_image_ref}
    BG->>DB: update_delta_contract(status=complete, gap_analysis, sf_image_ref, tasks)
    BG->>DB: update_contract(required_state, negative_indicators)

    User->>UI: Poll GET /delta-contracts/{id}
    UI-->>User: Gap analysis + inferred contract fields + SF image
```

### Memory Swap for Dual-Model Inference

When both Qwen2.5-VL (~14 GB) and FLUX.1-schnell (~12 GB) are used on the same device:

```mermaid
sequenceDiagram
    participant VLM as Oracle (device)
    participant CPU as System RAM
    participant FLUX as Flux Pipeline

    Note over VLM: oracle loaded, analyzing S0
    VLM->>CPU: oracle.model.to("cpu")
    Note over CPU: oracle in RAM, device free
    CPU->>FLUX: FluxPipeline.from_pretrained(...)
    Note over FLUX: ~12 GB on device
    FLUX->>FLUX: generate(prompt, 4 steps)
    FLUX->>CPU: gen.offload() — pipe.to("cpu")
    Note over CPU: Flux in RAM, device free
    CPU->>VLM: oracle.model.to(device)
    Note over VLM: oracle restored, device ready
```

---

## 3. LoRA Training Job

Submit labeled PASS/FAIL images to fine-tune a domain-specific LoRA adapter.

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as POST /training/jobs
    participant DB as SQLite
    participant BG as Background Task
    participant TRAIN as _train_lora()
    participant AR as Adapter Registry

    User->>UI: Fill adapter identity + hyperparams + upload images
    UI->>API: POST /training/jobs (multipart: identity + config + images)
    API->>API: Save pass_images → data/train/{job_id}/pass/
    API->>API: Save fail_images → data/train/{job_id}/fail/
    API->>DB: create_training_job(status=queued)
    API-->>UI: {job_id, status: queued}
    API->>BG: background_task(_run_training)

    BG->>DB: update_training_job(status=running)
    BG->>TRAIN: _train_lora(pass_refs, fail_refs, config, output_path)
    Note over TRAIN: Validates images exist<br/>Runs PEFT SFTTrainer loop<br/>(stub: creates output dir)
    TRAIN-->>BG: done / exception

    alt success
        BG->>AR: create_adapter(adapter_id, domain, path)
        BG->>DB: update_training_job(status=complete)
    else failure
        BG->>DB: update_training_job(status=failed, error=...)
    end

    User->>UI: Poll GET /training/jobs/{id} every 3s
    UI-->>User: Status badge update → complete / failed
```

---

## 4. Active Learning Loop

Continuous model improvement driven by low-confidence evaluations and human review.

```mermaid
flowchart TD
    EVAL[Image Evaluation<br/>POST /evaluate]
    CONF{Confidence in<br/>review band?}
    FLAG[flag_for_review]
    Q[Redis Queue]
    CELERY[process_flagged_evaluation<br/>Celery task]
    HUMAN[Human Reviewer<br/>approves / overrides]
    RATE{Success rate<br/>< threshold?}
    RETRAIN[trigger_lora_retrain<br/>Celery task]
    BOOT[bootstrap_contract_data<br/>Synthetic data generation]
    TRAIN[LoRA fine-tuning<br/>PEFT SFTTrainer]
    REGISTER[Register new adapter<br/>LoraAdapter table]
    VLLM[Hot-load adapter<br/>vLLM S-LoRA]

    EVAL --> CONF
    CONF -->|yes| FLAG
    CONF -->|no| END1[Store + return verdict]
    FLAG --> Q
    Q --> CELERY
    CELERY --> HUMAN
    HUMAN --> RATE
    RATE -->|no| END2[No action]
    RATE -->|yes| RETRAIN
    RETRAIN --> BOOT
    BOOT --> TRAIN
    TRAIN --> REGISTER
    REGISTER --> VLLM
    VLLM --> EVAL
```

**Thresholds** (configurable via `.env`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `HUMAN_REVIEW_LOWER` | 0.45 | Lower confidence band boundary |
| `HUMAN_REVIEW_UPPER` | 0.65 | Upper confidence band boundary |
| `LORA_RETRAIN_THRESHOLD` | 0.95 | Min accuracy before retrain triggers |
| `LORA_RETRAIN_WINDOW` | 100 | # of human-reviewed records to measure |

---

## 5. vLLM Juror Panel

For critical evaluations in production, three independent jurors vote:

```mermaid
flowchart LR
    IMG[Evidence Image]
    J1["Juror 1\nDomain LoRA\npositive assertion\nweight: 50%"]
    J2["Juror 2\nAdversarial LoRA\nlooks for failure signals\nweight: 20%"]
    J3["Juror 3\nBase model\nzero-shot tiebreaker\nweight: 30%"]
    VOTE["Weighted Consensus\n0.5 × J1 + 0.3 × J3 + 0.2 × J2"]
    VERDICT{Final verdict}

    IMG --> J1 & J2 & J3
    J1 & J2 & J3 --> VOTE
    VOTE --> VERDICT
    VERDICT -->|≥ strictness| PASS[✅ PASS]
    VERDICT -->|< strictness| FAIL[❌ FAIL]
```

---

## 6. Contract Registration

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as POST /contracts
    participant DB as SQLite

    User->>UI: Fill domain, name, target object, required state,\nnegative indicators, strictness coefficient,\noptional LoRA adapter ID
    UI->>API: POST /contracts (JSON)
    API->>DB: create_contract(ContractDefinition)
    DB-->>API: saved record with UUID
    API-->>UI: {id, domain, name, ...}
    UI-->>User: Contract registered — available in Evaluate tab
```
