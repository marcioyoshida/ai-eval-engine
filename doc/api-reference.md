# API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Contracts

### `POST /contracts`

Register a new visual assertion contract.

**Body** (JSON):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `domain` | string | ✓ | Namespace (e.g. `municipal_infrastructure`) |
| `name` | string | ✓ | Human label (e.g. `Pothole Repair`) |
| `target_object` | string | ✓ | Object being inspected (e.g. `pothole`) |
| `required_state` | string | ✓ | Plain-English description of passing state |
| `negative_indicators` | string[] | ✓ | List of failure signals |
| `strictness_coefficient` | float | | Confidence threshold 0.0–1.0 (default 0.80) |
| `lora_id` | string | | Adapter ID to use with vLLM backend |

**Response** `201`:
```json
{
  "id": "uuid",
  "domain": "municipal_infrastructure",
  "name": "Pothole Repair",
  "target_object": "pothole",
  "required_state": "fully filled with asphalt, level with road surface",
  "negative_indicators": ["loose gravel", "exposed base layer"],
  "strictness_coefficient": 0.85,
  "lora_id": null,
  "created_at": "2026-06-07T12:00:00Z",
  "active": true
}
```

---

### `GET /contracts`

List all active contracts.

**Response** `200`: `ContractDefinition[]`

---

### `GET /contracts/{domain}`

Filter contracts by domain.

**Response** `200`: `ContractDefinition[]`

---

## Delta Contracts

### `POST /delta-contracts`

Upload an S0 (current state) image. Triggers VLM gap analysis and optional SF image generation in the background.

**Body** (multipart/form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `s0_image` | file | ✓ | Current-state image |
| `domain` | string | ✓ | Contract domain |
| `target_object` | string | ✓ | Object to analyze |
| `name` | string | ✓ | Contract name |
| `strictness_coefficient` | float | | Default 0.80 |
| `lora_id` | string | | Optional LoRA adapter |

**Response** `202`:
```json
{
  "delta_id": "uuid",
  "contract_id": "uuid",
  "status": "pending"
}
```

The analysis runs asynchronously. Poll `GET /delta-contracts/{delta_id}` until `status == "complete"`.

---

### `GET /delta-contracts/{delta_id}`

Retrieve analysis results.

**Response** `200`:
```json
{
  "id": "uuid",
  "contract_id": "uuid",
  "s0_image_ref": "s0_abc123.jpg",
  "sf_image_ref": "sf_abc123.png",
  "gap_analysis": "The pothole has exposed aggregate and loose gravel...",
  "sf_description": "A smooth, dark asphalt patch level with surrounding road...",
  "tasks": [
    {
      "action": "Clean pothole cavity",
      "materials": ["blower", "broom"],
      "tools": ["air compressor"],
      "acceptance_criteria": "No loose debris or standing water"
    }
  ],
  "required_state": "smooth asphalt fill, level with surrounding road...",
  "negative_indicators": ["loose gravel visible", "exposed base layer", "water pooling"],
  "generation_status": "complete"
}
```

---

### `GET /delta-contracts`

List all completed delta contracts. Used by the UI for autocomplete suggestions.

**Response** `200`: `DeltaSummary[]` with `gap_analysis`, `sf_description`, `domain` fields.

---

## Evaluation

### `POST /evaluate`

Evaluate an image against a contract.

**Body** (multipart/form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `contract_id` | string | ✓ | Contract UUID |
| `file` | file | * | Image file upload |
| `image_url` | string | * | Remote image URL |

*Either `file` or `image_url` required.

**Response** `200`:
```json
{
  "passed": true,
  "confidence": 0.87,
  "rationale": "The pothole shows complete fill with dark asphalt, surface level with surrounding road. No loose aggregate visible.",
  "routed_to_human": false,
  "thinking": {
    "observations": ["Dark asphalt patch observed", "Surface appears level"],
    "positive_evidence": ["Uniform asphalt color", "No visible voids"],
    "negative_evidence": [],
    "reasoning": "All required state criteria appear met with high confidence."
  }
}
```

**`routed_to_human: true`** is returned when confidence falls in `[HUMAN_REVIEW_LOWER, HUMAN_REVIEW_UPPER]`.

---

## LoRA Adapters

### `GET /adapters`

List all registered LoRA adapters.

**Response** `200`:
```json
[
  {
    "id": "uuid",
    "name": "Pothole Detector v1",
    "adapter_id": "pothole-detector-v1",
    "domain": "municipal_infrastructure",
    "category": "road_surface",
    "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "adapter_path": "adapters/pothole-detector-v1",
    "status": "active",
    "created_at": "2026-06-07T12:00:00Z"
  }
]
```

---

### `POST /adapters`

Register an existing LoRA adapter (manually, without training).

**Body** (JSON or form): `name`, `adapter_id`, `domain`, `category?`, `base_model?`, `adapter_path?`, `notes?`

**Response** `201`: `LoraAdapter`

---

## Training Jobs

### `POST /training/jobs`

Submit a LoRA fine-tuning job.

**Body** (multipart/form-data):

| Group | Field | Type | Default | Description |
|-------|-------|------|---------|-------------|
| Identity | `adapter_name` | string | ✓ | Display name |
| | `adapter_id` | string | ✓ | Slug / vLLM model name |
| | `domain` | string | ✓ | Domain namespace |
| | `category` | string | | Sub-domain |
| | `base_model` | string | `Qwen/Qwen2.5-VL-7B-Instruct` | Base checkpoint |
| | `contract_id` | string | | Link to contract |
| | `notes` | string | | Free text |
| LoRA | `lora_rank` | int | `16` | Adapter rank r |
| | `lora_alpha` | int | `rank × 2` | Scaling factor |
| | `lora_dropout` | float | `0.05` | Regularization |
| | `target_modules` | string | `q_proj,v_proj` | Comma-separated |
| Hyperparams | `learning_rate` | float | `2e-4` | |
| | `num_epochs` | int | `3` | |
| | `batch_size` | int | `2` | |
| | `max_seq_length` | int | `2048` | |
| | `gradient_accumulation_steps` | int | `4` | |
| | `warmup_steps` | int | `10` | |
| | `save_steps` | int | `100` | |
| | `output_path` | string | `adapters/{adapter_id}` | Output directory |
| Images | `pass_images` | file[] | | PASS-labeled images |
| | `fail_images` | file[] | | FAIL-labeled images |

**Response** `201`:
```json
{
  "id": "uuid",
  "adapter_name": "Pothole Detector v1",
  "adapter_id": "pothole-detector-v1",
  "domain": "municipal_infrastructure",
  "status": "queued",
  "pass_count": 12,
  "fail_count": 8,
  "config": { "lora_rank": 16, "lora_alpha": 32, ... },
  "output_path": "adapters/pothole-detector-v1",
  "error": null,
  "created_at": "2026-06-07T12:00:00Z"
}
```

---

### `GET /training/jobs`

List all training jobs, newest first.

**Response** `200`: `TrainingJobResponse[]`

---

### `GET /training/jobs/{job_id}`

Get a single job. Poll this endpoint to track progress.

**Status values**: `queued` → `running` → `complete` / `failed`

**Response** `200`: `TrainingJobResponse`

---

## Utility

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /images/{filename}`

Serve images from `data/images/`. Used to display S0 and SF images in the UI.

### `GET /`

Serve the SPA (`static/index.html`).
