from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./eval_engine.db"
    redis_url: str = "redis://localhost:6379/0"

    # "local" uses HuggingFace transformers directly; "vllm" calls a running vLLM server
    inference_backend: str = "local"

    # vLLM server config (used when inference_backend=vllm)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "token"
    vllm_default_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Local model config (used when inference_backend=local)
    local_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Compute device: "auto" probes MPS → CUDA → CPU in that order.
    # Override with "mps", "cuda", or "cpu" to force a specific backend.
    device: str = "auto"

    # Confidence band that routes to human review queue
    human_review_lower: float = 0.45
    human_review_upper: float = 0.65

    # LoRA adapter registry success rate threshold; below this triggers auto-retrain
    lora_retrain_threshold: float = 0.95
    lora_retrain_window: int = 100

    # SF image generation via local Flux.1 (requires local-flux extras)
    generate_sf_image: bool = True
    flux_model_id: str = "black-forest-labs/FLUX.1-schnell"
    flux_num_steps: int | None = None       # None = auto (4 for schnell, 20 for dev)
    flux_guidance_scale: float | None = None  # None = auto (0.0 for schnell, 3.5 for dev)
    flux_image_width: int = 1024
    flux_image_height: int = 1024


settings = Settings()
