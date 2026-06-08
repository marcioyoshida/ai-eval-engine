"""Local Flux.1 text-to-image backend via diffusers.

Requires the 'local-flux' extras:
    poetry install --extras local-flux

Model variants:
  FLUX.1-schnell — 4 steps, guidance_scale=0.0  (fast, ~12 GB VRAM)
  FLUX.1-dev     — 20 steps, guidance_scale=3.5  (higher quality, ~24 GB VRAM)

Both need a HuggingFace account with accepted model terms:
    huggingface-cli login

Note: FluxImageGenerator is NOT a singleton. Callers must instantiate it,
generate, then call .offload() to release device memory before it goes
out of scope. This keeps peak memory bounded when Flux runs alongside VLM.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)


class FluxImageGenerator:
    """Wraps FluxPipeline for local SF image synthesis.

    Lifecycle: instantiate → generate() → offload() → (let go out of scope)
    """

    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-schnell"):
        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError as e:
            raise RuntimeError(
                "SF image generation requires the 'local-flux' extras: "
                "poetry install --extras local-flux"
            ) from e

        from core.config import settings
        from core.inference import _detect_device

        self._device = _detect_device(settings.device)
        self._is_schnell = "schnell" in model_id.lower()
        dtype = torch.float32 if self._device == "cpu" else torch.float16

        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)

        if self._device == "cuda":
            # Offload model components to CPU between forward passes — handles sub-24 GB cards.
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self._device)

        logger.info(
            "FluxImageGenerator loaded %s on device=%s dtype=%s",
            model_id, self._device, dtype,
        )

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> "Image":
        import torch
        from core.config import settings

        steps = num_steps or settings.flux_num_steps or (4 if self._is_schnell else 20)
        guidance = (
            guidance_scale
            if guidance_scale is not None
            else (
                settings.flux_guidance_scale
                if settings.flux_guidance_scale is not None
                else (0.0 if self._is_schnell else 3.5)
            )
        )

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )

        return result.images[0]

    def offload(self) -> None:
        """Move pipeline to CPU and release device cache. Call after generate()."""
        import torch
        try:
            self.pipe.to("cpu")
        except Exception:
            pass
        if self._device == "mps":
            torch.mps.empty_cache()
        elif self._device == "cuda":
            torch.cuda.empty_cache()
        logger.debug("FluxImageGenerator offloaded from %s", self._device)
