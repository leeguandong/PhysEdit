"""
PhysEdit Python API

Provides a simple Python interface for PhysEdit image editing.

Usage:
    from api import PhysEditAPI

    api = PhysEditAPI(model_path="checkpoints/ChronoEdit-14B-Diffusers")
    result = api.edit(image="input.jpg", prompt="Change the hat to red")
    result.save("output.jpg")
"""

import sys
import os
import numpy as np
import torch
from PIL import Image
from typing import Optional, Union
from pathlib import Path


class PhysEditAPI:
    """Python API for PhysEdit image editing."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        offload: bool = False,
    ):
        """
        Initialize the PhysEdit pipeline.

        Args:
            model_path: Path to ChronoEdit-14B-Diffusers checkpoint.
            device: Device to use ("cuda" or "cpu").
            offload: Enable model CPU offloading for lower VRAM.
        """
        self.device = device
        self.offload = offload

        # Add the ChronoEdit source to path
        chronoedit_root = os.environ.get(
            "CHRONOEDIT_ROOT",
            os.path.join(os.path.dirname(__file__), "..", "..", "common", "ChronoEdit")
        )
        if chronoedit_root not in sys.path:
            sys.path.insert(0, chronoedit_root)

        self._load_pipeline(model_path)

    def _load_pipeline(self, model_path: str):
        """Load the PhysEdit pipeline."""
        from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
        from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, UMT5EncoderModel
        from chronoedit_diffusers.transformer_chronoedit import ChronoEditTransformer3DModel
        from chronoedit_diffusers.pipeline_physedit import PhysEditPipeline

        vae = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.float32
        )
        image_encoder = CLIPVisionModel.from_pretrained(
            model_path, subfolder="image_encoder", torch_dtype=torch.float32
        )
        transformer = ChronoEditTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        image_processor = CLIPImageProcessor.from_pretrained(
            model_path, subfolder="image_processor"
        )

        self.pipe = PhysEditPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            image_processor=image_processor,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

        if self.offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

    def edit(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int = 42,
        enable_physedit: bool = True,
        srm_temperature: float = 0.1,
        rpfi_relaxation: float = 1.5,
        rpfi_enabled: bool = True,
        card_override_nr: Optional[int] = None,
        card_override_r: Optional[int] = None,
    ) -> Image.Image:
        """
        Edit an image using PhysEdit.

        Args:
            image: Input image path or PIL Image.
            prompt: Editing instruction.
            negative_prompt: Negative prompt for guidance.
            num_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            enable_physedit: Enable PhysEdit features.
            srm_temperature: SRM mask sharpness (lower = sharper).
            rpfi_relaxation: RPFI boundary relaxation factor.
            rpfi_enabled: Enable RPFI feature injection.
            card_override_nr: Override CARD reasoning steps.
            card_override_r: Override CARD reasoning token length.

        Returns:
            Edited PIL Image.
        """
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Calculate dimensions
        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))

        # Run pipeline
        generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

        output = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=5,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            enable_physedit=enable_physedit,
            srm_temperature=srm_temperature,
            rpfi_relaxation=rpfi_relaxation,
            rpfi_enabled=rpfi_enabled,
            card_override_nr=card_override_nr,
            card_override_r=card_override_r,
            offload_model=self.offload,
        )

        # Extract the edited frame (last frame)
        frames = output.frames[0]
        if isinstance(frames, np.ndarray):
            edited = frames[-1] if frames.ndim == 4 else frames
            if edited.dtype in (np.float32, np.float64):
                edited = (edited * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(edited)
        elif isinstance(frames, list):
            return frames[-1] if isinstance(frames[-1], Image.Image) else Image.fromarray(frames[-1])
        return frames
