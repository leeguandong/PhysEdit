# PhysEdit Pipeline
# Extended ChronoEdit pipeline with Spatial Reasoning Mask (SRM),
# Complexity-Adaptive Reasoning Depth (CARD), and Region-Preserving Feature Injection (RPFI)

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Union

import PIL
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import logging
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from chronoedit_diffusers.pipeline_chronoedit import ChronoEditPipeline
from chronoedit_diffusers.physedit_modules import (
    PhysEditController,
    ReasoningConfig,
)
from chronoedit._ext.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from chronoedit._ext.imaginaire.utils import log

logger = logging.get_logger(__name__)


class PhysEditPipeline(ChronoEditPipeline):
    """
    Extended ChronoEdit pipeline with spatially-adaptive temporal reasoning.

    Adds three innovations:
    1. Spatial Reasoning Mask (SRM): identifies which regions need reasoning
    2. Complexity-Adaptive Reasoning Depth (CARD): dynamically adjusts N_r and r
    3. Region-Preserving Feature Injection (RPFI): prevents identity drift

    Usage:
        pipe = PhysEditPipeline.from_pretrained(...)
        output = pipe(
            image=image,
            prompt="change the hat to red",
            enable_physedit=True,
            # Optionally override:
            # srm_temperature=0.1,
            # rpfi_relaxation=1.5,
            # card_mode="heuristic",  # or "learned"
        )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the PhysEdit controller
        self.pe_controller = PhysEditController(
            srm_temperature=0.1,
            srm_blur_kernel=5,
            srm_layer=min(12, len(self.transformer.blocks) - 1),
            rpfi_relaxation=1.5,
            use_heuristic_card=True,
        )

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 5,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # === PhysEdit parameters ===
        enable_physedit: bool = True,
        enable_temporal_reasoning: bool = False,
        num_temporal_reasoning_steps: int = 0,
        # SRM parameters
        srm_temperature: Optional[float] = None,
        srm_layer: Optional[int] = None,
        # CARD parameters
        card_mode: str = "heuristic",  # "heuristic" or "learned"
        card_override_nr: Optional[int] = None,
        card_override_r: Optional[int] = None,
        # RPFI parameters
        rpfi_enabled: bool = True,
        rpfi_relaxation: Optional[float] = None,
        offload_model: bool = False,
    ):
        """
        Extended call with PhysEdit features.

        When `enable_physedit=True`:
        - CARD predicts optimal (N_r, r) from the instruction
        - SRM computes spatial reasoning mask
        - RPFI injects clean features during reasoning

        When `enable_physedit=False`, falls back to original ChronoEdit behavior.
        """

        # Fallback to original ChronoEdit if PhysEdit is disabled
        if not enable_physedit:
            return super().__call__(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image_embeds=image_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                enable_temporal_reasoning=enable_temporal_reasoning,
                num_temporal_reasoning_steps=num_temporal_reasoning_steps,
                offload_model=offload_model,
            )

        # Update controller parameters if overridden
        if srm_temperature is not None:
            self.pe_controller.srm.temperature = srm_temperature
        if srm_layer is not None:
            self.pe_controller.srm.extraction_layer = srm_layer
        if rpfi_relaxation is not None:
            self.pe_controller.rpfi.relaxation_factor = rpfi_relaxation

        # =====================================================================
        # Stage 0: CARD - Predict complexity and reasoning configuration
        # =====================================================================
        prompt_str = prompt if isinstance(prompt, str) else prompt[0]

        if card_override_nr is not None and card_override_r is not None:
            reasoning_config = ReasoningConfig(
                num_reasoning_steps=card_override_nr,
                reasoning_token_length=card_override_r,
                complexity_level="custom",
                complexity_probs=torch.tensor([0.0, 0.0, 0.0]),
            )
        else:
            reasoning_config = self.pe_controller.predict_complexity(prompt_str)

        Nr = reasoning_config.num_reasoning_steps
        r = reasoning_config.reasoning_token_length

        logger.info(
            f"[PhysEdit] CARD prediction: complexity={reasoning_config.complexity_level}, "
            f"N_r={Nr}, r={r}"
        )

        # =====================================================================
        # Standard pipeline setup (from ChronoEditPipeline)
        # =====================================================================
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt, negative_prompt, image, height, width,
            prompt_embeds, negative_prompt_embeds, image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # Guardrail check
        if self.text_guardrail_runner is not None:
            if not guardrail_presets.run_text_guardrail(prompt_str, self.text_guardrail_runner):
                raise Exception(f"Guardrail blocked: {prompt_str}")

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if offload_model:
            self.text_encoder.cpu()

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        if image_embeds is None:
            image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1).to(transformer_dtype)

        if offload_model:
            self.image_encoder.cpu()

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latents
        num_channels_latents = self.vae.config.z_dim
        image_tensor = self.video_processor.preprocess(
            image, height=height, width=width
        ).to(device, dtype=torch.bfloat16)

        latents, condition = self.prepare_latents(
            image_tensor, batch_size * num_videos_per_prompt,
            num_channels_latents, height, width, num_frames,
            torch.bfloat16, device, generator, latents,
        )

        # Store reference latent and noise for RPFI
        reference_latent = condition[:, :num_channels_latents, :1, :, :].clone()
        reference_noise = latents[:, :, :1, :, :].clone()

        # =====================================================================
        # Stage 1: SRM - Compute spatial reasoning mask (pilot pass)
        # =====================================================================
        p_t, p_h, p_w = self.transformer.config.patch_size
        _, _, T_lat, H_lat, W_lat = latents.shape
        spatial_shape = (T_lat // p_t, H_lat // p_h, W_lat // p_w)

        # Use first timestep for pilot pass
        pilot_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
        pilot_timestep = timesteps[0].expand(latents.shape[0])

        spatial_mask = self.pe_controller.compute_spatial_mask(
            self.transformer, pilot_input, prompt_embeds,
            pilot_timestep, image_embeds, spatial_shape,
        )

        # Resize mask to latent spatial dimensions
        spatial_mask = torch.nn.functional.interpolate(
            spatial_mask, size=(H_lat, W_lat), mode='bilinear', align_corners=False
        )

        mask_coverage = spatial_mask.mean().item()
        logger.info(
            f"[PhysEdit] SRM mask coverage: {mask_coverage:.2%}"
        )

        # =====================================================================
        # Stage 2: Denoising loop with SRM + RPFI
        # =====================================================================
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if offload_model:
            torch.cuda.empty_cache()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # --- Reasoning token drop (same as ChronoEdit) ---
                if i == Nr and Nr > 0:
                    latents = latents[:, :, [0, -1]]
                    condition = condition[:, :, [0, -1]]

                    for j in range(len(self.scheduler.model_outputs)):
                        if self.scheduler.model_outputs[j] is not None:
                            if latents.shape[-3] != self.scheduler.model_outputs[j].shape[-3]:
                                self.scheduler.model_outputs[j] = self.scheduler.model_outputs[j][:, :, [0, -1]]
                    if self.scheduler.last_sample is not None:
                        self.scheduler.last_sample = self.scheduler.last_sample[:, :, [0, -1]]

                    logger.info(f"[PhysEdit] Dropped reasoning tokens at step {i}")

                # --- Forward pass ---
                self._current_timestep = t
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if offload_model:
                    torch.cuda.empty_cache()

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # ODE step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # --- RPFI: Region-Preserving Feature Injection ---
                if rpfi_enabled and i < Nr and Nr > 0:
                    t_float = t.item() / 1000.0 if t.item() > 1 else t.item()
                    latents = self.pe_controller.apply_rpfi(
                        latents=latents,
                        clean_reference=reference_latent,
                        reference_noise=reference_noise,
                        spatial_mask=spatial_mask,
                        timestep=t_float,
                        current_step=i,
                        total_reasoning_steps=Nr,
                    )

                # Callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # =====================================================================
        # Stage 3: Decode
        # =====================================================================
        if offload_model:
            self.transformer.cpu()
            torch.cuda.empty_cache()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

            video = self.vae.decode(latents, return_dict=False)[0]

            # Video guardrail
            if self.video_guardrail_runner is not None:
                for idx in range(video.shape[0]):
                    frames = ((video[idx] + 1) * 127.5).clamp(0.0, 255.0).to(torch.uint8)
                    frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
                    processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                    if processed_frames is None:
                        raise Exception("Guardrail blocked generation.")
                    processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 127.5 - 1.0
                    video[idx] = processed_video.to(video.device, dtype=video.dtype)

            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
