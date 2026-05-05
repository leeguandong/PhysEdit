# PhysEdit: Spatially-Adaptive Temporal Reasoning Modules
# Three core innovations: SRM (Spatial Reasoning Mask), CARD (Complexity-Adaptive Reasoning Depth),
# RPFI (Region-Preserving Feature Injection)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ReasoningConfig:
    """Adaptive reasoning configuration predicted by CARD."""
    num_reasoning_steps: int  # N_r
    reasoning_token_length: int  # r
    complexity_level: str  # "low", "medium", "high"
    complexity_probs: torch.Tensor  # softmax probabilities


# =============================================================================
# Component 1: Spatial Reasoning Mask (SRM)
# =============================================================================

class SpatialReasoningMask(nn.Module):
    """
    Generates a spatial mask indicating which regions require temporal reasoning,
    based on cross-attention maps between editing instructions and image features.

    The mask is computed during a pilot forward pass at the noisiest timestep.
    For local edits (e.g., "change hat color"), the mask concentrates on the
    relevant region. For global edits (e.g., "change style"), it covers the
    entire frame.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        blur_kernel_size: int = 5,
        extraction_layer: int = 12,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.blur_kernel_size = blur_kernel_size
        self.extraction_layer = extraction_layer
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    @torch.no_grad()
    def extract_cross_attention_map(
        self,
        transformer: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        spatial_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Extract cross-attention maps from the specified transformer layer.

        This hooks into the transformer's cross-attention computation at
        layer `self.extraction_layer` to capture the attention weights
        between text tokens and spatial positions.

        Args:
            transformer: The ChronoEdit transformer model.
            hidden_states: Input latent [B, C, T, H, W].
            encoder_hidden_states: Text embeddings [B, L, D].
            timestep: Current timestep.
            encoder_hidden_states_image: Optional image embeddings.
            spatial_shape: (T', H', W') post-patch spatial dimensions.

        Returns:
            Aggregated attention map [B, 1, H', W'] normalized to [0, 1].
        """
        attn_maps = []

        def hook_fn(module, input, output):
            # Capture the cross-attention weights
            # input[0] is hidden_states, input[1] is encoder_hidden_states
            query = module.to_q(input[0])
            key = module.to_k(input[1] if len(input) > 1 and input[1] is not None else input[0])

            if module.norm_q is not None:
                query = module.norm_q(query)
            if module.norm_k is not None:
                key = module.norm_k(key)

            query = query.unflatten(2, (module.heads, -1)).transpose(1, 2)
            key = key.unflatten(2, (module.heads, -1)).transpose(1, 2)

            # Compute attention scores (not softmax, just raw scores)
            scale = query.shape[-1] ** -0.5
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)  # [B, heads, spatial, text]

            # Average over heads, then over text tokens
            attn_map = attn_weights.mean(dim=1).mean(dim=-1)  # [B, spatial]
            attn_maps.append(attn_map)

        # Register hook on the target cross-attention layer
        target_block = transformer.blocks[self.extraction_layer]
        hook_handle = target_block.attn2.register_forward_hook(hook_fn)

        try:
            # Run a single forward pass
            transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=False,
            )
        finally:
            hook_handle.remove()

        if not attn_maps:
            # Fallback: return uniform mask
            if spatial_shape is not None:
                T, H, W = spatial_shape
                return torch.ones(hidden_states.shape[0], 1, H, W, device=hidden_states.device)
            return None

        attn_map = attn_maps[0]  # [B, spatial_tokens]
        return attn_map

    def generate_mask(
        self,
        attn_map: torch.Tensor,
        spatial_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Convert raw attention map to a smooth spatial mask.

        Args:
            attn_map: Raw attention [B, T*H*W].
            spatial_shape: (T', H', W') post-patch dimensions.

        Returns:
            Smooth mask [B, 1, H', W'] in [0, 1].
        """
        T, H, W = spatial_shape
        B = attn_map.shape[0]

        # Reshape to spatial dimensions, take the first temporal frame
        attn_3d = attn_map.reshape(B, T, H, W)
        attn_2d = attn_3d.mean(dim=1)  # Average over temporal dimension [B, H, W]

        # Adaptive thresholding (Eq. 3 in paper)
        mu = attn_2d.mean(dim=(-2, -1), keepdim=True)
        mask_raw = torch.sigmoid((attn_2d - mu) / self.temperature)

        # Ensure minimum and maximum mask ratios
        mask_ratio = mask_raw.mean(dim=(-2, -1), keepdim=True)
        if mask_ratio.min() < self.min_mask_ratio:
            # Boost the mask if it's too sparse
            threshold = torch.quantile(
                mask_raw.flatten(1), 1.0 - self.min_mask_ratio, dim=1, keepdim=True
            ).unsqueeze(-1)
            mask_raw = torch.where(mask_raw >= threshold, mask_raw, mask_raw * 0.5 + 0.5)

        # Gaussian blur for smooth boundaries
        mask = mask_raw.unsqueeze(1)  # [B, 1, H, W]
        if self.blur_kernel_size > 1:
            padding = self.blur_kernel_size // 2
            # Create Gaussian kernel
            sigma = self.blur_kernel_size / 3.0
            x = torch.arange(self.blur_kernel_size, device=mask.device, dtype=mask.dtype) - padding
            kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

            mask = F.pad(mask, [padding] * 4, mode='reflect')
            mask = F.conv2d(mask, kernel_2d)

        # Normalize to [0, 1]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        return mask


# =============================================================================
# Component 2: Complexity-Adaptive Reasoning Depth (CARD)
# =============================================================================

class ComplexityAdaptiveReasoningDepth(nn.Module):
    """
    Predicts edit complexity and dynamically determines the optimal
    reasoning configuration (N_r, r) for each editing instruction.

    Uses CLIP text and image embeddings to classify edits into
    low/medium/high complexity, then interpolates reasoning parameters.
    """

    # Default reasoning configurations per complexity level
    COMPLEXITY_CONFIGS = {
        "low": {"num_reasoning_steps": 3, "reasoning_token_length": 2},
        "medium": {"num_reasoning_steps": 8, "reasoning_token_length": 4},
        "high": {"num_reasoning_steps": 15, "reasoning_token_length": 8},
    }

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 3,
        configs: Optional[dict] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.configs = configs or self.COMPLEXITY_CONFIGS

        # Lightweight classifier: [text_emb; image_emb; text_emb * image_emb] -> 3 classes
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        self._class_names = ["low", "medium", "high"]

    def forward(
        self,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> ReasoningConfig:
        """
        Predict edit complexity and return adaptive reasoning configuration.

        Args:
            text_embedding: CLIP text embedding [B, D] or [D].
            image_embedding: CLIP image embedding [B, D] or [D].

        Returns:
            ReasoningConfig with adaptive N_r, r values.
        """
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
        if image_embedding.dim() == 1:
            image_embedding = image_embedding.unsqueeze(0)

        # Fuse embeddings: concat + element-wise product
        fused = torch.cat([
            text_embedding,
            image_embedding,
            text_embedding * image_embedding,
        ], dim=-1)

        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)  # [B, 3]

        # Soft interpolation of reasoning parameters (Eq. 6 in paper)
        nr_values = torch.tensor([
            self.configs["low"]["num_reasoning_steps"],
            self.configs["medium"]["num_reasoning_steps"],
            self.configs["high"]["num_reasoning_steps"],
        ], dtype=probs.dtype, device=probs.device)

        r_values = torch.tensor([
            self.configs["low"]["reasoning_token_length"],
            self.configs["medium"]["reasoning_token_length"],
            self.configs["high"]["reasoning_token_length"],
        ], dtype=probs.dtype, device=probs.device)

        nr_star = (probs * nr_values).sum(dim=-1)
        r_star = (probs * r_values).sum(dim=-1)

        # Round to integers
        nr_int = max(1, round(nr_star.item()))
        r_int = max(1, round(r_star.item()))

        # Determine complexity level
        pred_class = probs.argmax(dim=-1).item()
        complexity_level = self._class_names[pred_class]

        return ReasoningConfig(
            num_reasoning_steps=nr_int,
            reasoning_token_length=r_int,
            complexity_level=complexity_level,
            complexity_probs=probs.squeeze(0),
        )

    @classmethod
    def from_heuristic(cls, instruction: str) -> ReasoningConfig:
        """
        Rule-based fallback when CLIP embeddings are unavailable.
        Classifies based on instruction keywords.
        """
        instruction_lower = instruction.lower()

        # High complexity: physical actions, motion
        high_keywords = [
            "pick up", "picks up", "picking up",
            "grab", "throw", "push", "pull", "lift", "drop",
            "walk", "run", "jump", "dance", "drive", "fly", "swim",
            "open", "close", "fold", "unfold", "rotate", "flip",
            "superhero", "pose", "stance", "gesture", "action",
            "manipulate", "interact", "hand over", "catch",
        ]

        # Medium complexity: structural changes
        medium_keywords = [
            "add", "remove", "replace", "delete", "insert",
            "change background", "change the background",
            "put", "place", "move", "reposition",
            "extract", "resize", "crop", "expand",
            "set in", "set to",
        ]

        # Check keywords
        for kw in high_keywords:
            if kw in instruction_lower:
                return ReasoningConfig(
                    num_reasoning_steps=15,
                    reasoning_token_length=8,
                    complexity_level="high",
                    complexity_probs=torch.tensor([0.05, 0.15, 0.80]),
                )

        for kw in medium_keywords:
            if kw in instruction_lower:
                return ReasoningConfig(
                    num_reasoning_steps=8,
                    reasoning_token_length=4,
                    complexity_level="medium",
                    complexity_probs=torch.tensor([0.10, 0.75, 0.15]),
                )

        # Default: low complexity (color, style, texture changes)
        return ReasoningConfig(
            num_reasoning_steps=3,
            reasoning_token_length=2,
            complexity_level="low",
            complexity_probs=torch.tensor([0.80, 0.15, 0.05]),
        )


# =============================================================================
# Component 3: Region-Preserving Feature Injection (RPFI)
# =============================================================================

class RegionPreservingFeatureInjection(nn.Module):
    """
    Injects clean reference features into non-edited spatial regions
    during the temporal reasoning stage, preventing identity drift.

    At each reasoning timestep, masked regions receive the original
    reference features at the corresponding noise level, while
    unmasked (edited) regions undergo normal denoising.
    """

    def __init__(
        self,
        relaxation_factor: float = 1.5,
    ):
        super().__init__()
        self.relaxation_factor = relaxation_factor

    def compute_reference_at_noise_level(
        self,
        clean_latent: torch.Tensor,
        noise: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        """
        Compute the reference latent at a given noise level using
        the flow matching interpolation.

        Args:
            clean_latent: Clean reference latent z_c [B, C, h, w].
            noise: Reference noise (same seed as main noise) [B, C, h, w].
            timestep: Current timestep t in [0, 1].

        Returns:
            Noisy reference latent at level t.
        """
        return (1 - timestep) * clean_latent + timestep * noise

    def compute_effective_mask(
        self,
        spatial_mask: torch.Tensor,
        current_step: int,
        total_reasoning_steps: int,
    ) -> torch.Tensor:
        """
        Compute the effective mask with gradual relaxation.

        Early steps use strong preservation (mask close to 1 everywhere),
        gradually relaxing to the actual spatial mask.

        Args:
            spatial_mask: SRM mask [B, 1, H, W] in [0, 1].
            current_step: Current reasoning step index.
            total_reasoning_steps: Total N_r steps.

        Returns:
            Effective mask [B, 1, H, W].
        """
        alpha = min(1.0, (current_step / total_reasoning_steps) * self.relaxation_factor)
        effective_mask = alpha * spatial_mask + (1 - alpha) * torch.ones_like(spatial_mask)
        return effective_mask

    def inject(
        self,
        latents: torch.Tensor,
        clean_reference: torch.Tensor,
        reference_noise: torch.Tensor,
        spatial_mask: torch.Tensor,
        timestep: float,
        current_step: int,
        total_reasoning_steps: int,
    ) -> torch.Tensor:
        """
        Perform region-preserving feature injection.

        Args:
            latents: Current denoised latents [B, C, T, H, W] (includes reasoning tokens).
            clean_reference: Clean reference latent [B, C, 1, H, W].
            reference_noise: Noise for reference (consistent seed) [B, C, 1, H, W].
            spatial_mask: SRM mask [B, 1, H, W], 1=edit region, 0=preserve region.
            timestep: Current noise level t.
            current_step: Current reasoning step.
            total_reasoning_steps: Total N_r.

        Returns:
            Injected latents [B, C, T, H, W].
        """
        B, C, T, H, W = latents.shape

        # Compute reference at current noise level
        ref_noisy = self.compute_reference_at_noise_level(
            clean_reference.squeeze(2), reference_noise.squeeze(2), timestep
        )  # [B, C, H, W]

        # Compute effective mask with relaxation
        effective_mask = self.compute_effective_mask(
            spatial_mask, current_step, total_reasoning_steps
        )  # [B, 1, H, W]

        # Apply injection to all temporal frames except the first (reference frame)
        # The first frame (z_c) should already be clean
        injected = latents.clone()
        for t_idx in range(1, T):  # Skip the reference frame at index 0
            frame = latents[:, :, t_idx, :, :]  # [B, C, H, W]

            # Blend: edited regions keep denoised content, preserved regions get reference
            injected_frame = (
                effective_mask * frame +
                (1 - effective_mask) * ref_noisy
            )
            injected[:, :, t_idx, :, :] = injected_frame

        return injected


# =============================================================================
# Unified PhysEdit Controller
# =============================================================================

class PhysEditController:
    """
    Orchestrates all three components (SRM, CARD, RPFI) during inference.

    Usage:
        controller = PhysEditController()
        reasoning_config = controller.predict_complexity(instruction, image)
        spatial_mask = controller.compute_spatial_mask(transformer, latents, text_emb, ...)
        latents = controller.apply_rpfi(latents, ref_latent, noise, mask, t, step, Nr)
    """

    def __init__(
        self,
        srm_temperature: float = 0.1,
        srm_blur_kernel: int = 5,
        srm_layer: int = 12,
        rpfi_relaxation: float = 1.5,
        card_embed_dim: int = 768,
        use_heuristic_card: bool = True,
    ):
        self.srm = SpatialReasoningMask(
            temperature=srm_temperature,
            blur_kernel_size=srm_blur_kernel,
            extraction_layer=srm_layer,
        )
        self.rpfi = RegionPreservingFeatureInjection(
            relaxation_factor=rpfi_relaxation,
        )
        self.use_heuristic_card = use_heuristic_card

        if not use_heuristic_card:
            self.card = ComplexityAdaptiveReasoningDepth(embed_dim=card_embed_dim)
        else:
            self.card = None

    def predict_complexity(
        self,
        instruction: str,
        text_embedding: Optional[torch.Tensor] = None,
        image_embedding: Optional[torch.Tensor] = None,
    ) -> ReasoningConfig:
        """Predict edit complexity and return reasoning config."""
        if self.use_heuristic_card or text_embedding is None or image_embedding is None:
            return ComplexityAdaptiveReasoningDepth.from_heuristic(instruction)
        return self.card(text_embedding, image_embedding)

    def compute_spatial_mask(
        self,
        transformer: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        spatial_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """Compute spatial reasoning mask via cross-attention extraction."""
        attn_map = self.srm.extract_cross_attention_map(
            transformer, hidden_states, encoder_hidden_states,
            timestep, encoder_hidden_states_image, spatial_shape,
        )
        if attn_map is None or spatial_shape is None:
            return torch.ones(
                hidden_states.shape[0], 1,
                spatial_shape[1] if spatial_shape else 1,
                spatial_shape[2] if spatial_shape else 1,
                device=hidden_states.device,
            )
        return self.srm.generate_mask(attn_map, spatial_shape)

    def apply_rpfi(
        self,
        latents: torch.Tensor,
        clean_reference: torch.Tensor,
        reference_noise: torch.Tensor,
        spatial_mask: torch.Tensor,
        timestep: float,
        current_step: int,
        total_reasoning_steps: int,
    ) -> torch.Tensor:
        """Apply region-preserving feature injection."""
        return self.rpfi.inject(
            latents, clean_reference, reference_noise,
            spatial_mask, timestep, current_step, total_reasoning_steps,
        )
