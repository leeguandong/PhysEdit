#!/usr/bin/env python3
"""
PhysEdit: Spatially-Adaptive Temporal Reasoning for Region-Aware Image Editing

CLI entry point for running PhysEdit inference.

Usage:
    python physedit.py -i input.jpg -p "Change the hat to red" -o output.jpg \
        --model-path checkpoints/ChronoEdit-14B-Diffusers
"""

import argparse
import time
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from api import PhysEditAPI


def main():
    parser = argparse.ArgumentParser(
        description="PhysEdit: Spatially-Adaptive Temporal Reasoning for Image Editing"
    )

    # Required
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-p", "--prompt", required=True, help="Editing instruction")
    parser.add_argument("--model-path", required=True, help="Path to ChronoEdit-14B-Diffusers")

    # Output
    parser.add_argument("-o", "--output", default="output.jpg", help="Output image path")

    # Generation
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # SRM
    parser.add_argument("--srm-temperature", type=float, default=0.1, help="SRM mask sharpness")

    # CARD
    parser.add_argument("--card-nr", type=int, default=None, help="Override CARD Nr")
    parser.add_argument("--card-r", type=int, default=None, help="Override CARD r")

    # RPFI
    parser.add_argument("--rpfi-relaxation", type=float, default=1.5, help="RPFI relaxation factor")
    parser.add_argument("--disable-rpfi", action="store_true", help="Disable RPFI")

    # PhysEdit toggle
    parser.add_argument("--disable-sce", action="store_true", help="Disable PhysEdit")

    # Hardware
    parser.add_argument("--offload", action="store_true", help="CPU offload for lower VRAM")

    args = parser.parse_args()

    # Initialize
    print(f"Loading model from {args.model_path}...")
    api = PhysEditAPI(
        model_path=args.model_path,
        offload=args.offload,
    )

    # Run edit
    print(f"Editing: '{args.prompt}'")
    print(f"Input: {args.input}")

    start = time.time()
    result = api.edit(
        image=args.input,
        prompt=args.prompt,
        num_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        enable_physedit=not args.disable_sce,
        srm_temperature=args.srm_temperature,
        rpfi_relaxation=args.rpfi_relaxation,
        rpfi_enabled=not args.disable_rpfi,
        card_override_nr=args.card_nr,
        card_override_r=args.card_r,
    )
    elapsed = time.time() - start

    # Save
    result.save(args.output)
    print(f"Output saved to: {args.output}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
