<h3 align="center">PhysEdit: Adaptive Spatio-Temporal Reasoning for Physically-Consistent Region-Aware Image Editing</h3>

<p align="center">
  <a href="#">[Paper (under review at <i>The Visual Computer</i>)]</a> &nbsp;
  <a href="https://github.com/leeguandong/PhysEdit">[Code]</a>
</p>

<p align="center">
  Built on <a href="https://research.nvidia.com/labs/toronto-ai/chronoedit">ChronoEdit</a> by NVIDIA &amp; University of Toronto
</p>

---

> **Notice.** This repository contains the official implementation for the manuscript
> *"Adaptive Spatio-Temporal Reasoning for Physically-Consistent Region-Aware Image Editing"*
> currently under consideration at **The Visual Computer (Springer Nature)**.
> If you use this code or build on its ideas, please cite the manuscript (BibTeX below).

## Abstract

Physically-consistent image editing remains challenging due to heterogeneous edit
requirements and fixed inference schedules in existing methods. PhysEdit is a
region-aware framework with adaptive spatio-temporal reasoning, built on a
flow-matching video diffusion editor. It introduces two inference-time modules
that compose without retraining the backbone:

1. **Spatial Reasoning Mask (SRM)** — derives an instruction-conditioned spatial
   mask from cross-attention to confine reasoning to semantically relevant
   regions.
2. **Complexity-Adaptive Reasoning Depth (CARD)** — predicts per-instruction
   edit complexity and dynamically allocates the reasoning step count `N_r`
   and reasoning-token length `r` per sample.
3. **Region-Preserving Feature Injection (RPFI)** *(exploratory)* — anchors
   spatially-masked unedited regions to noise-matched reference features
   during reasoning.

On the full 737-case ImgEdit Basic-Edit Suite, PhysEdit delivers a **1.18×**
wall-clock speedup over a strong reasoning baseline while improving instruction
adherence (+0.7% CLIP-T), with appearance-level edits reaching **1.52×**.

## Installation

```bash
git clone https://github.com/leeguandong/PhysEdit.git
cd PhysEdit
pip install -r requirements.txt

# Download the ChronoEdit-14B base model (≈ 28 GB)
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('nvidia/ChronoEdit-14B-Diffusers', \
    local_dir='checkpoints/ChronoEdit-14B-Diffusers')"
```

## Usage (CLI)

```bash
# Basic PhysEdit editing (CARD auto-detects complexity)
python physedit.py \
    -i input.jpg \
    -p "Change the hat to a red cap" \
    -o output.jpg \
    --model-path checkpoints/ChronoEdit-14B-Diffusers

# High-complexity physical action edit
python physedit.py \
    -i robot.jpg \
    -p "The robot picks up the cup" \
    -o output.jpg \
    --model-path checkpoints/ChronoEdit-14B-Diffusers

# Manual complexity override (skip CARD auto-detection)
python physedit.py \
    -i input.jpg \
    -p "Change the background to a beach" \
    -o output.jpg \
    --card-nr 8 --card-r 4 \
    --model-path checkpoints/ChronoEdit-14B-Diffusers

# Disable PhysEdit (fall back to vanilla ChronoEdit)
python physedit.py \
    -i input.jpg \
    -p "Add sunglasses" \
    -o output.jpg \
    --disable-sce \
    --model-path checkpoints/ChronoEdit-14B-Diffusers

# Lower VRAM (~ 20 GB) via CPU offloading
python physedit.py ... --offload
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i`, `--input` | required | Input image path |
| `-p`, `--prompt` | required | Editing instruction |
| `-o`, `--output` | `output.jpg` | Output image path |
| `--model-path` | required | Path to ChronoEdit-14B-Diffusers |
| `--steps` | `30` | Number of inference steps |
| `--guidance-scale` | `5.0` | Classifier-free guidance scale |
| `--seed` | `42` | Random seed |
| `--srm-temperature` | `0.1` | SRM mask sharpness (lower = sharper) |
| `--rpfi-relaxation` | `1.5` | RPFI boundary relaxation factor |
| `--card-nr` | auto | Override CARD reasoning steps |
| `--card-r` | auto | Override CARD reasoning token length |
| `--disable-sce` | `False` | Disable all PhysEdit modules |
| `--offload` | `False` | Enable CPU offloading (saves VRAM) |

## Python API

```python
from api import PhysEditAPI

api = PhysEditAPI(model_path="checkpoints/ChronoEdit-14B-Diffusers")

# Simple edit (CARD auto-detects low complexity: Nr=3, r=2)
result = api.edit(
    image="input.jpg",
    prompt="Change the hat color to red",
)
result.save("output.jpg")

# Complex physical edit (CARD auto-detects high complexity: Nr=15, r=8)
result = api.edit(
    image="robot.jpg",
    prompt="The robot picks up the cup",
)
result.save("output_action.jpg")

# Manual configuration
result = api.edit(
    image="input.jpg",
    prompt="Change background to winter",
    card_override_nr=8,
    card_override_r=4,
    srm_temperature=0.15,
)
result.save("output_winter.jpg")
```

A full Python example is in [`examples/api_example.py`](examples/api_example.py)
and a quick CLI script in [`examples/run_example.sh`](examples/run_example.sh).

## Key Parameter Guide

### SRM Temperature (`--srm-temperature`)
- `0.05` — very sharp mask (strict local editing)
- `0.1` — default, good balance
- `0.3` — soft mask (broader reasoning region)
- `1.0` — nearly uniform (close to vanilla ChronoEdit)

### CARD Complexity Levels
| Level | `N_r` | `r` | Use Case |
|-------|------:|----:|----------|
| Low    | 3 | 2 | Color, texture, style changes |
| Medium | 8 | 4 | Add / remove objects, background change |
| High   | 15 | 8 | Physical actions, pose changes, interactions |

### RPFI Relaxation (`--rpfi-relaxation`)
- `1.0` — strong preservation (may show boundaries)
- `1.5` — default, smooth blending
- `2.0` — very gradual transition (best for complex edits)

## Hardware Requirements

| Configuration            | VRAM       | Speed (per sample) |
|--------------------------|-----------:|-------------------:|
| Full model (A100-80GB)   |  ~ 38 GB   | 64.3 s             |
| CPU offloading           |  ~ 20 GB   | ~ 90 s             |

## Repository Layout

```
PhysEdit/
├── physedit.py        # CLI entry (argparse → PhysEditAPI)
├── api.py             # PhysEditAPI: pipeline construction & edit() method
├── src/
│   ├── physedit_modules.py     # SRM, CARD, RPFI, PhysEditController
│   └── pipeline_physedit.py    # Reference copy of the diffusion pipeline
├── examples/
│   ├── api_example.py
│   └── run_example.sh
├── requirements.txt
└── README.md
```

> The runtime `PhysEditPipeline` is loaded from a sibling
> [`ChronoEdit`](https://github.com/nv-tlabs/ChronoEdit) repository (located via
> the `CHRONOEDIT_ROOT` environment variable). The copy in `src/` is for
> reference; behavioural changes must be applied to the
> `chronoedit_diffusers/` package consumed at runtime.

## Citation

```bibtex
@article{li2026physedit,
  title  = {Adaptive Spatio-Temporal Reasoning for Physically-Consistent
            Region-Aware Image Editing},
  author = {Li, Guandong and Ye, Mengxia},
  journal= {The Visual Computer (under review)},
  year   = {2026},
  note   = {Code: \url{https://github.com/leeguandong/PhysEdit}},
}

@inproceedings{wu2025chronoedit,
  title    = {ChronoEdit: Towards Temporal Reasoning for Image Editing and
              World Simulation},
  author   = {Wu, Jay Zhangjie and Ren, Xuanchi and Shen, Tianchang and others},
  booktitle= {International Conference on Learning Representations (ICLR)},
  year     = {2026},
}
```

If you find this code useful, please **cite the PhysEdit manuscript above** —
this repository is directly tied to the work currently under review at *The Visual
Computer*, and citations help us track the work's reach.

## Acknowledgments

- [ChronoEdit](https://research.nvidia.com/labs/toronto-ai/chronoedit) by NVIDIA & University of Toronto
- [Wan 2.1](https://github.com/Wan-AI/Wan2.1) video generation model by Alibaba
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
