# Alucard

A small (32M parameter) text-to-sprite generative model using flow matching. Generates 128x128 RGBA sprites from text prompts, with optional reference frame input for animation generation.

## Architecture

```
Text Prompt --> [Frozen CLIP ViT-B/32] --> 512-dim embedding
                                               |
                                          AdaLN-Zero (every ResBlock)
                                               |
Noise (128x128x4) --cat--> [UNet 32M params] --> Sprite (128x128x4 RGBA)
                    |
Previous Frame (128x128x4) or zeros
```

- **UNet** with 8-channel input (4 noisy RGBA + 4 reference RGBA), base channels 64, multipliers [1, 2, 4, 4]
- **AdaLN-Zero** conditioning from CLIP text embeddings + sinusoidal timestep
- **Flow matching** (rectified flow) training objective
- **Dual classifier-free guidance** for independent text and reference frame control
- Self-attention at 32x32 and 16x16 resolutions
- Gradient checkpointing for 16GB GPU training

## Two Modes

1. **Text to Sprite** - generate a sprite from a text prompt alone
2. **Text + Reference to Sprite** - generate the next animation frame conditioned on a previous frame and text describing the change

## Installation

```bash
pip install -e .
```

## Dataset Preparation

### 1. Prepare sprite images

Place 128x128 RGBA PNG sprites in a directory with `.txt` caption files:

```
data/processed/
    sprite_000001.png      # 128x128 RGBA
    sprite_000001.txt      # "a pixel art knight sprite, idle pose"
    sprite_000001.prev.png # optional: previous animation frame
```

### 2. Pre-compute CLIP embeddings

```bash
alucard-precompute --data-dir data/processed
```

This creates `.clip.pt` files containing 512-dim CLIP text embeddings for each caption.

### 3. Build dataset from public sources (optional)

```bash
python scripts/build_dataset.py
python scripts/process_extra_sources.py
python scripts/fix_captions_and_embed.py
```

## Training

```bash
alucard-train \
    --data-dir data/processed \
    --output-dir checkpoints \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-4 \
    --grad-accum 2 \
    --save-every 10 \
    --sample-every 10
```

### Training with Docker

```bash
docker build -t alucard .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints alucard \
    alucard-train --data-dir data/processed --output-dir checkpoints --epochs 200 --batch-size 64
```

### Resume training

```bash
alucard-train --data-dir data/processed --resume checkpoints/checkpoint_0050.pt
```

### VRAM usage (with gradient checkpointing)

| Batch Size | Peak VRAM |
|-----------|-----------|
| 32 | ~3.7 GB |
| 64 | ~7.3 GB |
| 96 | ~10.9 GB |

## Sampling

```bash
# Text-only generation
alucard-sample \
    --checkpoint checkpoints/checkpoint_0200.pt \
    --prompt "a pixel art knight sprite, idle pose" \
    --output knight.png

# Animation: generate next frame from reference
alucard-sample \
    --checkpoint checkpoints/checkpoint_0200.pt \
    --prompt "a pixel art knight sprite, walking, next frame" \
    --ref knight.png \
    --output knight_frame2.png

# Multiple samples
alucard-sample \
    --checkpoint checkpoints/checkpoint_0200.pt \
    --prompt "a pixel art dragon enemy sprite" \
    --num-samples 4 \
    --seed 42
```

### Sampling parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-steps` | 20 | Euler ODE integration steps |
| `--cfg-text` | 5.0 | Text guidance scale |
| `--cfg-ref` | 2.0 | Reference image guidance scale |

## Model Export

Convert a training checkpoint to safetensors for distribution:

```bash
python scripts/convert_to_safetensors.py \
    --checkpoint checkpoints/checkpoint_0200.pt \
    --output alucard_model.safetensors
```

## License

The model code is provided as-is. The training dataset is assembled from publicly available sources with the following licenses:

| Source | License |
|--------|---------|
| Kaggle Pixel Art | Apache 2.0 |
| Kenney Assets | CC0 |
| GameTileNet | CC0 / CC-BY |
| Pixel Art Nouns | HuggingFace |
| TinyHero | CC-BY-SA 3.0 |
