#!/usr/bin/env python3
"""
Fix captions for Kaggle sprites (classes 0-4) and pre-compute CLIP embeddings
for the entire dataset.
"""

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/home/evilsocket/Lab/genvania")
PROCESSED = BASE / "data" / "processed"

# Kaggle class labels (determined by visual inspection)
KAGGLE_CLASS_CAPTIONS = {
    0: [
        "a pixel art character sprite, standing pose, front view",
        "a small pixel art RPG character, facing forward",
        "a pixel art hero character sprite, idle stance",
        "a tiny pixel art adventurer sprite, front facing",
        "a pixel art character, standing idle, front view",
        "a retro pixel art character sprite",
        "a 16-bit style character sprite, standing pose",
        "a pixel art game character, idle animation frame",
    ],
    1: [
        "a pixel art creature sprite, small monster",
        "a pixel art slime creature, game enemy sprite",
        "a pixel art mushroom creature sprite",
        "a pixel art orb item, glowing sphere",
        "a small pixel art monster sprite, game enemy",
        "a pixel art creature, colorful game sprite",
        "a pixel art blob creature, RPG enemy sprite",
        "a pixel art small enemy sprite, cute creature",
    ],
    2: [
        "a pixel art fruit item sprite, food pickup",
        "a pixel art food item, collectible sprite",
        "a pixel art fruit sprite, game item",
        "a pixel art apple sprite, food item",
        "a pixel art berry sprite, collectible food",
        "a pixel art produce item sprite",
        "a pixel art fruit, game collectible item",
        "a pixel art food pickup sprite, small item",
    ],
    3: [
        "a pixel art weapon sprite, game equipment",
        "a pixel art staff weapon, RPG equipment sprite",
        "a pixel art armor piece, equipment sprite",
        "a pixel art sword sprite, weapon item",
        "a pixel art wand sprite, magical weapon",
        "a pixel art equipment sprite, RPG gear",
        "a pixel art weapon item, game loot sprite",
        "a pixel art armor sprite, protective equipment",
    ],
    4: [
        "a pixel art character sprite wielding a weapon, action pose",
        "a pixel art warrior sprite with sword, combat stance",
        "a pixel art hero sprite, armed character, side view",
        "a pixel art adventurer sprite holding a weapon",
        "a pixel art character in combat pose, armed with weapon",
        "a pixel art fighter sprite, battle ready stance",
        "a pixel art armed character sprite, RPG hero",
        "a pixel art swordsman sprite, attack pose",
    ],
}


def fix_kaggle_captions():
    """Fix captions for the first 20000 sprites (Kaggle source)."""
    logger.info("=== Fixing Kaggle captions ===")

    labels_path = BASE / "data" / "raw" / "kaggle_pixelart" / "sprites_labels.npy"
    if not labels_path.exists():
        logger.warning("Kaggle labels not found, skipping caption fix")
        return

    labels = np.load(labels_path)
    class_ids = np.argmax(labels, axis=1)

    random.seed(42)
    fixed = 0

    for i in range(min(20000, len(class_ids))):
        txt_path = PROCESSED / f"sprite_{i:06d}.txt"
        if not txt_path.exists():
            continue

        cls = int(class_ids[i])
        caption = random.choice(KAGGLE_CLASS_CAPTIONS[cls])
        txt_path.write_text(caption)
        fixed += 1

        if fixed % 5000 == 0:
            logger.info(f"  Fixed {fixed} captions...")

    logger.info(f"  Fixed {fixed} Kaggle captions")


def improve_kenney_captions():
    """Clean up Kenney captions - remove path noise, make more descriptive."""
    logger.info("=== Improving Kenney captions ===")

    # Kenney sprites start after Kaggle (20000) + GameTileNet (2645) + TinyHero (43) = ~22688
    improved = 0
    for txt_path in sorted(PROCESSED.glob("sprite_*.txt")):
        try:
            idx = int(txt_path.stem.split("_")[1])
            if idx < 22688:  # Skip non-Kenney
                continue

            caption = txt_path.read_text().strip()
            if "kenney" not in caption.lower():
                continue

            # Clean up the caption
            # Remove long path fragments
            parts = caption.split(", ")
            clean_parts = []
            for p in parts:
                p = p.strip()
                if len(p) > 80:
                    # Truncate long path segments, keep first meaningful part
                    p = p[:60].rsplit(" ", 1)[0]
                if p and "files)" not in p and "assets)" not in p:
                    clean_parts.append(p)

            if clean_parts:
                caption = ", ".join(clean_parts[:5])  # Max 5 parts
            else:
                caption = "pixel art game asset, kenney"

            txt_path.write_text(caption)
            improved += 1

        except Exception:
            continue

    logger.info(f"  Improved {improved} Kenney captions")


def precompute_clip_embeddings():
    """Pre-compute CLIP text embeddings for all sprites."""
    logger.info("=== Pre-computing CLIP embeddings ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    # Load CLIP
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    logger.info("  CLIP model loaded")

    # Find all sprites needing embeddings
    txt_files = sorted(PROCESSED.glob("sprite_*.txt"))
    txt_files = [f for f in txt_files if not f.stem.endswith(".prev")]

    # Filter to those without existing embeddings
    to_process = []
    for txt_path in txt_files:
        clip_path = txt_path.with_suffix(".clip.pt")
        if not clip_path.exists():
            to_process.append(txt_path)

    logger.info(f"  {len(to_process)} sprites need embeddings ({len(txt_files)} total)")

    if not to_process:
        logger.info("  All embeddings already computed")
        return

    # Process in batches
    batch_size = 256
    for i in range(0, len(to_process), batch_size):
        batch_paths = to_process[i : i + batch_size]
        captions = []
        for txt_path in batch_paths:
            caption = txt_path.read_text().strip()
            if not caption:
                caption = "a pixel art sprite"
            captions.append(caption)

        tokens = tokenizer(captions).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.cpu().float()

        for j, txt_path in enumerate(batch_paths):
            clip_path = txt_path.with_suffix(".clip.pt")
            torch.save(embeddings[j], clip_path)

        processed = min(i + batch_size, len(to_process))
        if processed % 5000 == 0 or processed == len(to_process):
            logger.info(f"  Embedded {processed}/{len(to_process)}")

    logger.info("  CLIP embedding computation complete")


def verify_dataset():
    """Verify dataset integrity."""
    logger.info("=== Verifying dataset ===")

    pngs = sorted([p for p in PROCESSED.glob("sprite_*.png") if ".prev." not in p.name])
    txts = sorted([p for p in PROCESSED.glob("sprite_*.txt")])
    clips = sorted([p for p in PROCESSED.glob("sprite_*.clip.pt")])
    prevs = sorted([p for p in PROCESSED.glob("sprite_*.prev.png")])

    logger.info(f"  Sprite images: {len(pngs)}")
    logger.info(f"  Captions: {len(txts)}")
    logger.info(f"  CLIP embeddings: {len(clips)}")
    logger.info(f"  Animation pairs: {len(prevs)}")

    # Check a few random samples
    import random
    random.seed(123)
    samples = random.sample(pngs, min(5, len(pngs)))
    for p in samples:
        txt_path = p.with_suffix(".txt")
        clip_path = p.with_suffix(".clip.pt")
        img = Image.open(p)
        caption = txt_path.read_text().strip() if txt_path.exists() else "MISSING"
        has_clip = clip_path.exists()
        logger.info(f"  {p.name}: {img.size} {img.mode} | clip={has_clip} | '{caption[:80]}'")

    # Check for mismatches
    png_stems = {p.stem for p in pngs}
    txt_stems = {p.stem for p in txts}
    clip_stems = {p.stem for p in clips}

    missing_txt = png_stems - txt_stems
    missing_clip = png_stems - clip_stems
    if missing_txt:
        logger.warning(f"  {len(missing_txt)} sprites missing captions!")
    if missing_clip:
        logger.warning(f"  {len(missing_clip)} sprites missing CLIP embeddings!")
    else:
        logger.info("  All sprites have captions and CLIP embeddings")


def main():
    fix_kaggle_captions()
    improve_kenney_captions()
    precompute_clip_embeddings()
    verify_dataset()


if __name__ == "__main__":
    main()
