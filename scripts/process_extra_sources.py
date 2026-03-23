#!/usr/bin/env python3
"""
Process additional sprite sources: Kenney assets and HuggingFace parquets.
Appends to existing data/processed/ directory.
"""

import io
import logging
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/home/evilsocket/Lab/genvania")
PROCESSED = BASE / "data" / "processed"


def count_existing() -> int:
    pngs = list(PROCESSED.glob("sprite_*.png"))
    prev_pngs = list(PROCESSED.glob("sprite_*.prev.png"))
    return len(pngs) - len(prev_pngs)


def save_sprite(img: Image.Image, caption: str, idx: int) -> None:
    name = f"sprite_{idx:06d}"
    img.save(PROCESSED / f"{name}.png")
    (PROCESSED / f"{name}.txt").write_text(caption.strip())


def sprite_to_128(img: Image.Image) -> Image.Image:
    return img.resize((128, 128), Image.NEAREST).convert("RGBA")


def is_valid_sprite(img: Image.Image) -> bool:
    """Check if image is a reasonable sprite (not too big, not empty)."""
    w, h = img.size
    if w < 8 or h < 8:
        return False
    if w > 512 or h > 512:
        return False
    # Check it's not entirely transparent or single-color
    rgba = img.convert("RGBA")
    alpha = np.array(rgba.split()[-1])
    if alpha.mean() < 5:
        return False
    return True


def extract_sprites_from_sheet(img: Image.Image, base_caption: str = "pixel art sprite") -> list[tuple[Image.Image, str]]:
    """Extract individual sprites from a sprite sheet."""
    w, h = img.size
    img_rgba = img.convert("RGBA")
    results = []

    for cell_size in [64, 48, 32, 24, 16]:
        cols = w // cell_size
        rows = h // cell_size
        if cols < 2 or rows < 1:
            continue
        if cols * rows > 100:
            continue

        extracted = 0
        for row in range(rows):
            for col in range(cols):
                x, y = col * cell_size, row * cell_size
                cell = img_rgba.crop((x, y, x + cell_size, y + cell_size))
                alpha = np.array(cell.split()[-1])
                if alpha.mean() < 10:
                    continue
                caption = f"{base_caption}, frame {extracted + 1}"
                results.append((cell, caption))
                extracted += 1

        if extracted >= 2:
            return results

    return []


# ---------------------------------------------------------------------------
# Kenney Assets
# ---------------------------------------------------------------------------

def process_kenney(start_idx: int) -> int:
    logger.info("=== Processing Kenney Assets ===")
    kenney_dir = BASE / "data" / "raw" / "kenney" / "kenney-master"

    if not kenney_dir.exists():
        logger.warning(f"Kenney dir not found: {kenney_dir}")
        return start_idx

    idx = start_idx
    count = 0

    for img_path in sorted(kenney_dir.rglob("*.png")):
        try:
            img = Image.open(img_path)
            w, h = img.size

            # Build caption from directory structure
            rel = img_path.relative_to(kenney_dir)
            parts = [p.replace("_", " ").replace("-", " ") for p in rel.parts[:-1]]
            name = img_path.stem.replace("_", " ").replace("-", " ")

            if w > 256 or h > 256:
                # Sprite sheet - extract individual sprites
                base_caption = f"pixel art game asset, kenney"
                if parts:
                    base_caption += f", {' '.join(parts)}"

                sprites = extract_sprites_from_sheet(img, base_caption)
                for sprite_img, caption in sprites:
                    if is_valid_sprite(sprite_img):
                        save_sprite(sprite_to_128(sprite_img), caption, idx)
                        idx += 1
                        count += 1
            elif is_valid_sprite(img):
                caption = f"pixel art game asset, kenney"
                if parts:
                    caption += f", {' '.join(parts)}"
                caption += f", {name}"

                save_sprite(sprite_to_128(img), caption, idx)
                idx += 1
                count += 1

            if count % 500 == 0 and count > 0:
                logger.info(f"  Kenney: {count} sprites processed...")

        except Exception:
            continue

    logger.info(f"  Kenney: {count} total sprites")
    return idx


# ---------------------------------------------------------------------------
# HuggingFace Parquet: pixel-art-nouns-2k
# ---------------------------------------------------------------------------

def process_pixel_art_nouns(start_idx: int) -> int:
    logger.info("=== Processing pixel-art-nouns-2k ===")
    parquet_path = BASE / "data" / "raw" / "hf_parquet" / "pixel_art_nouns.parquet"

    if not parquet_path.exists():
        logger.warning(f"Parquet not found: {parquet_path}")
        return start_idx

    import pandas as pd
    df = pd.read_parquet(parquet_path)
    logger.info(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    idx = start_idx
    count = 0

    for _, row in df.iterrows():
        try:
            # The image column contains the actual image data
            img_data = row.get("image")
            if img_data is None:
                continue

            # Handle different image formats in parquet
            if isinstance(img_data, dict):
                img_bytes = img_data.get("bytes")
                if img_bytes:
                    img = Image.open(io.BytesIO(img_bytes))
                else:
                    continue
            elif isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data))
            elif isinstance(img_data, Image.Image):
                img = img_data
            else:
                continue

            # Get caption/label
            text = row.get("text", "") or row.get("prompt", "") or row.get("label", "") or row.get("caption", "")
            if not text or not isinstance(text, str):
                text = "pixel art noun sprite"

            caption = f"pixel art: {text}"

            if is_valid_sprite(img):
                save_sprite(sprite_to_128(img), caption, idx)
                idx += 1
                count += 1

            if count % 500 == 0 and count > 0:
                logger.info(f"  pixel-art-nouns: {count} sprites processed...")

        except Exception:
            continue

    logger.info(f"  pixel-art-nouns: {count} total sprites")
    return idx


# ---------------------------------------------------------------------------
# TinyHero (already cloned by previous script)
# ---------------------------------------------------------------------------

def process_tinyhero_full(start_idx: int) -> int:
    """Process TinyHero with better sprite extraction."""
    logger.info("=== Processing TinyHero (full) ===")
    tinyhero_dir = BASE / "data" / "raw" / "tinyhero"

    if not tinyhero_dir.exists():
        logger.warning(f"TinyHero dir not found: {tinyhero_dir}")
        return start_idx

    idx = start_idx
    count = 0
    directions = {"front": "facing forward", "back": "facing away", "left": "facing left", "right": "facing right"}

    # Look for the actual dataset images
    for img_path in sorted(tinyhero_dir.rglob("*.png")):
        try:
            img = Image.open(img_path)
            w, h = img.size

            # Skip very large images (likely full sheets) and tiny ones
            if w > 256 or h > 256:
                # Try to extract sprites from sheet
                sprites = extract_sprites_from_sheet(img, "pixel art character sprite, tinyhero")
                for sprite_img, caption in sprites:
                    if is_valid_sprite(sprite_img):
                        # Detect direction
                        for d, desc in directions.items():
                            if d in str(img_path).lower():
                                caption += f", {desc}"
                                break
                        save_sprite(sprite_to_128(sprite_img), caption, idx)
                        idx += 1
                        count += 1
                continue

            if w < 8 or h < 8:
                continue

            if not is_valid_sprite(img):
                continue

            caption = "pixel art character sprite, tinyhero"
            for d, desc in directions.items():
                if d in str(img_path).lower():
                    caption += f", {desc}"
                    break

            name_clean = img_path.stem.replace("_", " ").replace("-", " ")
            if name_clean:
                caption += f", {name_clean}"

            save_sprite(sprite_to_128(img), caption, idx)
            idx += 1
            count += 1

        except Exception:
            continue

    logger.info(f"  TinyHero: {count} total sprites")
    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_idx = count_existing()
    logger.info(f"Starting from index {start_idx}")

    idx = start_idx
    idx = process_kenney(idx)
    idx = process_pixel_art_nouns(idx)
    idx = process_tinyhero_full(idx)

    total = count_existing()
    logger.info(f"\nTotal sprites: {total}")


if __name__ == "__main__":
    main()
