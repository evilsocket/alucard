#!/usr/bin/env python3
"""
Build the genvania sprite dataset from multiple public sources.

Sources:
1. Kaggle Pixel Art (ebrahimelgazar) - 89K 16x16 sprites with labels [Apache 2.0]
   - Downloaded via HuggingFace mirror or direct NumPy files
2. HuggingFace free-to-use-pixelart (bghira) - 7K pixel art images [MIT]
3. OpenGameArt CC0 via HuggingFace (nyuuzyou) - thousands of sprites [CC0]
4. TinyHero (AgaMiko) - 3.6K character sprites [CC-BY-SA 3.0]
5. GameTileNet - 2.1K labeled tiles [CC0/CC-BY]

Output structure:
    data/processed/
        sprite_00001.png      # 128x128 RGBA
        sprite_00001.txt      # text caption
        sprite_00001.prev.png # optional previous frame (for animation sequences)
"""

import argparse
import hashlib
import io
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def sprite_to_128(img: Image.Image) -> Image.Image:
    """Resize sprite to 128x128 using nearest-neighbor (preserves pixel art)."""
    return img.resize((128, 128), Image.NEAREST).convert("RGBA")


def save_sprite(img: Image.Image, caption: str, idx: int, prev_img: Image.Image | None = None) -> int:
    """Save a processed sprite with caption. Returns the index used."""
    name = f"sprite_{idx:06d}"
    img_path = PROCESSED_DIR / f"{name}.png"
    txt_path = PROCESSED_DIR / f"{name}.txt"

    img.save(img_path)
    txt_path.write_text(caption.strip())

    if prev_img is not None:
        prev_path = PROCESSED_DIR / f"{name}.prev.png"
        prev_img.save(prev_path)

    return idx


def count_existing() -> int:
    """Count existing processed sprites."""
    return len(list(PROCESSED_DIR.glob("sprite_*.png"))) - len(list(PROCESSED_DIR.glob("sprite_*.prev.png")))


# ---------------------------------------------------------------------------
# Source 1: Kaggle Pixel Art via NumPy arrays
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress."""
    if dest.exists():
        logger.info(f"  Already downloaded: {dest}")
        return

    logger.info(f"  Downloading {desc or url}...")
    req = Request(url, headers={"User-Agent": "genvania-dataset-builder/1.0"})

    with urlopen(req) as response, open(dest, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                print(f"\r  {downloaded}/{total} bytes ({pct}%)", end="", flush=True)
        print()


def process_kaggle_pixelart(start_idx: int, max_sprites: int = 30000) -> int:
    """
    Process the Kaggle Pixel Art dataset.
    Downloads sprites.npy and sprites_labels.npy from the dataset.
    """
    logger.info("=== Source 1: Kaggle Pixel Art (ebrahimelgazar) ===")

    kaggle_dir = RAW_DIR / "kaggle_pixelart"
    kaggle_dir.mkdir(exist_ok=True)

    # Try to download via kaggle API first
    sprites_npy = kaggle_dir / "sprites.npy"
    labels_npy = kaggle_dir / "sprites_labels.npy"

    if not sprites_npy.exists() or not labels_npy.exists():
        try:
            # Try kaggle CLI
            logger.info("  Attempting kaggle CLI download...")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", "ebrahimelgazar/pixel-art",
                 "-p", str(kaggle_dir), "--unzip"],
                check=True, capture_output=True, text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.warning(f"  Kaggle CLI failed: {e}")
            logger.info("  Trying HuggingFace mirror...")

            # Try loading via HuggingFace datasets
            try:
                from datasets import load_dataset
                ds = load_dataset("ebrahimelgazar/pixel-art", split="train")
                logger.info(f"  Loaded {len(ds)} samples from HuggingFace")

                idx = start_idx
                count = 0
                for item in ds:
                    if count >= max_sprites:
                        break
                    try:
                        img = item["image"]
                        if img is None:
                            continue
                        label = item.get("label", "pixel art sprite")
                        if isinstance(label, int):
                            label = f"a pixel art sprite, category {label}"
                        elif isinstance(label, str) and label:
                            label = f"a pixel art {label} sprite"
                        else:
                            label = "a pixel art sprite"

                        img_128 = sprite_to_128(img)
                        save_sprite(img_128, label, idx)
                        idx += 1
                        count += 1
                        if count % 1000 == 0:
                            logger.info(f"  Processed {count} sprites...")
                    except Exception as ex:
                        continue

                logger.info(f"  Processed {count} sprites from Kaggle/HF")
                return idx

            except Exception as e2:
                logger.warning(f"  HuggingFace load also failed: {e2}")
                logger.info("  Skipping Kaggle dataset")
                return start_idx

    # Process from NumPy files
    if sprites_npy.exists() and labels_npy.exists():
        logger.info("  Loading NumPy arrays...")
        sprites = np.load(sprites_npy)
        labels = np.load(labels_npy)
        logger.info(f"  Loaded {len(sprites)} sprites with {len(labels)} labels")

        idx = start_idx
        count = 0
        for i in range(min(len(sprites), max_sprites)):
            try:
                arr = sprites[i]
                if arr.ndim == 3 and arr.shape[2] == 3:
                    img = Image.fromarray(arr.astype(np.uint8), "RGB")
                elif arr.ndim == 3 and arr.shape[2] == 4:
                    img = Image.fromarray(arr.astype(np.uint8))
                elif arr.ndim == 2:
                    img = Image.fromarray(arr.astype(np.uint8), "L").convert("RGB")
                else:
                    continue

                label_val = labels[i] if i < len(labels) else "unknown"
                if isinstance(label_val, (int, np.integer)):
                    caption = f"a small pixel art sprite, category {int(label_val)}"
                else:
                    caption = f"a pixel art {label_val} sprite"

                img_128 = sprite_to_128(img)
                save_sprite(img_128, caption, idx)
                idx += 1
                count += 1
                if count % 1000 == 0:
                    logger.info(f"  Processed {count}/{min(len(sprites), max_sprites)}...")
            except Exception:
                continue

        logger.info(f"  Processed {count} sprites from Kaggle")
        return idx

    return start_idx


# ---------------------------------------------------------------------------
# Source 2: HuggingFace free-to-use-pixelart
# ---------------------------------------------------------------------------

def process_hf_pixelart(start_idx: int, max_sprites: int = 7000) -> int:
    """Process the free-to-use-pixelart dataset from HuggingFace."""
    logger.info("=== Source 2: HuggingFace free-to-use-pixelart (bghira) ===")

    try:
        from datasets import load_dataset
        ds = load_dataset("bghira/free-to-use-pixelart", split="train")
        logger.info(f"  Loaded {len(ds)} items")
    except Exception as e:
        logger.warning(f"  Failed to load: {e}")
        return start_idx

    idx = start_idx
    count = 0
    for item in ds:
        if count >= max_sprites:
            break
        try:
            img = item.get("image")
            if img is None:
                continue

            # Build caption from metadata
            title = item.get("title", "").strip()
            description = item.get("description", "").strip()

            if title:
                caption = f"pixel art: {title}"
                if description and len(description) < 200:
                    caption += f". {description}"
            else:
                caption = "a pixel art sprite"

            # Truncate overly long captions
            if len(caption) > 300:
                caption = caption[:297] + "..."

            img_128 = sprite_to_128(img)
            save_sprite(img_128, caption, idx)
            idx += 1
            count += 1
            if count % 500 == 0:
                logger.info(f"  Processed {count} sprites...")
        except Exception:
            continue

    logger.info(f"  Processed {count} sprites from HF pixelart")
    return idx


# ---------------------------------------------------------------------------
# Source 3: OpenGameArt CC0
# ---------------------------------------------------------------------------

def process_opengameart_cc0(start_idx: int, max_sprites: int = 10000) -> int:
    """Process OpenGameArt CC0 2D art from HuggingFace."""
    logger.info("=== Source 3: OpenGameArt CC0 (nyuuzyou) ===")

    try:
        from datasets import load_dataset
        ds = load_dataset("nyuuzyou/OpenGameArt-CC0", split="2d_art")
        logger.info(f"  Loaded {len(ds)} submissions")
    except Exception as e:
        logger.warning(f"  Failed to load: {e}")
        return start_idx

    idx = start_idx
    count = 0

    for item in ds:
        if count >= max_sprites:
            break
        try:
            # Each item has preview images and metadata
            images = item.get("previews", []) or []
            if not images:
                img = item.get("image")
                if img is not None:
                    images = [img]
                else:
                    continue

            title = item.get("title", "game sprite").strip()
            tags = item.get("tags", []) or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]

            # Build caption
            tag_str = ", ".join(tags[:5]) if tags else ""
            caption = f"pixel art game asset: {title}"
            if tag_str:
                caption += f" ({tag_str})"

            for img in images:
                if count >= max_sprites:
                    break
                if img is None:
                    continue

                try:
                    if isinstance(img, dict) and "path" in img:
                        continue  # skip URL references we can't fetch easily
                    if not isinstance(img, Image.Image):
                        continue

                    # Only process reasonable-sized images that look like sprites
                    w, h = img.size
                    if w > 1024 or h > 1024 or w < 8 or h < 8:
                        continue

                    # If it's a sprite sheet, try to extract individual sprites
                    if w > 256 or h > 256:
                        sprites = extract_sprites_from_sheet(img, title, tags)
                        for sprite_img, sprite_caption in sprites:
                            if count >= max_sprites:
                                break
                            img_128 = sprite_to_128(sprite_img)
                            save_sprite(img_128, sprite_caption, idx)
                            idx += 1
                            count += 1
                    else:
                        img_128 = sprite_to_128(img)
                        save_sprite(img_128, caption, idx)
                        idx += 1
                        count += 1

                    if count % 500 == 0:
                        logger.info(f"  Processed {count} sprites...")
                except Exception:
                    continue

        except Exception:
            continue

    logger.info(f"  Processed {count} sprites from OpenGameArt CC0")
    return idx


def extract_sprites_from_sheet(
    img: Image.Image, title: str = "", tags: list = None, min_size: int = 16, max_size: int = 128
) -> list[tuple[Image.Image, str]]:
    """
    Try to extract individual sprites from a sprite sheet.
    Uses simple grid detection based on common sprite sizes.
    """
    w, h = img.size
    img_rgba = img.convert("RGBA")
    results = []

    # Try common sprite sizes
    for cell_size in [64, 48, 32, 24, 16]:
        cols = w // cell_size
        rows = h // cell_size
        if cols < 2 or rows < 1:
            continue
        if cols * rows > 200:  # too many cells, probably wrong grid
            continue

        extracted = 0
        for row in range(rows):
            for col in range(cols):
                x = col * cell_size
                y = row * cell_size
                cell = img_rgba.crop((x, y, x + cell_size, y + cell_size))

                # Skip empty/mostly transparent cells
                alpha = np.array(cell.split()[-1])
                if alpha.mean() < 10:
                    continue

                tag_str = ", ".join((tags or [])[:3])
                caption = f"pixel art sprite from {title}" if title else "pixel art sprite"
                if tag_str:
                    caption += f" ({tag_str})"
                caption += f", {cell_size}x{cell_size} original"

                results.append((cell, caption))
                extracted += 1

        if extracted >= 2:  # found valid sprites
            return results

    return results


# ---------------------------------------------------------------------------
# Source 4: TinyHero
# ---------------------------------------------------------------------------

def process_tinyhero(start_idx: int) -> int:
    """Process TinyHero dataset from GitHub."""
    logger.info("=== Source 4: TinyHero (AgaMiko) ===")

    tinyhero_dir = RAW_DIR / "tinyhero"

    if not tinyhero_dir.exists():
        logger.info("  Cloning repository...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/AgaMiko/pixel_character_generator.git",
                 str(tinyhero_dir)],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"  Failed to clone: {e}")
            return start_idx

    # Find all sprite images
    sprite_dirs = {
        "front": "front view",
        "back": "back view",
        "left": "left view",
        "right": "right view",
    }

    idx = start_idx
    count = 0

    for img_path in sorted(tinyhero_dir.rglob("*.png")):
        try:
            img = Image.open(img_path)
            if img.size[0] < 8 or img.size[1] < 8:
                continue

            # Determine direction from path
            direction = ""
            for d, desc in sprite_dirs.items():
                if d in str(img_path).lower():
                    direction = desc
                    break

            caption = "a pixel art character sprite"
            if direction:
                caption += f", {direction}"

            # Extract info from filename/path
            name_parts = img_path.stem.lower().replace("_", " ").replace("-", " ")
            if name_parts and name_parts != img_path.stem:
                caption += f", {name_parts}"

            img_128 = sprite_to_128(img)
            save_sprite(img_128, caption, idx)
            idx += 1
            count += 1

        except Exception:
            continue

    logger.info(f"  Processed {count} sprites from TinyHero")
    return idx


# ---------------------------------------------------------------------------
# Source 5: GameTileNet
# ---------------------------------------------------------------------------

def process_gametilenet(start_idx: int) -> int:
    """Process GameTileNet dataset from GitHub."""
    logger.info("=== Source 5: GameTileNet ===")

    gtn_dir = RAW_DIR / "gametilenet"

    if not gtn_dir.exists():
        logger.info("  Cloning repository...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/RimiChen/2024-GameTileNet.git",
                 str(gtn_dir)],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"  Failed to clone: {e}")
            return start_idx

    idx = start_idx
    count = 0

    # Look for tile images and metadata
    for img_path in sorted(gtn_dir.rglob("*.png")):
        try:
            img = Image.open(img_path)
            w, h = img.size
            if w < 8 or h < 8 or w > 256 or h > 256:
                continue

            # Build caption from path structure
            parts = img_path.relative_to(gtn_dir).parts
            caption_parts = ["pixel art game tile"]
            for part in parts[:-1]:  # directories = categories
                clean = part.replace("_", " ").replace("-", " ")
                if clean and not clean.startswith("."):
                    caption_parts.append(clean)

            # Check for JSON metadata
            json_path = img_path.with_suffix(".json")
            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text())
                    if "name" in meta:
                        caption_parts.append(meta["name"])
                    if "tags" in meta and meta["tags"]:
                        caption_parts.extend(meta["tags"][:3])
                except Exception:
                    pass

            caption = ", ".join(caption_parts)
            img_128 = sprite_to_128(img)
            save_sprite(img_128, caption, idx)
            idx += 1
            count += 1

        except Exception:
            continue

    logger.info(f"  Processed {count} sprites from GameTileNet")
    return idx


# ---------------------------------------------------------------------------
# Source 6: Universal LPC Spritesheet Generator
# ---------------------------------------------------------------------------

def process_lpc(start_idx: int, max_sprites: int = 5000) -> int:
    """Process Universal LPC Spritesheet assets."""
    logger.info("=== Source 6: Universal LPC Spritesheet ===")

    lpc_dir = RAW_DIR / "lpc"

    if not lpc_dir.exists():
        logger.info("  Cloning repository...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/makrohn/Universal-LPC-spritesheet.git",
                 str(lpc_dir)],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"  Failed to clone: {e}")
            return start_idx

    idx = start_idx
    count = 0

    # LPC sprites are organized in sprite sheets
    # Each sheet typically has rows for different animations and columns for frames
    # Common layout: 64x64 per frame, with walk/attack/cast/etc.
    directions = ["down", "left", "right", "up"]
    actions = ["spellcast", "thrust", "walk", "slash", "shoot", "hurt"]

    for img_path in sorted(lpc_dir.rglob("*.png")):
        if count >= max_sprites:
            break
        try:
            img = Image.open(img_path).convert("RGBA")
            w, h = img.size

            # Extract category from path
            parts = img_path.relative_to(lpc_dir).parts
            category = " ".join(p.replace("_", " ") for p in parts[:-1] if not p.startswith("."))

            # Try to extract individual frames from sprite sheets
            # LPC standard: 64x64 per frame
            cell_size = 64
            if w >= cell_size and h >= cell_size:
                cols = w // cell_size
                rows = h // cell_size

                prev_cell = None
                for row in range(min(rows, 8)):
                    for col in range(min(cols, 8)):
                        if count >= max_sprites:
                            break
                        x = col * cell_size
                        y = row * cell_size
                        cell = img.crop((x, y, x + cell_size, y + cell_size))

                        # Skip empty cells
                        alpha = np.array(cell.split()[-1])
                        if alpha.mean() < 10:
                            prev_cell = None
                            continue

                        # Build caption
                        direction = directions[row % len(directions)] if row < len(directions) * 2 else ""
                        action = actions[row // len(directions)] if row // len(directions) < len(actions) else ""

                        caption = f"pixel art character sprite"
                        if category:
                            caption += f", {category}"
                        if action:
                            caption += f", {action}"
                        if direction:
                            caption += f" facing {direction}"
                        caption += f", frame {col + 1}"

                        cell_128 = sprite_to_128(cell)

                        # Save with animation pair if we have a previous frame
                        if prev_cell is not None:
                            save_sprite(cell_128, caption, idx, prev_img=prev_cell)
                        else:
                            save_sprite(cell_128, caption, idx)

                        prev_cell = cell_128
                        idx += 1
                        count += 1

                    prev_cell = None  # reset at row boundary

            if count % 500 == 0 and count > 0:
                logger.info(f"  Processed {count} sprites...")

        except Exception:
            continue

    logger.info(f"  Processed {count} sprites from LPC")
    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build genvania sprite dataset")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--max-total", type=int, default=50000, help="Maximum total sprites")
    parser.add_argument("--skip-sources", type=str, nargs="*", default=[],
                        help="Sources to skip (kaggle, hf_pixelart, opengameart, tinyhero, gametilenet, lpc)")
    args = parser.parse_args()

    global PROCESSED_DIR, RAW_DIR
    PROCESSED_DIR = Path(args.output_dir)
    RAW_DIR = Path(args.raw_dir)
    ensure_dirs()

    existing = count_existing()
    logger.info(f"Existing sprites: {existing}")

    idx = existing
    sources = [
        ("kaggle", process_kaggle_pixelart, {"max_sprites": 20000}),
        ("hf_pixelart", process_hf_pixelart, {"max_sprites": 7000}),
        ("opengameart", process_opengameart_cc0, {"max_sprites": 10000}),
        ("tinyhero", process_tinyhero, {}),
        ("gametilenet", process_gametilenet, {}),
        ("lpc", process_lpc, {"max_sprites": 5000}),
    ]

    for name, func, kwargs in sources:
        if name in args.skip_sources:
            logger.info(f"Skipping {name}")
            continue
        if idx >= args.max_total:
            logger.info(f"Reached max total ({args.max_total}), stopping")
            break

        remaining = args.max_total - idx
        if "max_sprites" in kwargs:
            kwargs["max_sprites"] = min(kwargs["max_sprites"], remaining)

        try:
            idx = func(idx, **kwargs)
        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total = count_existing()
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset complete: {total} sprites in {PROCESSED_DIR}")
    logger.info(f"{'='*60}")

    if total < 25000:
        logger.warning(f"Only {total} sprites collected, target was 25,000+")
        logger.info("Consider adding more sources or increasing limits")


if __name__ == "__main__":
    main()
