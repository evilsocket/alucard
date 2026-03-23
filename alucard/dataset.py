"""
Dataset for sprite generation training.

Expects a directory structure:
    data_dir/
        sprite_0001.png          # 128x128 RGBA sprite
        sprite_0001.txt          # text caption
        sprite_0001.clip.pt      # pre-computed CLIP embedding (512,)
        sprite_0001.prev.png     # optional: previous animation frame

Each sample yields:
    - image: (4, 128, 128) float tensor in [-1, 1] (RGBA)
    - text_emb: (512,) CLIP embedding
    - ref: (4, 128, 128) previous frame or zeros
    - has_ref: bool (whether a real reference was provided)
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgba(path: Path, size: int = 128) -> torch.Tensor:
    """Load image as RGBA, resize with nearest-neighbor, normalize to [-1, 1]."""
    img = Image.open(path).convert("RGBA")
    if img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1)  # (4, H, W)


def palette_swap(img: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    """Random palette shift on RGB channels (leave alpha untouched)."""
    rgb = img[:3]
    shift = (torch.rand(3, 1, 1) - 0.5) * 2 * strength
    rgb = (rgb + shift).clamp(-1, 1)
    return torch.cat([rgb, img[3:4]], dim=0)


class SpriteDataset(Dataset):
    """Dataset for sprite generation with optional animation frame pairs."""

    def __init__(
        self,
        data_dir: str | Path,
        image_size: int = 128,
        augment: bool = True,
        palette_swap_prob: float = 0.3,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment
        self.palette_swap_prob = palette_swap_prob

        # Find all sprites that have both image and CLIP embedding
        self.samples = []
        for img_path in sorted(self.data_dir.glob("*.png")):
            if img_path.stem.endswith(".prev"):
                continue  # skip reference frame files
            clip_path = img_path.with_suffix(".clip.pt")
            if clip_path.exists():
                self.samples.append(img_path)

        if not self.samples:
            raise ValueError(f"No valid samples found in {data_dir}. "
                             f"Expected .png files with matching .clip.pt embeddings.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.samples[idx]
        clip_path = img_path.with_suffix(".clip.pt")
        prev_path = img_path.parent / f"{img_path.stem}.prev.png"

        # Load image
        image = load_rgba(img_path, self.image_size)

        # Load CLIP embedding
        text_emb = torch.load(clip_path, map_location="cpu", weights_only=True)

        # Load reference (previous frame) if available
        has_ref = prev_path.exists()
        if has_ref:
            ref = load_rgba(prev_path, self.image_size)
        else:
            ref = torch.zeros_like(image)

        # Augmentation
        if self.augment:
            # Horizontal flip (apply same flip to both image and ref)
            if random.random() < 0.5:
                image = image.flip(-1)
                if has_ref:
                    ref = ref.flip(-1)

            # Palette swap
            if random.random() < self.palette_swap_prob:
                image = palette_swap(image)
                if has_ref:
                    ref = palette_swap(ref)

        return {
            "image": image,
            "text_emb": text_emb,
            "ref": ref,
            "has_ref": has_ref,
        }
