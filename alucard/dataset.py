"""
Dataset for sprite generation training.

Supports two formats:

Format 1 (individual files):
    data_dir/
        sprite_0001.png          # 128x128 RGBA sprite
        sprite_0001.clip.pt      # pre-computed CLIP embedding (512,)
        sprite_0001.prev.png     # optional: previous animation frame

Format 2 (consolidated embeddings):
    data_dir/
        sprite_0001.png          # 128x128 RGBA sprite
        sprite_0001.txt          # text caption (used if no .clip.pt)
        clip_embeddings.pt       # single file with all embeddings (N, 512)

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

        # Check for consolidated embeddings file
        consolidated_path = self.data_dir / "clip_embeddings.pt"
        self.clip_embeddings = None

        if consolidated_path.exists():
            self.clip_embeddings = torch.load(consolidated_path, map_location="cpu", weights_only=True)
            # Find all sprites by index (matching the consolidated embeddings order)
            self.samples = []
            for i in range(len(self.clip_embeddings)):
                img_path = self.data_dir / f"sprite_{i:06d}.png"
                if img_path.exists():
                    self.samples.append(img_path)
                else:
                    break
        else:
            # Fallback: find sprites with individual .clip.pt files
            self.samples = []
            for img_path in sorted(self.data_dir.glob("*.png")):
                if img_path.stem.endswith(".prev"):
                    continue
                clip_path = img_path.with_suffix(".clip.pt")
                if clip_path.exists():
                    self.samples.append(img_path)

        if not self.samples:
            raise ValueError(f"No valid samples found in {data_dir}. "
                             f"Expected .png files with .clip.pt embeddings or clip_embeddings.pt")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.samples[idx]
        prev_path = img_path.parent / f"{img_path.stem}.prev.png"

        # Load image
        image = load_rgba(img_path, self.image_size)

        # Load CLIP embedding
        if self.clip_embeddings is not None:
            text_emb = self.clip_embeddings[idx]
        else:
            clip_path = img_path.with_suffix(".clip.pt")
            text_emb = torch.load(clip_path, map_location="cpu", weights_only=True)

        # Load reference (previous frame) if available
        has_ref = prev_path.exists()
        if has_ref:
            ref = load_rgba(prev_path, self.image_size)
        else:
            ref = torch.zeros_like(image)

        # Augmentation
        if self.augment:
            if random.random() < 0.5:
                image = image.flip(-1)
                if has_ref:
                    ref = ref.flip(-1)

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
