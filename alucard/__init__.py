"""
Alucard: A small (32M param) text-to-sprite generative model.

Usage:
    from alucard import Alucard

    model = Alucard.from_pretrained("evilsocket/alucard")

    # Generate a sprite from text
    sprite = model("a pixel art knight sprite, idle pose")
    sprite.save("knight.png")

    # Generate next animation frame
    next_frame = model("walking, next frame", ref=sprite)
    next_frame.save("knight_walk.png")

    # Generate multiple samples
    sprites = model("a pixel art dragon", num_samples=4)
    for i, s in enumerate(sprites):
        s.save(f"dragon_{i}.png")
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from alucard.model import UNet
from alucard.sample import sample, tensor_to_rgba_image


class Alucard:
    """High-level wrapper for sprite generation."""

    def __init__(
        self,
        model: UNet,
        clip_model: nn.Module,
        tokenizer,
        device: torch.device | str = "cuda",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.clip_model = clip_model.to(self.device).eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cuda",
    ) -> "Alucard":
        """
        Load a pretrained Alucard model.

        Args:
            path: HuggingFace repo id (e.g. "evilsocket/alucard"),
                  local directory, or path to a .safetensors/.pt file.
            device: Device to load on ("cuda", "cpu", etc.)

        Returns:
            Alucard instance ready for generation.
        """
        import open_clip

        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        model = UNet()

        path_obj = Path(path)

        if path_obj.is_file():
            # Direct file path
            state_dict = cls._load_weights(path_obj, dev)
        elif path_obj.is_dir():
            # Local directory
            safetensors = path_obj / "alucard_model.safetensors"
            pt_file = path_obj / "best.pt"
            if safetensors.exists():
                state_dict = cls._load_weights(safetensors, dev)
            elif pt_file.exists():
                state_dict = cls._load_weights(pt_file, dev)
            else:
                raise FileNotFoundError(f"No model weights found in {path}")
        else:
            # HuggingFace repo id
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(repo_id=path, filename="alucard_model.safetensors")
            state_dict = cls._load_weights(Path(local_path), dev)

        model.load_state_dict(state_dict)
        model.eval()

        # Load CLIP text encoder
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        clip_model.eval()

        return cls(model, clip_model, tokenizer, device=dev)

    @staticmethod
    def _load_weights(path: Path, device: torch.device) -> dict:
        """Load weights from .safetensors or .pt file."""
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(path), device=str(device))
        else:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict):
                return ckpt.get("ema_model", ckpt.get("model", ckpt))
            return ckpt

    def encode_text(self, prompt: str | list[str]) -> torch.Tensor:
        """Encode text prompt(s) to CLIP embeddings."""
        if isinstance(prompt, str):
            prompt = [prompt]
        tokens = self.tokenizer(prompt).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    @staticmethod
    def load_ref(image: str | Path | Image.Image, size: int = 128) -> torch.Tensor:
        """Load a reference image as a tensor."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("RGBA")
        if image.size != (size, size):
            image = image.resize((size, size), Image.NEAREST)
        arr = np.array(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (4, H, W)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        ref: str | Path | Image.Image | None = None,
        num_samples: int = 1,
        num_steps: int = 20,
        cfg_text: float = 5.0,
        cfg_ref: float = 2.0,
        seed: int | None = None,
    ) -> Image.Image | list[Image.Image]:
        """
        Generate sprite(s) from a text prompt.

        Args:
            prompt: Text description of the sprite to generate.
            ref: Optional reference image (path, PIL Image, or previous output).
                 Used for animation frame generation.
            num_samples: Number of sprites to generate.
            num_steps: Number of Euler ODE integration steps.
            cfg_text: Text classifier-free guidance scale.
            cfg_ref: Reference image guidance scale.
            seed: Random seed for reproducibility.

        Returns:
            Single PIL Image if num_samples=1, otherwise list of PIL Images.
        """
        if seed is not None:
            torch.manual_seed(seed)

        text_emb = self.encode_text(prompt)
        text_emb = text_emb.expand(num_samples, -1)

        ref_tensor = None
        if ref is not None:
            ref_tensor = self.load_ref(ref).unsqueeze(0).expand(num_samples, -1, -1, -1).to(self.device)

        sprites = sample(
            self.model,
            text_emb,
            ref=ref_tensor,
            num_steps=num_steps,
            cfg_text=cfg_text,
            cfg_ref=cfg_ref,
            device=self.device,
        )

        images = [tensor_to_rgba_image(sprites[i]) for i in range(num_samples)]
        return images[0] if num_samples == 1 else images
