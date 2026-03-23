"""
Sampling / inference for flow matching sprite generation.

Uses Euler ODE solver to integrate the learned velocity field:
    x_{t+dt} = x_t + dt * v(x_t, t, text_emb, ref)

Starting from x_1 ~ N(0, I), integrate backward to x_0 (clean image).

Supports dual classifier-free guidance:
    v_guided = v_uncond + w_text * (v_text - v_uncond) + w_ref * (v_both - v_text)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import numpy as np


@torch.no_grad()
def sample(
    model: nn.Module,
    text_emb: torch.Tensor,
    ref: torch.Tensor | None = None,
    num_steps: int = 20,
    cfg_text: float = 5.0,
    cfg_ref: float = 2.0,
    device: torch.device | str = "cuda",
    image_size: int = 128,
) -> torch.Tensor:
    """
    Generate sprites using Euler ODE integration with dual CFG.

    Args:
        model: Trained UNet model
        text_emb: CLIP text embeddings (B, 512)
        ref: Optional reference/previous frame (B, 4, H, W)
        num_steps: Number of Euler steps
        cfg_text: Text guidance scale
        cfg_ref: Reference image guidance scale
        device: Device to run on
        image_size: Output image size

    Returns:
        Generated sprites (B, 4, image_size, image_size) in [-1, 1]
    """
    model.eval()
    B = text_emb.shape[0]
    text_emb = text_emb.to(device)

    if ref is not None:
        ref = ref.to(device)

    # Start from pure noise (t=1)
    x = torch.randn(B, 4, image_size, image_size, device=device)

    # Null embeddings for CFG
    null_text = model.null_text_emb.unsqueeze(0).expand(B, -1).to(text_emb.dtype)
    null_ref = torch.zeros(B, 4, image_size, image_size, device=device)

    # Euler integration from t=1 to t=0
    dt = -1.0 / num_steps
    use_cfg = cfg_text > 1.0 or (cfg_ref > 1.0 and ref is not None)

    for step in range(num_steps):
        t_val = 1.0 - step / num_steps
        t = torch.full((B,), t_val, device=device)

        if use_cfg:
            # Unconditional: no text, no ref
            v_uncond = model(x, t, null_text, null_ref)

            # Text only: text, no ref
            v_text = model(x, t, text_emb, null_ref)

            v_guided = v_uncond + cfg_text * (v_text - v_uncond)

            # If reference is provided, add ref guidance
            if ref is not None and cfg_ref > 1.0:
                v_both = model(x, t, text_emb, ref)
                v_guided = v_guided + cfg_ref * (v_both - v_text)
        else:
            v_guided = model(x, t, text_emb, ref)

        x = x + dt * v_guided

    return x.clamp(-1, 1)


def tensor_to_rgba_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a (4, H, W) tensor in [-1, 1] to an RGBA PIL Image."""
    arr = ((tensor + 1) * 127.5).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, "RGBA")


def main():
    parser = argparse.ArgumentParser(description="Generate sprites with alucard")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--ref", type=str, default=None, help="Path to reference/previous frame PNG")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of sampling steps")
    parser.add_argument("--cfg-text", type=float, default=5.0, help="Text guidance scale")
    parser.add_argument("--cfg-ref", type=float, default=2.0, help="Reference guidance scale")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Load model
    from alucard.model import UNet

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UNet().to(device)
    model.load_state_dict(ckpt.get("ema_model", ckpt.get("model")))
    model.eval()

    # Encode text with CLIP
    import open_clip

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()

    tokens = tokenizer([args.prompt]).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    # Expand for multiple samples
    text_emb = text_emb.expand(args.num_samples, -1)

    # Load reference if provided
    ref = None
    if args.ref:
        from alucard.dataset import load_rgba

        ref = load_rgba(Path(args.ref)).unsqueeze(0).expand(args.num_samples, -1, -1, -1).to(device)

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Generate
    sprites = sample(
        model, text_emb, ref,
        num_steps=args.num_steps,
        cfg_text=args.cfg_text,
        cfg_ref=args.cfg_ref,
        device=device,
    )

    # Save
    output_path = Path(args.output)
    if args.num_samples == 1:
        img = tensor_to_rgba_image(sprites[0])
        img.save(output_path)
        print(f"Saved: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for i in range(args.num_samples):
            p = output_path.parent / f"{output_path.stem}_{i:03d}{output_path.suffix}"
            img = tensor_to_rgba_image(sprites[i])
            img.save(p)
            print(f"Saved: {p}")


if __name__ == "__main__":
    main()
