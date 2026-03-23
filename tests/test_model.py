"""Tests for model, dataset, and training loop."""

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from alucard.model import UNet, timestep_embedding
from alucard.dataset import SpriteDataset, load_rgba
from alucard.train import flow_matching_loss
from alucard.sample import sample, tensor_to_rgba_image


def test_timestep_embedding():
    t = torch.tensor([0.0, 0.5, 1.0])
    emb = timestep_embedding(t, 128)
    assert emb.shape == (3, 128)
    # Different timesteps should produce different embeddings
    assert not torch.allclose(emb[0], emb[1])
    assert not torch.allclose(emb[1], emb[2])


def test_unet_shapes():
    model = UNet()
    B = 2
    x = torch.randn(B, 4, 128, 128)
    t = torch.rand(B)
    text = torch.randn(B, 512)

    # Without reference
    out = model(x, t, text)
    assert out.shape == (B, 4, 128, 128), f"Expected (2,4,128,128), got {out.shape}"

    # With reference
    ref = torch.randn(B, 4, 128, 128)
    out = model(x, t, text, ref)
    assert out.shape == (B, 4, 128, 128)


def test_unet_param_count():
    model = UNet()
    total = sum(p.numel() for p in model.parameters())
    # Should be ~32M params
    assert 25_000_000 < total < 50_000_000, f"Param count {total/1e6:.1f}M outside expected range"


def test_unet_output_zero_init():
    """Output should be near-zero at initialization (zero-init on output conv and gates)."""
    model = UNet()
    x = torch.randn(1, 4, 128, 128)
    t = torch.tensor([0.5])
    text = torch.randn(1, 512)

    with torch.no_grad():
        out = model(x, t, text)
    # With zero-init output conv, output should be near zero
    assert out.abs().mean() < 0.1, f"Initial output too large: {out.abs().mean():.4f}"


def test_flow_matching_loss():
    model = UNet()
    B = 4
    x_0 = torch.randn(B, 4, 128, 128)
    text_emb = torch.randn(B, 512)
    ref = torch.randn(B, 4, 128, 128)
    has_ref = torch.tensor([True, True, False, False])

    loss = flow_matching_loss(model, x_0, text_emb, ref, has_ref)
    assert loss.shape == ()
    assert loss.item() > 0
    assert torch.isfinite(loss)


def _create_test_dataset(tmp_dir: Path, n: int = 5):
    """Create a minimal dataset for testing."""
    for i in range(n):
        # Create random RGBA sprite
        arr = np.random.randint(0, 256, (128, 128, 4), dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")
        img.save(tmp_dir / f"sprite_{i:04d}.png")

        # Caption
        (tmp_dir / f"sprite_{i:04d}.txt").write_text(f"a pixel art sprite number {i}")

        # CLIP embedding (fake)
        emb = torch.randn(512)
        emb = emb / emb.norm()
        torch.save(emb, tmp_dir / f"sprite_{i:04d}.clip.pt")

        # Add previous frame for some
        if i > 0 and i % 2 == 0:
            arr_prev = np.random.randint(0, 256, (128, 128, 4), dtype=np.uint8)
            img_prev = Image.fromarray(arr_prev, "RGBA")
            img_prev.save(tmp_dir / f"sprite_{i:04d}.prev.png")


def test_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _create_test_dataset(tmp_dir, n=5)

        dataset = SpriteDataset(tmp_dir, augment=False)
        assert len(dataset) == 5

        item = dataset[0]
        assert item["image"].shape == (4, 128, 128)
        assert item["text_emb"].shape == (512,)
        assert item["ref"].shape == (4, 128, 128)
        assert isinstance(item["has_ref"], bool)

        # Check value range
        assert item["image"].min() >= -1.0
        assert item["image"].max() <= 1.0


def test_dataset_augmented():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _create_test_dataset(tmp_dir, n=5)

        dataset = SpriteDataset(tmp_dir, augment=True)
        item = dataset[0]
        assert item["image"].shape == (4, 128, 128)


def test_sample():
    model = UNet()
    model.eval()
    text_emb = torch.randn(2, 512)

    # Quick sample with few steps
    with torch.no_grad():
        sprites = sample(model, text_emb, num_steps=3, cfg_text=1.0, device="cpu")

    assert sprites.shape == (2, 4, 128, 128)
    assert sprites.min() >= -1.0
    assert sprites.max() <= 1.0


def test_sample_with_ref():
    model = UNet()
    model.eval()
    text_emb = torch.randn(2, 512)
    ref = torch.randn(2, 4, 128, 128)

    with torch.no_grad():
        sprites = sample(model, text_emb, ref=ref, num_steps=3, cfg_text=1.0, cfg_ref=1.0, device="cpu")

    assert sprites.shape == (2, 4, 128, 128)


def test_tensor_to_image():
    tensor = torch.randn(4, 128, 128).clamp(-1, 1)
    img = tensor_to_rgba_image(tensor)
    assert img.size == (128, 128)
    assert img.mode == "RGBA"


def test_overfit_single_sample():
    """Verify the training loop can overfit on a single sample."""
    torch.manual_seed(42)
    model = UNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Single sample
    x_0 = torch.randn(1, 4, 128, 128)
    text_emb = torch.randn(1, 512)
    ref = torch.zeros(1, 4, 128, 128)
    has_ref = torch.tensor([False])

    # Train for enough steps (flow matching has stochastic t sampling)
    losses = []
    for step in range(200):
        loss = flow_matching_loss(model, x_0, text_emb, ref, has_ref,
                                   text_drop_prob=0, ref_drop_prob=0, both_drop_prob=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Compare average of first 10 vs last 10 (reduces noise from stochastic t)
    early_avg = sum(losses[:10]) / 10
    late_avg = sum(losses[-10:]) / 10
    assert late_avg < early_avg, f"Loss didn't decrease: {early_avg:.4f} -> {late_avg:.4f}"
    print(f"Overfit test: loss {early_avg:.4f} -> {late_avg:.4f}")


if __name__ == "__main__":
    print("Running tests...")
    test_timestep_embedding()
    print("  [OK] timestep_embedding")
    test_unet_shapes()
    print("  [OK] unet_shapes")
    test_unet_param_count()
    print("  [OK] unet_param_count")
    test_unet_output_zero_init()
    print("  [OK] unet_output_zero_init")
    test_flow_matching_loss()
    print("  [OK] flow_matching_loss")
    test_dataset()
    print("  [OK] dataset")
    test_dataset_augmented()
    print("  [OK] dataset_augmented")
    test_sample()
    print("  [OK] sample")
    test_sample_with_ref()
    print("  [OK] sample_with_ref")
    test_tensor_to_image()
    print("  [OK] tensor_to_image")
    test_overfit_single_sample()
    print("  [OK] overfit_single_sample")
    print("\nAll tests passed!")
