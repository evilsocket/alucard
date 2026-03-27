"""
Training script for flow matching sprite generation.

Flow matching (rectified flow) objective:
    x_t = (1 - t) * x_0 + t * noise
    target = noise - x_0  (velocity)
    loss = MSE(model(x_t, t, text_emb, ref), target)

Supports dual classifier-free guidance dropout:
    - 10% text dropout (replace with null embedding)
    - 20% reference image dropout (replace with zeros)
    - 5% both dropped
"""

import argparse
import copy
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alucard.dataset import SpriteDataset
from alucard.model import UNet
from alucard.sample import sample

import sys

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[_FlushHandler(sys.stderr)])
logger = logging.getLogger(__name__)


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.lerp_(p, 1 - decay)


def flow_matching_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    text_emb: torch.Tensor,
    ref: torch.Tensor,
    has_ref: torch.Tensor,
    text_drop_prob: float = 0.10,
    ref_drop_prob: float = 0.20,
    both_drop_prob: float = 0.05,
    null_text_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute flow matching loss with dual CFG dropout."""
    B = x_0.shape[0]
    device = x_0.device

    # Sample timestep uniformly in [0, 1]
    t = torch.rand(B, device=device)

    # Sample noise
    noise = torch.randn_like(x_0)

    # Interpolate: x_t = (1 - t) * x_0 + t * noise
    t_expand = t[:, None, None, None]
    x_t = (1 - t_expand) * x_0 + t_expand * noise

    # Target velocity: v = noise - x_0
    target = noise - x_0

    # CFG dropout
    drop_rand = torch.rand(B, device=device)
    drop_both = drop_rand < both_drop_prob
    drop_text = (drop_rand >= both_drop_prob) & (drop_rand < both_drop_prob + text_drop_prob)
    drop_ref = (drop_rand >= both_drop_prob + text_drop_prob) & (
        drop_rand < both_drop_prob + text_drop_prob + ref_drop_prob
    )

    # Apply text dropout
    text_emb_masked = text_emb.clone()
    null_mask = drop_both | drop_text
    _null_emb = null_text_emb if null_text_emb is not None else model.null_text_emb
    if null_mask.any():
        text_emb_masked[null_mask] = _null_emb.to(text_emb.dtype)

    # Apply reference dropout (also drop for samples that have no ref)
    ref_masked = ref.clone()
    ref_null_mask = drop_both | drop_ref | ~has_ref
    if ref_null_mask.any():
        ref_masked[ref_null_mask] = 0.0

    # Forward
    pred = model(x_t, t, text_emb_masked, ref_masked)

    return F.mse_loss(pred, target)


def train(
    data_dir: str,
    output_dir: str = "checkpoints",
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-4,
    ema_decay: float = 0.9999,
    grad_accum: int = 2,
    save_every: int = 10,
    sample_every: int = 10,
    num_workers: int = 4,
    resume: str | None = None,
    wandb_project: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "samples").mkdir(exist_ok=True)

    # Dataset
    dataset = SpriteDataset(data_dir, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    # Model
    model = UNet().to(device)
    model.enable_gradient_checkpointing()
    ema_model = copy.deepcopy(model)
    ema_model.disable_gradient_checkpointing()
    ema_model.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Multi-GPU support
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logger.info(f"Using {num_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Helper to get underlying model (unwrap DataParallel)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    # LR scheduler
    total_steps = epochs * len(loader) // grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Resume
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_loss = ckpt.get("best_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}, best_loss {best_loss:.6f}")

    # Wandb
    if wandb_project:
        import wandb

        wandb.init(project=wandb_project, config={
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "ema_decay": ema_decay, "grad_accum": grad_accum,
            "total_params": total_params,
        })

    # Save config
    config = {
        "model": {"base_channels": 64, "channel_mults": [1, 2, 4, 4], "num_res_blocks": 2,
                   "attn_resolutions": [32, 16], "text_dim": 512, "image_size": 128},
        "training": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                      "ema_decay": ema_decay, "grad_accum": grad_accum},
    }
    (output_path / "config.json").write_text(json.dumps(config, indent=2))

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            x_0 = batch["image"].to(device)
            text_emb = batch["text_emb"].to(device)
            ref = batch["ref"].to(device)
            has_ref = batch["has_ref"].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = flow_matching_loss(model, x_0, text_emb, ref, has_ref,
                                          null_text_emb=raw_model.null_text_emb)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                update_ema(ema_model, raw_model, ema_decay)
                global_step += 1

            epoch_loss += loss.item() * grad_accum
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | Step: {global_step}")

        if wandb_project:
            import wandb

            wandb.log({"loss": avg_loss, "lr": current_lr, "epoch": epoch + 1}, step=global_step)

        # Save checkpoint on best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = output_path / "best.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "best_loss": best_loss,
                "model": raw_model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            logger.info(f"New best loss {best_loss:.6f}, saved: {ckpt_path}")

        # Save periodic checkpoint for resume
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ckpt_path = output_path / "latest.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "best_loss": best_loss,
                "model": raw_model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path} (epoch {epoch+1})")

        # Generate samples
        if (epoch + 1) % sample_every == 0:
            ema_model.eval()
            with torch.no_grad():
                # Use first batch's text embeddings for consistent comparison
                sample_batch = next(iter(loader))
                sample_text = sample_batch["text_emb"][:4].to(device)
                samples = sample(ema_model, text_emb=sample_text, num_steps=20, device=device)
                # Save as grid
                from torchvision.utils import save_image
                # Denormalize from [-1, 1] to [0, 1]
                samples_vis = (samples[:, :3] + 1) / 2  # RGB only for visualization
                save_image(samples_vis, output_path / "samples" / f"epoch_{epoch+1:04d}.png", nrow=2)
            model.train()

    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train alucard sprite generator")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to sprite dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name")
    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
