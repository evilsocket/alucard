"""
UNet with AdaLN-Zero conditioning for flow matching sprite generation.

Architecture:
- 8-channel input (4 noisy RGBA + 4 reference RGBA)
- 4-channel output (predicted velocity in RGBA space)
- ~40M parameters
- AdaLN-Zero conditioning from text (CLIP 512-dim) + timestep
- Self-attention at 32x32 and 16x16 resolutions
- Channel multipliers [1, 2, 4, 4] with base=128
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding. t: (B,) float in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """Residual block with AdaLN-Zero conditioning."""

    def __init__(self, channels: int, cond_dim: int, out_channels: int | None = None, dropout: float = 0.0):
        super().__init__()
        out_channels = out_channels or channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # AdaLN-Zero: separate modulations for each norm (different channel counts)
        self.adaln1 = nn.Linear(cond_dim, 2 * channels)  # shift, scale for norm1
        self.adaln2 = nn.Linear(cond_dim, 3 * out_channels)  # shift, scale, gate for norm2
        nn.init.zeros_(self.adaln1.weight)
        nn.init.zeros_(self.adaln1.bias)
        nn.init.zeros_(self.adaln2.weight)
        nn.init.zeros_(self.adaln2.bias)

        if channels != out_channels:
            self.skip_proj = nn.Conv2d(channels, out_channels, 1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1 = self.adaln1(c).chunk(2, dim=-1)
        shift2, scale2, gate = self.adaln2(c).chunk(3, dim=-1)

        h = self.norm1(x)
        h = h * (1 + scale1[:, :, None, None]) + shift1[:, :, None, None]
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = h * (1 + scale2[:, :, None, None]) + shift2[:, :, None, None]
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        h = h * gate[:, :, None, None]  # zero-init gate

        return self.skip_proj(x) + h


class SelfAttention(nn.Module):
    """Multi-head self-attention for 2D feature maps."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNetLevel(nn.Module):
    """One encoder or decoder level: res blocks + optional attention."""

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if isinstance(block, ResBlock):
                x = block(x, c)
            else:
                x = block(x)
        return x


class UNet(nn.Module):
    """
    UNet for flow matching sprite generation.

    Args:
        in_channels: Input channels (8 = 4 noisy RGBA + 4 reference RGBA)
        out_channels: Output channels (4 = predicted velocity RGBA)
        base_channels: Base channel count (128)
        channel_mults: Per-level channel multipliers
        num_res_blocks: ResBlocks per level
        attn_resolutions: Resolutions at which to apply self-attention
        text_dim: Dimension of text conditioning (CLIP = 512)
        dropout: Dropout rate
        image_size: Input image size (for determining attention resolutions)
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 4,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (32, 16),
        text_dim: int = 512,
        dropout: float = 0.0,
        image_size: int = 128,
    ):
        super().__init__()
        self.image_size = image_size
        num_levels = len(channel_mults)
        cond_dim = base_channels * 4  # 512-dim conditioning space

        # Time + text conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.null_text_emb = nn.Parameter(torch.randn(text_dim) * 0.02)

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Build encoder
        self.encoder_blocks = nn.ModuleList()  # flat list of (ResBlock, optional Attn)
        self.downsamplers = nn.ModuleList()
        skip_channels = [base_channels]  # track channels for skip connections
        ch = base_channels
        res = image_size

        for level in range(num_levels):
            out_ch = base_channels * channel_mults[level]
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, cond_dim, out_ch, dropout)])
                if res in attn_resolutions:
                    block.append(SelfAttention(out_ch))
                self.encoder_blocks.append(block)
                ch = out_ch
                skip_channels.append(ch)
            if level < num_levels - 1:
                self.downsamplers.append(Downsample(ch))
                skip_channels.append(ch)
                res //= 2

        # Bottleneck
        self.mid_block1 = ResBlock(ch, cond_dim, ch, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, cond_dim, ch, dropout)

        # Build decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level in reversed(range(num_levels)):
            out_ch = base_channels * channel_mults[level]
            for i in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                block = nn.ModuleList([ResBlock(ch + skip_ch, cond_dim, out_ch, dropout)])
                if res in attn_resolutions:
                    block.append(SelfAttention(out_ch))
                self.decoder_blocks.append(block)
                ch = out_ch
            if level > 0:
                self.upsamplers.append(Upsample(ch))
                res *= 2

        # Output
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        # Store structure info for forward pass
        self._num_levels = num_levels
        self._num_res_blocks = num_res_blocks
        self._use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self._use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self._use_gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy image (B, 4, H, W)
            t: Timestep (B,) in [0, 1]
            text_emb: CLIP text embedding (B, 512)
            ref: Reference/previous frame (B, 4, H, W) or None (uses zeros)

        Returns:
            Predicted velocity (B, 4, H, W)
        """
        if ref is None:
            ref = torch.zeros_like(x)

        # Concatenate noisy image and reference
        h = torch.cat([x, ref], dim=1)  # (B, 8, H, W)

        # Conditioning: time + text
        t_emb = timestep_embedding(t, self.time_mlp[0].in_features)
        c = self.time_mlp(t_emb) + self.text_proj(text_emb)

        # Input
        h = self.input_conv(h)
        skips = [h]

        # Encoder
        block_idx = 0
        down_idx = 0
        ckpt = self._use_gradient_checkpointing and self.training
        for level in range(self._num_levels):
            for _ in range(self._num_res_blocks):
                layers = self.encoder_blocks[block_idx]
                if ckpt:
                    h = checkpoint(layers[0], h, c, use_reentrant=False)
                else:
                    h = layers[0](h, c)
                for layer in layers[1:]:  # optional attention
                    if ckpt:
                        h = checkpoint(layer, h, use_reentrant=False)
                    else:
                        h = layer(h)
                skips.append(h)
                block_idx += 1
            if down_idx < len(self.downsamplers):
                h = self.downsamplers[down_idx](h)
                skips.append(h)
                down_idx += 1

        # Bottleneck
        if ckpt:
            h = checkpoint(self.mid_block1, h, c, use_reentrant=False)
            h = checkpoint(self.mid_attn, h, use_reentrant=False)
            h = checkpoint(self.mid_block2, h, c, use_reentrant=False)
        else:
            h = self.mid_block1(h, c)
            h = self.mid_attn(h)
            h = self.mid_block2(h, c)

        # Decoder
        block_idx = 0
        up_idx = 0
        for level in reversed(range(self._num_levels)):
            for _ in range(self._num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                layers = self.decoder_blocks[block_idx]
                if ckpt:
                    h = checkpoint(layers[0], h, c, use_reentrant=False)
                else:
                    h = layers[0](h, c)
                for layer in layers[1:]:
                    if ckpt:
                        h = checkpoint(layer, h, use_reentrant=False)
                    else:
                        h = layer(h)
                block_idx += 1
            if up_idx < len(self.upsamplers):
                h = self.upsamplers[up_idx](h)
                up_idx += 1

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)
