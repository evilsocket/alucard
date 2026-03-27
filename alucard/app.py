"""
Gradio web interface for Alucard sprite generation.

Usage:
    python -m alucard.app --checkpoint checkpoints/best.pt
    # or
    alucard-app --checkpoint checkpoints/best.pt
"""

import argparse

import gradio as gr
import torch
import numpy as np
from PIL import Image

from alucard import Alucard


def create_app(model: Alucard) -> gr.Blocks:
    def generate(
        prompt: str,
        ref_image: Image.Image | None,
        num_steps: int,
        cfg_text: float,
        cfg_ref: float,
        seed: int,
        num_samples: int,
    ) -> list[Image.Image]:
        if not prompt.strip():
            return []

        seed_val = seed if seed >= 0 else None
        ref = ref_image if ref_image is not None else None

        results = model(
            prompt,
            ref=ref,
            num_samples=num_samples,
            num_steps=num_steps,
            cfg_text=cfg_text,
            cfg_ref=cfg_ref,
            seed=seed_val,
        )

        if isinstance(results, Image.Image):
            results = [results]

        # Upscale 4x with nearest-neighbor for display
        upscaled = [img.resize((512, 512), Image.NEAREST) for img in results]
        return upscaled

    with gr.Blocks(title="Alucard - Sprite Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Alucard - Text to Sprite Generator")
        gr.Markdown("Generate 128x128 RGBA pixel art sprites from text prompts. "
                     "Optionally provide a reference image for animation frame generation.")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a pixel art knight sprite, idle pose, front view",
                    lines=2,
                )
                ref_image = gr.Image(
                    label="Reference Frame (optional - for animation)",
                    type="pil",
                    image_mode="RGBA",
                )
                with gr.Row():
                    num_samples = gr.Slider(1, 8, value=4, step=1, label="Number of Samples")
                    seed = gr.Number(value=42, label="Seed (-1 for random)", precision=0)
                with gr.Accordion("Advanced Settings", open=False):
                    num_steps = gr.Slider(5, 100, value=20, step=5, label="Sampling Steps")
                    cfg_text = gr.Slider(1.0, 15.0, value=5.0, step=0.5, label="Text Guidance (CFG)")
                    cfg_ref = gr.Slider(1.0, 10.0, value=2.0, step=0.5, label="Reference Guidance")

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Generated Sprites (4x upscaled for display)",
                    columns=4,
                    rows=2,
                    height="auto",
                    object_fit="contain",
                )

        gr.Examples(
            examples=[
                ["a pixel art knight character sprite, idle pose, front view"],
                ["a pixel art sword weapon sprite, game item"],
                ["a pixel art slime creature, game enemy sprite"],
                ["a pixel art fruit item sprite, food pickup"],
                ["a pixel art wizard character sprite, casting spell"],
                ["a pixel art treasure chest, game item, closed"],
                ["a pixel art dragon enemy, boss sprite"],
                ["a pixel art potion bottle, health item, red"],
            ],
            inputs=[prompt],
        )

        generate_btn.click(
            fn=generate,
            inputs=[prompt, ref_image, num_steps, cfg_text, cfg_ref, seed, num_samples],
            outputs=[gallery],
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Alucard Gradio web interface")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (.pt or .safetensors). "
                             "If not provided, downloads from HuggingFace.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    source = args.checkpoint or "evilsocket/alucard"
    print(f"Loading model from: {source}")
    model = Alucard.from_pretrained(source, device=args.device)
    print("Model loaded.")

    app = create_app(model)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
