"""
Pre-compute CLIP text embeddings for all captions in a sprite dataset.

Expects:
    data_dir/
        sprite_0001.png
        sprite_0001.txt   # text caption

Produces:
    data_dir/
        sprite_0001.clip.pt  # (512,) float32 tensor
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CLIP embeddings for sprite captions")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with .png and .txt files")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32", help="CLIP model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true", help="Overwrite existing embeddings")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data_dir = Path(args.data_dir)

    # Find all caption files
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt caption files found in {data_dir}")
        return

    # Filter to only those needing processing
    to_process = []
    for txt_path in txt_files:
        clip_path = txt_path.with_suffix(".clip.pt")
        png_path = txt_path.with_suffix(".png")
        if not png_path.exists():
            continue
        if clip_path.exists() and not args.force:
            continue
        to_process.append(txt_path)

    if not to_process:
        print("All embeddings already computed. Use --force to recompute.")
        return

    print(f"Computing embeddings for {len(to_process)} captions...")

    # Load CLIP
    import open_clip

    clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained="openai")
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    clip_model = clip_model.to(device).eval()

    # Process in batches
    for i in range(0, len(to_process), args.batch_size):
        batch_paths = to_process[i : i + args.batch_size]
        captions = []
        for txt_path in batch_paths:
            caption = txt_path.read_text().strip()
            captions.append(caption)

        tokens = tokenizer(captions).to(device)
        with torch.no_grad():
            embeddings = clip_model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.cpu().float()

        for j, txt_path in enumerate(batch_paths):
            clip_path = txt_path.with_suffix(".clip.pt")
            torch.save(embeddings[j], clip_path)

        processed = min(i + args.batch_size, len(to_process))
        print(f"  {processed}/{len(to_process)}")

    print("Done.")


if __name__ == "__main__":
    main()
