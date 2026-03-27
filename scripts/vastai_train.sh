#!/bin/bash
set -e

echo "=== Alucard Training on Vast.ai ==="

# Install alucard and dependencies
pip install -q pyarrow huggingface_hub
pip install -q git+https://github.com/evilsocket/alucard.git

# Setup directories
DATA_DIR=/workspace/data
CKPT_DIR=/workspace/checkpoints
mkdir -p $DATA_DIR $CKPT_DIR/samples

# Download and extract dataset
if [ ! -f "$DATA_DIR/clip_embeddings.pt" ]; then
    echo "=== Downloading dataset ==="
    python3 -u -c "
import io, torch
import pyarrow.parquet as pq
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
import open_clip

DATA = Path('$DATA_DIR')
idx = 0
for chunk_id in range(6):
    fname = f'data/train-{chunk_id:05d}-of-00006.parquet'
    local = hf_hub_download(repo_id='evilsocket/alucard-sprites', filename=fname, repo_type='dataset')
    table = pq.read_table(local)
    for row_idx in range(len(table)):
        img_struct = table.column('image')[row_idx].as_py()
        text = table.column('text')[row_idx].as_py()
        img = Image.open(io.BytesIO(img_struct['bytes']))
        img.save(DATA / f'sprite_{idx:06d}.png')
        (DATA / f'sprite_{idx:06d}.txt').write_text(text)
        idx += 1
    print(f'  Chunk {chunk_id + 1}/6: {len(table)} rows ({idx} total)', flush=True)
    del table
print(f'Extracted {idx} sprites', flush=True)

print('Computing CLIP embeddings...', flush=True)
device = torch.device('cuda')
clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model = clip_model.to(device).eval()

txt_files = sorted(DATA.glob('sprite_*.txt'))
all_embs = []
for i in range(0, len(txt_files), 512):
    batch = txt_files[i:i + 512]
    caps = [f.read_text().strip() for f in batch]
    tokens = tokenizer(caps).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        emb = clip_model.encode_text(tokens)
        emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().float()
    all_embs.append(emb)
    if (i + 512) % 50000 < 512:
        print(f'  {min(i + 512, len(txt_files))}/{len(txt_files)}', flush=True)

all_embs = torch.cat(all_embs, dim=0)
torch.save(all_embs, DATA / 'clip_embeddings.pt')
print(f'Saved clip_embeddings.pt: {all_embs.shape}', flush=True)
del clip_model, tokenizer, all_embs
torch.cuda.empty_cache()
"
fi

# Resume from checkpoint if available
RESUME=""
if [ -f "$CKPT_DIR/latest.pt" ]; then
    RESUME="--resume $CKPT_DIR/latest.pt"
    echo "=== Resuming from latest.pt ==="
elif [ -f "$CKPT_DIR/best.pt" ]; then
    RESUME="--resume $CKPT_DIR/best.pt"
    echo "=== Resuming from best.pt ==="
else
    echo "=== Starting fresh training ==="
fi

# Train - A100 can handle batch_size=64
python3 -u -m alucard.train \
    --data-dir $DATA_DIR \
    --output-dir $CKPT_DIR \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-4 \
    --ema-decay 0.9999 \
    --grad-accum 2 \
    --save-every 5 \
    --sample-every 10 \
    --num-workers 4 \
    $RESUME

echo "=== Training complete ==="
