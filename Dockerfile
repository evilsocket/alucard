FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || true

# Copy source
COPY alucard/ alucard/
COPY scripts/ scripts/

# Install package
RUN pip install --no-cache-dir -e .

# Default: run training
ENTRYPOINT ["python3", "-m"]
CMD ["alucard.train", "--help"]
