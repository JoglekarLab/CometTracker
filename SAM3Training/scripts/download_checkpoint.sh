#!/usr/bin/env bash
# Download the gated checkpoint after the user has run `hf auth login`.
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/nfs/turbo/umms-ajitj/checkpoints/sam3}"
mkdir -p "$CHECKPOINT_DIR"

python - "$CHECKPOINT_DIR" <<'PY'
from pathlib import Path
import shutil
import sys
from huggingface_hub import hf_hub_download

destination = Path(sys.argv[1]) / "sam3.pt"
cached = Path(hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt"))
if cached.resolve() != destination.resolve():
    shutil.copy2(cached, destination)
print(destination)
PY

echo "Set this for jobs:"
echo "export SAM3_CHECKPOINT=$CHECKPOINT_DIR/sam3.pt"

