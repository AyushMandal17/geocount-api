#!/usr/bin/env bash
# Usage: deploy/push_to_space.sh https://<hf-user>:<hf-token>@huggingface.co/spaces/<hf-user>/<space-name>
set -euo pipefail
SPACE_URL="${1:?pass the Space git URL (with your HF user:token)}"
TMP="$(mktemp -d)"
git clone "$SPACE_URL" "$TMP/space"
rsync -a --delete --exclude '.git' --exclude 'tests' --exclude 'docs' --exclude '.github' \
      --exclude '__pycache__' --exclude '.pytest_cache' ./ "$TMP/space/"
cp deploy/hf_README.md "$TMP/space/README.md"
cd "$TMP/space"
git add -A
git commit -m "Deploy $(date -u +%F)" || echo "nothing to commit"
git push
