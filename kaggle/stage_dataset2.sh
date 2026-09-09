#!/usr/bin/env bash
# Create (or version) a private copy of the bundle dataset under the SECOND Kaggle account
# (credential in C:/Users/yashr/.kaggle2/kaggle.json). Stages independently of kaggle/dataset.
#   bash kaggle/stage_dataset2.sh [message]
set -e
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
export KAGGLE_CONFIG_DIR="C:/Users/yashr/.kaggle2"
U2=$(sed -n 's/.*"username": *"\([^"]*\)".*/\1/p' "C:/Users/yashr/.kaggle2/kaggle.json")
[ -n "$U2" ] || { echo "could not read username from kaggle.json"; exit 1; }
D=kaggle/dataset2
rm -rf "$D/project"
mkdir -p "$D/project/data" "$D/project/results" "$D/weights"
cp -r src scripts tests reference "$D/project/"
cp pyproject.toml requirements.txt PRD.md "$D/project/"
cp -r data/processed "$D/project/data/processed"
cp -r results/tuning "$D/project/results/tuning"
[ -f "$D/weights/mat_pretrained_weights.pt" ] || cp data/pretrained/mat_pretrained_weights.pt "$D/weights/"
find "$D" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$D/project/src/molbench.egg-info"
cat > "$D/dataset-metadata.json" <<EOF
{
  "title": "molbench-bundle",
  "id": "$U2/molbench-bundle",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
du -sh "$D/project" "$D/weights"
if kaggle datasets status "$U2/molbench-bundle" >/dev/null 2>&1; then
  kaggle datasets version -p "$D" --dir-mode zip -m "${1:-update}"
else
  kaggle datasets create -p "$D" --dir-mode zip
fi
