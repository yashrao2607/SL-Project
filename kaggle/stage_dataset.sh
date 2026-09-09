#!/usr/bin/env bash
# Re-stage the Kaggle dataset from the current project state (code + processed data + tuning + checkpoint)
# and upload a new version:  bash kaggle/stage_dataset.sh "message"
# A fresh staging directory is used every time because Windows may keep a previous upload's files locked.
set -e
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
D="kaggle/dataset_$(date +%s)"
mkdir -p "$D/project/data" "$D/project/results" "$D/weights"
cp -r src scripts tests reference "$D/project/"
cp pyproject.toml requirements.txt PRD.md "$D/project/"
cp -r data/processed "$D/project/data/processed"
cp -r results/tuning "$D/project/results/tuning"
cp data/pretrained/mat_pretrained_weights.pt "$D/weights/"
find "$D" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$D/project/src/molbench.egg-info"
printf '{\n  "title": "molbench-bundle",\n  "id": "dipurao/molbench-bundle",\n  "licenses": [{"name": "CC0-1.0"}]\n}\n' > "$D/dataset-metadata.json"
du -sh "$D/project" "$D/weights"
ls -la "$D/weights"
[ "${1:-}" = "--no-upload" ] && exit 0
kaggle datasets version -p "$D" --dir-mode zip -m "${1:-update}"
echo "STAGE_DONE $D"
