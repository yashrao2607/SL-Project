"""Create molbench_colab_bundle.zip for notebooks/colab_tier2.ipynb (code + processed data + splits + features)."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["pyproject.toml", "requirements.txt", "PRD.md", "src", "scripts", "tests", "reference",
           "data/processed", "results/tuning"]


def main():
    out = ROOT / "molbench_colab_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            if p.is_file():
                z.write(p, item)
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and "__pycache__" not in f.parts and not f.name.endswith(".pyc"):
                        z.write(f, f.relative_to(ROOT).as_posix())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
