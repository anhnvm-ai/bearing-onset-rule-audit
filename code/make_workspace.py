"""Lay this repository out so the analysis scripts resolve their relative paths.

Every script computes the project root as the parent of its own directory and reads results/tables, data/ and
colab_bundle/ from there. This copies code/scripts and code/src to the repository root, writes the record-index files
where the scripts expect them, and creates the folders into which the exporters write.

    python code/make_workspace.py            # into the repository root
    python code/make_workspace.py --dest DIR # into DIR (results/, data/ and plan/ are copied there too)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=REPO)
    dest = ap.parse_args().dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("scripts", "src"):
        shutil.copytree(HERE / name, dest / name, dirs_exist_ok=True)
        print(f"  {name}/ -> {dest / name}")
    if dest != REPO:
        for name in ("results", "data", "plan", "environment"):
            shutil.copytree(REPO / name, dest / name, dirs_exist_ok=True)
            print(f"  {name}/ -> {dest / name}")
    # the exporters write LaTeX files and figures into the manuscript folders, which are not released
    for name in ("paper/access_resub/figures", "paper/access_rev/figures", "paper/ieee", "paper/journal"):
        (dest / name).mkdir(parents=True, exist_ok=True)
    for f in sorted((REPO / "data" / "derived").glob("record_index_*.csv")):
        out = dest / "colab_bundle" / f.stem.replace("record_index_", "")
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / "records.csv")
        print(f"  {f.name} -> {out / 'records.csv'}")
    print(f"\nworkspace ready in {dest}")
    print("The raw windows (colab_bundle/<dataset>/windows.npy) are not released; data/README.md says how to rebuild them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
