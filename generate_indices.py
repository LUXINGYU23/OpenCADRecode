#!/usr/bin/env python3
"""
Dataset index generation script.

Generates train_index.json / val_index.json (and optionally test_index.json) plus a merged index.json
under the dataset root. Each entry follows the same schema as utils.utils.load_data_index() expects
when it falls back to directory scanning:

[
  {
    "sample_id": "batch_0000_part123",   # unique id (batch prefix + stem or plain stem)
    "code_path": "data/train/batch_0000/part123.py",  # path to .py file
    "relative_path": "batch_0000/part123.py",         # path relative to split dir
    "split": "train"
  },
  ...
]

Usage:
  python generate_indices.py --data-root data --splits train val
  python generate_indices.py --data-root data --splits train val --overwrite
  python generate_indices.py --data-root data --splits train val test --pattern "batch_*"

Notes:
- Will scan both top-level *.py files and any batch_* style subdirectories (pattern configurable).
- Skips files whose stem appears in error_samples.json (if present in split directory).
- Safe by default: will not overwrite existing index files unless --overwrite is passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Iterable, Set

DEFAULT_BATCH_PATTERN = "batch_*"


def load_error_samples(split_dir: Path) -> Set[str]:
    f = split_dir / "error_samples.json"
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(str(x) for x in data)
        except Exception:
            pass
    return set()


def iter_code_files(split_dir: Path, batch_pattern: str) -> Iterable[Path]:
    # batch subfolders
    for sub in split_dir.glob(batch_pattern):
        if sub.is_dir():
            for py in sub.glob("*.py"):
                yield py
    # top-level .py files
    for py in split_dir.glob("*.py"):
        if py.is_file():
            yield py


def build_index_for_split(root: Path, split: str, batch_pattern: str) -> List[Dict[str, Any]]:
    split_dir = root / split
    if not split_dir.exists():
        print(f"[WARN] Split directory missing: {split_dir}")
        return []
    error_samples = load_error_samples(split_dir)
    entries: List[Dict[str, Any]] = []
    for py_file in iter_code_files(split_dir, batch_pattern):
        stem = py_file.stem
        if stem in error_samples:
            continue
        # sample id logic: if inside batch dir, prefix with folder name
        try:
            rel_to_split = py_file.relative_to(split_dir)
        except ValueError:
            rel_to_split = py_file.name
        parts = rel_to_split.parts
        if len(parts) > 1:  # inside subdir
            sample_id = f"{parts[0]}_{stem}"
        else:
            sample_id = stem
        entries.append({
            "sample_id": sample_id,
            "code_path": str(py_file),  # keep relative-with-root path
            "relative_path": str(rel_to_split),
            "split": split
        })
    entries.sort(key=lambda x: x["sample_id"])
    print(f"[INFO] Split {split}: {len(entries)} samples (excluded {len(error_samples)} errors)")
    return entries


def write_index(file_path: Path, data: List[Dict[str, Any]], overwrite: bool):
    if file_path.exists() and not overwrite:
        print(f"[SKIP] {file_path} exists (use --overwrite to replace)")
        return
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {file_path} ({len(data)} entries)")


def parse_args():
    p = argparse.ArgumentParser(description="Generate dataset index json files for splits.")
    p.add_argument("--data-root", default="data", help="Root data directory containing split subfolders")
    p.add_argument("--splits", nargs="*", default=["train", "val"], help="Splits to index (e.g. train val test)")
    p.add_argument("--batch-pattern", default=DEFAULT_BATCH_PATTERN, help="Glob pattern for batch subdirectories")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing index files")
    p.add_argument("--merged-name", default="index.json", help="Filename for merged index")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    if not root.exists():
        raise SystemExit(f"Data root not found: {root}")

    merged: List[Dict[str, Any]] = []
    for split in args.splits:
        entries = build_index_for_split(root, split, args.batch_pattern)
        if not entries:
            continue
        split_index_file = root / f"{split}_index.json"
        write_index(split_index_file, entries, args.overwrite)
        merged.extend(entries)

    if merged:
        merged_file = root / args.merged_name
        write_index(merged_file, merged, args.overwrite)
    else:
        print("[WARN] No entries generated; nothing written.")


if __name__ == "__main__":
    main()
