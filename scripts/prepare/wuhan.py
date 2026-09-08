#!/usr/bin/env python3
"""Prepare the Wuhan temporal-pair dataset for RefRWKV.

The command copies (or, with ``--move``, moves) only the data files and keeps
the source repository's ``.git`` directory out of the project dataset root.
Validation is split by complete temporal-pair directories, never by patches.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


# These two directions have no reverse-pair counterpart in the official test
# folder. Keeping the validation protocol fixed makes the small dataset's
# results reproducible and avoids consuming a published test direction.
DEFAULT_VAL_PAIRS = ("20180109_20190522", "20191114_20210118")


def prepare(
    source: Path,
    destination: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    val_pairs: tuple[str, ...] | None = DEFAULT_VAL_PAIRS,
    move: bool = False,
) -> dict:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not (source / "train").is_dir() or not (source / "test").is_dir():
        raise FileNotFoundError("source must contain train/ and test/ directories")
    if destination == source or destination.is_relative_to(source):
        raise ValueError("destination must not be source or below source")
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        target = destination / split
        target.mkdir(exist_ok=True)
        for item in sorted((source / split).iterdir()):
            out = target / item.name
            if out.exists():
                continue
            operation = shutil.move if move else shutil.copytree
            operation(str(item), str(out))
    readme = source / "README.md"
    if readme.is_file() and not (destination / "README.md").exists():
        if move:
            shutil.move(str(readme), str(destination / "README.md"))
        else:
            shutil.copy2(readme, destination / "README.md")

    existing_manifest = destination / "split_manifest.json"
    if existing_manifest.is_file():
        try:
            saved = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved = None
        if isinstance(saved, dict) and all(
            isinstance(saved.get("splits", {}).get(split), list) for split in ("train", "val", "test")
        ):
            return saved

    train_dirs = sorted(path for path in (destination / "train").iterdir() if path.is_dir())
    if val_pairs is None:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1")
        count = max(1, int(round(len(train_dirs) * val_fraction)))
        held_out = sorted(random.Random(seed).sample(train_dirs, count), key=lambda path: path.name)
        selection: dict[str, object] = {
            "method": "seeded_random_complete_pair_holdout",
            "seed": seed,
            "val_fraction": val_fraction,
        }
    else:
        names = tuple(str(name) for name in val_pairs)
        known = {path.name: path for path in train_dirs}
        missing = sorted(set(names).difference(known))
        if missing:
            raise FileNotFoundError(
                "requested validation pairs are not present in train: " + ", ".join(missing)
            )
        held_out = [known[name] for name in names]
        selection = {"method": "fixed_complete_pair_holdout", "val_pairs": list(names)}
    val_dir = destination / "val"
    val_dir.mkdir(exist_ok=True)
    for item in held_out:
        target = val_dir / item.name
        if item.parent == val_dir:
            continue
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(item), str(target))
    manifest = {
        "selection": selection,
        "splits": {
            split: sorted(path.name for path in (destination / split).iterdir() if path.is_dir())
            for split in ("train", "val", "test")
        },
    }
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-pair",
        action="append",
        default=None,
        help="complete temporal-pair directory to hold out; repeatable. Omit to use the Wuhan fixed split.",
    )
    parser.add_argument(
        "--random-val",
        action="store_true",
        help="use --seed/--val-fraction instead of the fixed Wuhan validation pairs",
    )
    parser.add_argument("--move", action="store_true")
    args = parser.parse_args()
    val_pairs = None if args.random_val else tuple(args.val_pair or DEFAULT_VAL_PAIRS)
    print(json.dumps(prepare(args.source, args.destination, val_fraction=args.val_fraction, seed=args.seed, val_pairs=val_pairs, move=args.move), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
