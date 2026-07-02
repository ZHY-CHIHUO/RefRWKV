#!/usr/bin/env python
"""Export PyTorch Lightning/TensorBoard training scalars to CSV.

Usage:
    python scripts/export_tb_csv.py --logdir logs/sd2_control_ldm --output logs/sd2_control_ldm/scalars.csv
    python scripts/export_tb_csv.py --logdir logs --output train_scalars.csv
    如果你只想导出某些 tag，比如train/G_total,train/D_total
    python scripts/export_tb_csv.py --logdir logs/sd2_control_ldm --output logs/sd2_control_ldm/scalars.csv --tags train/G_total,train/D_total
"""

import argparse
import csv
import glob
import os
from pathlib import Path

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError as exc:
    raise ImportError(
        "tensorboard is required to parse event logs. Install it with `pip install tensorboard`."
    ) from exc


def find_event_files(logdir: Path):
    if logdir.is_file():
        return [logdir]

    files = list(logdir.rglob("events.out.tfevents.*"))
    return files


def load_scalars(event_path: Path):
    ea = event_accumulator.EventAccumulator(
        str(event_path),
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.GRAPH: 0,
        },
    )
    ea.Reload()
    return ea.scalars.Keys(), ea


def write_csv(output_path: Path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "step", "wall_time", "value"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar logs to CSV."
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="TensorBoard log directory or event file path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output file path. Defaults to <logdir>/scalars.csv.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Optional comma-separated scalar tags to export.",
    )
    args = parser.parse_args()

    logdir = Path(args.logdir)
    event_files = find_event_files(logdir)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {logdir}.")

    if args.output is None:
        output_path = logdir / "scalars.csv"
    else:
        output_path = Path(args.output)

    tags_filter = None
    if args.tags:
        tags_filter = {t.strip() for t in args.tags.split(",") if t.strip()}

    rows = []
    seen_tags = set()
    for event_file in sorted(event_files):
        tags, ea = load_scalars(event_file)
        for tag in tags:
            if tags_filter is not None and tag not in tags_filter:
                continue
            seen_tags.add(tag)
            values = ea.Scalars(tag)
            for item in values:
                rows.append([tag, item.step, item.wall_time, item.value])

    if not rows:
        raise RuntimeError(
            f"No scalar data exported from {len(event_files)} event files."
        )

    rows.sort(key=lambda x: (x[0], x[1]))
    write_csv(output_path, rows)

    print(
        f"Exported {len(rows)} scalar rows for {len(seen_tags)} tags to {output_path}"
    )
    if tags_filter is not None:
        print(f"Tags exported: {sorted(seen_tags)}")


if __name__ == "__main__":
    main()
