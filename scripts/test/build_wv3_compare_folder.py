#!/usr/bin/env python3
"""Build a self-contained WV3 RDM-PAN vs FusionMamba comparison folder."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/zhy/PROJECT/RefRWKV")
OUT = ROOT / "experiments/vis/wv3_rdm_vs_fusionmamba"
TEST = ROOT / "experiments/test/refsr"
RDM_PREV = TEST / "rdm_pan_preview/tile16_ov8"
FM_VIS = ROOT / "experiments/vis/fusion_mamba_official"
COST_JSON = TEST / "compare_pan_cost.json"

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)

PAPER = {
    "label": "Paper FusionMamba (reported)",
    "model": "fusion_mamba",
    "ckpt": "paper table",
    "infer": "official (paper)",
    "psnr": (39.374, 2.973),
    "q2n": (0.922, 0.084),
    "sam_deg": (2.843, 0.577),
    "ergas": (2.092, 0.510),
    "d_lambda": (0.0186, 0.0078),
    "d_s": (0.0269, 0.0058),
    "qnr": (0.9550, 0.0110),
    "note": "Numbers from the FusionMamba paper, not re-evaluated here.",
}

HIST_279 = {
    "label": "RDM-PAN epoch 279 (deleted ckpt)",
    "model": "rdm_pan",
    "ckpt": "epoch=0279 (file gone)",
    "infer": "LR tile=16 overlap=8",
    "d_lambda": (0.0219, None),
    "d_s": (0.0378, None),
    "qnr": (0.9412, None),
    "note": "Historical FR-only numbers from an earlier eval; checkpoint was deleted.",
}


def mean_std(rec: dict, key: str):
    val = rec.get(key)
    if isinstance(val, dict) and "mean" in val:
        return float(val["mean"]), (None if val.get("std") is None else float(val["std"]))
    if isinstance(val, (int, float)):
        return float(val), None
    return None, None


def fmt_pair(pair, digits=4):
    if not pair or pair[0] is None:
        return ""
    mean, std = pair
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def ckpt_short(path: str) -> str:
    if not path:
        return ""
    name = Path(path).name
    if "epoch=" in name:
        return name.replace(".ckpt", "")
    return name


def infer_desc(rec: dict) -> str:
    tile = rec.get("eval_tile_size")
    ov = rec.get("eval_tile_overlap")
    split = rec.get("split")
    if tile in (None, 0):
        return "no explicit tile (runner default / full image)"
    if int(tile) >= 1024 and int(ov or 0) == 0:
        if split == "test":
            return "full image (tile=1024 is a no-op on 64×64 LR / 256 PAN)"
        return "attempted full image on OrigScale (tile=1024, overlap=0)"
    if int(tile) == 64 and int(ov or 0) == 0:
        return "official FR cut: LR tile=64 overlap=0 (PAN 256)"
    if int(tile) == 16 and int(ov or 0) == 8:
        return "RDM-style tiles: LR tile=16 overlap=8 (PAN 64, 49 tiles on 256)"
    if int(tile) == 16 and int(ov or 0) == 0:
        return "no-overlap tiles: LR tile=16 overlap=0 (PAN 64)"
    return f"LR tile={tile} overlap={ov}"


def ds_protocol(rec: dict, path: str) -> str:
    ds, _ = mean_std(rec, "d_s")
    if ds is None:
        return "n/a"
    name = path.lower()
    if "toolbox" in name:
        return "toolbox MATLAB imresize (paper-matching)"
    if ds >= 0.05:
        return "torch bicubic Ds (NOT paper protocol; Ds inflated)"
    return "toolbox MATLAB imresize (paper-matching)"


def completeness(rec: dict) -> str:
    mets = set(rec.get("metrics") or [])
    split = rec.get("split")
    rr = {"psnr", "q2n", "ergas"} & mets and ({"sam_deg", "sam_rad", "sam"} & mets)
    fr = {"d_lambda", "d_s", "qnr"} <= mets
    if split == "test" and rr:
        return "full RR"
    if split == "test_hard" and fr:
        return "full FR"
    if mets <= {"psnr"} or mets == {"psnr"}:
        return "PSNR only (not a paper-table row)"
    if mets <= {"psnr", "ssim"}:
        return "PSNR/SSIM only (not a paper-table row)"
    return "partial: " + ",".join(rec.get("metrics") or [])


def load_runs():
    rows = []
    for p in sorted(TEST.rglob("metrics.json")):
        rel = str(p.relative_to(TEST))
        if rel.startswith("rdm_stf/"):
            continue
        data = json.loads(p.read_text())
        rec = data[0] if isinstance(data, list) else data
        ds_mean, ds_std = mean_std(rec, "d_s")
        row = {
            "run_dir": rel.replace("/metrics.json", ""),
            "model": rec.get("model") or "",
            "split": rec.get("split") or "",
            "checkpoint": rec.get("checkpoint") or "",
            "ckpt_name": ckpt_short(rec.get("checkpoint") or ""),
            "tile": rec.get("eval_tile_size"),
            "overlap": rec.get("eval_tile_overlap"),
            "infer": infer_desc(rec),
            "samples": rec.get("samples"),
            "metrics": ",".join(rec.get("metrics") or []),
            "completeness": completeness(rec),
            "ds_protocol": ds_protocol(rec, rel),
            "psnr": mean_std(rec, "psnr"),
            "ssim": mean_std(rec, "ssim"),
            "q2n": mean_std(rec, "q2n"),
            "sam_deg": mean_std(rec, "sam_deg"),
            "sam_rad": mean_std(rec, "sam_rad"),
            "ergas": mean_std(rec, "ergas"),
            "d_lambda": mean_std(rec, "d_lambda"),
            "d_s": (ds_mean, ds_std),
            "qnr": mean_std(rec, "qnr"),
            "source_json": str(p),
        }
        rows.append(row)
    return rows


def md_escape(text) -> str:
    return str(text).replace("|", "\\|")


def write_csv(path: Path, rows: list[dict]):
    fields = [
        "run_dir", "model", "split", "ckpt_name", "checkpoint",
        "tile", "overlap", "infer", "samples", "metrics",
        "completeness", "ds_protocol",
        "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
        "q2n_mean", "q2n_std", "sam_deg_mean", "sam_deg_std",
        "ergas_mean", "ergas_std",
        "d_lambda_mean", "d_lambda_std", "d_s_mean", "d_s_std",
        "qnr_mean", "qnr_std",
        "source_json",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {
                "run_dir": r["run_dir"],
                "model": r["model"],
                "split": r["split"],
                "ckpt_name": r["ckpt_name"],
                "checkpoint": r["checkpoint"],
                "tile": r["tile"],
                "overlap": r["overlap"],
                "infer": r["infer"],
                "samples": r["samples"],
                "metrics": r["metrics"],
                "completeness": r["completeness"],
                "ds_protocol": r["ds_protocol"],
                "source_json": r["source_json"],
            }
            for key in ("psnr", "ssim", "q2n", "sam_deg", "ergas", "d_lambda", "d_s", "qnr"):
                mean, std = r[key]
                out[f"{key}_mean"] = "" if mean is None else f"{mean:.10g}"
                out[f"{key}_std"] = "" if std is None else f"{std:.10g}"
            w.writerow(out)


def table_md(headers, body) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in body:
        lines.append("| " + " | ".join(md_escape(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def add_caption_bar(im: Image.Image, text: str, width: int) -> Image.Image:
    bar_h = 28
    canvas = Image.new("RGB", (width, im.height + bar_h), (18, 18, 18))
    canvas.paste(im, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), text, fill=(255, 255, 255), font=FONT_BOLD)
    return canvas


def hstack_labeled(parts: list[tuple[Image.Image, str]]) -> Image.Image:
    labeled = []
    for im, label in parts:
        rgb = im.convert("RGB")
        labeled.append(add_caption_bar(rgb, label, rgb.width))
    h = max(x.height for x in labeled)
    w = sum(x.width for x in labeled)
    out = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for tile in labeled:
        out.paste(tile, (x, 0))
        x += tile.width
    return out


def contact_sheet(images: list[Image.Image], cols: int, pad: int = 8) -> Image.Image:
    if not images:
        raise ValueError("no images")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + pad * (cols + 1), rows * h + pad * (rows + 1)), (30, 30, 30))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        if im.size != (w, h):
            im = im.resize((w, h), Image.Resampling.BILINEAR)
        sheet.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return sheet


def copy_png(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_images():
    sources = []
    rr_strips = []
    fr_strips = []
    for i in range(20):
        idx = f"{i:02d}"
        rdm_rr_name = f"test_wv3_multiExm1_{i:06d}.png"
        rdm_fr_name = f"test_wv3_OrigScale_multiExm1_{i:06d}.png"
        rdm_gt = RDM_PREV / "test" / f"test_wv3_multiExm1_{i:06d}_gt.png"
        rdm_rr = RDM_PREV / "test" / rdm_rr_name
        rdm_fr = RDM_PREV / "test_hard" / rdm_fr_name
        fm_rr = FM_VIS / "WV3_rr/FusionMamba" / f"{idx}.png"
        fm_gt = FM_VIS / "WV3_rr/GT" / f"{idx}.png"
        fm_fr = FM_VIS / "WV3_fr/FusionMamba" / f"{idx}.png"
        for p in (rdm_gt, rdm_rr, rdm_fr, fm_rr, fm_gt, fm_fr):
            if not p.exists():
                raise FileNotFoundError(p)

        rr_dir = OUT / "images/rr"
        fr_dir = OUT / "images/fr"
        copy_png(rdm_gt, rr_dir / f"{idx}_gt.png")
        copy_png(rdm_rr, rr_dir / f"{idx}_rdm_pan.png")
        copy_png(fm_rr, rr_dir / f"{idx}_fusion_mamba.png")
        copy_png(rdm_fr, fr_dir / f"{idx}_rdm_pan.png")
        copy_png(fm_fr, fr_dir / f"{idx}_fusion_mamba.png")

        gt = Image.open(rdm_gt)
        rdm = Image.open(rdm_rr)
        fm = Image.open(fm_rr)
        strip = hstack_labeled([(gt, f"{idx}  GT"), (rdm, f"{idx}  RDM-PAN"), (fm, f"{idx}  FusionMamba")])
        strip.save(rr_dir / f"{idx}_strip.png")
        rr_strips.append(strip)

        rdm_h = Image.open(rdm_fr)
        fm_h = Image.open(fm_fr)
        # FR official vis is 512; RDM preview on this machine used 256-cut batches
        # in cost script, but preview test_hard images should be 512 if dataset is OrigScale.
        if rdm_h.size != fm_h.size:
            fm_h_resized = fm_h.resize(rdm_h.size, Image.Resampling.BILINEAR)
            note = f"{idx}  FusionMamba (resized {fm_h.size}->{rdm_h.size})"
            strip_fr = hstack_labeled([(rdm_h, f"{idx}  RDM-PAN"), (fm_h_resized, note)])
        else:
            strip_fr = hstack_labeled([(rdm_h, f"{idx}  RDM-PAN"), (fm_h, f"{idx}  FusionMamba")])
        strip_fr.save(fr_dir / f"{idx}_strip.png")
        fr_strips.append(strip_fr)

        sources.append({
            "index": idx,
            "gt_rr": str(rdm_gt),
            "gt_rr_official_vis": str(fm_gt),
            "rdm_rr": str(rdm_rr),
            "fusion_mamba_rr": str(fm_rr),
            "rdm_fr": str(rdm_fr),
            "fusion_mamba_fr": str(fm_fr),
            "rdm_rr_size": list(rdm.size),
            "fm_rr_size": list(fm.size),
            "rdm_fr_size": list(rdm_h.size),
            "fm_fr_size": list(Image.open(fm_fr).size),
        })

    # Smaller contact sheets for overview
    rr_small = [im.resize((im.width // 2, im.height // 2), Image.Resampling.BILINEAR) for im in rr_strips]
    fr_small = [im.resize((im.width // 2, im.height // 2), Image.Resampling.BILINEAR) for im in fr_strips]
    contact_sheet(rr_small, cols=2).save(OUT / "images/rr/_contact_strips.png")
    contact_sheet(fr_small, cols=2).save(OUT / "images/fr/_contact_strips.png")
    (OUT / "images/sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    return sources


def load_cost():
    if not COST_JSON.exists():
        return []
    return json.loads(COST_JSON.read_text())


def write_tables(rows, sources):
    tables = OUT / "tables"
    write_csv(tables / "all_runs.csv", rows)

    # Headline RR / FR
    def find(model, ckpt_sub, split, tile, ov, completeness_prefix=None):
        hits = []
        for r in rows:
            if r["model"] != model or r["split"] != split:
                continue
            if ckpt_sub not in (r["ckpt_name"] or "") and ckpt_sub not in (r["checkpoint"] or ""):
                continue
            if tile is not None and r["tile"] != tile:
                continue
            if ov is not None and r["overlap"] != ov:
                continue
            if completeness_prefix and not r["completeness"].startswith(completeness_prefix):
                continue
            hits.append(r)
        return hits

    fm420_rr = find("fusion_mamba", "420.ckpt", "test", 1024, 0, "full RR")[0]
    fm420_fr = find("fusion_mamba", "420.ckpt", "test_hard", 64, 0, "full FR")
    # prefer toolbox Ds
    fm420_fr_tb = [r for r in fm420_fr if "inflated" not in r["ds_protocol"]][0]
    fm_last_rr = find("fusion_mamba", "last.ckpt", "test", 1024, 0, "full RR")[0]
    fm_last_fr = [r for r in find("fusion_mamba", "last.ckpt", "test_hard", 64, 0, "full FR") if "inflated" not in r["ds_protocol"]][0]
    def first(*queries):
        for args in queries:
            hits = find(*args)
            if hits:
                return hits[0]
        raise FileNotFoundError(f"no metrics for {queries}")

    # last.ckpt == epoch 439; the last_ov8 dump may have been deleted.
    rdm_last_rr = first(
        ("rdm_pan", "last.ckpt", "test", 16, 8, "full RR"),
        ("rdm_pan", "epoch=0439", "test", 16, 8, "full RR"),
    )
    rdm_last_fr = first(
        ("rdm_pan", "last.ckpt", "test_hard", 16, 8, "full FR"),
        ("rdm_pan", "epoch=0439", "test_hard", 16, 8, "full FR"),
    )
    rdm409_rr = find("rdm_pan", "epoch=0409", "test", 16, 8, "full RR")[0]
    rdm409_fr = find("rdm_pan", "epoch=0409", "test_hard", 16, 8, "full FR")[0]
    rdm419_rr = find("rdm_pan", "epoch=0419", "test", 16, 8, "full RR")[0]
    rdm419_fr = find("rdm_pan", "epoch=0419", "test_hard", 16, 8, "full FR")[0]
    rdm439_rr = find("rdm_pan", "epoch=0439", "test", 16, 8, "full RR")[0]
    rdm439_fr = find("rdm_pan", "epoch=0439", "test_hard", 16, 8, "full FR")[0]

    headline = [
        ("Paper FusionMamba", "paper table", "official (paper)", PAPER, PAPER),
        ("FusionMamba official 420", "420.ckpt", "RR full image; FR LR tile=64 ov=0, toolbox Ds", fm420_rr, fm420_fr_tb),
        ("FusionMamba trained last", "last.ckpt", "RR full image; FR LR tile=64 ov=0, toolbox Ds", fm_last_rr, fm_last_fr),
        ("RDM-PAN last (= epoch 439)", "last.ckpt / epoch=0439", "LR tile=16 overlap=8 (both RR and FR)", rdm_last_rr, rdm_last_fr),
        ("RDM-PAN epoch 419", "epoch=0419-step=127260", "LR tile=16 overlap=8", rdm419_rr, rdm419_fr),
        ("RDM-PAN epoch 409", "epoch=0409-step=124230", "LR tile=16 overlap=8", rdm409_rr, rdm409_fr),
    ]

    rr_headers = ["Model", "Checkpoint", "Inference", "PSNR↑", "Q2n↑", "SAM°↓", "ERGAS↓"]
    rr_body = []
    for name, ckpt, infer, rr, _fr in headline:
        src = rr if name != "Paper FusionMamba" else PAPER
        rr_body.append([
            name, ckpt, infer,
            fmt_pair(src["psnr"] if isinstance(src, dict) and "psnr" in src else src.get("psnr")),
            fmt_pair(src["q2n"], 4),
            fmt_pair(src["sam_deg"], 4),
            fmt_pair(src["ergas"], 4),
        ])
        if name == "Paper FusionMamba":
            rr_body[-1][3] = fmt_pair(PAPER["psnr"], 3)
            rr_body[-1][4] = fmt_pair(PAPER["q2n"], 3)
            rr_body[-1][5] = fmt_pair(PAPER["sam_deg"], 3)
            rr_body[-1][6] = fmt_pair(PAPER["ergas"], 3)

    # Fix paper digits already set; for dict rows use fmt
    rr_body[0] = [
        "Paper FusionMamba", "paper table", "official (paper)",
        "39.374±2.973", "0.922±0.084", "2.843±0.577", "2.092±0.510",
    ]
    rr_body[1] = [
        "FusionMamba official 420", "420.ckpt", "full image (tile=1024 no-op)",
        fmt_pair(fm420_rr["psnr"]), fmt_pair(fm420_rr["q2n"]), fmt_pair(fm420_rr["sam_deg"]), fmt_pair(fm420_rr["ergas"]),
    ]
    rr_body[2] = [
        "FusionMamba trained last", "last.ckpt", "full image (tile=1024 no-op)",
        fmt_pair(fm_last_rr["psnr"]), fmt_pair(fm_last_rr["q2n"]), fmt_pair(fm_last_rr["sam_deg"]), fmt_pair(fm_last_rr["ergas"]),
    ]
    rr_body[3] = [
        "RDM-PAN last (= epoch 439)", "last.ckpt", "LR tile=16 overlap=8",
        fmt_pair(rdm_last_rr["psnr"]), fmt_pair(rdm_last_rr["q2n"]), fmt_pair(rdm_last_rr["sam_deg"]), fmt_pair(rdm_last_rr["ergas"]),
    ]
    rr_body[4] = [
        "RDM-PAN epoch 419", "epoch=0419-step=127260", "LR tile=16 overlap=8",
        fmt_pair(rdm419_rr["psnr"]), fmt_pair(rdm419_rr["q2n"]), fmt_pair(rdm419_rr["sam_deg"]), fmt_pair(rdm419_rr["ergas"]),
    ]
    rr_body[5] = [
        "RDM-PAN epoch 409", "epoch=0409-step=124230", "LR tile=16 overlap=8",
        fmt_pair(rdm409_rr["psnr"]), fmt_pair(rdm409_rr["q2n"]), fmt_pair(rdm409_rr["sam_deg"]), fmt_pair(rdm409_rr["ergas"]),
    ]

    fr_body = [
        ["Paper FusionMamba", "paper table", "official (paper)", "0.0186±0.0078", "0.0269±0.0058", "0.9550±0.0110", "paper"],
        ["FusionMamba official 420", "420.ckpt", "LR tile=64 overlap=0",
         fmt_pair(fm420_fr_tb["d_lambda"]), fmt_pair(fm420_fr_tb["d_s"]), fmt_pair(fm420_fr_tb["qnr"]),
         fm420_fr_tb["ds_protocol"]],
        ["FusionMamba trained last", "last.ckpt", "LR tile=64 overlap=0",
         fmt_pair(fm_last_fr["d_lambda"]), fmt_pair(fm_last_fr["d_s"]), fmt_pair(fm_last_fr["qnr"]),
         fm_last_fr["ds_protocol"]],
        ["RDM-PAN last (= epoch 439)", "last.ckpt", "LR tile=16 overlap=8",
         fmt_pair(rdm_last_fr["d_lambda"]), fmt_pair(rdm_last_fr["d_s"]), fmt_pair(rdm_last_fr["qnr"]),
         rdm_last_fr["ds_protocol"]],
        ["RDM-PAN epoch 419", "epoch=0419-step=127260", "LR tile=16 overlap=8",
         fmt_pair(rdm419_fr["d_lambda"]), fmt_pair(rdm419_fr["d_s"]), fmt_pair(rdm419_fr["qnr"]),
         rdm419_fr["ds_protocol"]],
        ["RDM-PAN epoch 409", "epoch=0409-step=124230", "LR tile=16 overlap=8",
         fmt_pair(rdm409_fr["d_lambda"]), fmt_pair(rdm409_fr["d_s"]), fmt_pair(rdm409_fr["qnr"]),
         rdm409_fr["ds_protocol"]],
        ["RDM-PAN epoch 279 (deleted)", "epoch=0279 (gone)", "LR tile=16 overlap=8",
         "0.0219", "0.0378", "0.9412", "historical FR-only; ckpt deleted"],
    ]

    rr_md = (
        "# Reduced-Resolution (RR) — PanCollection WV3 `test`\n\n"
        "- Split: 20 images, LR MS `8×64×64`, PAN `1×256×256`, HR `8×256×256`, scale x4.\n"
        "- Metrics: PSNR / Q2n / SAM° / ERGAS vs HR.\n"
        "- FusionMamba official protocol is **full 256 image** (tile=1024 is a no-op).\n"
        "- RDM-PAN uses **LR tile=16 / overlap=8** because Bi-WKV CUDA is unstable at T=256.\n\n"
        + table_md(rr_headers, rr_body)
        + "\n`rdm_pan_last_ov8` is numerically identical to `rdm_pan_ckpts/epoch0439` (`last.ckpt` = epoch 439).\n"
    )
    fr_md = (
        "# Full-Resolution (FR) — PanCollection WV3 `test_hard`\n\n"
        "- Split: 20 OrigScale images, LR MS `8×128×128`, PAN `1×512×512`, no HR GT.\n"
        "- Metrics: Dλ / Ds / QNR.\n"
        "- **Headline Ds must use the toolbox MATLAB `imresize` protocol.** Torch bicubic Ds (~0.07) is the wrong protocol and is listed only in the all-runs appendix.\n"
        "- FusionMamba official FR cut is LR tile=64 / overlap=0 (PAN 256). RDM-PAN is LR tile=16 / overlap=8.\n\n"
        + table_md(["Model", "Checkpoint", "Inference", "Dλ↓", "Ds↓", "QNR↑", "Ds protocol"], fr_body)
        + "\nEpoch 279 is **not** on disk; FR numbers are from an earlier eval log.\n"
    )
    (tables / "rr_metrics.md").write_text(rr_md)
    (tables / "fr_metrics.md").write_text(fr_md)

    cost = load_cost()
    cost_body = []
    for c in cost:
        flops_g = c["flops"] / 1e9
        params_m = c["params"] / 1e6
        note = c.get("protocol") or ""
        if c["split"] == "test_hard":
            note += " | COST SCRIPT INPUT WAS 64×64 MS / 256 PAN, not OrigScale 128/512"
        cost_body.append([
            c["model"],
            Path(c["checkpoint"]).name,
            c["infer"],
            c["split"],
            c.get("tiles"),
            f"{params_m:.3f}M",
            f"{flops_g:.2f}G",
            f"{c['ms_mean']:.1f}",
            f"{c['peak_mem_mb']:.1f}",
            note,
        ])
    cost_md = (
        "# Params / FLOPs / runtime\n\n"
        "Source: `experiments/test/refsr/compare_pan_cost.json`.\n\n"
        "- FlopCounterMode **undercounts** Mamba / WKV CUDA kernels. Treat **ms/img** as the real cost.\n"
        "- The cost script labelled `test_hard` but the batched tensors were still **64×64 MS / 256×256 PAN**, so the “FR” cost row is the same spatial size as RR.\n"
        "- RDM-PAN cannot do a fair full-image CUDA run at 256 because Bi-WKV CUDA breaks at T=256; tiled 16/8 is the production path.\n\n"
        + table_md(
            ["Model", "Ckpt", "Infer", "Split label", "Tiles", "Params", "FLOPs", "ms/img", "Peak MB", "Note"],
            cost_body,
        )
        + "\nSame 256-size image: FusionMamba official ~**52 ms**; same 16/8 tiling FusionMamba ~**2.4 s**, RDM-PAN ~**2.06 s**.\n"
    )
    (tables / "cost.md").write_text(cost_md)

    all_headers = [
        "Run dir", "Model", "Split", "Checkpoint", "Tile", "Overlap", "Inference",
        "Complete?", "Ds protocol", "PSNR", "Q2n", "SAM°", "ERGAS", "SSIM", "Dλ", "Ds", "QNR",
    ]
    all_body = []
    for r in rows:
        all_body.append([
            r["run_dir"], r["model"], r["split"], r["ckpt_name"],
            "" if r["tile"] is None else r["tile"],
            "" if r["overlap"] is None else r["overlap"],
            r["infer"], r["completeness"], r["ds_protocol"],
            fmt_pair(r["psnr"]), fmt_pair(r["q2n"]), fmt_pair(r["sam_deg"]),
            fmt_pair(r["ergas"]), fmt_pair(r["ssim"]),
            fmt_pair(r["d_lambda"]), fmt_pair(r["d_s"]), fmt_pair(r["qnr"]),
        ])
    all_md = (
        "# Every PanCollection WV3 metrics.json on disk\n\n"
        "Includes mid-run RDM-PAN checkpoints, the trained FusionMamba `last.ckpt`, official `420.ckpt`, "
        "and incomplete PSNR-only rows. STF / Wuhan runs are excluded.\n\n"
        "Duplicates: `rdm_pan_last_ov8` == `rdm_pan_ckpts/epoch0439`; "
        "`compare_wv3_fm_official/fusion_mamba_420` RR == `compare_wv3_fullimage/fusion_mamba_420` RR; "
        "`fusion_mamba_420_qnr_toolbox` FR == official-420 FR toolbox Ds; "
        "`compare_wv3_pan/fusion_mamba` last FR == `compare_wv3_fm_official/fusion_mamba_last` FR.\n\n"
        + table_md(all_headers, all_body)
    )
    (tables / "all_runs.md").write_text(all_md)

    size_note = ""
    if sources:
        s0 = sources[0]
        size_note = (
            f"- RR PNG size: RDM {s0['rdm_rr_size']}, FusionMamba vis {s0['fm_rr_size']}, GT {s0['rdm_rr_size']}.\n"
            f"- FR PNG size: RDM {s0['rdm_fr_size']}, FusionMamba vis {s0['fm_fr_size']}.\n"
        )
        if s0["rdm_fr_size"] != s0["fm_fr_size"]:
            size_note += (
                "- FR strip resizes FusionMamba to the RDM preview size so the two panels line up. "
                "Original FusionMamba FR PNGs are kept unresized as `images/fr/XX_fusion_mamba.png`.\n"
            )

    readme = f"""# WV3 comparison: RDM-PAN vs FusionMamba

Local-only visual + metric dump (PNG is gitignored). Rebuild with:

```bash
/home/zhy/miniconda3/envs/rwkv7/bin/python scripts/test/build_wv3_compare_folder.py
```

## What is in this folder

- `images/rr/` — 20 reduced-res samples: GT, RDM-PAN (`last.ckpt`, tile 16/8), FusionMamba official vis, and `GT | RDM | FM` strips.
- `images/fr/` — 20 full-res samples: RDM-PAN (`last.ckpt`, tile 16/8) and FusionMamba official vis (no GT).
- `tables/` — headline RR/FR tables, cost table, and **every** on-disk `metrics.json` for this task.

Image sources (copied, not re-inferred):

- RDM-PAN RGB: `experiments/test/refsr/rdm_pan_preview/tile16_ov8/{{test,test_hard}}` (clean RGB, no red grid; `*_grad.png` skipped).
- FusionMamba RGB: `experiments/vis/fusion_mamba_official/WV3_{{rr,fr}}/FusionMamba`.
- GT: RDM `*_gt.png` from the same `test` split (official vis GT is the same 20 RR images).

{size_note}

## Headline (use these numbers)

Reduced-resolution (`test`, 20 images):

{table_md(rr_headers, rr_body)}

Full-resolution (`test_hard`, 20 images; **toolbox Ds**):

{table_md(["Model", "Checkpoint", "Inference", "Dλ↓", "Ds↓", "QNR↑", "Ds protocol"], fr_body)}

## Cost (256-size image)

{table_md(["Model", "Ckpt", "Infer", "Split label", "Tiles", "Params", "FLOPs", "ms/img", "Peak MB", "Note"], cost_body)}

Read `tables/cost.md` before quoting FLOPs. The cost script’s `test_hard` rows are **not** 512 PAN.

## How to read the other runs

Full appendix: `tables/all_runs.md` and `tables/all_runs.csv`.

Conditions that change the number:

| Condition | What it means |
| --- | --- |
| FusionMamba RR `tile=1024 ov=0` | Full 256 image, official protocol. Matches the paper RR table. |
| FusionMamba FR `tile=64 ov=0` + toolbox Ds | Official FR cut. Matches the paper FR table for 420. |
| FusionMamba FR with Ds ≈ 0.07 / QNR ≈ 0.91 | **Wrong Ds protocol** (torch bicubic). Do not compare to the paper. |
| FusionMamba `tile=16 ov=8` | Forced to RDM tiling. RR stays close; FR Ds in that folder used the wrong protocol. |
| FusionMamba `tile=16 ov=0` (`compare_wv3_tile64_nooverlap`) | No-overlap tiles. RR drops (~39.02 / 38.96). |
| RDM-PAN any full-metric row | Always LR tile=16 overlap=8. CUDA Bi-WKV is not used on the full 256 sequence. |
| `rdm_pan_seams/full` and `compare_wv3_tiled` / remote `fusion_mamba/.../test` | Incomplete (PSNR or PSNR/SSIM only). Not paper-table rows. |
| epoch 279 | Checkpoint deleted; FR-only historical numbers. |

## Takeaway

- **RR:** RDM-PAN last is slightly ahead of official FusionMamba 420 (PSNR 39.399 vs 39.343, SAM 2.782 vs 2.816, ERGAS 2.058 vs 2.079). Q2n is a tie (~0.919).
- **FR:** FusionMamba 420 still leads (QNR 0.9569 vs RDM 0.9385). The gap is Dλ and Ds, not RR distortion.
- **Speed:** On the same 256 image, official FusionMamba is ~50× faster than tiled RDM-PAN, because RDM must emit 49 tiles. Under the *same* 16/8 tiling, RDM is a bit faster than FusionMamba (2.06 s vs 2.40 s) and uses fewer parameters (0.625M vs 0.735M).
"""
    (OUT / "README.md").write_text(readme)


def main():
    if OUT.exists():
        # keep the folder but rebuild contents
        for sub in ("tables", "images"):
            p = OUT / sub
            if p.exists():
                shutil.rmtree(p)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "images/rr").mkdir(parents=True, exist_ok=True)
    (OUT / "images/fr").mkdir(parents=True, exist_ok=True)
    rows = load_runs()
    sources = build_images()
    write_tables(rows, sources)
    n_rr = len(list((OUT / "images/rr").glob("*.png")))
    n_fr = len(list((OUT / "images/fr").glob("*.png")))
    print(f"wrote {OUT}")
    print(f"  runs={len(rows)}  rr_pngs={n_rr}  fr_pngs={n_fr}")
    print(f"  README={OUT / 'README.md'}")


if __name__ == "__main__":
    main()
