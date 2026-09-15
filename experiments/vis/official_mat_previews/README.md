# Official `.mat` RGB previews

Source: `D:\Download\Results` (`/mnt/d/Download/Results`).

Layout:

```
experiments/vis/official_mat_previews/
  WV3_rr/<Model>/00.png ... 19.png   _contact.png
  WV3_fr/<Model>/...
  GF2_rr/<Model>/...
  GF2_fr/<Model>/...
  QB_rr/<Model>/...
  QB_fr/<Model>/...
  cave_x4/<Model>/...
  Results_for_Hyper_pan/<Model>/...
  WV3_rr/GT/                         PanCollection reduced GT
  WV3_rr/GT_vs_FusionMamba/          side-by-side
```

- 8-band (WV3): true color R/G/B = bands 5/3/2
- 4-band (GF2/QB): R/G/B = bands 3/2/1
- CAVE / hyperspectral: equally spaced false color
- `_contact.png` is the 20-image montage for that model
- Stretch is per-image 1–99 percentile, for viewing only (not for metrics)

Re-run / resume:

```bash
bash scripts/shortcuts/test/mat_to_rgb_preview.sh
```
