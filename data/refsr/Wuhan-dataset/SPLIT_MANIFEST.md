# Wuhan split manifest

The original release contains 10 temporal-pair directories under `train` and
5 official test pairs.  Two complete temporal pairs were held out from the
training set for validation (20%), leaving eight training pairs:

| split | temporal-pair directories |
| --- | ---: |
| train | 8 |
| val | 2 |
| test (official) | 5 |

Validation holdout (fixed complete-pair selection; it does not consume a
pair whose reverse direction is already in the official test list):

- `20180109_20190522`
- `20191114_20210118`

The source copy at `/home/zhy/PROJECT/Wuhan-dataset` is left untouched.  Each
pair contains `G_date1`, `G_date2`, `L_date1`, and `L_date2` GeoTIFF files;
`G`/`L` are already aligned at 1000×1000 pixels.  The loader uses tensor
`scale=1` and normalizes uint16 values by 11848.
