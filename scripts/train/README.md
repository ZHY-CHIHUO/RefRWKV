# `scripts/train/`

这里是训练 Python 入口：

| 文件 | 内容 |
|---|---|
| `sr.py` | SwinIR、EDSR、RCAN、HAT、MambaIRv2 等单图 SR。 |
| `refsr.py` | TTSR、MASA-SR、DATSR 等 direct RefSR baseline。 |
| `refsrwkv.py` | RefSRWKV。 |
| `refdiffrwkv.py` | RefDiffRWKV。 |

入口接收训练 YAML 和可选 `--overrides`。常用模型/数据组合可通过 `scripts/shortcuts/train/` 的 shell 快捷脚本启动。
