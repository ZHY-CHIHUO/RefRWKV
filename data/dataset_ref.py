import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import zoom
import json
import warnings


class RefPairedEnviMemmapDataset(Dataset):
    """
    遥感参考超分配对数据集（高分六号 PMS/WFV 专用）
    支持传入 ratios 进行空间按行划分。
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: int = 256,
        scale: int = 8,
        mode: str = "train",          # 'train', 'val', 'test'
        stride_ratio: float = 0.5,
        use_memmap: bool = True,
        augment: bool = True,
        ratios: tuple = None,
        norm_stats_file: str = None,
    ):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir
        self.patch_size = patch_size
        self.scale = scale
        self.mode = mode
        self.lr_patch_size = patch_size // scale
        self.augment = augment and (mode == "train")
        self.ratios = ratios
        self.norm_stats_file = norm_stats_file

        # 固定文件名
        self.input_name1 = "wfv1"
        self.target_name1 = "pms1"
        self.input_name2 = "wfv2"
        self.target_name2 = "pms2"

        # 解析头文件、空间参考、内存映射
        self.file_meta = self._parse_all_headers()
        required_files = [
            self.input_name1,
            self.target_name1,
            self.input_name2,
            self.target_name2,
        ]
        for name in required_files:
            if name not in self.file_meta:
                raise FileNotFoundError(f"File '{name}' not found in {self.raw_dir}")

        ref_meta = self.file_meta[self.target_name1]
        self.ref_lines = ref_meta["lines"]
        self.ref_samples = ref_meta["samples"]

        t2_meta = self.file_meta[self.target_name2]
        if t2_meta["lines"] != self.ref_lines or t2_meta["samples"] != self.ref_samples:
            warnings.warn(
                "Target images have different sizes. Patches will be aligned to target1."
            )

        for low_name in [self.input_name1, self.input_name2]:
            low_meta = self.file_meta[low_name]
            scale_row = low_meta["lines"] / self.ref_lines
            scale_col = low_meta["samples"] / self.ref_samples
            expected = 1.0 / scale
            if not np.isclose(scale_row, expected, atol=0.01) or not np.isclose(
                scale_col, expected, atol=0.01
            ):
                warnings.warn(
                    f"'{low_name}' scale ({scale_row:.4f}, {scale_col:.4f}) "
                    f"differs from expected 1/{scale} = {expected}"
                )

        self.use_memmap = use_memmap
        if use_memmap:
            self._init_mmaps()

        # 1. 生成完整样本列表（按步长）
        self.samples = self._build_sample_list(stride_ratio)

        # 2. 若指定了比例，则按 mode 过滤行区域
        if self.ratios is not None:
            self._apply_spatial_split()

        # 3. 归一化参数（优先使用外部文件）
        self.norm_params = self._load_or_compute_norm_params()

        # 打印信息
        print(f"RefPaired Dataset ready: {len(self.samples)} samples")
        print(f"  Mode: {self.mode} | LR1: {self.input_name1}  HR1: {self.target_name1}")
        print(f"  LR2: {self.input_name2}  HR2: {self.target_name2}")
        print(f"  Patch sizes: HR {patch_size}x{patch_size}, LR {self.lr_patch_size}x{self.lr_patch_size}")
        if self.augment:
            print(f"  Data augmentation: ON (random flip & rot90)")
        if self.ratios is not None:
            print(f"  Spatial split ratios: {self.ratios}")

    # ========== 空间划分 ==========
    def _apply_spatial_split(self):
        """
        根据 self.ratios 和 self.mode 按行坐标过滤样本。
        ratios 格式：(train_ratio, val_ratio, test_ratio)，三者之和可以 ≤ 1。
        剩余行（如果有）将被丢弃。
        """
        train_r, val_r, test_r = self.ratios
        total_ratio = train_r + val_r + test_r
        if total_ratio > 1.0 + 1e-6:
            raise ValueError(f"Ratios sum {total_ratio} exceeds 1.0, got {self.ratios}")

        rows = [s[0] for s in self.samples]
        max_row = max(rows)

        # 计算各个区域的结束行（累积比例）
        train_end = int(max_row * train_r)
        val_end   = int(max_row * (train_r + val_r))
        test_end  = int(max_row * (train_r + val_r + test_r))

        if self.mode == "train":
            row_min, row_max = 0, train_end
        elif self.mode == "val":
            row_min, row_max = train_end + 1, val_end
        elif self.mode == "test":
            row_min, row_max = val_end + 1, test_end
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # 如果总比例 < 1，test_end < max_row，后续行被丢弃
        new_samples = [(r, c) for (r, c) in self.samples if row_min <= r <= row_max]
        print(f"  Spatial split (mode={self.mode}): kept {len(new_samples)} of {len(self.samples)} "
            f"samples (rows {row_min}-{row_max})")
        self.samples = new_samples

    # ========== 归一化参数管理 ==========
    def _load_or_compute_norm_params(self):
        # 优先使用外部指定的归一化文件
        if self.norm_stats_file is not None:
            stats_path = Path(self.norm_stats_file)
        else:
            stats_path = self.data_dir / "norm_stats.json"

        if stats_path.exists():
            with open(stats_path, "r") as f:
                norm_params = json.load(f)
            print(f"Loaded normalization stats from {stats_path}")
            return norm_params
        else:
            print(f"{stats_path} not found. Computing from scratch for {self.data_dir}...")
            norm_params = self._compute_norm_params_full_scan()
            with open(stats_path, "w") as f:
                json.dump(norm_params, f, indent=2)
            print(f"Saved normalization stats to {stats_path}")
            return norm_params

    def _compute_norm_params_full_scan(self):
        input_files = [self.input_name1, self.input_name2]
        output_files = [self.target_name1, self.target_name2]
        input_min, input_max = self._merge_stats_for_files(input_files)
        output_min, output_max = self._merge_stats_for_files(output_files)
        input_min, input_max = self._fix_zero_range(input_min, input_max)
        output_min, output_max = self._fix_zero_range(output_min, output_max)
        return {
            "input": {"min": input_min.tolist(), "max": input_max.tolist()},
            "output": {"min": output_min.tolist(), "max": output_max.tolist()},
        }

    def _merge_stats_for_files(self, file_names):
        first_meta = self.file_meta[file_names[0]]
        n_bands = first_meta["bands"]
        global_min = np.full(n_bands, np.inf)
        global_max = np.full(n_bands, -np.inf)
        for name in file_names:
            meta = self.file_meta[name]
            if meta["bands"] != n_bands:
                raise ValueError(f"Inconsistent bands: {name}")
            shape = meta["shape"]
            dtype = meta["dtype"]
            dat_path = self.raw_dir / f"{name}.dat"
            mmap = np.memmap(dat_path, dtype=dtype, mode="r", shape=shape)
            lines = shape[1]
            samples = shape[2]
            print(f"  Scanning {name} ({lines} lines, {samples} samples)...")
            for r in range(lines):
                row_data = mmap[:, r, :]
                row_min = row_data.min(axis=1)
                row_max = row_data.max(axis=1)
                global_min = np.minimum(global_min, row_min)
                global_max = np.maximum(global_max, row_max)
                if (r + 1) % 1000 == 0:
                    print(f"    Processed {r+1}/{lines} lines")
            del mmap
            print(f"  Finished {name}")
        return global_min, global_max

    @staticmethod
    def _fix_zero_range(min_vals, max_vals):
        min_vals = min_vals.copy()
        max_vals = max_vals.copy()
        for i in range(len(min_vals)):
            if max_vals[i] - min_vals[i] == 0:
                if min_vals[i] != 0:
                    max_vals[i] = min_vals[i] + 1.0
                else:
                    max_vals[i] = 1.0
        return min_vals, max_vals

    # ==================== 内存映射与头文件解析（不变） ====================
    def _init_mmaps(self):
        self.mmaps = {}
        for name in [
            self.input_name1,
            self.target_name1,
            self.input_name2,
            self.target_name2,
        ]:
            dat_path = self.raw_dir / f"{name}.dat"
            meta = self.file_meta[name]
            self.mmaps[name] = np.memmap(
                dat_path, dtype=meta["dtype"], mode="r", shape=meta["shape"]
            )

    def _parse_envi_hdr(self, hdr_path):
        info = {}
        with open(hdr_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if "{" in val:
                    items = val.replace("{", "").replace("}", "").split(",")
                    parsed = []
                    for item in items:
                        item = item.strip()
                        if not item:
                            continue
                        try:
                            parsed.append(float(item) if "." in item else int(item))
                        except ValueError:
                            parsed.append(item)
                    info[key] = parsed
                else:
                    try:
                        info[key] = int(val)
                    except ValueError:
                        try:
                            info[key] = float(val)
                        except ValueError:
                            info[key] = val
        return info

    @staticmethod
    def _envi_dtype_to_numpy(dtype_code):
        mapping = {
            1: np.uint8,
            2: np.int16,
            3: np.int32,
            4: np.float32,
            5: np.float64,
            12: np.uint16,
            13: np.uint32,
            14: np.int64,
            15: np.uint64,
        }
        return mapping.get(dtype_code, np.float32)

    def _parse_all_headers(self):
        meta_dict = {}
        for hdr_path in self.raw_dir.glob("*.hdr"):
            name = hdr_path.stem
            info = self._parse_envi_hdr(hdr_path)
            samples = info.get("samples")
            lines = info.get("lines")
            bands = info.get("bands", 1)
            interleave = info.get("interleave", "bsq").lower()
            dtype_code = info.get("data type", 4)
            dtype = self._envi_dtype_to_numpy(dtype_code)
            if interleave == "bsq":
                shape = (bands, lines, samples)
            elif interleave == "bil":
                shape = (lines, bands, samples)
            elif interleave == "bip":
                shape = (lines, samples, bands)
            else:
                raise ValueError(f"Unsupported interleave: {interleave}")
            meta_dict[name] = {
                "shape": shape,
                "dtype": dtype,
                "bands": bands,
                "lines": lines,
                "samples": samples,
            }
        return meta_dict

    def _build_sample_list(self, stride_ratio):
        max_row = self.ref_lines - self.patch_size
        max_col = self.ref_samples - self.patch_size
        if max_row < 0 or max_col < 0:
            raise ValueError("Patch size larger than image dimensions")
        stride = (
            int(self.patch_size * stride_ratio)
            if self.mode == "train"
            else self.patch_size
        )
        stride = max(1, stride)
        samples = []
        for r in range(0, max_row + 1, stride):
            for c in range(0, max_col + 1, stride):
                samples.append((r, c))
        return samples

    def __len__(self):
        return len(self.samples)

    def _extract_patch(self, name, row, col, output_size):
        meta = self.file_meta[name]
        bands = meta["bands"]
        lines = meta["lines"]
        samples = meta["samples"]
        scale_row = lines / self.ref_lines
        scale_col = samples / self.ref_samples
        read_h = int(np.ceil(output_size / scale_row))
        read_w = int(np.ceil(output_size / scale_col))
        orig_row_start = int(row * scale_row)
        orig_col_start = int(col * scale_col)
        orig_row_start = max(0, min(orig_row_start, lines - read_h))
        orig_col_start = max(0, min(orig_col_start, samples - read_w))
        if self.use_memmap:
            mmap = self.mmaps[name]
        else:
            dat_path = self.raw_dir / f"{name}.dat"
            mmap = np.memmap(
                dat_path, dtype=meta["dtype"], mode="r", shape=meta["shape"]
            )
        patch = mmap[
            :,
            orig_row_start : orig_row_start + read_h,
            orig_col_start : orig_col_start + read_w,
        ].copy()
        if patch.shape[1] != output_size or patch.shape[2] != output_size:
            zoom_h = output_size / patch.shape[1]
            zoom_w = output_size / patch.shape[2]
            patch = zoom(patch, (1, zoom_h, zoom_w), order=1)
        return patch

    def _normalize(self, img, img_type):
        img = img.astype(np.float32)
        C = img.shape[0]
        mins = np.array(self.norm_params[img_type]["min"][:C])
        maxs = np.array(self.norm_params[img_type]["max"][:C])
        ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
        for c in range(C):
            img[c] = (img[c] - mins[c]) / ranges[c]
        img = np.clip(img, 0.0, 1.0)
        return img

    def denormalize_output(self, norm_img):
        is_tensor = torch.is_tensor(norm_img)
        if is_tensor:
            norm_img = norm_img.cpu().numpy()
        orig_shape = norm_img.shape
        if len(orig_shape) == 4:
            B, C, H, W = orig_shape
            norm_img = norm_img.reshape(B, C, -1)
        else:
            C, H, W = orig_shape
            norm_img = norm_img.reshape(C, -1)
        mins = np.array(self.norm_params["output"]["min"][:C]).reshape(-1, 1)
        maxs = np.array(self.norm_params["output"]["max"][:C]).reshape(-1, 1)
        ranges = maxs - mins
        orig = norm_img * ranges + mins
        if len(orig_shape) == 4:
            orig = orig.reshape(B, C, H, W)
        else:
            orig = orig.reshape(C, H, W)
        if is_tensor:
            orig = torch.from_numpy(orig)
        return orig

    # ==================== 新增数据增强核心方法 ====================
    def _augment(self, lr1, hr1, lr2, hr2):
        """
        对四张影像（numpy数组，形状 [C, H, W]）同步施加随机空间变换。
        包含：水平翻转、垂直翻转、90°倍数旋转。
        返回：变换后的四个数组（保证内存连续）。
        """
        # 随机水平翻转（概率0.5）
        if np.random.rand() > 0.5:
            lr1 = np.flip(lr1, axis=2)
            hr1 = np.flip(hr1, axis=2)
            lr2 = np.flip(lr2, axis=2)
            hr2 = np.flip(hr2, axis=2)

        # 随机垂直翻转（概率0.5）
        if np.random.rand() > 0.5:
            lr1 = np.flip(lr1, axis=1)
            hr1 = np.flip(hr1, axis=1)
            lr2 = np.flip(lr2, axis=1)
            hr2 = np.flip(hr2, axis=1)

        # 随机90°旋转（0, 90, 180, 270 度等概率）
        k = np.random.randint(0, 4)
        if k > 0:
            lr1 = np.rot90(lr1, k, axes=(1, 2))
            hr1 = np.rot90(hr1, k, axes=(1, 2))
            lr2 = np.rot90(lr2, k, axes=(1, 2))
            hr2 = np.rot90(hr2, k, axes=(1, 2))

        # 保证转换为Tensor时内存连续，避免报错
        return (
            np.ascontiguousarray(lr1),
            np.ascontiguousarray(hr1),
            np.ascontiguousarray(lr2),
            np.ascontiguousarray(hr2),
        )

    # ==================== 核心数据获取（集成增强） ====================
    def __getitem__(self, idx):
        row, col = self.samples[idx]

        # 提取原始patch
        lr1_raw = self._extract_patch(self.input_name1, row, col, self.lr_patch_size)
        hr1_raw = self._extract_patch(self.target_name1, row, col, self.patch_size)
        lr2_raw = self._extract_patch(self.input_name2, row, col, self.lr_patch_size)
        hr2_raw = self._extract_patch(self.target_name2, row, col, self.patch_size)

        # 归一化到 [0,1]
        lr1 = self._normalize(lr1_raw, "input")
        hr1 = self._normalize(hr1_raw, "output")
        lr2 = self._normalize(lr2_raw, "input")
        hr2 = self._normalize(hr2_raw, "output")

        # -------- 数据增强（训练时同步变换）---------
        if self.augment:
            lr1, hr1, lr2, hr2 = self._augment(lr1, hr1, lr2, hr2)

        # 转换为PyTorch Tensor
        return (
            torch.from_numpy(lr1),
            torch.from_numpy(hr1),
            torch.from_numpy(lr2),
            torch.from_numpy(hr2),
        )


# ================================= 使用示例 =================================
def main():
    # ======================== 1. 数据集测试参数 ========================
    data_dirs = [
        "/mnt/sda/home/zhangheyi/projects/RWKV/data/raw/data0",
    ]
    patch_size = 256
    scale = 8
    ratios = (0.7, 0.15, 0.15)          # 训练:验证:测试
    batch_size = 4

    # ======================== 2. 生成全局归一化文件（如果尚未存在） ========================
    # 方法1：手动合并已有的 norm_stats.json（推荐）
    # merge_existing_norm_stats(data_dirs, output_path="./norm_stats_global.json")
    #
    # 方法2：从训练区域重新统计（第一次运行时耗时，但更精确）
    # 此处我们使用已有的合并脚本结果，假设文件已存在
    global_norm_path = "./norm_stats_global.json"
    if not Path(global_norm_path).exists():
        print("⚠️ 全局归一化文件不存在，将使用各数据集内部 norm_stats.json（不推荐混合训练）")
        global_norm_path = None  # 让每个数据集自行加载

    # ======================== 3. 创建训练/验证/测试数据集 ========================
    # --- 训练集：所有目录的训练区域合并 ---
    train_datasets = []
    for d in data_dirs:
        ds = RefPairedEnviMemmapDataset(
            data_dir=d,
            patch_size=patch_size,
            scale=scale,
            mode="train",
            stride_ratio=1.0,          # 训练时有重叠
            augment=True,
            ratios=ratios,
            norm_stats_file=global_norm_path
        )
        train_datasets.append(ds)
    if len(train_datasets) > 1:
        from torch.utils.data import ConcatDataset
        full_train = ConcatDataset(train_datasets)
    else:
        full_train = train_datasets[0]

    # --- 验证集：所有目录的验证区域合并 ---
    val_datasets = []
    for d in data_dirs:
        ds = RefPairedEnviMemmapDataset(
            data_dir=d,
            patch_size=patch_size,
            scale=scale,
            mode="val",
            stride_ratio=1.0,          # 验证时无重叠，会被内部忽略
            augment=False,
            ratios=ratios,
            norm_stats_file=global_norm_path
        )
        val_datasets.append(ds)
    if len(val_datasets) > 1:
        full_val = ConcatDataset(val_datasets)
    else:
        full_val = val_datasets[0]

    # --- 测试集 ---
    test_datasets = []
    for d in data_dirs:
        ds = RefPairedEnviMemmapDataset(
            data_dir=d,
            patch_size=patch_size,
            scale=scale,
            mode="test",
            stride_ratio=1.0,
            augment=False,
            ratios=ratios,
            norm_stats_file=global_norm_path
        )
        test_datasets.append(ds)
    if len(test_datasets) > 1:
        full_test = ConcatDataset(test_datasets)
    else:
        full_test = test_datasets[0]

    print("\n====== 数据集划分结果 ======")
    print(f"训练集样本总数: {len(full_train)}")
    print(f"验证集样本总数: {len(full_val)}")
    print(f"测试集样本总数: {len(full_test)}")

    # ======================== 4. DataLoader 测试 ========================
    from torch.utils.data import DataLoader

    # 定义不带互换的 collate（因为我们要查看原始数据）
    def simple_collate(batch):
        lr1, hr1, lr2, hr2 = zip(*batch)
        return (
            torch.stack(lr1, 0),
            torch.stack(hr1, 0),
            torch.stack(lr2, 0),
            torch.stack(hr2, 0),
        )

    train_loader = DataLoader(
        full_train, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=simple_collate
    )
    val_loader = DataLoader(
        full_val, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=simple_collate
    )

    # 取一个 batch 观察
    print("\n====== 训练集 batch 测试 ======")
    for lr1, hr1, lr2, hr2 in train_loader:
        print(f"LR1 shape: {lr1.shape} | range: [{lr1.min():.3f}, {lr1.max():.3f}]")
        print(f"HR1 shape: {hr1.shape} | range: [{hr1.min():.3f}, {hr1.max():.3f}]")
        print(f"LR2 shape: {lr2.shape} | range: [{lr2.min():.3f}, {lr2.max():.3f}]")
        print(f"HR2 shape: {hr2.shape} | range: [{hr2.min():.3f}, {hr2.max():.3f}]")
        print("说明：lr1/hr1 为参考对，lr2/hr2 为目标对（或交换后身份改变）")
        break

    print("\n====== 验证集 batch 测试 ======")
    for lr1, hr1, lr2, hr2 in val_loader:
        print(f"LR1 shape: {lr1.shape} | range: [{lr1.min():.3f}, {lr1.max():.3f}]")
        print(f"HR1 shape: {hr1.shape} | range: [{hr1.min():.3f}, {hr1.max():.3f}]")
        print(f"LR2 shape: {lr2.shape} | range: [{lr2.min():.3f}, {lr2.max():.3f}]")
        print(f"HR2 shape: {hr2.shape} | range: [{hr2.min():.3f}, {hr2.max():.3f}]")
        break

    # 验证数据不重叠：打印 train/val/test 的样本行坐标范围（近似）
    # 为了不破坏 Dataset 封装，这里只从第一个子集（data0）查看
    if len(train_datasets) > 0:
        train_rows = [r for r, _ in train_datasets[0].samples]
        print(f"\nData0 训练集行范围: {min(train_rows)} - {max(train_rows)}")
    if len(val_datasets) > 0:
        val_rows = [r for r, _ in val_datasets[0].samples]
        print(f"Data0 验证集行范围: {min(val_rows)} - {max(val_rows)}")
    if len(test_datasets) > 0:
        test_rows = [r for r, _ in test_datasets[0].samples]
        print(f"Data0 测试集行范围: {min(test_rows)} - {max(test_rows)}")
    print("三者应没有重叠，且连续覆盖全部行。")

if __name__ == "__main__":
    main()