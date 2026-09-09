# `kernels/`

这里保存模型使用的 CUDA/C++ 加速内核。目前只有 `wkv/`：

- `bi_wkv.cpp`：PyTorch/C++ 绑定；
- `bi_wkv_kernel.cu`：WKV 前向和反向 CUDA kernel；
- `layers.py`：PyTorch 层封装；
- `runtime.py`：编译、加载和运行后端。

模型通过统一的 WKV 接口调用这些文件。没有 CUDA toolkit 时仍可检查配置和导入部分代码，但不能运行需要编译扩展的训练。
