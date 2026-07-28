/******************************************************************************
 * Copyright (c) 2025 Shanghai AI Lab.
 * Modified: forward kernel uses shared memory instead of __shfl_sync
 ******************************************************************************/

#include <torch/extension.h>
#include <cuda.h>
#include <ATen/cuda/Atomic.cuh>
#include <cuda_runtime.h>
#include <vector>

#define MIN_VALUE (-1e38)
#define CHANNEL_LEN 8
#define EPS (1e-6)
#define TOKEN_SPLIT 32


template <typename scalar_t>
__global__ void bi_wkv_cuda_forward_kernel(
    size_t B,
    size_t T,
    size_t C,
    const scalar_t* __restrict__ _w,
    const scalar_t* __restrict__ _u,
    const scalar_t* __restrict__ _k,
    const scalar_t* __restrict__ _v,
    scalar_t* __restrict__ const _y
    ) {
    const int idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int channel_id = threadIdx.y;
    const int token_id = threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _T = (T + TOKEN_SPLIT - 1) / TOKEN_SPLIT;
    const int _t = _T * token_id;
    const int _offset = _b * T * C + _c;
    const int _t_lim = T - _t;
    const int _tokenLength = min(_t_lim, _T);
    scalar_t u = _u[_c];
    scalar_t w = _w[_c];
    const scalar_t *__restrict__ const k = _k + _offset;
    const scalar_t *__restrict__ const v = _v + _offset;
    scalar_t *__restrict__ const y = _y + _offset;

    // ★ 新增：shared memory（替代 __shfl_sync）
    __shared__ scalar_t Sa[TOKEN_SPLIT][CHANNEL_LEN];
    __shared__ scalar_t Sb[TOKEN_SPLIT][CHANNEL_LEN];
    __shared__ scalar_t Sc[TOKEN_SPLIT][CHANNEL_LEN];
    __shared__ scalar_t Sd[TOKEN_SPLIT][CHANNEL_LEN];
    __shared__ scalar_t So1[TOKEN_SPLIT][CHANNEL_LEN];
    __shared__ scalar_t So2[TOKEN_SPLIT][CHANNEL_LEN];

    // ── 第一阶段：段内双向扫描（不变）──
    scalar_t a = 0, b = 0, c = 0, d = 0;
    scalar_t o1 = MIN_VALUE, o2 = MIN_VALUE;
    for (int i = _t; i < (_t + _tokenLength); i++){
        const int ii = i * C;
        scalar_t no = max(o1, k[ii] - w * (i - _t));
        scalar_t e1 = exp(o1 - no);
        scalar_t e3 = exp(k[ii] - w * (i - _t) - no);
        c = e1 * c + e3 * v[ii];
        d = e1 * d + e3;
        o1 = no;
        const int ni = 2 * _t + _tokenLength - 1 - i;
        const int nini = ni * C;
        const int exp_w = _t + _tokenLength - ni;
        no = max(o2, k[nini] - w * exp_w);
        scalar_t e2 = exp(o2 - no);
        e3 = exp(k[nini] - w * exp_w - no);
        a = e2 * a + e3 * v[nini];
        b = e2 * b + e3;
        o2 = no;
    }

    // ★ 写入 shared memory（替代 __shfl_sync 的数据来源）
    __syncthreads();
    Sa[token_id][channel_id] = a;
    Sb[token_id][channel_id] = b;
    Sc[token_id][channel_id] = c;
    Sd[token_id][channel_id] = d;
    So1[token_id][channel_id] = o1;
    So2[token_id][channel_id] = o2;
    __syncthreads();

    // ★ 第二阶段：段间聚合（从 shared memory 读取，替代 __shfl_sync）
    scalar_t a2 = 0, b2 = 0, c2 = 0, d2 = 0;
    scalar_t o3 = MIN_VALUE, o4 = MIN_VALUE;

    // 反向段聚合（原来用 __shfl_sync(0xffffffff, o2, i) 等）
    for (int i = 0; i < token_id; i++) {
        const int exp_w = (token_id - i - 1) * _T;
        scalar_t no = max(So2[i][channel_id] - w * exp_w, o4);
        a2 = a2 * exp(o4 - no) + Sa[i][channel_id] * exp(So2[i][channel_id] - w * exp_w - no);
        b2 = b2 * exp(o4 - no) + Sb[i][channel_id] * exp(So2[i][channel_id] - w * exp_w - no);
        o4 = no;
    }
    a = a2;
    b = b2;
    o2 = o4;

    // 正向段聚合（原来用 __shfl_sync(0xffffffff, o1, i) 等）
    for (int i = token_id; i < TOKEN_SPLIT; i++){
        const int exp_w = (i - token_id) * _T;
        scalar_t no = max(So1[i][channel_id] - w * exp_w, o3);
        c2 = c2 * exp(o3 - no) + Sc[i][channel_id] * exp(So1[i][channel_id] - w * exp_w - no);
        d2 = d2 * exp(o3 - no) + Sd[i][channel_id] * exp(So1[i][channel_id] - w * exp_w - no);
        o3 = no;
    }

    // ── 第三阶段：最终输出（不变）──
    c = c2 - exp(k[_t * C] - o3) * v[_t * C];
    d = d2 - exp(k[_t * C] - o3);
    o1 = o3;
    for (int i = _t; i < (_t + _tokenLength); i++) {
        const int ii = i * C;
        scalar_t no = max(o1, u + k[ii]);
        no = max(no, o2);
        scalar_t e1 = exp(o1 - no);
        scalar_t e2 = exp(o2 - no);
        scalar_t e3 = exp(u + k[ii] - no);
        y[ii] = (c * e1 + a * e2 + e3 * v[ii])/(d * e1 + b * e2 + e3 + EPS);
        const int ii2 = ((i + 1) % T) * C;
        no = max(o2 - w, k[ii]);
        e2 = exp(o2 - w - no);
        e3 = exp(k[ii] - no);
        a = e2 * a + e3 * v[ii];
        b = e2 * b + e3;
        o2 = no;
        no = max(o1 + w, k[ii2] + w);
        e1 = exp(o1 + w - no);
        e3 = exp(k[ii2] + w - no);
        c = e1 * c - e3 * v[ii2];
        d = e1 * d - e3;
        o1 = no;
    }
}


// ═══════════════════════════════════════════════════════════
// Backward kernel：完全不动（已经用 shared memory，没有 bug）
// ═══════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void bi_wkv_cuda_backward_kernel(
    size_t B, 
    size_t T, 
    size_t C,
    const scalar_t *__restrict__ const _w, 
    const scalar_t *__restrict__ const _u, 
    const scalar_t *__restrict__ const _k, 
    const scalar_t *__restrict__ const _v, 
    const scalar_t *__restrict__ const _gy,
    scalar_t *__restrict__ const _gw, 
    scalar_t *__restrict__ const _gu, 
    scalar_t *__restrict__ const _gk, 
    scalar_t *__restrict__ const _gv,
    scalar_t *__restrict__ const _y,
    scalar_t *__restrict__ const _z,
    scalar_t *__restrict__ const _zexp
    ) {

    const int idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int channel_id = threadIdx.y;
    const int token_id = threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _T = (T + TOKEN_SPLIT - 1) / TOKEN_SPLIT;
    const int _t = _T * token_id;
    const int _offset = _b * T * C + _c;
    const int _t_lim = T - _t;
    const int _tokenLength = min(_t_lim, _T);
    scalar_t u = _u[_c];
    scalar_t w = _w[_c];
    const scalar_t *__restrict__ const k = _k + _offset;
    const scalar_t *__restrict__ const v = _v + _offset;
    const scalar_t *__restrict__ const gy = _gy + _offset;
    scalar_t *__restrict__ const y = _y + _offset;
    scalar_t *__restrict__ const z = _z + _offset;
    scalar_t *__restrict__ const zexp = _zexp + _offset;
    scalar_t *__restrict__ const gk = _gk + _offset;
    scalar_t *__restrict__ const gv = _gv + _offset;

    __shared__ scalar_t Sa[TOKEN_SPLIT][CHANNEL_LEN], Sb[TOKEN_SPLIT][CHANNEL_LEN];
    __shared__ scalar_t Sdadw[TOKEN_SPLIT][CHANNEL_LEN], Sdbdw[TOKEN_SPLIT][CHANNEL_LEN];
    __shared__ scalar_t So2[TOKEN_SPLIT][CHANNEL_LEN];
    scalar_t a = 0, b = 0, c = 0, d = 0;
    scalar_t dadw = 0, dbdw = 0, dcdw = 0, dddw = 0;
    scalar_t o1 = MIN_VALUE, o2 = MIN_VALUE;
    for (int i = _t; i < (_t + _tokenLength); i++){
        const int ii = i * C;
        scalar_t no = max(o1, k[ii] - w * (i - _t));
        scalar_t e1 = exp(o1 - no);
        scalar_t e3 = exp(k[ii] - w * (i - _t) - no);
        dcdw = dcdw * e1 - (i - _t) * e3 * v[ii];
        dddw = dddw * e1 - (i - _t) * e3;
        c = e1 * c + e3 * v[ii];
        d = e1 * d + e3;
        o1 = no;
        const int ni = 2 * _t + _tokenLength - 1 - i;
        const int nini = ni * C;
        const int exp_w = _t + _tokenLength - ni;
        no = max(o2, k[nini] - w * exp_w);
        scalar_t e2 = exp(o2 - no);
        e3 = exp(k[nini] - w * exp_w - no);
        dadw = dadw * e2 - exp_w * e3 * v[nini];
        dbdw = dbdw * e2 - exp_w * e3;
        a = e2 * a + e3 * v[nini];
        b = e2 * b + e3;
        o2 = no;
    }
    __syncthreads();
    So2[token_id][channel_id] = o2;
    Sa[token_id][channel_id] = a;
    Sb[token_id][channel_id] = b;
    Sdadw[token_id][channel_id] = dadw;
    Sdbdw[token_id][channel_id] = dbdw;
    __syncthreads();
    a = 0;
    b = 0;
    dadw = 0;
    dbdw = 0;
    o2 = MIN_VALUE;

    scalar_t a2 = 0, b2 = 0, c2 = 0, d2 = 0, dadw2 = 0, dbdw2 = 0, dcdw2 = 0, dddw2 = 0;
    scalar_t o3 = MIN_VALUE, o4 = MIN_VALUE;

    for (int i = 0; i < token_id; i++){
        const int exp_w = (token_id - i - 1) * _T;
        scalar_t no = max(So2[i][channel_id] - w * exp_w, o2);
        a = a * exp(o2 - no) + Sa[i][channel_id] * exp(So2[i][channel_id] - w * exp_w - no);
        b = b * exp(o2 - no) + Sb[i][channel_id] * exp(So2[i][channel_id] - w * exp_w - no);
        dadw = dadw * exp(o2 - no) + (Sdadw[i][channel_id] - exp_w * Sa[i][channel_id])
            * exp(So2[i][channel_id] - w * exp_w - no);
        dbdw = dbdw * exp(o2 - no) + (Sdbdw[i][channel_id] - exp_w * Sb[i][channel_id])
            * exp(So2[i][channel_id] - w * exp_w - no);
        o2 = no;
    }
    __syncthreads();
    So2[token_id][channel_id] = o1;
    Sa[token_id][channel_id] = c;
    Sb[token_id][channel_id] = d;
    Sdadw[token_id][channel_id] = dcdw;
    Sdbdw[token_id][channel_id] = dddw;
    __syncthreads();
    c = 0;
    d = 0;
    dcdw = 0;
    dddw = 0;
    o1 = MIN_VALUE;
    for (int i = token_id; i < TOKEN_SPLIT; i++){
        const int exp_w = (i - token_id) * _T;
        scalar_t no = max(So2[i][channel_id] - w * exp_w, o1);
        c = c * exp(o1 - no) + Sa[i][channel_id] * exp(So2[i][channel_id] - w * exp_w - no);
        d = d * exp(o1 - no) + Sb[i][channel_id] * exp(So2[i][channel_id] - w * exp_w - no);
        dcdw = dcdw * exp(o1 - no) + (Sdadw[i][channel_id] - exp_w * Sa[i][channel_id])
             * exp(So2[i][channel_id] - w * exp_w - no);
        dddw = dddw * exp(o1 - no) + (Sdbdw[i][channel_id] - exp_w * Sb[i][channel_id])
             * exp(So2[i][channel_id] - w * exp_w - no);
        o1 = no;
    }
    c -= exp(k[_t * C] - o1) * v[_t * C];
    d -= exp(k[_t * C] - o1);

    scalar_t gw = 0, gu = 0;
    scalar_t gc = 0, gd = 0, ga = 0, gb = 0;
    scalar_t go1 = MIN_VALUE, go2 = MIN_VALUE;
    for (int i = _t; i < (_t + _tokenLength); i++) {
        const int ii = i * C;
        scalar_t no = max(o1, u + k[ii]);
        no = max(no, o2);
        scalar_t e1 = exp(o1 - no);
        scalar_t e2 = exp(o2 - no);
        scalar_t e3 = exp(u + k[ii] - no);
        scalar_t num = (c * e1 + a * e2 + e3 * v[ii]);
        scalar_t iden = 1 / (d * e1 + b * e2 + e3 + EPS);
        y[ii] = num * iden;
        z[ii] = iden;
        zexp[ii] = -no;
        gw += gy[ii] * (dadw - dbdw * (num * iden)) * iden * e2;
        gw += gy[ii] * (dcdw - dddw * (num * iden)) * iden * e1;
        gu += gy[ii] * (v[ii] - (num * iden)) * e3 * iden;
        gk[ii] = gy[ii] * iden * (v[ii] - (num * iden)) * e3;
        gv[ii] = gy[ii] * iden * e3;
        scalar_t gno = max(- w + go1, -no);
        e1 = exp(- w + go1 - gno);
        e3 = gy[ii] * iden  * exp(- no - gno);
        gc = e1 * gc + e3 * (num * iden);
        gd = e1 * gd + e3;
        go1 = gno;

        const int ii2 = ((i + 1) % T) * C;
        no = max(o2 - w, k[ii]);
        e2 = exp(o2 - w - no);
        e3 = exp(k[ii] - no);
        dadw = e2 * (dadw - a);
        dbdw = e2 * (dbdw - b);
        a = e2 * a + e3 * v[ii];
        b = e2 * b + e3;
        o2 = no;
        no = max(o1 + w, k[ii2] + w);
        e1 = exp(o1 + w - no);
        e3 = exp(k[ii2] + w - no);
        dcdw = e1 * (c + dcdw) - e3 * v[ii2];
        dddw = e1 * (d + dddw) - e3;
        c = e1 * c - e3 * v[ii2];
        d = e1 * d - e3;
        o1 = no;
    }
    atomicAdd(&_gw[_c], gw);
    atomicAdd(&_gu[_c], gu);
    for (int i = _t + _tokenLength - 1; i >=_t ; i--) {
        const int ii = i * C;
        scalar_t gno = max(-w + go2, zexp[ii]);
        scalar_t e2 = exp(-w + go2 - gno);
        scalar_t e3 = gy[ii] * z[ii] * exp(zexp[ii] - gno);
        ga = e2 * ga + e3 * y[ii];
        gb = e2 * gb + e3;
        go2 = gno;
    }
    __syncthreads();
    Sa[token_id][channel_id] = gc;
    Sb[token_id][channel_id] = gd;
    So2[token_id][channel_id] = go1;
    __syncthreads();
    gc = 0;
    gd = 0;
    go1 = MIN_VALUE;
    for (int i = 0; i < token_id; i++){
        const int exp_w = (token_id - i - 1) * _T;
        scalar_t gno = max(So2[i][channel_id] - w * exp_w, go1);
        gc = gc * exp(go1 - gno) + Sa[i][channel_id] * exp(So2[i][channel_id] - w * exp_w - gno);
        gd = gd * exp(go1 - gno) + Sb[i][channel_id] * exp(So2[i][channel_id] - w * exp_w - gno);
        go1 = gno;
    }

    __syncthreads();
    Sa[token_id][channel_id] = ga;
    Sb[token_id][channel_id] = gb;
    So2[token_id][channel_id] = go2;
    __syncthreads();
    ga = 0;
    gb = 0;
    go2 = MIN_VALUE;
    for (int i = token_id + 1; i < TOKEN_SPLIT; i++){
        const int exp_w = (i - token_id - 1) * _T;
        scalar_t gno = max(So2[i][channel_id] - w * exp_w, go2);
        ga = ga * exp(go2 - gno) + Sa[i][channel_id] * exp(So2[i][channel_id] - w * exp_w - gno);
        gb = gb * exp(go2 - gno) + Sb[i][channel_id] * exp(So2[i][channel_id] - w * exp_w - gno);
        go2 = gno;
    }

    for (int i = _t; i < (_t + _tokenLength); i++) {
        const int ii = i * C;
        const int ni = 2 * _t + _tokenLength - 1 - i;
        const int nini = ni * C;
        gk[ii] += exp(k[ii] + go1) * (gd * v[ii] - gc);
        gk[nini] += exp(k[nini] + go2) * (gb * v[nini] - ga);
        gv[ii] += exp(k[ii] + go1) * gd;
        gv[nini] += exp(k[nini] + go2) * gb;
        scalar_t gno = max(-w + go1, zexp[ii]);
        scalar_t e1 = exp(-w + go1 - gno);
        scalar_t e3 = gy[ii] * z[ii]  * exp(zexp[ii] - gno);
        gc = e1 * gc + e3 * y[ii];
        gd = e1 * gd + e3;
        go1 = gno;
        gno = max(-w + go2, zexp[nini]);
        scalar_t e2 = exp(-w + go2 - gno);
        e3 = gy[nini] * z[nini] * exp(zexp[nini] - gno);
        ga = e2 * ga + e3 * y[nini];
        gb = e2 * gb + e3;
        go2 = gno;
    }
}


// ═══════════════════════════════════════════════════════════
// Host 函数（不变）
// ═══════════════════════════════════════════════════════════

torch::Tensor bi_wkv_cuda_forward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v) {

    const auto batch_size = k.size(0);
    const auto num_tokens = k.size(1);
    const auto num_channels = k.size(2);

    auto y = torch::zeros_like(k);
    
    assert(num_channels >= CHANNEL_LEN);
    const dim3 threads(TOKEN_SPLIT, CHANNEL_LEN);
    assert(batch_size * num_channels % threads.y == 0);
    const dim3 blocks(batch_size * num_channels / threads.y);

    AT_DISPATCH_FLOATING_TYPES(k.scalar_type(), "bi_wkv_forward_cuda", ([&] {
        bi_wkv_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            batch_size,
            num_tokens,
            num_channels,
            w.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>());
    }));
    return y;
}

std::vector<torch::Tensor> bi_wkv_cuda_backward(
    torch::Tensor w, 
    torch::Tensor u, 
    torch::Tensor k, 
    torch::Tensor v, 
    torch::Tensor gy) {

    const auto batch_size = k.size(0);
    const auto num_tokens = k.size(1);
    const auto num_channels = k.size(2);
    auto gw = torch::zeros_like(w);
    auto gu = torch::zeros_like(u);
    auto gk = torch::zeros_like(k);
    auto gv = torch::zeros_like(v);
    auto y = torch::zeros_like(k);
    auto z = torch::zeros_like(k);
    auto zexp = torch::zeros_like(k);

    assert(num_channels >= CHANNEL_LEN);
    const dim3 threads(TOKEN_SPLIT, CHANNEL_LEN);
    assert(batch_size * num_channels % threads.y == 0);
    const dim3 blocks(batch_size * num_channels / threads.y);

    AT_DISPATCH_FLOATING_TYPES(k.scalar_type(), "bi_wkv_backward_cuda", ([&] {
        bi_wkv_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            batch_size,
            num_tokens,
            num_channels,
            w.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            gy.data_ptr<scalar_t>(),
            gw.data_ptr<scalar_t>(),
            gu.data_ptr<scalar_t>(),
            gk.data_ptr<scalar_t>(),
            gv.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            z.data_ptr<scalar_t>(),
            zexp.data_ptr<scalar_t>());
    }));
    return {gw, gu, gk, gv};
}