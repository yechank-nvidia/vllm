/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Fused per-head QK RMSNorm + RoPE + KV-cache write.
 *
 * Replaces this sequence of a decoder attention layer with one kernel:
 *
 *     q, k, v = qkv.split(...)
 *     q = q_norm(q); k = k_norm(k)            # per-head RMSNorm
 *     q, k = rotary_emb(positions, q, k)      # optional NeoX RoPE over the
 *                                             # full head (none: norm only)
 *     reshape_and_cache_flash(k, v, kv_cache, slot_mapping)
 *
 * The arithmetic follows what torch.compile generates for the unfused ops:
 * fp32 sum of squares, x * rsqrt(sum / head_dim + eps) * weight in fp32 (no
 * rounding between the norm and RoPE), RoPE in fp32 with the cos/sin table
 * values, one rounding to the model dtype at the store.
 *
 * Layout follows the MiniMax-M3 kernel (fused_minimax_m3_qknorm_rope_kv_insert)
 * with wider accesses: one work item is one (token, head slot); a group of
 * head_dim / 8 lanes owns one head, 8 dims (16 bytes) per lane, so a warp
 * covers 32 * 8 / head_dim heads at once. Each lane holds dims [4l, 4l + 4)
 * and [head_dim / 2 + 4l, head_dim / 2 + 4l + 4) of its head, so both members
 * of every NeoX rotation pair sit in one lane (no shuffles). For large batches
 * each thread takes kIters items and issues all their loads first, so enough
 * bytes are in flight to stream HBM. Slots per token: [0, nq) Q heads,
 * [nq, nq + nkv) K heads and, when the KV cache is given,
 * [nq + nkv, nq + 2 * nkv) V heads.
 *   Q: norm + RoPE -> q_out
 *   K: norm + RoPE -> k_out and the key cache
 *   V: copied into the value cache
 * qkv is read only. Tokens at or past slot_mapping.size(0), or with a negative
 * slot, are not written to the cache (same rule as reshape_and_cache_flash).
 */

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

#include "torch_utils.h"

#include "../cuda_compat.h"
#include "type_convert.cuh"
#include "dispatch_utils.h"

#ifndef FINAL_MASK
  #ifdef USE_ROCM
    #define FINAL_MASK 0xffffffffffffffffULL
  #else
    #define FINAL_MASK 0xffffffffu
  #endif
#endif

namespace vllm {
namespace fused_qk_norm_rope_kv {

namespace {
inline int getSMVersion() {
  auto* props = get_device_prop();
  return props->major * 10 + props->minor;
}
inline int getSMCount() { return get_device_prop()->multiProcessorCount; }
}  // namespace

constexpr int kWarp = 32;
constexpr int kElems = 8;  // dims per lane: one 16-byte access of 16-bit values
constexpr int kBlockSize = 256;
constexpr int kLargeIters = 2;  // items per thread for large batches

// Grid: 1D. Each thread handles kIters work items; a work item is one
// (token, slot) pair owned by a group of kHeadDim / kElems lanes. kRope:
// NeoX RoPE over the full head; otherwise norm only.
template <typename scalar_t, int kHeadDim, bool kRope, bool kInsertKV,
          int kIters>
__global__ void __launch_bounds__(kBlockSize) fusedQKNormRopeKVInsertKernel(
    scalar_t const* __restrict__ qkv,       // [N, qkv_stride]
    scalar_t* __restrict__ q_out,           // [N, nq * kHeadDim] contiguous
    scalar_t* __restrict__ k_out,           // [N, nkv * kHeadDim] contiguous
    scalar_t const* __restrict__ q_norm_w,  // [kHeadDim]
    scalar_t const* __restrict__ k_norm_w,  // [kHeadDim]
    scalar_t const* __restrict__ cos_sin_cache,  // [max_pos, kHeadDim]
    int64_t const* __restrict__ positions,       // [N]
    int64_t const* __restrict__ slot_mapping,    // [num_slot_tokens]
    scalar_t* __restrict__ kv_cache,  // logical [nb, 2, block_size, nkv, hd]
    float const eps, int const num_tokens, int const num_slot_tokens,
    int const nq, int const nkv, int const block_size, int64_t const qkv_stride,
    int64_t const kv_s_block, int64_t const kv_s_kv, int64_t const kv_s_token,
    int64_t const kv_s_head) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  // _typeConvert<BFloat16> needs sm80+; discard the bf16 body below that.
  if constexpr (std::is_same_v<scalar_t, torch::headeronly::BFloat16>) {
    return;
  } else {
#endif
    using Converter = vllm::_typeConvert<scalar_t>;
    using P2 = typename Converter::packed_hip_type;
    constexpr int kLanesPerHead = kHeadDim / kElems;  // 8, 16 or 32
    constexpr int kHeadsPerWarp = kWarp / kLanesPerHead;
    constexpr int kHalf = kHeadDim / 2;
    static_assert(kLanesPerHead <= kWarp && kWarp % kLanesPerHead == 0);

    int const laneId = threadIdx.x % kWarp;
    int const group = laneId / kLanesPerHead;
    // Head dims this lane holds: [off0, off0 + 4) and [off1, off1 + 4).
    int const off0 = (laneId % kLanesPerHead) * (kElems / 2);
    int const off1 = kHalf + off0;

    // 32-bit index math (the host checks num_items < 2^31): 64-bit integer
    // division would dominate this short kernel.
    uint32_t const warpIdx =
        blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp;
    uint32_t const slots_per_token = nq + nkv + (kInsertKV ? nkv : 0);
    uint32_t const num_items =
        static_cast<uint32_t>(num_tokens) * slots_per_token;
    if (warpIdx * kIters * kHeadsPerWarp >= num_items) return;  // warp-uniform

    auto ld = [&](scalar_t const* p) -> uint4 {  // this lane's 8 values
      uint2 const a = *reinterpret_cast<uint2 const*>(p + off0);
      uint2 const b = *reinterpret_cast<uint2 const*>(p + off1);
      return make_uint4(a.x, a.y, b.x, b.y);
    };
    auto st = [&](scalar_t* p, uint4 const v) {
      *reinterpret_cast<uint2*>(p + off0) = make_uint2(v.x, v.y);
      *reinterpret_cast<uint2*>(p + off1) = make_uint2(v.z, v.w);
    };
    auto unpack = [](uint4 const& v, float (&f)[kElems]) {
      auto const* p = reinterpret_cast<P2 const*>(&v);
#pragma unroll
      for (int i = 0; i < kElems / 2; i++) {
        float2 const f2 = Converter::convert(p[i]);
        f[2 * i] = f2.x;
        f[2 * i + 1] = f2.y;
      }
    };
    auto pack = [](float const(&f)[kElems]) -> uint4 {
      uint4 v;
      auto* p = reinterpret_cast<P2*>(&v);
#pragma unroll
      for (int i = 0; i < kElems / 2; i++) {
        p[i] = Converter::convert(make_float2(f[2 * i], f[2 * i + 1]));
      }
      return v;
    };

    int tok[kIters], slot[kIters];
    bool valid[kIters], isQ[kIters], isK[kIters], isV[kIters];
#pragma unroll
    for (int it = 0; it < kIters; it++) {
      uint32_t const item = (warpIdx * kIters + it) * kHeadsPerWarp + group;
      valid[it] = item < num_items;
      uint32_t const safe = valid[it] ? item : 0u;
      tok[it] = static_cast<int>(safe / slots_per_token);
      slot[it] = static_cast<int>(safe - tok[it] * slots_per_token);
      isQ[it] = slot[it] < nq;
      isK[it] = !isQ[it] && slot[it] < nq + nkv;
      isV[it] = !isQ[it] && !isK[it];  // only when kInsertKV
    }

    // PDL: wait for the producer of qkv (the QKV GEMM) before reading global
    // memory. No-op when the launch did not enable PDL.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

    // Issue every global load up front (qkv, norm weight, position, cos/sin)
    // so their latencies overlap; values stay packed until used.
    uint4 rawX[kIters], rawW[kIters], rawCS[kIters];
#pragma unroll
    for (int it = 0; it < kIters; it++) {
      rawX[it] = make_uint4(0, 0, 0, 0);
      rawW[it] = rawX[it];
      rawCS[it] = rawX[it];
      if (valid[it]) {
        rawX[it] = ld(qkv + static_cast<int64_t>(tok[it]) * qkv_stride +
                      slot[it] * kHeadDim);
        if (!isV[it]) rawW[it] = ld(isQ[it] ? q_norm_w : k_norm_w);
        if constexpr (kRope) {
          if (!isV[it]) {
            // cos[off0 .. off0 + 4) | sin[off0 .. off0 + 4)
            scalar_t const* c = cos_sin_cache + positions[tok[it]] * kHeadDim;
            uint2 const cv = *reinterpret_cast<uint2 const*>(c + off0);
            uint2 const sv = *reinterpret_cast<uint2 const*>(c + kHalf + off0);
            rawCS[it] = make_uint4(cv.x, cv.y, sv.x, sv.y);
          }
        }
      }
    }

#pragma unroll
    for (int it = 0; it < kIters; it++) {
      int const head =
          isQ[it] ? slot[it] : (isK[it] ? slot[it] - nq : slot[it] - nq - nkv);
      // Warps whose groups all hold V (or nothing) only copy V.
      bool const allV = __all_sync(FINAL_MASK, isV[it] || !valid[it]);
      uint4 outX = rawX[it];
      if (!allV) {
        float x[kElems];
        unpack(rawX[it], x);
        // Every lane of the warp runs the shuffles below (full masks); V and
        // invalid items drop the result.
        float sumsq = 0.0f;
#pragma unroll
        for (int i = 0; i < kElems; i++) sumsq = __fmaf_rn(x[i], x[i], sumsq);
#pragma unroll
        for (int mask = kLanesPerHead / 2; mask > 0; mask >>= 1) {
          sumsq += __shfl_xor_sync(FINAL_MASK, sumsq, mask, kWarp);
        }
        float const rms_rcp =
            rsqrtf(sumsq / static_cast<float>(kHeadDim) + eps);
        float w[kElems];
        unpack(rawW[it], w);
#pragma unroll
        for (int i = 0; i < kElems; i++) x[i] = (x[i] * rms_rcp) * w[i];

        if constexpr (kRope) {
          float cs[kElems];
          unpack(rawCS[it], cs);  // cos in [0, 4), sin in [4, 8)
#pragma unroll
          for (int i = 0; i < kElems / 2; i++) {
            float const x1 = x[i], x2 = x[kElems / 2 + i];
            float const c = cs[i], s = cs[kElems / 2 + i];
            // x1 * cos - x2 * sin, x2 * cos + x1 * sin (fixed FMA form)
            x[i] = __fmaf_rn(x1, c, -__fmul_rn(x2, s));
            x[kElems / 2 + i] = __fmaf_rn(x2, c, __fmul_rn(x1, s));
          }
        }
        if (!isV[it]) outX = pack(x);
      }

      if (!valid[it]) continue;
      if (!isV[it]) {
        st(isQ[it]
               ? q_out + (static_cast<int64_t>(tok[it]) * nq + head) * kHeadDim
               : k_out +
                     (static_cast<int64_t>(tok[it]) * nkv + head) * kHeadDim,
           outX);
      }
      if constexpr (kInsertKV) {
        if (!isQ[it] && tok[it] < num_slot_tokens) {
          int64_t const sm = slot_mapping[tok[it]];
          if (sm >= 0) {  // padded / unscheduled tokens have negative slots
            int64_t b, t;
            if (sm <= INT32_MAX) {  // cheap 32-bit division
              uint32_t const b32 = static_cast<uint32_t>(sm) / block_size;
              b = b32;
              t = static_cast<uint32_t>(sm) - b32 * block_size;
            } else {
              b = sm / block_size;
              t = sm % block_size;
            }
            // K: the normed/roped value (same bits as k_out); V: the input.
            st(kv_cache + b * kv_s_block + (isK[it] ? 0 : kv_s_kv) +
                   t * kv_s_token + head * kv_s_head,
               outX);
          }
        }
      }
    }

    // PDL: let a dependent kernel launch early. No-op without PDL.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

struct Args {
  float eps;
  int num_tokens, num_slot_tokens, nq, nkv, block_size;
  int64_t qkv_stride, kv_s_block, kv_s_kv, kv_s_token, kv_s_head;
};

template <typename scalar_t, int kHeadDim, bool kRope, bool kInsertKV,
          int kIters>
void launchKernel(scalar_t const* qkv, scalar_t* q_out, scalar_t* k_out,
                  scalar_t const* q_norm_w, scalar_t const* k_norm_w,
                  scalar_t const* cos_sin_cache, int64_t const* positions,
                  int64_t const* slot_mapping, scalar_t* kv_cache,
                  Args const& a, cudaStream_t stream) {
  constexpr int kHeadsPerWarp = kWarp / (kHeadDim / kElems);
  constexpr int kItemsPerBlock = (kBlockSize / kWarp) * kHeadsPerWarp * kIters;
  int64_t const slots_per_token = a.nq + a.nkv + (kInsertKV ? a.nkv : 0);
  int64_t const num_items =
      static_cast<int64_t>(a.num_tokens) * slots_per_token;
  int const grid =
      static_cast<int>((num_items + kItemsPerBlock - 1) / kItemsPerBlock);
  if (grid == 0) return;
  auto kernel = fusedQKNormRopeKVInsertKernel<scalar_t, kHeadDim, kRope,
                                              kInsertKV, kIters>;
#ifndef USE_ROCM
  // PDL (SM90+): the kernel may launch while the QKV GEMM drains; it waits
  // in cudaGridDependencySynchronize() before reading qkv.
  static int const sm_version = getSMVersion();
  cudaLaunchConfig_t config;
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = (sm_version >= 90) ? 1 : 0;
  cudaLaunchKernelEx(&config, kernel, qkv, q_out, k_out, q_norm_w, k_norm_w,
                     cos_sin_cache, positions, slot_mapping, kv_cache, a.eps,
                     a.num_tokens, a.num_slot_tokens, a.nq, a.nkv, a.block_size,
                     a.qkv_stride, a.kv_s_block, a.kv_s_kv, a.kv_s_token,
                     a.kv_s_head);
#else
  kernel<<<grid, kBlockSize, 0, stream> > >(
      qkv, q_out, k_out, q_norm_w, k_norm_w, cos_sin_cache, positions,
      slot_mapping, kv_cache, a.eps, a.num_tokens, a.num_slot_tokens, a.nq,
      a.nkv, a.block_size, a.qkv_stride, a.kv_s_block, a.kv_s_kv, a.kv_s_token,
      a.kv_s_head);
#endif
}

template <typename scalar_t, int kHeadDim, bool kRope, bool kInsertKV>
void launchIters(scalar_t const* qkv, scalar_t* q_out, scalar_t* k_out,
                 scalar_t const* q_norm_w, scalar_t const* k_norm_w,
                 scalar_t const* cos_sin_cache, int64_t const* positions,
                 int64_t const* slot_mapping, scalar_t* kv_cache, Args const& a,
                 cudaStream_t stream) {
  // Small batches (decode): one item per thread, most blocks, least latency.
  // Large batches (prefill): more items per thread so loads overlap.
  constexpr int kHeadsPerWarp = kWarp / (kHeadDim / kElems);
  static int const sm_count = getSMCount();
  int64_t const slots_per_token = a.nq + a.nkv + (kInsertKV ? a.nkv : 0);
  int64_t const num_items =
      static_cast<int64_t>(a.num_tokens) * slots_per_token;
  int64_t const one_wave =
      static_cast<int64_t>(sm_count) * (2048 / kWarp) * kHeadsPerWarp;
  if (num_items > 2 * one_wave) {
    launchKernel<scalar_t, kHeadDim, kRope, kInsertKV, kLargeIters>(
        qkv, q_out, k_out, q_norm_w, k_norm_w, cos_sin_cache, positions,
        slot_mapping, kv_cache, a, stream);
  } else {
    launchKernel<scalar_t, kHeadDim, kRope, kInsertKV, 1>(
        qkv, q_out, k_out, q_norm_w, k_norm_w, cos_sin_cache, positions,
        slot_mapping, kv_cache, a, stream);
  }
}

template <typename scalar_t, int kHeadDim>
void dispatchFlags(bool const rope, bool const insert_kv, scalar_t const* qkv,
                   scalar_t* q_out, scalar_t* k_out, scalar_t const* q_norm_w,
                   scalar_t const* k_norm_w, scalar_t const* cos_sin_cache,
                   int64_t const* positions, int64_t const* slot_mapping,
                   scalar_t* kv_cache, Args const& a, cudaStream_t stream) {
  auto go = [&](auto rope_c, auto insert_c) {
    launchIters<scalar_t, kHeadDim, decltype(rope_c)::value,
                decltype(insert_c)::value>(qkv, q_out, k_out, q_norm_w,
                                           k_norm_w, cos_sin_cache, positions,
                                           slot_mapping, kv_cache, a, stream);
  };
  auto with_insert = [&](auto rope_c) {
    if (insert_kv) {
      go(rope_c, std::true_type{});
    } else {
      go(rope_c, std::false_type{});
    }
  };
  if (rope) {
    with_insert(std::true_type{});
  } else {
    with_insert(std::false_type{});
  }
}

}  // namespace fused_qk_norm_rope_kv
}  // namespace vllm

void fused_qk_norm_rope_kv_insert(
    torch::stable::Tensor const& qkv,       // [N, (nq + 2 * nkv) * head_dim]
    torch::stable::Tensor& q_out,           // [N, nq * head_dim]
    torch::stable::Tensor& k_out,           // [N, nkv * head_dim]
    torch::stable::Tensor const& q_weight,  // [head_dim]
    torch::stable::Tensor const& k_weight,  // [head_dim]
    std::optional<torch::stable::Tensor> cos_sin_cache,  // [P, head_dim]
    torch::stable::Tensor const& positions,              // [N] i64
    int64_t num_heads, int64_t num_kv_heads, double eps,
    std::optional<torch::stable::Tensor> slot_mapping,  // [M <= N]
    std::optional<torch::stable::Tensor> kv_cache) {    // [nb, 2, bs, nkv, hd]
  using torch::headeronly::ScalarType;
  namespace F = vllm::fused_qk_norm_rope_kv;
  STD_TORCH_CHECK(qkv.is_cuda() && qkv.dim() == 2 && qkv.stride(1) == 1,
                  "qkv must be a 2D CUDA tensor with contiguous rows");
  STD_TORCH_CHECK(qkv.scalar_type() == ScalarType::Half ||
                      qkv.scalar_type() == ScalarType::BFloat16,
                  "qkv must be float16 or bfloat16");
  int64_t const nq = num_heads;
  int64_t const nkv = num_kv_heads;
  STD_TORCH_CHECK(nq > 0 && nkv > 0, "num_heads and num_kv_heads must be > 0");
  STD_TORCH_CHECK(qkv.size(1) % (nq + 2 * nkv) == 0,
                  "qkv width must be (num_heads + 2 * num_kv_heads) * "
                  "head_dim");
  int64_t const head_dim = qkv.size(1) / (nq + 2 * nkv);
  STD_TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                  "head_dim must be 64, 128 or 256");
  int64_t const num_tokens = qkv.size(0);
  STD_TORCH_CHECK(qkv.stride(0) % 8 == 0 &&
                      reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0,
                  "qkv must be 16-byte aligned with a row stride that is a "
                  "multiple of 8 elements");

  auto check_out = [&](torch::stable::Tensor const& t, int64_t heads,
                       char const* name) {
    STD_TORCH_CHECK(t.is_cuda() && t.is_contiguous() &&
                        t.scalar_type() == qkv.scalar_type() &&
                        reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                    name,
                    " must be a 16-byte aligned contiguous CUDA tensor of qkv "
                    "dtype");
    STD_TORCH_CHECK(t.numel() == num_tokens * heads * head_dim, name,
                    " must have num_tokens * heads * head_dim elements");
  };
  check_out(q_out, nq, "q_out");
  check_out(k_out, nkv, "k_out");
  STD_TORCH_CHECK(
      q_weight.scalar_type() == qkv.scalar_type() &&
          k_weight.scalar_type() == qkv.scalar_type() &&
          q_weight.is_contiguous() && k_weight.is_contiguous() &&
          reinterpret_cast<uintptr_t>(q_weight.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(k_weight.data_ptr()) % 16 == 0,
      "q/k norm weights must be 16-byte aligned, contiguous and of qkv dtype");
  STD_TORCH_CHECK(q_weight.numel() == head_dim && k_weight.numel() == head_dim,
                  "q/k norm weights must have head_dim elements");
  STD_TORCH_CHECK(
      positions.is_cuda() && positions.scalar_type() == ScalarType::Long &&
          positions.is_contiguous() && positions.numel() >= num_tokens,
      "positions must be contiguous int64 CUDA with >= num_tokens "
      "entries");

  bool const rope = cos_sin_cache.has_value();
  if (rope) {
    STD_TORCH_CHECK(
        cos_sin_cache->is_cuda() && cos_sin_cache->is_contiguous() &&
            cos_sin_cache->dim() == 2 &&
            cos_sin_cache->scalar_type() == qkv.scalar_type() &&
            reinterpret_cast<uintptr_t>(cos_sin_cache->data_ptr()) % 16 == 0,
        "cos_sin_cache must be a 16-byte aligned contiguous 2D CUDA tensor of "
        "qkv dtype");
    STD_TORCH_CHECK(cos_sin_cache->size(1) == head_dim,
                    "only NeoX RoPE over the full head is supported: "
                    "cos_sin_cache must have head_dim columns");
  }

  bool const insert_kv = kv_cache.has_value();
  F::Args a{};
  a.eps = static_cast<float>(eps);
  a.num_tokens = static_cast<int>(num_tokens);
  a.nq = static_cast<int>(nq);
  a.nkv = static_cast<int>(nkv);
  a.block_size = 1;
  a.qkv_stride = qkv.stride(0);
  if (insert_kv) {
    STD_TORCH_CHECK(slot_mapping.has_value() && slot_mapping->is_cuda() &&
                        slot_mapping->scalar_type() == ScalarType::Long &&
                        slot_mapping->is_contiguous(),
                    "kv_cache needs a contiguous int64 CUDA slot_mapping");
    STD_TORCH_CHECK(slot_mapping->numel() <= num_tokens,
                    "slot_mapping has more entries than qkv rows");
    STD_TORCH_CHECK(
        kv_cache->is_cuda() && kv_cache->scalar_type() == qkv.scalar_type(),
        "kv_cache must be a CUDA tensor of qkv dtype (unquantized "
        "cache only)");
    STD_TORCH_CHECK(
        kv_cache->dim() == 5 && kv_cache->size(1) == 2 &&
            kv_cache->size(3) == nkv && kv_cache->size(4) == head_dim &&
            kv_cache->stride(4) == 1 && kv_cache->stride(3) % 8 == 0 &&
            kv_cache->stride(2) % 8 == 0 && kv_cache->stride(1) % 8 == 0 &&
            kv_cache->stride(0) % 8 == 0 &&
            reinterpret_cast<uintptr_t>(kv_cache->data_ptr()) % 16 == 0,
        "kv_cache must be a 16-byte aligned [num_blocks, 2, block_size, "
        "num_kv_heads, head_dim] tensor with contiguous head_dim");
    a.num_slot_tokens = static_cast<int>(slot_mapping->numel());
    a.block_size = static_cast<int>(kv_cache->size(2));
    a.kv_s_block = kv_cache->stride(0);
    a.kv_s_kv = kv_cache->stride(1);
    a.kv_s_token = kv_cache->stride(2);
    a.kv_s_head = kv_cache->stride(3);
  }
  if (num_tokens == 0) return;
  STD_TORCH_CHECK(num_tokens * (nq + 2 * nkv) < (int64_t{1} << 31),
                  "too many tokens for one launch");

  const torch::stable::accelerator::DeviceGuard device_guard(
      qkv.get_device_index());
  auto stream = get_current_cuda_stream(qkv.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      qkv.scalar_type(), "fused_qk_norm_rope_kv_insert", [&] {
        using st = scalar_t;
        auto call = [&](auto head_dim_c) {
          constexpr int kHD = decltype(head_dim_c)::value;
          F::dispatchFlags<st, kHD>(
              rope, insert_kv, reinterpret_cast<st const*>(qkv.data_ptr()),
              reinterpret_cast<st*>(q_out.data_ptr()),
              reinterpret_cast<st*>(k_out.data_ptr()),
              reinterpret_cast<st const*>(q_weight.data_ptr()),
              reinterpret_cast<st const*>(k_weight.data_ptr()),
              rope ? reinterpret_cast<st const*>(cos_sin_cache->data_ptr())
                   : nullptr,
              reinterpret_cast<int64_t const*>(positions.data_ptr()),
              insert_kv
                  ? reinterpret_cast<int64_t const*>(slot_mapping->data_ptr())
                  : nullptr,
              insert_kv ? reinterpret_cast<st*>(kv_cache->data_ptr()) : nullptr,
              a, stream);
        };
        if (head_dim == 64) {
          call(std::integral_constant<int, 64>{});
        } else if (head_dim == 128) {
          call(std::integral_constant<int, 128>{});
        } else {
          call(std::integral_constant<int, 256>{});
        }
      });
}
