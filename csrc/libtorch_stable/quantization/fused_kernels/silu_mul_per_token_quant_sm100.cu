// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda/ptx>
#include <cuda_runtime.h>

#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>

#include "libtorch_stable/torch_utils.h"

namespace {

constexpr int kWidth = 27392;
constexpr int kPack = 16;
constexpr int kThreads = 256;
constexpr int kFullBytes = kWidth * 4;
constexpr int kStageWidth = 8192;
constexpr int kStageBytes = kStageWidth * 4;
constexpr float kInvFp8Max = 1.0f / 448.0f;
constexpr float kMinScale = 1.0f / (448.0f * 512.0f);
namespace ptx = cuda::ptx;

__device__ __forceinline__ float div_full(float a, float b) {
  float result;
  asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

__device__ __forceinline__ uint4 load128(const void* ptr) {
  uint4 value;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
  return value;
}

__device__ __forceinline__ void store128(void* ptr, uint4 value) {
  asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z),
                 "r"(value.w)
               : "memory");
}

__device__ __forceinline__ uint32_t quant4(const float* f, float inv) {
  uint32_t q;
  const float a = __fmul_rn(f[0], inv), b = __fmul_rn(f[1], inv);
  const float c = __fmul_rn(f[2], inv), d = __fmul_rn(f[3], inv);
  asm("{ .reg .b16 lo, hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi}; }"
      : "=r"(q)
      : "f"(a), "f"(b), "f"(c), "f"(d));
  return q;
}

__device__ __forceinline__ float silu_product(float gate, float up) {
  return __fmul_rn(div_full(gate, __fadd_rn(expf(-gate), 1.0f)), up);
}

__device__ __forceinline__ void product8(const uint16_t* gate,
                                         const uint16_t* up, float* y,
                                         uint32_t& peak) {
  const uint4 g = load128(gate), u = load128(up);
  const uint32_t gs[] = {g.x, g.y, g.z, g.w};
  const uint32_t us[] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    y[j * 2] = silu_product(__uint_as_float(gs[j] << 16),
                            __uint_as_float(us[j] << 16));
    y[j * 2 + 1] = silu_product(__uint_as_float(gs[j] & 0xffff0000u),
                                __uint_as_float(us[j] & 0xffff0000u));
    peak = max(peak, __float_as_uint(y[j * 2]) & 0x7fffffffu);
    peak = max(peak, __float_as_uint(y[j * 2 + 1]) & 0x7fffffffu);
  }
}

__device__ __forceinline__ void load_stage(uint16_t* dst, const uint16_t* src,
                                           int stage, uint64_t* barrier) {
  const int off = stage * kStageWidth;
  const int nbytes = min(kWidth - off, kStageWidth) * sizeof(uint16_t);
  ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta,
                                 ptx::space_shared, barrier, nbytes * 2);
  ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, dst, src + off,
                     nbytes, barrier);
  ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, dst + kStageWidth,
                     src + kWidth + off, nbytes, barrier);
}

template <bool Staged>
__global__ __launch_bounds__(kThreads) void tma_row(const uint16_t* x,
                                                    uint8_t* q, float* scales,
                                                    int64_t stride_x) {
  constexpr int kTileWidth = Staged ? kStageWidth : kWidth;
  constexpr int kStages = (kWidth + kTileWidth - 1) / kTileWidth;
  constexpr int kIters = (kTileWidth / kPack + kThreads - 1) / kThreads;
  const int row = blockIdx.x, tid = threadIdx.x;
  const auto* x_row = x + static_cast<int64_t>(row) * stride_x;
  auto* q_row = q + static_cast<int64_t>(row) * kWidth;
  extern __shared__ __align__(16) uint16_t buffer[];
  __shared__ uint64_t barrier[Staged ? 2 : 1];
  // Overlap the next copies with FP32 computation in the staged schedule.
  if (tid == 0) {
    ptx::mbarrier_init(&barrier[0], 1);
    if constexpr (Staged) {
      ptx::mbarrier_init(&barrier[1], 1);
    }
    asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
    if constexpr (Staged) {
      load_stage(buffer, x_row, 0, &barrier[0]);
      load_stage(buffer + 2 * kStageWidth, x_row, 1, &barrier[1]);
    } else {
      ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta,
                                     ptx::space_shared, &barrier[0],
                                     kFullBytes);
      ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, buffer, x_row,
                         kFullBytes, &barrier[0]);
    }
  }
  __syncthreads();

  float y[kStages][kIters][kPack];
  // Positive FP32 values sort by their unsigned bits, with NaNs above inf.
  uint32_t peak = 0;
#pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    const int slot = stage % 2;
    while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                          &barrier[slot], stage / 2)) {
    }
    const auto* stage_buffer = buffer + slot * 2 * kTileWidth;
#pragma unroll
    for (int i = 0; i < kIters; ++i) {
      const int local_col = (tid + i * kThreads) * kPack;
      if (local_col < kTileWidth && stage * kTileWidth + local_col < kWidth) {
        const uint16_t* gate = stage_buffer + local_col;
        const uint16_t* up = gate + kTileWidth;
        product8(gate, up, y[stage][i], peak);
        product8(gate + 8, up + 8, y[stage][i] + 8, peak);
      }
    }
    if (stage + 2 < kStages) {
      __syncthreads();
      if (tid == 0) {
        load_stage(buffer + slot * 2 * kTileWidth, x_row, stage + 2,
                   &barrier[slot]);
      }
    }
  }

  __shared__ uint32_t warp_peaks[kThreads / 32];
  __shared__ float inv_scale;
  const uint32_t warp_peak = __reduce_max_sync(0xffffffffu, peak);
  if (tid % 32 == 0) {
    warp_peaks[tid / 32] = warp_peak;
  }
  __syncthreads();
  if (tid < 32) {
    const uint32_t p = tid < kThreads / 32 ? warp_peaks[tid] : 0;
    peak = __reduce_max_sync(0xffffffffu, p);
    if (tid == 0) {
      const float fp8_scale = __fmul_rn(__uint_as_float(peak), kInvFp8Max);
      const float scale =
          isnan(fp8_scale) ? fp8_scale : fmaxf(fp8_scale, kMinScale);
      scales[row] = scale;
      inv_scale = div_full(1.0f, scale);
    }
  }
  __syncthreads();

#pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
#pragma unroll
    for (int i = 0; i < kIters; ++i) {
      const int local_col = (tid + i * kThreads) * kPack;
      const int col = stage * kTileWidth + local_col;
      if (local_col < kTileWidth && col < kWidth) {
        const float* product = y[stage][i];
        store128(q_row + col, make_uint4(quant4(product, inv_scale),
                                         quant4(product + 4, inv_scale),
                                         quant4(product + 8, inv_scale),
                                         quant4(product + 12, inv_scale)));
      }
    }
  }
}

template <bool Staged>
void launch(const uint16_t* x, uint8_t* q, float* scales, int64_t rows,
            int64_t stride, int device, cudaStream_t stream) {
  constexpr int SharedBytes = Staged ? 2 * kStageBytes : kFullBytes;
  // CUDA function attributes are per device. Set them off the launch path
  // after the first call, with no stream synchronization or device allocation.
  static std::deque<std::once_flag> initialized(device_properties.size());
  std::call_once(initialized[device], [] {
    auto status = cudaFuncSetAttribute(
        tma_row<Staged>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        SharedBytes);
    STD_TORCH_CHECK(status == cudaSuccess, "SiLU FP8 shared-memory setup: ",
                    cudaGetErrorString(status));
  });
  tma_row<Staged>
      <<<rows, kThreads, SharedBytes, stream>>>(x, q, scales, stride);
  const auto status = cudaPeekAtLastError();
  STD_TORCH_CHECK(status == cudaSuccess,
                  "SiLU FP8 launch: ", cudaGetErrorString(status));
}

}  // namespace

void silu_mul_per_token_quant_sm100(const torch::stable::Tensor& input,
                                    torch::stable::Tensor& out,
                                    torch::stable::Tensor& scales,
                                    bool staged) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(input.is_cuda() && out.is_cuda() && scales.is_cuda(),
                  "SiLU FP8 expects CUDA tensors");
  const auto device = input.get_device_index();
  STD_TORCH_CHECK(
      out.get_device_index() == device && scales.get_device_index() == device,
      "SiLU FP8 tensors must be on the same device");
  STD_TORCH_CHECK(input.dim() == 2 && input.size(1) == 2 * kWidth &&
                      input.scalar_type() == ScalarType::BFloat16 &&
                      input.stride(1) == 1 && input.stride(0) % 8 == 0,
                  "SiLU FP8 expects a BF16 [M, 54784] input with aligned rows");
  const auto rows = input.size(0);
  STD_TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max(),
                  "Too many SiLU FP8 rows for one launch");
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::Float8_e4m3fn &&
                      out.is_contiguous() && out.numel() == rows * kWidth &&
                      scales.scalar_type() == ScalarType::Float &&
                      scales.is_contiguous() && scales.numel() == rows,
                  "SiLU FP8 expects contiguous E4M3 output and FP32 scales");
  if (rows == 0) return;

  auto* x = static_cast<const uint16_t*>(input.data_ptr());
  auto* q = static_cast<uint8_t*>(out.data_ptr());
  auto* s = scales.mutable_data_ptr<float>();
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(x) % 16 == 0 &&
                      reinterpret_cast<uintptr_t>(q) % 16 == 0,
                  "SiLU FP8 input and output must be 16-byte aligned");

  const torch::stable::accelerator::DeviceGuard guard(device);
  const auto* prop = get_device_prop();
  STD_TORCH_CHECK(prop->major == 10 && prop->minor == 0,
                  "SiLU FP8 bulk-copy kernel requires SM100");
  const auto stream = get_current_cuda_stream(device);
  if (staged) {
    launch<true>(x, q, s, rows, input.stride(0), device, stream);
  } else {
    launch<false>(x, q, s, rows, input.stride(0), device, stream);
  }
}
