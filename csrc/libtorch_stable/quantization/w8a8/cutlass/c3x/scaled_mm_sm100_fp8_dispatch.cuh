#pragma once

#include <torch/headeronly/util/shim_utils.h>

#include "scaled_mm.cuh"
#include "cutlass_gemm_caller.cuh"
#include "libtorch_stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

/**
 * This file defines Gemm kernel configurations for SM100 (fp8) based on the
 * Gemm shape.
 */

namespace vllm {

using c3x::cutlass_gemm_caller;

template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule, bool swap_ab_ = false,
          typename TileScheduler_ = void>
struct cutlass_3x_gemm_sm100_fp8 {
  using ElementAB = ElementAB_;
  using ElementC = ElementD_;
  using ElementD = ElementD_;
  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;

  using EVTCompute = typename Epilogue::EVTCompute;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  // Compile-time swap_ab flag
  static constexpr bool swap_ab = swap_ab_;

  // -----------------------------------------------------------
  // Layout definitions
  // -----------------------------------------------------------
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;

  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutD>::type;

  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;

  // -----------------------------------------------------------
  // Collective epilogue (conditionally swap operands and layouts)
  // -----------------------------------------------------------
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc, float, ElementC,
          conditional_t<swap_ab, LayoutC_Transpose, LayoutC>, AlignmentCD,
          ElementD, conditional_t<swap_ab, LayoutD_Transpose, LayoutD>,
          AlignmentCD, EpilogueSchedule, EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);

  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  // -----------------------------------------------------------
  // Collective mainloop (conditionally swap operands and layouts)
  // -----------------------------------------------------------
  using CollectiveMainloop = conditional_t<
      swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutB_T, AlignmentAB,             // Swapped B (as A)
          ElementAB, LayoutA_T, AlignmentAB,  // Swapped A (as B)
          ElementAcc, TileShape, ClusterShape, Stages,
          KernelSchedule>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutA, AlignmentAB, ElementAB, LayoutB, AlignmentAB, ElementAcc,
          TileShape, ClusterShape, Stages, KernelSchedule>::CollectiveOp>;

  // -----------------------------------------------------------
  // Kernel definition
  // -----------------------------------------------------------
  using GemmKernel = enable_sm100_to_sm120<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileScheduler_>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_default {
  // M in (256, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_256, _128, _128>;
  using ClusterShape = Shape<_2, _2, _1>;
  using Cutlass3xGemm =
      conditional_t<EnableBias,
                    cutlass_3x_gemm_sm100_fp8<
                        InType, OutType, c3x::ScaledEpilogueBias, TileShape,
                        ClusterShape, KernelSchedule, EpilogueSchedule>,
                    cutlass_3x_gemm_sm100_fp8<
                        InType, OutType, c3x::ScaledEpilogue, TileShape,
                        ClusterShape, KernelSchedule, EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_M256 {
  // M in (64, 256]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      conditional_t<EnableBias,
                    cutlass_3x_gemm_sm100_fp8<
                        InType, OutType, c3x::ScaledEpilogueBias, TileShape,
                        ClusterShape, KernelSchedule, EpilogueSchedule>,
                    cutlass_3x_gemm_sm100_fp8<
                        InType, OutType, c3x::ScaledEpilogue, TileShape,
                        ClusterShape, KernelSchedule, EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_M64_swap_ab {
  // This config is for M in (16, 64] and K >= 4096
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _64, _256>;
  using ClusterShape = Shape<_4, _1, _1>;

  // Use ScaledEpilogueColumnBias instead of ScaledEpilogueBias when doing swap
  // AB
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                                TileShape, ClusterShape, KernelSchedule,
                                EpilogueSchedule, true>,
      cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule,
                                true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_M64 {
  // This config is for M = 64 and K < 4096 (do not enable swap AB in such case)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _64, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  using Cutlass3xGemm =
      conditional_t<EnableBias,
                    cutlass_3x_gemm_sm100_fp8<
                        InType, OutType, c3x::ScaledEpilogueBias, TileShape,
                        ClusterShape, KernelSchedule, EpilogueSchedule>,
                    cutlass_3x_gemm_sm100_fp8<
                        InType, OutType, c3x::ScaledEpilogue, TileShape,
                        ClusterShape, KernelSchedule, EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_M16_swap_ab {
  // M in [1, 16]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_4, _1, _1>;

  // Use ScaledEpilogueColumnBias instead of ScaledEpilogueBias when doing swap
  // AB
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                                TileShape, ClusterShape, KernelSchedule,
                                EpilogueSchedule, true>,
      cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule,
                                true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_few_tiles_swap_ab {
  // Small M (see sm100_fp8_select_config). Swap A/B so N fills the 256-row
  // 2-SM MMA tile and M is split into 128-column tiles, which pads M less.
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_256, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;

  // Use ScaledEpilogueColumnBias instead of ScaledEpilogueBias when doing swap
  // AB
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                                TileShape, ClusterShape, KernelSchedule,
                                EpilogueSchedule, true>,
      cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule,
                                true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_few_tiles_stream_k {
  // Small M whose data-parallel tiles leave the last wave almost empty (see
  // sm100_fp8_select_config). CUTLASS's heuristic decides per launch: if the
  // tiles of this config leave a partial last wave, the stream-K scheduler
  // splits their K loop so all SMs get work, which needs a reduction
  // workspace; if they fit in whole waves, it runs data-parallel. Split
  // tiles are reduced in a fixed order, so results are deterministic but not
  // bitwise equal to the data-parallel configs.
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_256, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using TileScheduler = cutlass::gemm::StreamKScheduler;
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogueBias,
                                TileShape, ClusterShape, KernelSchedule,
                                EpilogueSchedule, false, TileScheduler>,
      cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule,
                                false, TileScheduler>>;
};

template <typename InType, typename OutType, bool EnableBias,
          typename ClusterShape_ = Shape<_2, _2, _1>>
struct sm100_fp8_config_large_tiles {
  // 256x256 2-SM tile (see sm100_fp8_select_config). Per FLOP it loads less
  // A and B data than the 256x128 default, so it is faster when its CTAs
  // fill the waves about as well. Data-parallel, no workspace. Cluster 2x2
  // for M <= 1024 (few waves), 2x1 above (sm100_fp8_config_large_m); each
  // was the faster one in that range on GB200.
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_256, _256, _128>;
  using ClusterShape = ClusterShape_;
  using Cutlass3xGemm =
      conditional_t<EnableBias,
                    cutlass_3x_gemm_sm100_fp8<
                        InType, OutType, c3x::ScaledEpilogueBias, TileShape,
                        ClusterShape, KernelSchedule, EpilogueSchedule>,
                    cutlass_3x_gemm_sm100_fp8<
                        InType, OutType, c3x::ScaledEpilogue, TileShape,
                        ClusterShape, KernelSchedule, EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
using sm100_fp8_config_large_m =
    sm100_fp8_config_large_tiles<InType, OutType, EnableBias,
                                 Shape<_2, _1, _1>>;

// Shape-aware choice only below this M (decode and small prefill chunks).
inline constexpr uint32_t kSm100Fp8FewTilesMaxM = 1024;
// Stream-K only pays off when each tile has a long K loop to split.
inline constexpr uint32_t kSm100Fp8StreamKMinK = 8192;

enum class Sm100Fp8Config {
  kM16SwapAB,
  kM64SwapAB,
  kM64,
  kM256,
  kDefault,
  kFewTilesSwapAB,
  kFewTilesStreamK,
  kLargeTiles,
  kLargeM,
};

// Number of CTAs a config launches for an (m x n) output. With an even
// cluster M and a tile M of 128 or 256, CUTLASS uses the 2-SM MMA, where one
// tile spans two CTAs.
template <typename Config>
inline uint32_t sm100_fp8_num_ctas(uint32_t m, uint32_t n) {
  using TileShape = typename Config::TileShape;
  using ClusterShape = typename Config::ClusterShape;
  constexpr uint32_t tile_m = cute::size<0>(TileShape{});
  constexpr uint32_t tile_n = cute::size<1>(TileShape{});
  constexpr uint32_t cluster_m = cute::size<0>(ClusterShape{});
  constexpr uint32_t cluster_n = cute::size<1>(ClusterShape{});
  constexpr uint32_t ctas_per_tile_m =
      (cluster_m % 2 == 0 && tile_m % 128 == 0) ? 2 : 1;
  auto round_up = [](uint32_t x, uint32_t y) { return (x + y - 1) / y * y; };
  uint32_t const grid_m =
      round_up((m + tile_m - 1) / tile_m * ctas_per_tile_m, cluster_m);
  uint32_t const grid_n = round_up((n + tile_n - 1) / tile_n, cluster_n);
  return grid_m * grid_n;
}

// Output elements per SM over all waves of a config: whole waves x the
// output elements of one CTA. Used to compare tile sizes, since a bigger
// tile does more work per CTA but may leave more of its last wave empty.
template <typename Config>
inline uint64_t sm100_fp8_wave_work(uint32_t m, uint32_t n, uint32_t sm_count) {
  using TileShape = typename Config::TileShape;
  using ClusterShape = typename Config::ClusterShape;
  constexpr uint64_t tile_m = cute::size<0>(TileShape{});
  constexpr uint64_t tile_n = cute::size<1>(TileShape{});
  constexpr uint64_t cluster_m = cute::size<0>(ClusterShape{});
  constexpr uint64_t ctas_per_tile_m =
      (cluster_m % 2 == 0 && tile_m % 128 == 0) ? 2 : 1;
  uint64_t const waves =
      (sm100_fp8_num_ctas<Config>(m, n) + sm_count - 1) / sm_count;
  return waves * (tile_m / ctas_per_tile_m) * tile_n;
}

// Pick a config from the GEMM shape. For 64 < m <= 1024, each data-parallel
// config is costed by its wave work (sm100_fp8_wave_work), which counts both a
// nearly empty last wave and the rows a tile pads beyond M. The M-only config,
// the 256x128 default and swap-AB 256x128 compete on that cost; bigger tiles
// win ties because they load less data per FLOP. The 256x256 tile wins if its
// wave work is at most 1/8 more. If the chosen config leaves its last wave
// less than a quarter full and K is long, the stream-K scheduler spreads the
// tail over all SMs. Above m = 1024, the 256x256 tile (2x1 cluster) replaces
// the default under the same 1/8 rule.
template <typename InType, typename OutType, bool EnableBias>
inline Sm100Fp8Config sm100_fp8_select_config(uint32_t m, uint32_t n,
                                              uint32_t k, uint32_t sm_count) {
  if (m <= 16) {
    return Sm100Fp8Config::kM16SwapAB;
  }
  if (m <= 64) {
    // do not enable swap AB for m == 64 and k < 4096
    return (m == 64 && k < 4096) ? Sm100Fp8Config::kM64
                                 : Sm100Fp8Config::kM64SwapAB;
  }
  using M256 = sm100_fp8_config_M256<InType, OutType, EnableBias>;
  using Default = sm100_fp8_config_default<InType, OutType, EnableBias>;
  using FewTilesSwapAB =
      sm100_fp8_config_few_tiles_swap_ab<InType, OutType, EnableBias>;
  using LargeTiles = sm100_fp8_config_large_tiles<InType, OutType, EnableBias>;
  using LargeM = sm100_fp8_config_large_m<InType, OutType, EnableBias>;
  uint64_t const dflt = sm100_fp8_wave_work<Default>(m, n, sm_count);
  if (m > kSm100Fp8FewTilesMaxM) {
    return 8 * sm100_fp8_wave_work<LargeM>(m, n, sm_count) <= 9 * dflt
               ? Sm100Fp8Config::kLargeM
               : Sm100Fp8Config::kDefault;
  }
  uint64_t const large = sm100_fp8_wave_work<LargeTiles>(m, n, sm_count);
  // Swap-AB computes the (n x m) problem.
  uint64_t const swap = sm100_fp8_wave_work<FewTilesSwapAB>(n, m, sm_count);
  Sm100Fp8Config best = Sm100Fp8Config::kDefault;
  uint64_t best_work = dflt;
  uint32_t best_ctas = sm100_fp8_num_ctas<Default>(m, n);
  if (swap < best_work) {
    best = Sm100Fp8Config::kFewTilesSwapAB;
    best_work = swap;
    best_ctas = sm100_fp8_num_ctas<FewTilesSwapAB>(n, m);
  }
  if (m <= 256 && sm100_fp8_wave_work<M256>(m, n, sm_count) < best_work) {
    best = Sm100Fp8Config::kM256;
    best_work = sm100_fp8_wave_work<M256>(m, n, sm_count);
    best_ctas = sm100_fp8_num_ctas<M256>(m, n);
  }
  if (8 * large <= 9 * best_work) {
    return Sm100Fp8Config::kLargeTiles;
  }
  if (m > 128 && k >= kSm100Fp8StreamKMinK && best_ctas % sm_count != 0 &&
      best_ctas % sm_count <= sm_count / 4) {
    return Sm100Fp8Config::kFewTilesStreamK;
  }
  return best;
}

// Launches Gemm and returns true. With workspace_optional, returns false
// without launching when Gemm's workspace cannot be allocated.
template <typename Gemm, typename... EpilogueArgs>
bool try_cutlass_gemm_caller_sm100_fp8(bool workspace_optional,
                                       torch::stable::Tensor& out,
                                       torch::stable::Tensor const& a,
                                       torch::stable::Tensor const& b,
                                       EpilogueArgs&&... epilogue_params) {
  static constexpr bool swap_ab = Gemm::swap_ab;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;

  int32_t m = a.size(0), n = b.size(1), k = a.size(1);
  auto prob_shape =
      swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

  StrideA a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(
      StrideC{},
      swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args =
      swap_ab ? typename GemmKernel::MainloopArguments{b_ptr, b_stride, a_ptr,
                                                       a_stride}
              : typename GemmKernel::MainloopArguments{a_ptr, a_stride, b_ptr,
                                                       b_stride};

  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr, c_stride, c_ptr, c_stride};

  return c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape,
                                              mainloop_args, epilogue_args, {},
                                              workspace_optional);
}

template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller_sm100_fp8(torch::stable::Tensor& out,
                                   torch::stable::Tensor const& a,
                                   torch::stable::Tensor const& b,
                                   EpilogueArgs&&... epilogue_params) {
  try_cutlass_gemm_caller_sm100_fp8<Gemm>(
      false, out, a, b, std::forward<EpilogueArgs>(epilogue_params)...);
}

template <typename InType, typename OutType, bool EnableBias,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm100_fp8_dispatch(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  STD_TORCH_CHECK(a.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);

  using Cutlass3xGemmDefault =
      typename sm100_fp8_config_default<InType, OutType,
                                        EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM16SwapAB =
      typename sm100_fp8_config_M16_swap_ab<InType, OutType,
                                            EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM64SwapAB =
      typename sm100_fp8_config_M64_swap_ab<InType, OutType,
                                            EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM64 =
      typename sm100_fp8_config_M64<InType, OutType, EnableBias>::Cutlass3xGemm;

  using Cutlass3xGemmM256 =
      typename sm100_fp8_config_M256<InType, OutType,
                                     EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmFewTilesSwapAB =
      typename sm100_fp8_config_few_tiles_swap_ab<InType, OutType,
                                                  EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmFewTilesStreamK =
      typename sm100_fp8_config_few_tiles_stream_k<InType, OutType,
                                                   EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmLargeTiles =
      typename sm100_fp8_config_large_tiles<InType, OutType,
                                            EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmLargeM =
      typename sm100_fp8_config_large_m<InType, OutType,
                                        EnableBias>::Cutlass3xGemm;

  uint32_t const m = a.size(0);
  uint32_t const n = b.size(1);
  uint32_t const k = a.size(1);
  uint32_t const sm_count = get_device_prop()->multiProcessorCount;

  switch (
      sm100_fp8_select_config<InType, OutType, EnableBias>(m, n, k, sm_count)) {
    case Sm100Fp8Config::kM16SwapAB:
      return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM16SwapAB>(
          out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
    case Sm100Fp8Config::kM64SwapAB:
      return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM64SwapAB>(
          out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
    case Sm100Fp8Config::kM64:
      return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM64>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
    case Sm100Fp8Config::kM256:
      return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM256>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
    case Sm100Fp8Config::kFewTilesSwapAB:
      return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmFewTilesSwapAB>(
          out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
    case Sm100Fp8Config::kFewTilesStreamK:
      // Stream-K may need a reduction workspace that the data-parallel configs
      // do not. If it cannot be allocated, run the M-only config instead.
      if (try_cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmFewTilesStreamK>(
              true, out, a, b, a_scales, b_scales, args...)) {
        return;
      }
      if (m <= 256) {
        return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM256>(
            out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
      }
      return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmDefault>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
    case Sm100Fp8Config::kLargeTiles:
      return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmLargeTiles>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
    case Sm100Fp8Config::kLargeM:
      return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmLargeM>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
    case Sm100Fp8Config::kDefault:
    default:
      return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmDefault>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  }
}

template <typename InType, typename OutType, bool EnableBias,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm100_fp8_batch_invariant_dispatch(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  STD_TORCH_CHECK(a.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);

  using Cutlass3xGemmM64SwapAB =
      typename sm100_fp8_config_M64_swap_ab<InType, OutType,
                                            EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM64 =
      typename sm100_fp8_config_M64<InType, OutType, EnableBias>::Cutlass3xGemm;

  // keep the CUTLASS config independent of M for batch invariance
  uint32_t const k = a.size(1);
  if (k < 4096) {
    return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM64>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  }
  return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM64SwapAB>(
      out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
}

template <bool EnableBias, typename... EpilogueArgs>
void cutlass_scaled_mm_sm100_fp8_epilogue(torch::stable::Tensor& out,
                                          torch::stable::Tensor const& a,
                                          torch::stable::Tensor const& b,
                                          torch::stable::Tensor const& a_scales,
                                          torch::stable::Tensor const& b_scales,
                                          EpilogueArgs&&... epilogue_args) {
  STD_TORCH_CHECK(a.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);

  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    return cutlass_gemm_sm100_fp8_dispatch<cutlass::float_e4m3_t,
                                           cutlass::bfloat16_t, EnableBias>(
        out, a, b, a_scales, b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    return cutlass_gemm_sm100_fp8_dispatch<cutlass::float_e4m3_t,
                                           cutlass::half_t, EnableBias>(
        out, a, b, a_scales, b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

template <bool EnableBias, typename... EpilogueArgs>
void cutlass_scaled_mm_sm100_fp8_batch_invariant_epilogue(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, EpilogueArgs&&... epilogue_args) {
  STD_TORCH_CHECK(a.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);

  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    return cutlass_gemm_sm100_fp8_batch_invariant_dispatch<
        cutlass::float_e4m3_t, cutlass::bfloat16_t, EnableBias>(
        out, a, b, a_scales, b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    return cutlass_gemm_sm100_fp8_batch_invariant_dispatch<
        cutlass::float_e4m3_t, cutlass::half_t, EnableBias>(
        out, a, b, a_scales, b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

}  // namespace vllm
