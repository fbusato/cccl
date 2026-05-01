#include <cuda/std/__simd_> // IWYU pragma: keep
#include <cuda/std/array>

#if _CCCL_HAS_SIMD_F32X2()

namespace simd = cuda::std::simd;

using Vec_f32_4 = simd::basic_vec<float, simd::fixed_size<4>>;

extern "C" __global__ void test_operator_minus_f32_4(const float* lhs, const float* rhs, float* out)
{
  const cuda::std::array<float, 4> lhs_values{lhs[0], lhs[1], lhs[2], lhs[3]};
  const cuda::std::array<float, 4> rhs_values{rhs[0], rhs[1], rhs[2], rhs[3]};

  const Vec_f32_4 lhs_vec(lhs_values);
  const Vec_f32_4 rhs_vec(rhs_values);
  const Vec_f32_4 result = lhs_vec - rhs_vec;

  out[0] = result[0];
  out[1] = result[1];
  out[2] = result[2];
  out[3] = result[3];
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : test_operator_minus_f32_4
; SM100: {{.*FADD2.*}}
; SM100: {{.*FADD2.*}}

*/

#endif // _CCCL_HAS_SIMD_F32X2()
