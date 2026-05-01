#include <cuda/std/__simd_> // IWYU pragma: keep
#include <cuda/std/array>

#if _CCCL_HAS_SIMD_F32X2()

namespace simd = cuda::std::simd;

using Vec_f32_4 = simd::basic_vec<float, simd::fixed_size<4>>;

extern "C" __global__ void test_operator_unary_minus_f32_4(const float* in, float* out)
{
  const cuda::std::array<float, 4> values{in[0], in[1], in[2], in[3]};

  const Vec_f32_4 vec(values);
  const Vec_f32_4 result = -vec;

  out[0] = result[0];
  out[1] = result[1];
  out[2] = result[2];
  out[3] = result[3];
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : test_operator_unary_minus_f32_4
; SM100: {{.*FADD2.*}}
; SM100: {{.*FADD2.*}}

*/

#endif // _CCCL_HAS_SIMD_F32X2()
