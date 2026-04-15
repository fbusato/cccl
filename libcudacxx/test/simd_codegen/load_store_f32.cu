#include <cuda/std/__simd/load.h>
#include <cuda/std/__simd/store.h>

namespace simd = cuda::std::simd;

// --- 8-byte tier (SM80): float, N=2 ---

using Vec_f32_2 = simd::basic_vec<float, simd::fixed_size<2>>;

__global__ void test_load_f32_2(const float* in, float* out)
{
  Vec_f32_2 v = simd::unchecked_load<Vec_f32_2>(in, Vec_f32_2::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_f32_2::size(), simd::flag_aligned);
}

// --- 16-byte tier (SM100): float, N=4 ---

using Vec_f32_4 = simd::basic_vec<float, simd::fixed_size<4>>;

__global__ void test_load_f32_4(const float* in, float* out)
{
  Vec_f32_4 v = simd::unchecked_load<Vec_f32_4>(in, Vec_f32_4::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_f32_4::size(), simd::flag_aligned);
}

/*

; SM8X-LABEL: .visible .entry {{.*}}test_load_f32_2{{.*}}(
; SM8X: {{.*}}ld.global.v4.u16{{.*}}
; SM8X: {{.*}}st.global.v4.u16{{.*}}

; SM10X-LABEL: .visible .entry {{.*}}test_load_f32_4{{.*}}(
; SM10X: {{.*}}ld.global.v4.b32{{.*}}
; SM10X: {{.*}}st.global.v4.b32{{.*}}

*/
