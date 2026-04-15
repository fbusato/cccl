#include <cuda/std/__simd/load.h>
#include <cuda/std/__simd/store.h>
#include <cuda/std/cstdint>

namespace simd = cuda::std::simd;

// --- 8-byte tier (SM80): int8_t, N=8 ---

using Vec_i8_8 = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<8>>;

__global__ void test_load_i8_8(const cuda::std::int8_t* in, cuda::std::int8_t* out)
{
  Vec_i8_8 v = simd::unchecked_load<Vec_i8_8>(in, Vec_i8_8::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i8_8::size(), simd::flag_aligned);
}

// --- 16-byte tier (SM100): int8_t, N=16 ---

using Vec_i8_16 = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<16>>;

__global__ void test_load_i8_16(const cuda::std::int8_t* in, cuda::std::int8_t* out)
{
  Vec_i8_16 v = simd::unchecked_load<Vec_i8_16>(in, Vec_i8_16::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i8_16::size(), simd::flag_aligned);
}

/*

; SM8X-LABEL: .visible .entry {{.*}}test_load_i8_8{{.*}}(
; SM8X: {{.*}}ld.global.v4.u16{{.*}}
; SM8X: {{.*}}st.global.v4.u16{{.*}}

; SM10X-LABEL: .visible .entry {{.*}}test_load_i8_16{{.*}}(
; SM10X: {{.*}}ld.global.v4.b32{{.*}}
; SM10X: {{.*}}st.global.v4.b32{{.*}}

*/
