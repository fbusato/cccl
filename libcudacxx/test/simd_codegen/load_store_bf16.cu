#include <cuda/std/__simd/load.h>
#include <cuda/std/__simd/store.h>

#if _CCCL_HAS_NVBF16()

namespace simd = cuda::std::simd;

// --- 8-byte tier (SM80): __nv_bfloat16, N=4 ---

using Vec_bf16_4 = simd::basic_vec<__nv_bfloat16, simd::fixed_size<4>>;

__global__ void test_load_bf16_4(const __nv_bfloat16* in, __nv_bfloat16* out)
{
  Vec_bf16_4 v = simd::unchecked_load<Vec_bf16_4>(in, Vec_bf16_4::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_bf16_4::size(), simd::flag_aligned);
}

// --- 16-byte tier (SM100): __nv_bfloat16, N=8 ---

using Vec_bf16_8 = simd::basic_vec<__nv_bfloat16, simd::fixed_size<8>>;

__global__ void test_load_bf16_8(const __nv_bfloat16* in, __nv_bfloat16* out)
{
  Vec_bf16_8 v = simd::unchecked_load<Vec_bf16_8>(in, Vec_bf16_8::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_bf16_8::size(), simd::flag_aligned);
}

/*

; SM8X-LABEL: .visible .entry {{.*}}test_load_bf16_4{{.*}}(
; SM8X: {{.*}}ld.global.v4.u16{{.*}}
; SM8X: {{.*}}st.global.v4.u16{{.*}}

; SM10X-LABEL: .visible .entry {{.*}}test_load_bf16_8{{.*}}(
; SM10X: {{.*}}ld.global.v4.b32{{.*}}
; SM10X: {{.*}}st.global.v4.b32{{.*}}

*/

#endif // _CCCL_HAS_NVBF16()
