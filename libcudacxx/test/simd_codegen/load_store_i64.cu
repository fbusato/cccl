#include <cuda/std/__simd/load.h>
#include <cuda/std/__simd/store.h>
#include <cuda/std/cstdint>

namespace simd = cuda::std::simd;

// --- 16-byte tier (SM100): int64_t, N=2 ---
// 8-byte tier skipped: N=1

using Vec_i64_2 = simd::basic_vec<cuda::std::int64_t, simd::fixed_size<2>>;

__global__ void test_load_i64_2(const cuda::std::int64_t* in, cuda::std::int64_t* out)
{
  Vec_i64_2 v = simd::unchecked_load<Vec_i64_2>(in, Vec_i64_2::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_i64_2::size(), simd::flag_aligned);
}

/*

; SM10X-LABEL: .visible .entry {{.*}}test_load_i64_2{{.*}}(
; SM10X: {{.*}}ld.global.v4.b32{{.*}}
; SM10X: {{.*}}st.global.v4.b32{{.*}}

*/
