#include <cuda/std/__simd/load.h>
#include <cuda/std/__simd/store.h>

namespace simd = cuda::std::simd;

// --- 16-byte tier (SM100): double, N=2 ---
// 8-byte tier skipped: N=1

using Vec_f64_2 = simd::basic_vec<double, simd::fixed_size<2>>;

__global__ void test_load_f64_2(const double* in, double* out)
{
  Vec_f64_2 v = simd::unchecked_load<Vec_f64_2>(in, Vec_f64_2::size(), simd::flag_aligned);
  simd::unchecked_store(v, out, Vec_f64_2::size(), simd::flag_aligned);
}

/*

; SM10X-LABEL: .visible .entry {{.*}}test_load_f64_2{{.*}}(
; SM10X: {{.*}}ld.global.v4.b32{{.*}}
; SM10X: {{.*}}st.global.v4.b32{{.*}}

*/
