#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <unittest/unittest.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <unittest/cuda/testframework.h>
#endif

using namespace unittest;

struct SumTupleFunctor
{
  template <typename Tuple>
  _CCCL_HOST_DEVICE Tuple operator()(const Tuple& lhs, const Tuple& rhs)
  {
    using thrust::get;

    return thrust::make_tuple(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
  }
};

struct MakeTupleFunctor
{
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE thrust::tuple<T1, T2> operator()(T1& lhs, T2& rhs)
  {
    return thrust::make_tuple(lhs, rhs);
  }
};

template <typename T>
struct TestTupleScan
{
  void operator()(const size_t n)
  {
    using namespace thrust;

    host_vector<T> h_t1 = unittest::random_integers<T>(n);
    host_vector<T> h_t2 = unittest::random_integers<T>(n);

    // initialize input
    host_vector<tuple<T, T>> h_input(n);
    thrust::transform(h_t1.begin(), h_t1.end(), h_t2.begin(), h_input.begin(), MakeTupleFunctor());
    device_vector<tuple<T, T>> d_input = h_input;

    // allocate output
    tuple<T, T> zero(0, 0);
    host_vector<tuple<T, T>> h_output(n, zero);
    device_vector<tuple<T, T>> d_output(n, zero);

    // inclusive_scan
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), SumTupleFunctor());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), SumTupleFunctor());
    ASSERT_EQUAL_QUIET(h_output, d_output);

    // exclusive_scan
    tuple<T, T> init(13, 17);
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), init, SumTupleFunctor());
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), init, SumTupleFunctor());

    ASSERT_EQUAL_QUIET(h_output, d_output);
  }
};
VariableUnitTest<TestTupleScan, IntegralTypes> TestTupleScanInstance;
