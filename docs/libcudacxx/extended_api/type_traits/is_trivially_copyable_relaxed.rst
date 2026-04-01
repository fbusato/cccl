.. _libcudacxx-extended-api-type_traits-is_trivially_copyable_relaxed:

``cuda::is_trivially_copyable_relaxed``
=======================================

Defined in the ``<cuda/type_traits>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   struct is_trivially_copyable_relaxed;

   template <typename T>
   constexpr bool is_trivially_copyable_relaxed_v = is_trivially_copyable_relaxed<T>::value;

   } // namespace cuda

``cuda::is_trivially_copyable_relaxed<T>`` is a type trait that extends ``cuda::std::is_trivially_copyable`` to also recognize CUDA extended floating-point scalar and vector types as trivially copyable.

A type ``T`` satisfies ``cuda::is_trivially_copyable_relaxed`` if any of the following holds:

- ``T`` is trivially copyable.
- ``T`` is an extended floating-point scalar type (e.g. ``__half``, ``__nv_bfloat16``, ``__nv_fp8_e4m3``).
- ``T`` is an extended floating-point vector type (e.g. ``__half2``, ``__nv_bfloat162``, ``__nv_fp8x2_e4m3``).

The trait also propagates through composite types:

- C-style arrays: ``T[N]`` and ``T[]`` are relaxed trivially copyable when ``T`` is.
- ``cuda::std::array<T, N>``: relaxed trivially copyable when ``T`` is.
- ``cuda::std::pair<T1, T2>``: relaxed trivially copyable when both ``T1`` and ``T2`` are.
- ``cuda::std::tuple<Ts...>``: relaxed trivially copyable when all ``Ts...`` are.

``const``, ``volatile``, and ``const volatile`` qualifications are handled transparently.

Example
-------

.. code:: cuda

   #include <cuda/type_traits>
   #include <cuda/std/array>
   #include <cuda/std/tuple>
   #include <cuda/std/utility>

   #include <cuda_fp16.h>

   // Standard trivially copyable types
   static_assert(cuda::is_trivially_copyable_relaxed_v<int>);
   static_assert(cuda::is_trivially_copyable_relaxed_v<float>);

   // Extended floating-point types
   static_assert(cuda::is_trivially_copyable_relaxed_v<__half>);
   static_assert(cuda::is_trivially_copyable_relaxed_v<__nv_bfloat16>);
   static_assert(cuda::is_trivially_copyable_relaxed_v<__half2>);

   // Composite types containing extended floating-point types
   static_assert(cuda::is_trivially_copyable_relaxed_v<__half[4]>);
   static_assert(cuda::is_trivially_copyable_relaxed_v<cuda::std::array<__half, 4>>);
   static_assert(cuda::is_trivially_copyable_relaxed_v<cuda::std::pair<__half, int>>);
   static_assert(cuda::is_trivially_copyable_relaxed_v<cuda::std::tuple<__half, float, double>>);
