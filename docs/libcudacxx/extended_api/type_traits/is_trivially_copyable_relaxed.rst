.. _libcudacxx-extended-api-type_traits-is_trivially_copyable_relaxed:

``cuda::is_trivially_copyable_relaxed``
=======================================

Defined in the ``<cuda/type_traits>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   constexpr bool is_trivially_copyable_relaxed_v = /* see below */;

   template <typename T>
   using is_trivially_copyable_relaxed = cuda::std::bool_constant<is_trivially_copyable_relaxed_v<T>>;

   } // namespace cuda

``cuda::is_trivially_copyable_relaxed_v<T>`` is a variable template that extends ``cuda::std::is_trivially_copyable`` to also recognize CUDA extended floating-point scalar and vector types as trivially copyable.

A type ``T`` satisfies ``cuda::is_trivially_copyable_relaxed`` if any of the following holds:

- ``T`` is trivially copyable.
- ``T`` is an extended floating-point scalar type (e.g. ``__half``, ``__nv_bfloat16``, ``__nv_fp8_e4m3``).
- ``T`` is an extended floating-point vector type (e.g. ``__half2``, ``__nv_bfloat162``, ``__nv_fp8x2_e4m3``).

The trait also propagates through composite types:

- C-style arrays: ``T[N]`` and ``T[]`` are relaxed trivially copyable when ``T`` is.
- ``cuda::std::array<T, N>``: relaxed trivially copyable when ``T`` is.
- ``cuda::std::pair<T1, T2>``: relaxed trivially copyable when both ``T1`` and ``T2`` are and the object has no padding.
- ``cuda::std::tuple<Ts...>``: relaxed trivially copyable when all ``Ts...`` are and the object has no padding.

``const`` qualification is handled transparently, while ``volatile`` is compile-time dependent.

Custom Specialization
---------------------

Users may specialize ``cuda::is_trivially_copyable_relaxed_v`` for their own types whose memory representation is safe to copy with ``memcpy`` but that the compiler does not consider trivially copyable.

... warning::

    Users are responsible for ensuring that the type is actually trivially copyable when specializing this variable template. Otherwise, the behavior is undefined.

A common case is a type that wraps extended floating-point fields and provides user-defined copy operations
solely to add ``__host__ __device__`` annotations:

.. code:: cuda

    struct HalfWrapper {
        __half value;
    };

    struct NonTriviallyCopyable {
        __host__ __device__ NonTriviallyCopyable(const NonTriviallyCopyable&) {}
    };

    // Specializing the variable template
    template <>
    constexpr bool cuda::is_trivially_copyable_relaxed_v<HalfWrapper> = true;

    template <>
    constexpr bool cuda::is_trivially_copyable_relaxed_v<NonTriviallyCopyable> = true;

    static_assert(cuda::is_trivially_copyable_relaxed_v<HalfWrapper>);
    static_assert(cuda::is_trivially_copyable_relaxed_v<NonTriviallyCopyable>);

Examples
--------

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

   // Padding-free composite types containing extended floating-point types
   static_assert(cuda::is_trivially_copyable_relaxed_v<__half[4]>);
   static_assert(cuda::is_trivially_copyable_relaxed_v<cuda::std::array<__half, 4>>);
   static_assert(cuda::is_trivially_copyable_relaxed_v<cuda::std::pair<__half, __half>>);
   static_assert(cuda::is_trivially_copyable_relaxed_v<cuda::std::tuple<__half, __half>>);

   // Composites with padding are not trivially copyable relaxed
   static_assert(!cuda::is_trivially_copyable_relaxed_v<cuda::std::pair<__half, int>>);
