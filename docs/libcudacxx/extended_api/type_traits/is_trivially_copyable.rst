.. _libcudacxx-extended-api-type_traits-is_trivially_copyable:

``cuda::is_trivially_copyable``
=======================================

Defined in the ``<cuda/type_traits>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   constexpr bool is_trivially_copyable_v = /* see below */;

   template <typename T>
   using is_trivially_copyable = cuda::std::bool_constant<is_trivially_copyable_v<T>>;

   } // namespace cuda

``cuda::is_trivially_copyable_v<T>`` is a variable template that extends ``cuda::std::is_trivially_copyable`` to also recognize CUDA extended floating-point scalar and vector types as trivially copyable.

A type ``T`` satisfies ``cuda::is_trivially_copyable`` if any of the following holds:

- ``T`` is trivially copyable.
- ``T`` is an extended floating-point scalar type (e.g. ``__half``, ``__nv_bfloat16``, ``__nv_fp8_e4m3``).
- ``T`` is an extended floating-point vector type (e.g. ``__half2``, ``__nv_bfloat162``, ``__nv_fp8x2_e4m3``).

The trait also propagates through composite types:

- C-style arrays: ``T[N]`` and ``T[]`` are trivially copyable when ``T`` is.
- ``cuda::std::array<T, N>``: trivially copyable when ``T`` is also trivially copyable.
- ``cuda::std::pair<T1, T2>``: trivially copyable when both ``T1`` and ``T2`` are.
- ``cuda::std::tuple<Ts...>``: trivially copyable when all ``Ts...`` are.
- Aggregates: trivially copyable when all members are.

``const`` qualification is handled transparently, while ``volatile`` is compiler dependent.

Custom Specialization
---------------------

Users may specialize ``cuda::is_trivially_copyable_v`` for types whose semantics allow copying with ``memcpy``, but which the compiler does not consider to be trivially copyable.

A `trivially copyable <https://en.cppreference.com/w/cpp/language/classes.html>`__ class is a class that

- has at least one eligible copy constructor, move constructor, copy assignment operator, or move assignment operator,
- each of its eligible copy constructors is trivial
- each eligible move constructor is trivial
- each eligible copy assignment operator is trivial
- each eligible move assignment operator is trivial, and
- has a non-deleted trivial destructor.

.. warning::

    The user is responsible for ensuring that the type is actually trivially copyable when specializing this variable template. Otherwise, the behavior is undefined.

A common case is a user-declared type that has a copy constructor but is technically trivially copyable:

.. code:: cuda

    struct NonTriviallyCopyable {
        int x;
        __host__ __device__ NonTriviallyCopyable(const NonTriviallyCopyable&) {}
    };

    // Specializing the variable template
    template <>
    constexpr bool cuda::is_trivially_copyable_v<NonTriviallyCopyable> = true;

    static_assert(cuda::is_trivially_copyable_v<NonTriviallyCopyable>);

Examples
--------

.. code:: cuda

   #include <cuda/type_traits>
   #include <cuda/std/array>
   #include <cuda/std/tuple>
   #include <cuda/std/utility>

   #include <cuda_fp16.h>

   struct UserType {
     __half x, y;
   };

   // Standard trivially copyable types
   static_assert(cuda::is_trivially_copyable_v<int>);
   static_assert(cuda::is_trivially_copyable_v<float>);

   // Extended floating-point types
   static_assert(cuda::is_trivially_copyable_v<__half>);
   static_assert(cuda::is_trivially_copyable_v<__nv_bfloat16>);
   static_assert(cuda::is_trivially_copyable_v<__half2>);

   // Padding-free composite types containing extended floating-point types
   static_assert(cuda::is_trivially_copyable_v<__half[4]>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::array<__half, 4>>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::pair<__half, __half>>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::tuple<__half, __half>>);

   // Composites with padding are not trivially copyable
   static_assert(!cuda::is_trivially_copyable_v<cuda::std::pair<__half, int>>);
