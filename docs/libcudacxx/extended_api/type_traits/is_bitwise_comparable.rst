.. _libcudacxx-extended-api-type_traits-is_bitwise_comparable:

``cuda::is_bitwise_comparable``
===============================

Defined in the ``<cuda/type_traits>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   constexpr bool is_bitwise_comparable_v = /* see below */;

   template <typename T>
   using is_bitwise_comparable = cuda::std::bool_constant<is_bitwise_comparable_v<T>>;

   } // namespace cuda

``cuda::is_bitwise_comparable_v<T>`` is a user-specializable variable template that forwards to ``cuda::std::has_unique_object_representations`` but excludes extended floating-point scalar and vector types.

The trait also propagates through composite types:

- C-style arrays: ``T[N]`` and ``T[]`` are bitwise comparable when ``T`` is.
- ``cuda::std::array<T, N>``: bitwise comparable when ``T`` is.
- ``cuda::std::pair<T1, T2>``: bitwise comparable when both ``T1`` and ``T2`` are and the object has no padding.
- ``cuda::std::tuple<Ts...>``: bitwise comparable when all ``Ts...`` are and the object has no padding.

``const``, ``volatile``, and ``const volatile`` qualifications are handled transparently.

... warning::

    ``cuda::is_bitwise_comparable_v`` cannot recognize structures (or classes) that contain extended floating-point types as bitwise comparable. Structures that contain extended floating-point types are detected as bitwise comparable by default.

Custom Specialization
---------------------

Users may specialize ``cuda::is_bitwise_comparable_v`` for their own types to indicate that two objects with the same value always have the same object  representation, even when the compiler cannot determine this automatically.
The specialization must be provided for the unqualified type; cv-qualified forms are handled automatically.

.. code:: cuda

   struct MyType {
     double value;
   };

   template <>
   constexpr bool cuda::is_bitwise_comparable_v<MyType> = true;

   static_assert(cuda::is_bitwise_comparable_v<MyType>);
   static_assert(cuda::is_bitwise_comparable_v<const MyType>);


... warning::

    Users are responsible for ensuring that the type is actually bitwise comparable when specializing this variable template. Otherwise, the behavior is undefined.

Examples
--------

.. code:: cuda

   #include <cuda/type_traits>

   // Integer types have unique object representations
   static_assert(cuda::is_bitwise_comparable_v<int>);
   static_assert(cuda::is_bitwise_comparable_v<unsigned>);
   static_assert(cuda::is_bitwise_comparable_v<char>);
   static_assert(cuda::is_bitwise_comparable_v<int[4]>);

   // Floating-point types do not
   static_assert(!cuda::is_bitwise_comparable_v<float>);
   static_assert(!cuda::is_bitwise_comparable_v<double>);
