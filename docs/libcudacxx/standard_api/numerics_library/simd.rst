.. _libcudacxx-standard-api-numerics-simd:

``<cuda/std/__simd_>``
=======================

The internal ``<cuda/std/__simd_>`` header provides a partial implementation of the C++ Standard Library data-parallel types and operations in ``cuda::std::simd``.
The public ``<cuda/simd>`` header includes this partial implementation together with the CUDA-specific extensions in ``cuda::simd``, which are documented in the :ref:`SIMD extended API <libcudacxx-extended-api-simd>`.

Feature-test macros
-------------------

libcu++ uses the ``__cccl_lib_`` prefix for library feature-test macros to avoid conflicts with the host standard library.
Including either ``<cuda/simd>`` or ``<cuda/std/version>`` defines:

- ``__cccl_lib_simd`` as ``202411L``.
- ``__cccl_lib_simd_complex`` as ``202502L``.
- ``__cccl_lib_simd_permutations`` as ``202506L``.

The current C++29 ``__cpp_lib_simd`` value is not advertised because it includes ``iota``.
Likewise, ``__cccl_lib_simd_bitops`` is not defined because the corresponding C++29 operations are not implemented.

Differences from the C++ Standard
---------------------------------

Mathematical functions
^^^^^^^^^^^^^^^^^^^^^^

The C++ Standard specifies most ``std::simd`` mathematical functions as ``constexpr``. The corresponding ``cuda::std::simd`` functions are not ``constexpr`` when the scalar function used to implement them is not ``constexpr`` in libcu++.

The following C++ Standard special mathematical functions are not provided for ``cuda::std::simd``:

- Associated orthogonal polynomials: ``assoc_laguerre``, ``assoc_legendre``, and ``sph_legendre``.
- Beta and elliptic integrals: ``beta``, ``comp_ellint_1``, ``comp_ellint_2``, ``comp_ellint_3``, ``ellint_1``,  ``ellint_2``, and ``ellint_3``.
- Bessel and Neumann functions: ``cyl_bessel_i``, ``cyl_bessel_j``, ``cyl_bessel_k``, ``cyl_neumann``, ``sph_bessel``, and ``sph_neumann``.
- Other special functions: ``expint``, ``hermite``, ``laguerre``, ``legendre``, and ``riemann_zeta``.

C++29 features
^^^^^^^^^^^^^^

The C++29 ``iota``, ``bit_repeat``, ``bit_compress``, and ``bit_expand`` SIMD features are not provided.

Type restrictions
^^^^^^^^^^^^^^^^^

``long double`` and complex types whose value type is not a vectorizable floating-point type are not supported. These restrictions match the set of vectorizable types specified by the C++ Standard.
