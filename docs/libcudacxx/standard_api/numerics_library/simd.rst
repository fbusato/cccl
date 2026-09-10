.. _libcudacxx-standard-api-numerics-simd:

``<cuda/simd>``
===============

The ``<cuda/simd>`` header provides the C++ Standard Library data-parallel types and operations in ``cuda::std::simd``. CUDA-specific extensions in ``cuda::simd`` are documented in the :ref:`SIMD extended API <libcudacxx-extended-api-simd>`.

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
