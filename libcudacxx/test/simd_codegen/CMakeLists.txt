add_custom_target(libcudacxx.test.simd.ptx)

find_program(filecheck "FileCheck")

if (filecheck)
  message("-- ${filecheck} found... building simd codegen tests")
else()
  return()
endif()

find_program(cuobjdump "cuobjdump" REQUIRED)
find_program(bash "bash" REQUIRED)

set(dump_and_check_script "${CMAKE_CURRENT_SOURCE_DIR}/../atomic_codegen/dump_and_check.bash")

# Files with both SM8X (8-byte) and SM10X (16-byte) checks
set(simd_codegen_common_tests
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i8.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i16.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i32.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_f16.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_bf16.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_f32.cu"
)

# Files with SM10X-only checks (N=1 skipped for 8-byte tier)
set(simd_codegen_sm100_only_tests
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_i64.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/load_store_f64.cu"
)

# simd_codegen_add_tests(arch prefix test_list)
#   For each .cu in test_list, build a static library at the given arch
#   and POST_BUILD run dump_and_check.bash with the given FileCheck prefix.
function(simd_codegen_add_tests arch prefix test_list)
  foreach (test_path IN LISTS ${test_list})
    cmake_path(GET test_path FILENAME test_file)
    cmake_path(REMOVE_EXTENSION test_file LAST_ONLY OUTPUT_VARIABLE test_name)

    set(target_name "simd_codegen_sm${arch}_${test_name}")

    add_library(${target_name} STATIC "${test_path}")

    set_target_properties(
      ${target_name}
      PROPERTIES
        CUDA_ARCHITECTURES "${arch}"
    )

    # Clang stopped emitting PTX in clang20. Add flags to re-enable it.
    if (
      CMAKE_CUDA_COMPILER_ID STREQUAL Clang
      AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 20
    )
      target_compile_options(${target_name} PRIVATE "--cuda-include-ptx=sm_${arch}")
    endif()

    target_compile_options(${target_name} PRIVATE "-Wno-comment")

    target_include_directories(
      ${target_name}
      PRIVATE "${libcudacxx_SOURCE_DIR}/include"
    )
    add_dependencies(libcudacxx.test.simd.ptx ${target_name})

    add_custom_command(
      TARGET libcudacxx.test.simd.ptx
      POST_BUILD
      # gersemi: off
      COMMAND
        "${dump_and_check_script}"
          $<TARGET_FILE:${target_name}>
          "${test_path}"
          ${prefix}
      # gersemi: on
    )
  endforeach()
endfunction()

if (NOT "NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # SM80: 8-byte vectorized (always built)
  simd_codegen_add_tests(80 SM8X simd_codegen_common_tests)

  # SM100: 16-byte vectorized (CTK >= 12.9)
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9)
    set(simd_codegen_sm100_tests ${simd_codegen_common_tests} ${simd_codegen_sm100_only_tests})
    simd_codegen_add_tests(100 SM10X simd_codegen_sm100_tests)
  endif()
endif()
