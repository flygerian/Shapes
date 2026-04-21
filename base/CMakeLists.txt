cmake_minimum_required(VERSION 3.10)
project(shapes C CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Copy compile_commands.json to root automatically
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_BINARY_DIR}/compile_commands.json
      ${CMAKE_SOURCE_DIR}/compile_commands.json
    DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
    COMMENT "Copying compile_commands.json to source root"
  )
endif()

# Default to Debug build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

add_compile_options(
  -march=native -c
  $<$<CONFIG:Debug>:-O0>
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:RelWithDebInfo>:-O2>
)
# Enable AddressSanitizer in Debug builds (can be disabled with -DENABLE_ASAN=OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" ON)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND ENABLE_ASAN)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-fsanitize=address> $<$<COMPILE_LANGUAGE:C>:-fno-omit-frame-pointer> $<$<COMPILE_LANGUAGE:C>:-g>)
  add_link_options(-fsanitize=address)
endif()
# OpenBLAS location changed from older installs; clear stale cache so configure
# always resolves against the current in-repo install path.
unset(OPENBLAS_LIB CACHE)
find_library(OPENBLAS_LIB
  NAMES openblas
  PATHS ${CMAKE_SOURCE_DIR}/build/openblas/lib
  NO_DEFAULT_PATH
)
if(NOT OPENBLAS_LIB)
  message(FATAL_ERROR "OpenBLAS not found in ${CMAKE_SOURCE_DIR}/build/openblas/lib. Run 'make init'.")
endif()

find_package(CUDAToolkit REQUIRED)

include_directories (${CMAKE_SOURCE_DIR}/build/openblas/include)

# Memory library
add_library(utils_lib
  utils_lib/memory.c
  utils_lib/array.c
  utils_lib/map.c
  utils_lib/bitset.c
)
target_include_directories(utils_lib PUBLIC ${CMAKE_SOURCE_DIR})

add_library(shapes_cuda
    cuda/access.cu
    cuda/binary_op.cu
    cuda/cast.cu
    cuda/creation.cu
    cuda/conv_bias.cu
    cuda/accumulate.cu
    cuda/cross_entropy.cu
    cuda/gemm.cu
    cuda/im2col.cu
    cuda/optimizer.cu
    cuda/pool.cu
    cuda/reduction.cu
    cuda/unary.cu
)
target_include_directories(shapes_cuda PUBLIC ${CMAKE_SOURCE_DIR}) 


# Core library
add_library(shapes_core
    context.c
    layer/dense.c
    layer/batch_norm.c
    layer/im2col.c
    layer/im2col_v2.c
    layer/conv.c
    layer/pool.c
    loss/cross_entropy.c
    optimizer/sgd.c
    optimizer/adam.c
    tensor/tensor.c
    tensor/binary_op.c
    tensor/matrix_ops.c
    tensor/shape_ops.c
    tensor/access.c
    tensor/reduction.c
    tensor/creation.c
    tensor/destruction.c
    tensor/unary.c
    tensor/cast.c
    tensor/accumulate.c
    tensor/gemm.c
    nn/backward.c
    nn/dense.c
    nn/embedding.c
    nn/mse.c
    nn/sgd.c
    nn/training.c
    nn/batchnorm.c
    nn/cross_enthropy.c
    nn/adam.c
    nn/tanh.c
    cmd/examples/makemore_3.c
    cmd/examples/makemore_5.c
)
target_include_directories(shapes_core PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(shapes_core shapes_cuda utils_lib ${OPENBLAS_LIB} CUDA::cublas CUDA::cudart stdc++ m)

# Main executable
add_executable(shapes
    cmd/main/main.c
)
target_link_libraries(shapes shapes_core)

# Enable CTest
enable_testing()

option(BUILD_TESTS "Build test executables" OFF)
if(BUILD_TESTS)
  add_executable(shapes_test
      cmd/test/main.c
      cmd/test/memory_test.c
      cmd/test/tensor_test.c
      cmd/test/unary_test.c
      cmd/test/cast_test.c
      cmd/test/sgd_test.c
      cmd/test/adam_test.c
      cmd/test/layer_test.c
  )
  target_link_libraries(shapes_test shapes_core utils_lib)
  add_test(NAME shapes_tests COMMAND shapes_test)
endif()

# ---------- Linting & Formatting ----------

# Collect all project source files (exclude vendored OpenBLAS)
file(GLOB_RECURSE ALL_SOURCES
  ${CMAKE_SOURCE_DIR}/*.c
  ${CMAKE_SOURCE_DIR}/*.h
)
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*/OpenBLAS/.*")
list(FILTER ALL_SOURCES EXCLUDE REGEX ".*/build/.*")

# make format  -- auto-format all source files in-place
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${ALL_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-format on all source files"
    VERBATIM
  )

  # make check-format -- verify formatting without modifying files (CI-friendly)
  add_custom_target(check-format
    COMMAND ${CLANG_FORMAT} --dry-run --Werror ${ALL_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking clang-format compliance"
    VERBATIM
  )
endif()

# make lint -- run clang-tidy static analysis
find_program(CLANG_TIDY clang-tidy)
if(CLANG_TIDY)
  add_custom_target(lint
    COMMAND ${CLANG_TIDY} -p ${CMAKE_BINARY_DIR} ${ALL_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-tidy on all source files"
    VERBATIM
  )
endif()
