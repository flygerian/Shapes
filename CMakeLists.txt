cmake_minimum_required(VERSION 3.10)

option(SHAPES_ENABLE_CUDA "Enable CUDA support" OFF)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(SHAPES_ENABLE_CUDA)
  # NVCC 13.2 is incompatible with GCC 16.1.1; use GCC 15 if available.
  if(EXISTS /usr/bin/g++-15)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-15)
  endif()

  project(shapes C CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 86)
else()
  project(shapes C)
endif()

add_subdirectory("deps/raylib")
add_subdirectory("deps/OpenBLAS")
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
  -march=native
  $<$<CONFIG:Debug>:-O0>
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-ffast-math>
  $<$<CONFIG:RelWithDebInfo>:-O2>
)
# Enable AddressSanitizer in Debug builds (can be disabled with -DENABLE_ASAN=OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" ON)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND ENABLE_ASAN)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-fsanitize=address> $<$<COMPILE_LANGUAGE:C>:-fno-omit-frame-pointer> $<$<COMPILE_LANGUAGE:C>:-g>)
  add_link_options(-fsanitize=address)
endif()

add_library(olib
  olib/memory.c
  olib/array.c
  olib/map.c
  olib/bitset.c
  olib/file.c
)

if(SHAPES_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_compile_definitions(SHAPES_HAS_CUDA)

  add_library(shapescuda
      shapescuda/access.cu
      shapescuda/binary_op.cu
      shapescuda/cast.cu
      shapescuda/creation.cu
      shapescuda/conv_bias.cu
      shapescuda/accumulate.cu
      shapescuda/cross_entropy.cu
      shapescuda/gemm.cu
      shapescuda/im2col.cu
      shapescuda/optimizer.cu
      shapescuda/pool.cu
      shapescuda/reduction.cu
      shapescuda/unary.cu
      shapescuda/cuda_memory.c
  )
  target_include_directories(shapescuda PUBLIC ${CMAKE_SOURCE_DIR}/olib)
  target_include_directories(shapescuda PUBLIC ${CMAKE_SOURCE_DIR}/shapes)
  target_link_libraries(shapescuda olib CUDA::cublas CUDA::cudart stdc++ m)
endif()

# Core library
add_library(shapes
    shapes/context.c
    shapes/device.c
    shapes/dense.c
    shapes/batch_norm.c
    shapes/im2col_v2.c
    shapes/conv.c
    shapes/pool.c
    shapes/cross_entropy.c
    shapes/optimizer/sgd.c
    shapes/optimizer/adam.c
    shapes/internal.c
    shapes/binary_op.c
    shapes/shape_ops.c
    shapes/access.c
    shapes/reduction.c
    shapes/creation.c
    shapes/unary.c
    shapes/cast.c
    shapes/accumulate.c
    shapes/gemm.c
    shapes/cuda_stubs.c
)

target_include_directories(shapes PUBLIC ${CMAKE_SOURCE_DIR}/olib)
target_include_directories(shapes PUBLIC ${CMAKE_SOURCE_DIR}/shapescuda)

if(SHAPES_ENABLE_CUDA)
  target_include_directories(shapes PUBLIC ${CMAKE_SOURCE_DIR}/shapescuda)
  target_link_libraries(shapes shapescuda raylib olib openblas CUDA::cublas CUDA::cudart stdc++ m)
elseif (APPLE)
  target_link_libraries(shapes raylib olib openblas stdc++ m "-framework Accelerate")
else() 
  target_link_libraries(shapes raylib olib openblas stdc++)
endif()

if(APPLE)
  target_compile_definitions(shapes PRIVATE ACCELERATE_NEW_LAPACK)
endif()

add_library(shapesnn
    shapesnn/backward.c
    shapesnn/dense.c
    shapesnn/embedding.c
    shapesnn/mse.c
    shapesnn/sgd.c
    shapesnn/training.c
    shapesnn/batchnorm.c
    shapesnn/cross_enthropy.c
    shapesnn/adam.c
    shapesnn/tanh.c
    shapesnn/relu.c
    shapesnn/maxpool2d.c
    shapesnn/adaptive_avg_pool2d.c
    shapesnn/conv2d.c
    shapesnn/sequential.c
    shapesnn/utils.c
    shapesnn/flatten.c
    shapesnn/softmax.c
    shapesnn/persistence.c
    shapesnn/safetensors.c
)

target_link_libraries(shapesnn shapes)
target_include_directories(shapes PUBLIC ${CMAKE_SOURCE_DIR}/shapes)

# Main executable
add_executable(makemore
    examples/makemore.c
)

add_executable(vgg10
    examples/vgg-10.c
)

target_link_libraries(makemore shapes shapesnn openblas)
target_include_directories(makemore PUBLIC ${CMAKE_SOURCE_DIR}/shapesnn)
target_include_directories(makemore PUBLIC ${CMAKE_SOURCE_DIR}/deps)

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
  target_link_libraries(shapes_test shapes_core olib)
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
