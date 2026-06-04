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

# add_subdirectory("deps/OpenBLAS")

include(ExternalProject)
set(OPENBLAS_INSTALL_DIR ${CMAKE_BINARY_DIR}/openblas-install)

# Create it so CMake stops complaining at configure time
file(MAKE_DIRECTORY ${OPENBLAS_INSTALL_DIR}/include)

add_library(openblas STATIC IMPORTED GLOBAL)
add_dependencies(openblas openblas_ext)
set_target_properties(openblas PROPERTIES
    IMPORTED_LOCATION ${OPENBLAS_INSTALL_DIR}/lib/libopenblas.a
    INTERFACE_INCLUDE_DIRECTORIES ${OPENBLAS_INSTALL_DIR}/include
)
ExternalProject_Add(
    openblas_ext
    SOURCE_DIR  ${CMAKE_SOURCE_DIR}/deps/OpenBLAS
    BINARY_DIR  ${CMAKE_BINARY_DIR}/openblas-build
    INSTALL_DIR ${CMAKE_BINARY_DIR}/openblas-install
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_TESTING=OFF
    BUILD_BYPRODUCTS
        ${CMAKE_BINARY_DIR}/openblas-install/lib/libopenblas.a
)

# Wrap it in an imported target your other targets can link against
add_dependencies(openblas openblas_ext)
set_target_properties(openblas PROPERTIES
    IMPORTED_LOCATION ${OPENBLAS_INSTALL_DIR}/lib/libopenblas.a
    INTERFACE_INCLUDE_DIRECTORIES ${OPENBLAS_INSTALL_DIR}/include
)
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
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND ENABLE_ASAN AND NOT SHAPES_ENABLE_CUDA)
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
      shapes/shapescuda/access.cu
      shapes/shapescuda/binary_op.cu
      shapes/shapescuda/cast.cu
      shapes/shapescuda/creation.cu
      shapes/shapescuda/conv_bias.cu
      shapes/shapescuda/accumulate.cu
      shapes/shapescuda/cross_entropy.cu
      shapes/shapescuda/gemm.cu
      shapes/shapescuda/im2col.cu
      shapes/shapescuda/optimizer.cu
      shapes/shapescuda/pool.cu
      shapes/shapescuda/reduction.cu
      shapes/shapescuda/unary.cu
      shapes/shapescuda/cuda_memory.c
  )
  target_include_directories(shapescuda PUBLIC ${CMAKE_SOURCE_DIR}/olib)
  target_include_directories(shapescuda PUBLIC ${CMAKE_SOURCE_DIR}/shapes)
  target_link_libraries(shapescuda olib CUDA::cublas CUDA::cudart stdc++ m)
endif()

# Core library
add_library(shapes
    shapes/base/context.c
    shapes/base/device.c
    shapes/base/dense.c
    shapes/base/batch_norm.c
    shapes/base/im2col_v2.c
    shapes/base/conv.c
    shapes/base/pool.c
    shapes/base/cross_entropy.c
    shapes/base/optimizer/sgd.c
    shapes/base/optimizer/adam.c
    shapes/base/internal.c
    shapes/base/binary_op.c
    shapes/base/shape_ops.c
    shapes/base/access.c
    shapes/base/reduction.c
    shapes/base/creation.c
    shapes/base/unary.c
    shapes/base/cast.c
    shapes/base/accumulate.c
    shapes/base/gemm.c
)

if(NOT SHAPES_ENABLE_CUDA)
  target_sources(shapes PRIVATE shapes/base/cuda_stubs.c)
endif()

target_include_directories(shapes PUBLIC ${CMAKE_SOURCE_DIR}/olib)
target_include_directories(shapes PUBLIC ${CMAKE_SOURCE_DIR}/shapes)
target_include_directories(shapes PUBLIC ${CMAKE_SOURCE_DIR}/shapes/base)
target_include_directories(shapes PUBLIC ${CMAKE_SOURCE_DIR}/shapes/shapescuda)


add_library(shapesnn
    shapes/shapesnn/backward.c
    shapes/shapesnn/dense.c
    shapes/shapesnn/embedding.c
    shapes/shapesnn/mse.c
    shapes/shapesnn/sgd.c
    shapes/shapesnn/training.c
    shapes/shapesnn/batchnorm.c
    shapes/shapesnn/cross_enthropy.c
    shapes/shapesnn/adam.c
    shapes/shapesnn/tanh.c
    shapes/shapesnn/relu.c
    shapes/shapesnn/maxpool2d.c
    shapes/shapesnn/adaptive_avg_pool2d.c
    shapes/shapesnn/conv2d.c
    shapes/shapesnn/sequential.c
    shapes/shapesnn/utils.c
    shapes/shapesnn/flatten.c
    shapes/shapesnn/softmax.c
    shapes/shapesnn/persistence.c
    shapes/shapesnn/safetensors.c
)

target_include_directories(shapesnn PUBLIC ${CMAKE_SOURCE_DIR}/shapes)

if(SHAPES_ENABLE_CUDA)
  target_link_libraries(shapes PRIVATE shapescuda olib openblas CUDA::cublas CUDA::cudart stdc++ m)
  target_link_libraries(shapesnn PRIVATE shapes olib CUDA::cublas CUDA::cudart stdc++ m)
elseif (APPLE)
  target_link_libraries(shapes olib openblas stdc++ m "-framework Accelerate")
  target_link_libraries(shapesnn shapes olib stdc++ m "-framework Accelerate")
else() 
  target_link_libraries(shapes olib openblas stdc++)
  target_link_libraries(shapesnn shapes olib)
endif()

if(APPLE)
  target_compile_definitions(shapes PRIVATE ACCELERATE_NEW_LAPACK)
endif()
# Main executable
add_executable(makemore
    examples/makemore.c
)

add_executable(vgg10
    examples/vgg-10.c
)

if(SHAPES_ENABLE_CUDA)
  target_link_libraries(makemore shapes shapesnn openblas CUDA::cublas CUDA::cudart stdc++ m)
  target_link_libraries(vgg10 shapes shapesnn openblas CUDA::cublas CUDA::cudart stdc++ m)
else()
  target_link_libraries(makemore shapes shapesnn openblas m)
  target_link_libraries(vgg10 shapes shapesnn openblas m)
endif()

target_include_directories(makemore PUBLIC ${CMAKE_SOURCE_DIR}/shapes/shapesnn)
target_include_directories(makemore PUBLIC ${CMAKE_SOURCE_DIR}/deps)
target_include_directories(vgg10 PUBLIC ${CMAKE_SOURCE_DIR}/shapes/shapesnn)
target_include_directories(vgg10 PUBLIC ${CMAKE_SOURCE_DIR}/deps)

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
