cmake_minimum_required(VERSION 3.10)
project(shapes C)

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

# Enable AddressSanitizer in Debug builds (can be disabled with -DENABLE_ASAN=OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" ON)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
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

include_directories (${CMAKE_SOURCE_DIR}/build/openblas/include)

# Memory library
add_library(shapes_memory
    memory.c
)
target_include_directories(shapes_memory PUBLIC ${CMAKE_SOURCE_DIR})

# Core library
add_library(shapes_core
    common.c
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
)
target_include_directories(shapes_core PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(shapes_core shapes_memory ${OPENBLAS_LIB} m)

# Main executable
add_executable(shapes
    cmd/main/main.c
)
target_link_libraries(shapes shapes_core)

# Test executable
add_executable(shapes_test
    cmd/test/main.c
    cmd/test/memory_test.c
    cmd/test/tensor_test.c
    cmd/test/unary_test.c
    cmd/test/cast_test.c
)
target_link_libraries(shapes_test shapes_core)

# Enable CTest
enable_testing()
add_test(NAME shapes_tests COMMAND shapes_test)

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
