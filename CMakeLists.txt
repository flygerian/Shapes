cmake_minimum_required(VERSION 3.10)
project(shapes C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Default to Debug build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Enable AddressSanitizer in Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
endif()

find_library(OPENBLAS_LIB
  NAMES openblas
  PATHS ${CMAKE_SOURCE_DIR}/OpenBLAS/install/lib
  NO_DEFAULT_PATH
)

include_directories (${CMAKE_SOURCE_DIR}/OpenBLAS/install/include)

# Memory library
add_library(shapes_memory
    memory.c
)
target_include_directories(shapes_memory PUBLIC ${CMAKE_SOURCE_DIR})

# Core library
add_library(shapes_core
    tensor/tensor.c
)
target_include_directories(shapes_core PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(shapes_core shapes_memory)

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
)
target_link_libraries(shapes_test shapes_core)

# Enable CTest
enable_testing()
add_test(NAME shapes_tests COMMAND shapes_test)
