cmake_minimum_required(VERSION 3.10)
project(shapes C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Core library
add_library(shapes_core
    memory.c
    tensor/tensor.c
)
target_include_directories(shapes_core PUBLIC ${CMAKE_SOURCE_DIR})

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
add_test(NAME memory_tests COMMAND shapes_test)
