# AGENTS.md - Shapes Tensor Library

## Project Overview

Shapes is a from-scratch tensor computation library written in C99, similar in concept to
NumPy/PyTorch. It includes a custom arena-based memory allocator, multi-dimensional tensor
operations (creation, indexing, slicing, reshape, transpose, broadcasting, matmul, dot product,
reduction), and an autograd computation graph scaffold. OpenBLAS is vendored for BLAS operations.

## Build System

CMake (minimum 3.10), C99 standard (`-std=gnu99`). Debug builds enable AddressSanitizer.

```bash
# Configure (from project root)
cmake -S . -B build

# Build all targets
make -C build

# Or equivalently, from the build/ directory:
cmake --build .
```

### Build Targets

| Target          | Type           | Description                          |
|-----------------|----------------|--------------------------------------|
| `shapes_memory` | Static library | Arena memory allocator (`memory.c`)  |
| `shapes_core`   | Static library | Core tensor lib (depends on memory + OpenBLAS) |
| `shapes`        | Executable     | Main application (`cmd/main/main.c`) |
| `shapes_test`   | Executable     | Test suite                           |

## Testing

Custom test framework defined in `cmd/test/test.h` with macros: `ASSERT`, `ASSERT_EQ`,
`ASSERT_NEQ`, `ASSERT_NULL`, `ASSERT_NOT_NULL`, `TEST_SUMMARY`.

```bash
# Run all tests via CTest
ctest --test-dir build

# Run all tests directly
./build/shapes_test

# Run with verbose output
ctest --test-dir build --verbose
```

There is a single test executable (`shapes_test`) that runs all tests. There is no mechanism
to run a single test in isolation -- all tests run together via `run_memory_tests()` and
`run_tensor_tests()` in `cmd/test/main.c`. To test a specific area, temporarily comment out
the other `run_*` call in `cmd/test/main.c` and rebuild.

Test files:
- `cmd/test/main.c` -- test runner entry point
- `cmd/test/memory_test.c` -- arena allocator tests (18 tests)
- `cmd/test/tensor_test.c` -- tensor operation tests (60+ tests)

### Writing Tests

Each test is a `static void` function. Group related tests in a `run_*_tests()` function
declared in a corresponding `*_test.h` header. Register new test groups in `cmd/test/main.c`.

```c
static void test_my_feature(void) {
  Memory *mem = initializeMemory();
  Context ctx = {.memory = mem};
  // ... setup and operations ...
  ASSERT_EQ(actual, expected, "descriptive failure message");
}

void run_my_tests(void) {
  test_my_feature();
}
```

## Linting & Formatting

Two tools are configured: **clang-format** (style) and **clang-tidy** (static analysis).
Config files: `.clang-format` and `.clang-tidy` in the project root.

```bash
# Auto-format all source files in-place
make -C build format

# Check formatting without modifying files (returns non-zero on violations)
make -C build check-format

# Run clang-tidy static analysis
make -C build lint
```

Run `make format` before committing. All new code must pass `check-format`.

## Project Structure

```
CMakeLists.txt          # Build configuration
common.h / common.c     # Core type definitions (Tensor, Dim, Value, Context, Dtype, etc.)
memory.h / memory.c     # Arena memory allocator (1MB arena, block headers/footers)
result/result.h         # Error enum (Result type: OK, ERR_*)
tensor/
  tensor.h              # Public tensor API (all exported functions)
  tensor_internal.h     # Internal helpers (not part of public API)
  tensor.c              # Core utilities (indexing, copying, broadcasting)
  value.h               # Type-generic value macros (VALUE_SET, VALUE_GET_FROM_ARR, etc.)
  blas.h                # OpenBLAS wrapper macros (BLAS_GEMM, BLAS_DOT)
  binary_op.c           # Element-wise add, subtract, multiply, divide
  matrix_ops.c          # MatMul, Dot (via OpenBLAS)
  shape_ops.c           # Slice, Reshape, Transpose, Squeeze, UnSqueeze
  access.c              # GetAt, AssignValueAt
  reduction.c           # Sum
  creation.c            # T_Zeros, T_Int, Clone
  destruction.c         # FreeTensor
grad/
  grad.h                # Autograd header
  binary_op_backwards.c # Backward pass stubs for binary ops
visual/
  visual.h / visual.c   # Terminal UI (raw mode, box drawing)
cmd/
  main/main.c           # Application entry point
  test/                 # Test suite (see Testing section)
OpenBLAS/               # Vendored OpenBLAS (installed artifacts in build/openblas/)
```

## Code Style and Conventions

### Language Standard
- C99 (`CMAKE_C_STANDARD 99`). Do not use C11-specific features (e.g. `_Generic`, anonymous structs/unions, `_Static_assert`).

### Naming Conventions
- **Public API functions**: `shapes_` prefix with `PascalCase` -- `shapes_Add`, `shapes_Subtract`, `shapes_MatMul`, `shapes_GetAt`
- **Category prefixes**: `shapes_layer_*` (layer ops), `shapes_loss_*` (loss ops), `shapes_optimizer_*` (optimizer ops)
- **Creation functions**: `shapes_Make_*` prefix -- `shapes_Make_ZerosTensor`, `shapes_Make_FloatTensor`, `shapes_Make_RandomTensor`
- **Internal/static functions**: `camelCase` -- `binaryOp`, `isOutOfBounds`, `copyToContiguous`, `unravel_index`
- **Internal/static functions**: `camelCase` -- `binaryOp`, `isOutOfBounds`, `copyToContiguous`, `unravel_index`
- **Types (structs, enums, typedefs)**: `PascalCase` -- `Tensor`, `Dim`, `Value`, `Context`, `Memory`, `Result`, `GraphNode`
- **Enum values**: `UPPER_SNAKE_CASE` for errors (`ERR_DIM_MISMATCH`), `PascalCase` for dtypes (`F32`, `U8`), `UPPER_SNAKE_CASE` with `OP_` prefix for ops (`OP_ADD`)
- **Type aliases**: lowercase -- `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `tensor_size_t`, `dim_t`, `multiplier_t`
- **Macros**: `UPPER_SNAKE_CASE` -- `ARENA`, `BLOCK_HEADER`, `GROW_ARRAY`, `VALUE_SET`, `BLAS_GEMM`
- **Local variables**: `camelCase` -- `flatIdx`, `destCoords`, `numOfDims`, `blockToReturn`
- **Struct fields**: `camelCase` -- `numOfDims`, `isView`, `isContigous`, `blockSize`

### Header Guards
Use `#ifndef`/`#define`/`#endif` style, not `#pragma once`. Pattern: `shapes_<module>_h`.
```c
#ifndef shapes_tensor_h
#define shapes_tensor_h
// ...
#endif
```

### Includes
- Project headers use quoted includes with relative paths: `#include "common.h"`, `#include "../memory.h"`
- System headers use angle brackets: `#include <stddef.h>`, `#include <string.h>`
- Order: project headers first, then system headers (no strict sorting enforced)

### Error Handling
- Functions that can fail return `Result` (an enum defined in `result/result.h`)
- Success is `OK`. Errors are `ERR_*` prefixed constants
- Validate inputs at the top of functions with early returns
- Check for NULL pointers, dimension mismatches, out-of-bounds, dtype mismatches before proceeding
```c
Result MyOp(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }
  // ... operation ...
  return OK;
}
```

### Memory Management
- All allocations go through the custom arena allocator: `allocate(ctx->memory, size)`
- Never use `malloc`/`calloc`/`free` directly except in `memory.c` itself
- The `Context` struct carries a `Memory*` pointer -- pass `Context*` to all allocating functions
- `initializeMemory()` creates a 1MB arena; `DestroyContext()` / `freeMemory()` releases it
- Use `GROW_ARRAY` macro for dynamic array resizing
- Do not manually free individual allocations (no `freeAlloc`, `FreeTensor`, `freeIfContingousCopy`).
  The arena is freed wholesale when the context is destroyed or reset via `resetArena()`.

### Tensor Patterns
- Tensor creation functions return `Tensor*` (heap-allocated via arena): `T_Zeros`, `T_Int`
- Operation functions take output as a pointer parameter and write via `*dest = (Tensor){...}`:
  `Add(ctx, a, b, &result)`, `Reshape(ctx, source, &dest, newShape)`
- Views (Slice, Transpose, Squeeze, UnSqueeze) share the underlying `values` pointer with
  `.isView = true` and `.isContigous = false`
- Non-contiguous tensors are copied to contiguous before operations via `copyToContiguous()`
- Use designated initializers for struct literals:
  `(Dim){.dims = dims, .numOfDims = 2}`, `(Value){.dtype = U8, .as.u8 = 5}`

### Context Rules For Ops
- Every tensor op executes in the `Context *ctx` passed into that op. The op must not pick a different execution context internally.
- Materialize operands into the target context with `materializeTensorOnContext(ctx, ...)`. This applies to all real tensor operands used by the op.
- Do not create temporary fallback contexts, including stack-local CPU shims or arena-allocated synthetic contexts, to run part of an op elsewhere.
- Backend dispatch is based only on the passed-in `ctx`. If that backend cannot implement the op, return `ERR_NO_OP`.
- Do not hide CPU fallback work inside CUDA ops or CUDA fallback work inside CPU ops. The Go layer is responsible for turning `ERR_NO_OP` into a panic.
- Cross-context view creation is disallowed. If a view op such as `GetTensorAt` would need to materialize a temporary base tensor in another context, fail instead of returning a view with unclear ownership/lifetime semantics.

### Macros for Type-Generic Operations
The `value.h` header provides macros that switch on `Dtype` to handle all numeric types:
- `VALUE_SET(arr, idx, v)` -- write a Value to a typed array
- `VALUE_GET_FROM_ARR(arr, idx, v, dt)` -- read from a typed array into a Value
- `VALUE_BINOP(dest, a, b, op)` -- apply a binary operator across matching types
- `VALUE(type, data)` -- construct a Value literal
- Do NOT add new type-dispatch patterns; extend these macros instead

### Formatting
- Enforced by clang-format (`.clang-format` in project root). Run `make -C build format`.
- 2-space indentation, no tabs
- 100-column line limit
- Opening braces on same line as control structures and function signatures (K&R / Attach style)
- `do { ... } while(0)` wrapper for multi-statement macros
- Backslash-continuation for multi-line macros, aligned to column
- No trailing whitespace
