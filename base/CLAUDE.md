# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shapes is a from-scratch tensor computation library written in C99, implementing multi-dimensional tensor operations with automatic differentiation (autograd). Think NumPy/PyTorch but in pure C with a custom arena-based memory allocator and OpenBLAS for BLAS operations.

## Quick Reference

### Build Commands
```bash
# Initial setup
cmake -S . -B build

# Build all targets
make -C build

# Build and run tests
ctest --test-dir build

# Run tests directly
./build/shapes_test

# Auto-format code (REQUIRED before commits)
make -C build format

# Check formatting (CI-friendly, non-modifying)
make -C build check-format

# Static analysis
make -C build lint

# Run main executable
./build/shapes
```

### Testing
- Single test executable (`shapes_test`) runs all tests together
- No mechanism to run individual tests in isolation
- To test specific area: comment out other `run_*_tests()` calls in `cmd/test/main.c` and rebuild
- Test framework uses custom macros: `ASSERT`, `ASSERT_EQ`, `ASSERT_NEQ`, `ASSERT_NULL`, `ASSERT_NOT_NULL`, `TEST_SUMMARY`
- Each test is a static void function grouped in `run_*_tests()` functions

## Architecture

### Core Concepts

**Memory Management**: Custom arena allocator (`memory.h/c`) with 1MB arena. ALL allocations go through `allocate(ctx->memory, size)` — never use malloc/free directly. Context struct carries `Memory*` pointer everywhere.

**Computation Graph**: Tensors can be part of a computation graph for autograd. Each tensor optionally has a `GraphNode* computation` that tracks:
- Output tensor
- Gradient tensor
- Input tensors (array)
- Backward pass function pointer
- Operation type (OP_ADD, OP_MULTIPLY, etc.)
- Operation metadata (e.g., power value for Pow)

**Context**: The `Context` struct is the central state container passed to all operations:
```c
typedef struct Context {
  Memory *memory;      // Arena allocator
  bool grad;           // Enable gradient tracking
  ScreenConfig *screenConfig;  // Terminal visualization config
} Context;
```

**Result Pattern**: Functions that can fail return `Result` enum (OK or ERR_*). Always validate inputs at function top with early returns.

### Module Breakdown

**tensor/**: Core tensor operations
- `binary_op.c`: Element-wise Add, Subtract, Multiply, Divide with broadcasting
- `matrix_ops.c`: MatMul, Dot (wraps OpenBLAS)
- `shape_ops.c`: Slice, Reshape, Transpose, Squeeze, UnSqueeze
- `reduction.c`: Sum operation
- `creation.c`: T_Zeros, T_Int, T_Float, Clone
- `access.c`: GetAt, AssignValueAt for element access
- `value.h`: Type-generic macros (VALUE_SET, VALUE_GET_FROM_ARR, VALUE_BINOP)
- `blas.h`: OpenBLAS wrapper macros

**grad/**: Autograd system
- `backward.c`: Main backward pass orchestration, builds topological sort of computation graph
- `binary_op_backwards.c`: Backward implementations for Add, Subtract, Multiply, Divide
- `activation_backward.c`: Backward for Tanh and other activations
- Construction functions: `ConstructBinopBackwardpass`, `ConstructTanhBackwardpass`, etc.

**visual/**: Terminal visualization for computation graphs (uses raw terminal mode)

**common.h**: Core type definitions (Tensor, Dim, Value, GraphNode, Context, Dtype)

### Key Patterns

**Tensor Views vs Copies**:
- Shape operations (Slice, Transpose, Squeeze, UnSqueeze) create views: `.isView = true`, `.isContigous = false`
- Views share underlying `values` pointer
- Non-contiguous tensors are copied to contiguous before operations via internal `copyToContiguous()`

**Tensor Creation vs Operations**:
- Creation functions return `Tensor*` (heap-allocated): `T_Zeros`, `T_Int`, `T_Float`
- Operations take output as pointer parameter: `Add(ctx, a, b, &result)`, `Reshape(ctx, src, &dest, shape)`
- Operations write via assignment: `*dest = (Tensor){...}`

**Broadcasting**: Binary ops support NumPy-style broadcasting. Shape compatibility checked before operation.

**Type-Generic Operations**: Use macros from `value.h` for type-generic operations. Do NOT add new type-dispatch patterns — extend existing macros instead.

**Autograd Flow**:
1. Operations check `ctx->grad` flag
2. If enabled, construct GraphNode with backward function
3. `InitComputationGraph` builds topological order from output tensor
4. `Backward` traverses graph in reverse order, calling backward functions
5. Gradients accumulate in `node->grad` tensors

## Code Conventions

### Language
C99 standard. No C11 features (no `_Generic`, anonymous structs/unions, `_Static_assert`).

### Naming
- Public API: `PascalCase` (Add, MatMul, T_Zeros, FreeTensor)
- Internal/static: `camelCase` (binaryOp, unravel_index, copyToContiguous)
- Types: `PascalCase` (Tensor, Context, GraphNode)
- Type aliases: lowercase (u8, u32, i32, f32, tensor_size_t, dim_t)
- Macros: `UPPER_SNAKE_CASE` (ARENA, GROW_ARRAY, VALUE_SET, BLAS_GEMM)
- Struct fields: `camelCase` (numOfDims, isView, isContigous)

### Header Guards
Use `#ifndef`/`#define`/`#endif` style with pattern `shapes_<module>_h`, not `#pragma once`.

### Includes
- Project headers: quoted with relative paths (`#include "common.h"`, `#include "../memory.h"`)
- System headers: angle brackets (`#include <stddef.h>`)

### Formatting
Enforced by clang-format. Run `make -C build format` before all commits. Key points:
- 2-space indentation, no tabs
- 100-column line limit
- K&R brace style (opening brace same line)
- Use designated initializers: `(Dim){.dims = dims, .numOfDims = 2}`

## Important Files

- `AGENTS.md`: Comprehensive development guide with full conventions and patterns
- `.clang-format`, `.clang-tidy`: Tooling configuration
- `CMakeLists.txt`: Build configuration, includes format/lint/check-format targets
- `common.h`: Central type definitions, must read for understanding core types
- `shapes.h`: Complete public C API
- `grad/grad.h`: Autograd system API
