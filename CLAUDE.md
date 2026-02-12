# CLAUDE.md

This file provides guidance to Claude Code when working with the Shapes multi-language tensor library.

## Project Architecture

Shapes is a **multi-language tensor computation library** with a high-performance C core and language-specific bindings.

### Directory Structure

```
/home/olabode/repos/Shapes/
├── base/          # C99 core implementation
├── go/            # Go bindings + high-level API
└── [future]/      # Additional language bindings
```

**Key concept**: The `base/` directory contains a complete, standalone C tensor library. Other directories contain language bindings that call into the C core via FFI (Foreign Function Interface).

### Component Roles

**base/ (C Core)**
- Performance-critical tensor operations
- Custom arena memory allocator
- Autograd system with computation graph
- OpenBLAS integration for BLAS operations
- **Authoritative implementation**: This is the source of truth for tensor behavior

**go/ (Go Bindings)**
- CGo bindings to expose C functions to Go
- Idiomatic Go API wrapping C types
- Higher-level abstractions: layers, optimizers, training utilities
- **Builds on C core**: Calls C functions for tensor ops, adds Go-specific features

**Future language bindings**
- Similar pattern: FFI bindings + language-specific high-level features

## Development Guidelines

### General Principles

1. **Core vs. Bindings**: Performance-critical operations belong in C (`base/`). Language-specific ergonomics and high-level features belong in bindings.

2. **Shared Core Philosophy**: All languages use the same C core. Don't reimplement tensor operations in each language—bind to the C version.

3. **Layered Design**:
   - Layer 1: C tensor primitives (add, multiply, matmul, etc.)
   - Layer 2: Language bindings (FFI, memory management bridges)
   - Layer 3: High-level features (layers, training loops, utilities)

4. **Documentation**: Each component has its own documentation:
   - `base/CLAUDE.md` - C library development guide
   - `base/AGENTS.md` - Comprehensive C development reference
   - This file - Overall project architecture

### Working with the C Core

When modifying `base/`:
- Follow C99 standard strictly (see `base/CLAUDE.md` for conventions)
- Use the custom arena allocator for all allocations
- Run `make -C base/build format` before committing
- Test changes with `./base/build/shapes_test`
- Consider impact on all language bindings when changing C API

**Build commands** (from `base/` directory):
```bash
cmake -S . -B build
make -C build              # Build
make -C build format       # Auto-format (REQUIRED before commit)
ctest --test-dir build     # Run tests
```

### Working with Go Bindings

When modifying `go/`:
- Use CGo to interface with C core
- Wrap C types (`Tensor*`, `Context`, etc.) in Go structs
- Manage memory carefully across Go/C boundary:
  - C arena allocator manages C memory
  - Go garbage collector manages Go memory
  - Use finalizers or explicit Free methods for hybrid objects
- Build higher-level features in pure Go when possible
- Follow Go conventions for the Go API even if C API differs

**Module**: `github.com/yourusername/shapes`

### CGo Integration Strategy

The Go library should:
1. Import C headers and link against C library
2. Create Go wrapper types that hold C pointers
3. Provide Go methods that call C functions via CGo
4. Handle type conversions (Go slices ↔ C arrays, etc.)
5. Implement higher-level Go-native features on top

Example pattern:
```go
// #include "../../base/tensor/tensor.h"
// #include "../../base/common.h"
import "C"

type Tensor struct {
    cTensor *C.Tensor
    ctx     *Context
}

func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
    // Call C function via CGo
    var result C.Tensor
    resultCode := C.Add(t.ctx.cCtx, t.cTensor, other.cTensor, &result)
    // Handle errors, wrap result
}
```

## When Working on Different Components

### Adding a new tensor operation

1. **Implement in C first** (`base/tensor/`)
   - Write the C function following existing patterns
   - Add tests in `base/cmd/test/`
   - Document in `base/tensor/tensor.h`

2. **Expose to Go** (`go/`)
   - Create CGo binding
   - Wrap in idiomatic Go API
   - Add Go tests

3. **Update documentation**
   - Update relevant CLAUDE.md sections
   - Add examples to README if significant

### Adding a high-level feature (e.g., neural network layer)

1. **Decide on location**: If it can be built purely using existing tensor ops, implement in the language binding (e.g., Go)

2. **Implement in binding language**:
   - Use existing tensor operations as primitives
   - Follow language conventions
   - Add comprehensive tests

3. **Don't add to C core** unless it requires new low-level primitives

## Current State

- ✅ C core: Fully functional with tensor ops, autograd, memory allocator
- ✅ C build system: CMake with format/lint/test targets
- ✅ Go module: Initialized, ready for implementation
- 🚧 Go bindings: Not yet implemented
- 🚧 Go high-level API: Not yet implemented

## Quick Reference

### File Locations
- C library: `base/`
- C conventions: `base/CLAUDE.md`, `base/AGENTS.md`
- Go module: `go/`
- Project README: `README.md` (root)

### Common Tasks
- Format C code: `make -C base/build format`
- Test C code: `./base/build/shapes_test`
- Build Go code: `cd go && go build ./...`
- Test Go code: `cd go && go test ./...`

## Important Notes for Claude

1. **Don't duplicate tensor operations**: If a tensor operation exists in C, bind to it rather than reimplementing in Go.

2. **Respect the architecture**: C for performance-critical code, Go for ergonomics and high-level features.

3. **Memory management is critical**: The C library uses a custom arena allocator. Go code must carefully manage the lifecycle of C objects.

4. **Keep C API stable**: Changes to the C API affect all language bindings. Consider compatibility.

5. **Each language has its own conventions**: C code follows C99 + custom patterns (see `base/CLAUDE.md`). Go code should follow standard Go conventions.
