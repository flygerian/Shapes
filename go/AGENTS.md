# AGENTS.md - Shapes Go Bindings

## Overview

Go bindings for the Shapes tensor library. The root package (`shapes`) provides the primary API
via CGo wrapping the C core. Sub-packages build higher-level features (layers, optimizers, loss)
using only the root package's Go API -- they never call C directly.

Module: `github.com/flygerian/shapes`

## Build & Test

```bash
# From repo root:
make test-go

# Single test:
cd go && LD_LIBRARY_PATH=$PWD/../base/OpenBLAS/install/lib go test -v -run TestName ./...

# All Go tests (sub-packages included):
cd go && LD_LIBRARY_PATH=$PWD/../base/OpenBLAS/install/lib go test -v ./...
```

`LD_LIBRARY_PATH` must include the OpenBLAS install dir for both tests and runtime.

## Package Structure

```
go/                          # Root package: "shapes"
  context.go                 # Context type (wraps C Context + Go context.Context)
  error.go                   # C Result -> Go error/string mapping
  tensor.go                  # Tensor type, Shape/Range aliases, Dtype enum
  wrapped_tensor.go          # WrappedTensor: fluent API without explicit context
  creation.go                # Tensor creation: Zeros, Int, Float, FromFloat32, FromInt8, ...
  creation_utils.go          # Generic flatten/validate helpers for nested slices
  access.go                  # Get (coord & tensor indexing), Item
  binary_ops.go              # Plus, Minus, Times, Divide, AddInPlace
  unary.go                   # Pow, Exp, Negate
  reduction.go               # Sum
  matrix_ops.go              # Mul (MatMul), Dot
  shape_ops.go               # Slice, Reshape, Transpose, Squeeze, UnSqueeze, ...
  grad.go                    # Backward, ComputationGraphNode, graph building
  grad_binary.go             # Backward implementations for binary/unary/reduction ops
  grad_shape.go              # Backward implementations for shape ops

  activation/                # Tanh, etc. (pure Go, no C calls)
  layer/                     # Dense layer (pure Go closure)
  loss/                      # MSE loss (pure Go)
  optimizer/                 # SGD optimizer (pure Go closure), ZeroGrad
  extract/                   # Extract trainable parameters from computation graph
  visual/                    # Terminal visualization of tensors and graphs
  examples/                  # Training examples
  cmd/main/                  # Entry point binary
```

### Dependency Direction

- **Root package** depends only on C (via CGo). No Go package dependencies.
- **Sub-packages** (`activation/`, `layer/`, `loss/`, `optimizer/`, `extract/`, `visual/`)
  depend on the root `shapes` package only. They compose operations through Go, never calling
  C directly.
- **`optimizer/`** additionally depends on `extract/`.

## Dual API: Tensor vs WrappedTensor

Every operation exists in two forms:

1. **`*Tensor` methods** -- low-level, require explicit `ctx *Context`:
   ```go
   func (t *Tensor) Plus(ctx *Context, other *Tensor) *Tensor
   ```

2. **`*WrappedTensor` methods** -- high-level fluent API, context is embedded:
   ```go
   func (wt *WrappedTensor) Plus(other *WrappedTensor) *WrappedTensor
   ```

WrappedTensor methods always delegate to Tensor methods with this template:
```go
func (wt *WrappedTensor) OpName(args...) *WrappedTensor {
    wt.validateSameContext(other)  // for binary ops
    return wt.context.Wrap(wt.tensor.OpName(wt.context, args...))
}
```

Creation functions also mirror this: free functions return `*Tensor`, Context methods return
`*WrappedTensor`:
```go
func Zeros(ctx *Context, shape Shape) *Tensor           // free function
func (c *Context) Zeros(shape Shape) *WrappedTensor      // Context method
```

When adding a new operation, implement both forms.

## CGo Bridging Conventions

### Static inline wrapper functions

Every file that calls C defines **static inline wrappers** in the CGo preamble. These wrappers
allocate the output on the arena and return via an out-parameter:

```c
static inline Result wrap_Add(Context *ctx, Tensor *a, Tensor *b, Tensor **out) {
    Tensor *dest = allocate(ctx->memory, sizeof(Tensor));
    Result r = Add(ctx, a, b, dest);
    *out = dest;
    return r;
}
```

### CGo flags in every file

Every file with CGo repeats the full flag set:
```go
/*
#cgo CFLAGS: -I../base
#cgo LDFLAGS: -L../base/build -L../base/OpenBLAS/install/lib -lshapes_core -lshapes_memory -lopenblas -lm
*/
```

### Cross-package type bridging via unsafe.Pointer

Since each Go package has its own `C` namespace, C types are bridged through `unsafe.Pointer`:
```go
(*C.Context)(ctx.UnsafePtr())
(*C.Memory)(ctx.UnsafeMemory())
```

### Fixed-arity wrappers for C variadic functions

CGo cannot call C variadic functions directly. Generate fixed-arity wrappers (1-8 dimensions)
and dispatch via `switch ndims` in Go (see `shape_ops.go`).

## Error Handling: Panic, Not Error Returns

**The public API never returns `error`.** C result codes are converted to panics:

```go
result := C.wrap_Add((*C.Context)(ctx.UnsafePtr()), t.cTensor, other.cTensor, &dest)
if result != C.OK {
    panic("shapes: " + resultString(uint32(result)))
}
```

Rationale: failures are programming errors (dimension mismatches, null tensors, out-of-bounds),
not recoverable runtime conditions. All panic messages use the `"shapes: "` prefix.

Data validation in creation functions also panics directly:
```go
panic("shapes: data length does not match shape")
```

There is no custom error type. The `resultString()` helper maps C `Result` enum values to
human-readable strings.

## Naming Conventions

### Go methods vs C functions

Binary ops are renamed to operator-like names. Creation functions strip the `T_` prefix:

| C Function | Go Method | Notes |
|---|---|---|
| `Add` | `Plus` | Renamed |
| `Subtract` | `Minus` | Renamed |
| `Multiply` | `Times` | Renamed |
| `Divide` | `Divide` | Same |
| `MatMul` | `Mul` | Shortened |
| `T_Zeros` | `Zeros` | Prefix stripped |
| `T_Float` | `Float` | Prefix stripped |

All other ops keep their C names (`Pow`, `Exp`, `Negate`, `Sum`, `Reshape`, `Transpose`, etc.).

### Type aliases

- `Shape` = `[]uint32`
- `Range` = `[]uint32`
- `Dtype` = `int` enum with `Dtype` prefix: `DtypeF32`, `DtypeI8`, etc.
- `OpType` = `int` enum with `Op` prefix: `OpAdd`, `OpMultiply`, `OpDense`, etc.

### Internal helpers

Unexported functions use camelCase: `dim()`, `track()`, `resultString()`, `ptrOffset()`,
`shapeOf()`, `leafNode()`, `attachNode()`, `markIntermediate()`, `topo()`, `buildGraph()`.

### Import aliases

- Root package imported as `shapes` in sub-packages:
  ```go
  import shapes "github.com/flygerian/shapes"
  ```
- Standard `context` aliased as `stdctx` in `context.go` to avoid collision:
  ```go
  import stdctx "context"
  ```

## Memory Management

### Arena-based: all allocation in preamble C, never in Go

All C memory is allocated through the arena, but Go code never calls the arena directly. Arena
calls (`allocate(ctx->memory, sizeof(Tensor))`) belong exclusively in the `static inline`
preamble C wrapper functions. Go code only sees the resulting pointers returned by those wrappers.

### Context owns all memory

`Context.Close()` frees the entire arena. All Go-side pointers to C tensors are nilled out to
prevent use-after-free. **Always `defer ctx.Close()`.**

### Tensor tracking

Every tensor creation calls `track()` which registers the C pointer with the Context:
```go
func track(ctx *Context, t *Tensor) *Tensor {
    ctx.Track((*unsafe.Pointer)(unsafe.Pointer(&t.cTensor)))
    return t
}
```

### Intermediate tensors in backward pass

Temporary tensors created during backward are marked with `markIntermediate()` and freed in
bulk via `FreeIntermediates()` after backward completes.

## Context Derivation

Create derived contexts that share the same C context/memory but with different flags:

- `ctx.NoGrad()` -- disable gradient tracking (for allocating grad tensors)
- `ctx.Fused()` -- prevent internal ops from building separate backward passes (used in layers)
- `ctx.NoGraph()` -- disable computation graph building (used in backward/optimizer steps)

Derived contexts pass `validateSameContext` checks because they share the same underlying C pointer.

## Functional Options

`Context` creation uses the functional options pattern:
```go
ctx := shapes.New(context.Background(), shapes.WithGrad(true))
```

Always pass `context.Background()` instead of `nil`.

## Backward Pass Conventions

Every backward function follows this signature and pattern:
```go
func opBackward(ctx *Context, node *ComputationGraphNode) {
    // 1. Extract inputs from node.Inputs
    // 2. Compute local gradients
    // 3. Accumulate: input.Computation.Grad = input.Grad().Plus(ctx, gradX)
    // 4. Mark temporaries: markIntermediate(ctx, ...)
}
```

Gradients are always **accumulated** (added), never replaced. Op-specific data (power exponent,
dim index, transpose axes) is stored in `node.Computation.Metadata` as a typed value and
retrieved via type assertion.

## Layers and Optimizers: Closure Pattern

Layers and optimizers return function closures, not structs:

```go
// Layer: returns func(input) -> output
func Dense(inputSize int, outputSize int) func(*WrappedTensor) *WrappedTensor

// Optimizer: returns func(graph) -> void
func SGD(ctx *Context, lr float32) func(*ComputationGraph)
```

Parameters are lazily initialized on first call (captured in the closure).

## Test Conventions

### Setup/teardown

Every test creates and defers closing a context:
```go
func TestXxx(t *testing.T) {
    ctx := shapes.New(context.Background())
    defer ctx.Close()
    // ...
}
```

### Assertions

No assertion library. Use standard `t.Errorf` / `t.Fatalf` directly. For float comparison,
use the `approxEq` helper (copy-pasted per package since it's unexported):
```go
func approxEq(a, b, tol float32) bool {
    return float32(math.Abs(float64(a-b))) < tol
}
```

### Panic testing

```go
defer func() {
    if r := recover(); r == nil {
        t.Fatal("expected panic for ..., got nil")
    }
}()
```

### Iteration

Use `for i := range uint32(n)` (Go 1.22+), not C-style `for i := 0; i < n; i++`.

### Subtests and table-driven tests

Use `t.Run` for grouping. Use table-driven tests for parameterized cases (see `grad_test.go`).

## Adding a New Tensor Operation (Checklist)

1. Implement in C first (`base/tensor/`)
2. Add a `static inline wrap_OpName(...)` C wrapper in the CGo preamble of the appropriate Go file
3. Add a `*Tensor` method (takes `ctx *Context`, panics on error, calls `track()`)
4. Add a `*WrappedTensor` method (delegates to Tensor method, calls `validateSameContext` for binary ops)
5. Add a creation shortcut on `*Context` if it's a creation function
6. If the op needs backward: add an `OpType` constant, implement `opBackward()`, register in the backward dispatch, store any metadata
7. Add tests covering both Tensor and WrappedTensor forms
