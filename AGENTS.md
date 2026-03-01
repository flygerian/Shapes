# AGENTS.md - Shapes Multi-Language Tensor Library

## Build / Test / Lint
- First-time setup: `make init` (clones OpenBLAS submodule and builds it).
- Build everything: `make build` (C library via CMake + Go bindings via CGo).
- Run all tests: `make test` (C via `ctest --test-dir base/build`, Go via `go test -v ./...` in `go/`).
- Run C tests only: `make test-direct` (runs `base/build/shapes_test`; no single-test isolation — comment out other `run_*` calls in `base/cmd/test/main.c` to narrow scope).
- Run Go tests only: `make test-go`. Run a single Go test: `cd go && LD_LIBRARY_PATH=$PWD/../base/build/openblas/lib go test -v -run TestName ./...`.
- Format C code (required before commit): `make format`. Lint: `make lint`.

## Architecture
- `base/` — C99 core: arena allocator (`memory.{c,h}`), tensor ops (`tensor/`), autograd (`grad/`), types (`common.{c,h}`, `result/result.h`). Vendored OpenBLAS. Source of truth for all tensor behavior. See `base/AGENTS.md` for detailed C conventions.
- `go/` — Go bindings (module `github.com/flygerian/shapes`) via CGo wrapping the C core. Root package provides `Context`, `Tensor`, and `WrappedTensor`. Sub-packages: `activation/`, `layer/`, `loss/`, `optimizer/`, `extract/`, `visual/`. See `go/AGENTS.md` for detailed Go conventions.
- New tensor ops go in C first (`base/tensor/`), then get exposed via Go bindings. High-level features (layers, training) belong in Go, not C.

## Code Style
- **C**: C99 strict. PascalCase public API (`Add`, `MatMul`), camelCase internals/locals/fields, UPPER_SNAKE_CASE macros. `Result` return type for fallible ops; early-return on error. All allocations via arena (`allocate(ctx->memory, size)`) — never raw `malloc`. 2-space indent, K&R braces, 100-col limit. `#ifndef` header guards (`shapes_<module>_h`).
- **Go**: Standard Go conventions. Dual API: `*Tensor` methods (explicit `ctx`) and `*WrappedTensor` methods (fluent, embedded context). Panics on error (never returns `error`). CGo wrappers use `static inline` C functions in preamble; cross-package bridging via `unsafe.Pointer`. Layers/optimizers are closures, not structs. `LD_LIBRARY_PATH` must include OpenBLAS install dir for tests/runtime. Prefer `for i := range n` over C-style loops. Use `context.Background()` instead of `nil` when calling `shapes.New()`. See `go/AGENTS.md` for full conventions.
- Also see `CLAUDE.md` (root) and `base/CLAUDE.md` for additional context.
