# Shapes

A high-performance tensor computation library with automatic differentiation, built from scratch in C with multi-language bindings.

## Architecture

Shapes is designed as a multi-language project with a fast C core and language-specific bindings that provide idiomatic interfaces and higher-level abstractions.

```
Shapes/
├── base/          # Core C implementation
│   ├── tensor/    # Low-level tensor operations
│   ├── grad/      # Autograd system
│   ├── memory.c   # Custom arena allocator
│   └── ...
├── go/            # Go bindings + high-level API
│   └── ...
└── [future]/      # Python, Rust, or other language bindings
```

### Core Components

**Base (C99)**
- Custom arena-based memory allocator
- Multi-dimensional tensor operations (creation, indexing, slicing, reshape, transpose, broadcasting)
- BLAS operations via OpenBLAS (MatMul, Dot)
- Automatic differentiation (autograd) with computation graph
- Element-wise operations with NumPy-style broadcasting

**Go Bindings**
- CGo-based bindings to the C tensor library
- Idiomatic Go API for tensor operations
- Higher-level abstractions (layers, optimizers, etc.)
- Type-safe tensor operations

## Quick Start

### First Time Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Shapes

# Initialize submodules and build OpenBLAS
make init

# Build everything
make build

# Run tests
make test
```

### Development Workflow

```bash
# Build entire project
make build

# Run all tests
make test

# Format code before committing
make format

# Run static analysis
make lint

# Clean build artifacts
make clean

# Show all available commands
make help
```

### C Library (from base/ directory)

```bash
cd base
cmake -S . -B build
make -C build

# Run tests
./build/shapes_test

# Run main executable
./build/shapes
```

### Go Library

```bash
cd go
go build ./...
go test ./...
```

## Development

### C Development
See `base/AGENTS.md` and `base/CLAUDE.md` for detailed C development guidelines including:
- Build system (CMake)
- Testing framework
- Code conventions
- Memory management patterns
- Tensor operation patterns

### Go Development
The Go library uses CGo to interface with the C core. Development guidelines:
- Use `cgo` to call C functions
- Wrap C types in idiomatic Go types
- Handle memory management carefully across the Go/C boundary
- Build higher-level features (layers, etc.) in pure Go on top of tensor primitives

## Design Philosophy

1. **Performance-critical code in C**: Low-level tensor operations, memory management, and autograd are implemented in C for maximum performance
2. **Language-specific ergonomics**: Each language binding provides idiomatic APIs that feel natural to developers in that ecosystem
3. **Layered abstractions**: Higher-level features (neural network layers, training loops) are built in the binding languages on top of fast tensor primitives
4. **Shared core**: All languages use the same battle-tested C core, ensuring consistent behavior and performance

## Project Status

- ✅ C core library with tensor operations and autograd
- 🚧 Go bindings (in progress)
- 📋 Python bindings (planned)
- 📋 Additional language bindings (TBD)

## Contributing

When working on this project:
- C code follows C99 standard with custom conventions (see `base/CLAUDE.md`)
- Run `make format` before committing C changes
- Run `make test` to verify all tests pass
- Each language binding should maintain its own testing and documentation
- Keep the C core focused on performance; add language-specific features in bindings
- Use the root-level `Makefile` for convenient build orchestration

## License

