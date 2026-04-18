# Shapes Multi-language Tensor Library
#
# Root Makefile for building all components

.PHONY: all help init build build-openblas build-base build-go run debug test test-go clean format lint check-format

# Detect number of CPU cores
NPROC := $(shell nproc 2>/dev/null || echo 4)

# Directories
BASE_DIR := base
OPENBLAS_DIR := $(BASE_DIR)/OpenBLAS
BUILD_DIR := $(BASE_DIR)/build
GO_DIR := go

# OpenBLAS
OPENBLAS_INSTALL := $(BUILD_DIR)/openblas
OPENBLAS_LIB := $(OPENBLAS_INSTALL)/lib/libopenblas.a

# Go
GO_BINARY := $(GO_DIR)/main
OPENBLAS_LIB_DIR := $(OPENBLAS_INSTALL)/lib
CUDA_INCLUDE_DIR := /opt/cuda/targets/x86_64-linux/include
CUDA_LIB_DIR := /opt/cuda/targets/x86_64-linux/lib
GO_CGO_CFLAGS := -O0 -g -I$(abspath $(BASE_DIR)) -I$(CUDA_INCLUDE_DIR)
GO_CGO_LDFLAGS := -L$(abspath $(BUILD_DIR)) -L$(abspath $(OPENBLAS_LIB_DIR)) -L$(CUDA_LIB_DIR) -lshapes_core -lshapes_memory -lopenblas -lcublas -lcudart -lstdc++ -lm
GO_RUNTIME_LD_LIBRARY_PATH := $$PWD/../$(OPENBLAS_LIB_DIR):$(CUDA_LIB_DIR)

# Default target
all: build

help:
	@echo "Shapes Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build everything (default)"
	@echo "  init           - Initialize submodules and build OpenBLAS"
	@echo "  build          - Build C library and Go bindings"
	@echo "  build-openblas - Build OpenBLAS from submodule"
	@echo "  build-base     - Build C library only (Release mode)"
	@echo "  build-go       - Build Go bindings"
	@echo "  run            - Build and run the Go binary"
	@echo "  debug          - Build and debug the Go binary with Delve"
	@echo "  test           - Run all tests (C + Go)"
	@echo "  test-go        - Run Go tests only"
	@echo "  clean          - Clean all build artifacts"
	@echo "  format         - Auto-format C code"
	@echo "  check-format   - Check C code formatting (non-modifying)"
	@echo "  lint           - Run static analysis on C code"
	@echo ""
	@echo "Quick start:"
	@echo "  make init      # First time setup"
	@echo "  make build     # Build everything"
	@echo "  make test      # Run tests"

# Initialize submodules and build OpenBLAS
init:
	@echo "==> Initializing git submodules..."
	git submodule update --init --recursive
	@echo "==> Building OpenBLAS (this may take a few minutes)..."
	$(MAKE) build-openblas
	@echo "==> Initialization complete!"

# Build OpenBLAS
build-openblas: $(OPENBLAS_LIB)

$(OPENBLAS_LIB):
	@echo "==> Building OpenBLAS..."
	cd $(OPENBLAS_DIR) && $(MAKE) -j$(NPROC)
	@echo "==> Installing OpenBLAS to local prefix: $(OPENBLAS_INSTALL)"
	mkdir -p $(OPENBLAS_INSTALL)
	cd $(OPENBLAS_DIR) && $(MAKE) PREFIX=$$(realpath ../build/openblas) install
	@echo "==> Cleaning OpenBLAS build tree..."
	cd $(OPENBLAS_DIR) && $(MAKE) clean
	@echo "==> OpenBLAS build complete!"

# Configure and build everything
build: build-base build-go

build-base: $(OPENBLAS_LIB)
	@echo "==> Configuring C library..."
	cmake -S $(BASE_DIR) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug
	@echo "==> Building C library..."
	$(MAKE) -C $(BUILD_DIR) -j$(NPROC)
	@echo "==> C build complete!"

# Build Go bindings
build-go: build-base
	@echo "==> Building Go bindings..."
	cd $(GO_DIR) && CGO_CFLAGS='$(GO_CGO_CFLAGS)' CGO_LDFLAGS='$(GO_CGO_LDFLAGS)' go build -gcflags="all=-N -l" -ldflags="-compressdwarf=false" -v ./cmd/main
	@echo "==> Go build complete!"
	@echo ""
	@echo "To run: cd go && LD_LIBRARY_PATH=$(GO_RUNTIME_LD_LIBRARY_PATH) ./main"

# Run the Go binary
run: build-go
	cd $(GO_DIR) && LD_LIBRARY_PATH=$(GO_RUNTIME_LD_LIBRARY_PATH) ./main

# Debug the Go binary with Delve (ASan disabled to avoid CGo/Go runtime conflicts)
debug: $(OPENBLAS_LIB)
	@echo "==> Configuring C library (ASan disabled for Go debug)..."
	cmake -S $(BASE_DIR) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=OFF
	@echo "==> Building C library..."
	$(MAKE) -C $(BUILD_DIR) -j$(NPROC)
	@echo "==> Building Go binary with debug flags..."
	cd $(GO_DIR) && CGO_CFLAGS='$(GO_CGO_CFLAGS)' CGO_LDFLAGS='$(GO_CGO_LDFLAGS)' go build -gcflags='all=-N -l' -o main ./cmd/main
	@echo "==> Launching Delve debugger..."
	cd $(GO_DIR) && LD_LIBRARY_PATH=$(GO_RUNTIME_LD_LIBRARY_PATH) dlv exec ./main


debug-gdb: $(OPENBLAS_LIB)
	@echo "==> Configuring C library (ASan disabled for Go debug)..."
	cmake -S $(BASE_DIR) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=OFF
	@echo "==> Building C library..."
	$(MAKE) -C $(BUILD_DIR) -j$(NPROC)
	@echo "==> Building Go binary with debug flags..."
	cd $(GO_DIR) && CGO_CFLAGS='$(GO_CGO_CFLAGS)' CGO_LDFLAGS='$(GO_CGO_LDFLAGS)' go build -gcflags='all=-N -l' -o main ./cmd/main
	@echo "==> Launching Delve debugger..."
	cd $(GO_DIR) && LD_LIBRARY_PATH=$(GO_RUNTIME_LD_LIBRARY_PATH) gdb ./main

# Run all tests
test: build-base build-go
	@echo "==> Running C tests..."
	cd $(BASE_DIR) && ctest --test-dir build --output-on-failure
	@echo "==> Running Go tests..."
	cd $(GO_DIR) && CGO_CFLAGS='$(GO_CGO_CFLAGS)' CGO_LDFLAGS='$(GO_CGO_LDFLAGS)' LD_LIBRARY_PATH=$(GO_RUNTIME_LD_LIBRARY_PATH) go test -v ./...
	@echo "==> All tests passed!"

# Run Go tests only
test-go: build-go
	@echo "==> Running Go tests..."
	cd $(GO_DIR) && CGO_CFLAGS='$(GO_CGO_CFLAGS)' CGO_LDFLAGS='$(GO_CGO_LDFLAGS)' LD_LIBRARY_PATH=$(GO_RUNTIME_LD_LIBRARY_PATH) go test -v ./...
	@echo "==> Go tests passed!"

# Run tests directly (alternative to ctest)
test-direct: build-base
	@echo "==> Running C tests directly..."
	$(BUILD_DIR)/shapes_test
	@echo "==> All tests passed!"

# Format C code
format:
	@echo "==> Formatting C code..."
	$(MAKE) -C $(BUILD_DIR) format
	@echo "==> Code formatted!"

# Check formatting without modifying files
check-format:
	@echo "==> Checking C code formatting..."
	$(MAKE) -C $(BUILD_DIR) check-format
	@echo "==> Format check complete!"

# Run static analysis
lint:
	@echo "==> Running static analysis..."
	$(MAKE) -C $(BUILD_DIR) lint
	@echo "==> Linting complete!"

# Clean build artifacts
clean:
	@echo "==> Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f $(GO_BINARY)
	cd $(GO_DIR) && go clean
	cd $(OPENBLAS_DIR) && $(MAKE) clean 2>/dev/null || true
	@echo "==> Clean complete!"

# Deep clean (including OpenBLAS)
distclean: clean
	@echo "==> Deep cleaning (including OpenBLAS)..."
	cd $(OPENBLAS_DIR) && git clean -fdx
	@echo "==> Deep clean complete!"

# Rebuild everything from scratch
rebuild: clean build

# Check if OpenBLAS is built
check-openblas:
	@if [ ! -f $(OPENBLAS_LIB) ]; then \
		echo "Error: OpenBLAS not built. Run 'make init' first."; \
		exit 1; \
	fi
	@echo "OpenBLAS is built."
