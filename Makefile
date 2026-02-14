# Shapes Multi-language Tensor Library
# Root Makefile for building all components

.PHONY: all help init build build-openblas build-base build-go test test-go clean format lint check-format

# Detect number of CPU cores
NPROC := $(shell nproc 2>/dev/null || echo 4)

# Directories
BASE_DIR := base
OPENBLAS_DIR := $(BASE_DIR)/OpenBLAS
BUILD_DIR := $(BASE_DIR)/build
GO_DIR := go

# OpenBLAS
OPENBLAS_LIB := $(OPENBLAS_DIR)/libopenblas.a
OPENBLAS_INSTALL := $(OPENBLAS_DIR)/install

# Go
GO_BINARY := $(GO_DIR)/main
OPENBLAS_LIB_DIR := $(OPENBLAS_INSTALL)/lib

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
	@echo "==> Installing OpenBLAS to local prefix..."
	cd $(OPENBLAS_DIR) && $(MAKE) PREFIX=$$(pwd)/install install
	@echo "==> OpenBLAS build complete!"

# Configure and build everything
build: build-base build-go

build-base: $(OPENBLAS_LIB)
	@echo "==> Configuring C library (Release mode)..."
	cmake -S $(BASE_DIR) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@echo "==> Building C library..."
	$(MAKE) -C $(BUILD_DIR) -j$(NPROC)
	@echo "==> C build complete!"

# Build Go bindings
build-go: build-base
	@echo "==> Building Go bindings..."
	cd $(GO_DIR) && go build -v ./cmd/main
	@echo "==> Go build complete!"
	@echo ""
	@echo "To run: cd go && LD_LIBRARY_PATH=$$PWD/../$(OPENBLAS_LIB_DIR) ./main"

# Run all tests
test: build-base build-go
	@echo "==> Running C tests..."
	cd $(BASE_DIR) && ctest --test-dir build --output-on-failure
	@echo "==> Running Go tests..."
	cd $(GO_DIR) && LD_LIBRARY_PATH=$$PWD/../$(OPENBLAS_LIB_DIR) go test -v ./...
	@echo "==> All tests passed!"

# Run Go tests only
test-go: build-go
	@echo "==> Running Go tests..."
	cd $(GO_DIR) && LD_LIBRARY_PATH=$$PWD/../$(OPENBLAS_LIB_DIR) go test -v ./...
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
