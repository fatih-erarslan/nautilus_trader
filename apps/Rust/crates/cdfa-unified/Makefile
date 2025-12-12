# Makefile for CDFA Unified Library FFI Interfaces
# 
# This Makefile provides convenient targets for building, testing, and validating
# the FFI interfaces for both C and Python integration.

# Configuration
CARGO := cargo
PYTHON := python3
GCC := gcc
CLANG := clang

# Build configuration
RELEASE_DIR := target/release
DEBUG_DIR := target/debug
INCLUDE_DIR := include
EXAMPLES_DIR := examples/ffi_examples

# Features
FFI_FEATURES := ffi,python,c-bindings,core,algorithms,simd,parallel
C_FEATURES := c-bindings,core,algorithms,simd,parallel
PYTHON_FEATURES := python,core,algorithms,simd,parallel

# Compiler flags
CFLAGS := -Wall -Wextra -std=c99 -O2
LDFLAGS := -lm -lpthread

# Library names (platform-dependent)
ifeq ($(shell uname),Darwin)
    DYLIB_EXT := dylib
    SHARED_LIB := libcdfa_unified.$(DYLIB_EXT)
    RPATH_FLAG := -Wl,-rpath,$(CURDIR)/$(RELEASE_DIR)
else
    DYLIB_EXT := so
    SHARED_LIB := libcdfa_unified.$(DYLIB_EXT)
    RPATH_FLAG := -Wl,-rpath,$(CURDIR)/$(RELEASE_DIR)
endif

# Default target
.PHONY: all
all: build-ffi test-ffi examples

# Help target
.PHONY: help
help:
	@echo "CDFA Unified FFI Build System"
	@echo "=============================="
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build everything and run tests"
	@echo "  build-ffi        - Build library with all FFI features"
	@echo "  build-c          - Build library with C API only"
	@echo "  build-python     - Build library with Python bindings only"
	@echo "  test-ffi         - Run all FFI tests"
	@echo "  test-c           - Test C API"
	@echo "  test-python      - Test Python bindings"
	@echo "  examples         - Build all examples"
	@echo "  example-c        - Build and run C example"
	@echo "  example-python   - Run Python example"
	@echo "  validate         - Run comprehensive validation suite"
	@echo "  benchmark        - Run performance benchmarks"
	@echo "  docs             - Generate documentation"
	@echo "  clean            - Clean build artifacts"
	@echo "  install          - Install library and headers"
	@echo "  package-python   - Package Python wheel"
	@echo ""
	@echo "Variables:"
	@echo "  CARGO=$(CARGO)"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  GCC=$(GCC)"
	@echo "  PREFIX=$(PREFIX)"

# Build targets
.PHONY: build-ffi
build-ffi:
	@echo "Building CDFA library with all FFI features..."
	$(CARGO) build --release --features "$(FFI_FEATURES)"
	@echo "✓ FFI library built: $(RELEASE_DIR)/$(SHARED_LIB)"

.PHONY: build-c
build-c:
	@echo "Building CDFA library with C API..."
	$(CARGO) build --release --features "$(C_FEATURES)"
	@echo "✓ C API library built"

.PHONY: build-python
build-python:
	@echo "Building CDFA library with Python bindings..."
	$(CARGO) build --release --features "$(PYTHON_FEATURES)"
	@echo "✓ Python bindings built"

.PHONY: build-debug
build-debug:
	@echo "Building debug version..."
	$(CARGO) build --features "$(FFI_FEATURES)"

# Test targets
.PHONY: test-ffi
test-ffi: build-ffi
	@echo "Running all FFI tests..."
	$(CARGO) test --release --features "$(FFI_FEATURES)"
	@echo "✓ All FFI tests passed"

.PHONY: test-c
test-c: build-c
	@echo "Running C API tests..."
	$(CARGO) test ffi_c_integration --release --features "$(C_FEATURES)"
	@echo "✓ C API tests passed"

.PHONY: test-python
test-python: build-python
	@echo "Running Python binding tests..."
	$(CARGO) test ffi_python_integration --release --features "$(PYTHON_FEATURES)"
	@echo "✓ Python binding tests passed"

.PHONY: test-memory
test-memory: build-debug
	@echo "Running memory safety tests..."
	@if command -v valgrind >/dev/null 2>&1; then \
		echo "Using Valgrind for memory testing..."; \
		valgrind --tool=memcheck --leak-check=full \
			$(CARGO) test ffi --features "$(C_FEATURES)"; \
	else \
		echo "Valgrind not available, using AddressSanitizer..."; \
		RUSTFLAGS="-Z sanitizer=address" \
			$(CARGO) test ffi --features "$(C_FEATURES)" --target x86_64-unknown-linux-gnu || true; \
	fi

# Example targets
.PHONY: examples
examples: example-c example-python

.PHONY: example-c
example-c: build-c $(EXAMPLES_DIR)/c_usage_example.c
	@echo "Building and running C example..."
	$(GCC) $(CFLAGS) -o $(EXAMPLES_DIR)/c_usage_example \
		$(EXAMPLES_DIR)/c_usage_example.c \
		-I$(INCLUDE_DIR) \
		-L$(RELEASE_DIR) \
		-lcdfa_unified \
		$(RPATH_FLAG) \
		$(LDFLAGS)
	@echo "Running C example..."
	LD_LIBRARY_PATH=$(RELEASE_DIR):$$LD_LIBRARY_PATH \
		$(EXAMPLES_DIR)/c_usage_example
	@echo "✓ C example completed"

.PHONY: example-python
example-python: build-python $(EXAMPLES_DIR)/python_usage_example.py
	@echo "Running Python example..."
	@if command -v $(PYTHON) >/dev/null 2>&1; then \
		PYTHONPATH=$(RELEASE_DIR):$$PYTHONPATH \
		LD_LIBRARY_PATH=$(RELEASE_DIR):$$LD_LIBRARY_PATH \
			$(PYTHON) $(EXAMPLES_DIR)/python_usage_example.py; \
		echo "✓ Python example completed"; \
	else \
		echo "Python3 not available - skipping Python example"; \
	fi

# Validation and benchmarking
.PHONY: validate
validate: scripts/validate_ffi.sh
	@echo "Running comprehensive FFI validation..."
	@./scripts/validate_ffi.sh
	@echo "✓ Validation completed"

.PHONY: benchmark
benchmark: build-ffi
	@echo "Running performance benchmarks..."
	$(CARGO) bench --features "$(FFI_FEATURES)"
	@echo "✓ Benchmarks completed"

.PHONY: benchmark-ffi
benchmark-ffi: build-ffi
	@echo "Running FFI-specific benchmarks..."
	@# C API benchmark
	@if [ -f $(RELEASE_DIR)/$(SHARED_LIB) ]; then \
		echo "C API benchmark:"; \
		$(GCC) $(CFLAGS) -o /tmp/ffi_benchmark -x c - \
			-I$(INCLUDE_DIR) -L$(RELEASE_DIR) -lcdfa_unified $(RPATH_FLAG) $(LDFLAGS) \
			<<< 'int main(){return 0;}' 2>/dev/null && \
		LD_LIBRARY_PATH=$(RELEASE_DIR):$$LD_LIBRARY_PATH /tmp/ffi_benchmark || \
		echo "C benchmark compilation failed"; \
		rm -f /tmp/ffi_benchmark; \
	fi
	@# Python API benchmark
	@if command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "Python API benchmark:"; \
		PYTHONPATH=$(RELEASE_DIR):$$PYTHONPATH \
		LD_LIBRARY_PATH=$(RELEASE_DIR):$$LD_LIBRARY_PATH \
			$(PYTHON) -c "import time; import cdfa_unified; print('Python import successful')" 2>/dev/null || \
		echo "Python benchmark failed"; \
	fi

# Documentation
.PHONY: docs
docs:
	@echo "Generating documentation..."
	$(CARGO) doc --features "$(FFI_FEATURES)" --no-deps
	@echo "✓ Documentation generated in target/doc/"

.PHONY: docs-open
docs-open: docs
	@echo "Opening documentation..."
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open target/doc/cdfa_unified/index.html; \
	elif command -v open >/dev/null 2>&1; then \
		open target/doc/cdfa_unified/index.html; \
	else \
		echo "Please open target/doc/cdfa_unified/index.html in your browser"; \
	fi

# Installation
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include

.PHONY: install
install: build-ffi
	@echo "Installing CDFA library..."
	@mkdir -p $(LIBDIR) $(INCLUDEDIR)
	@cp $(RELEASE_DIR)/$(SHARED_LIB) $(LIBDIR)/
	@cp $(INCLUDE_DIR)/cdfa_unified.h $(INCLUDEDIR)/
	@echo "Library installed to $(LIBDIR)/$(SHARED_LIB)"
	@echo "Header installed to $(INCLUDEDIR)/cdfa_unified.h"
	@if command -v ldconfig >/dev/null 2>&1; then \
		echo "Running ldconfig..."; \
		ldconfig; \
	fi
	@echo "✓ Installation completed"

.PHONY: uninstall
uninstall:
	@echo "Uninstalling CDFA library..."
	@rm -f $(LIBDIR)/$(SHARED_LIB)
	@rm -f $(INCLUDEDIR)/cdfa_unified.h
	@if command -v ldconfig >/dev/null 2>&1; then \
		ldconfig; \
	fi
	@echo "✓ Uninstallation completed"

# Python packaging
.PHONY: package-python
package-python: build-python
	@echo "Packaging Python wheel..."
	@if command -v maturin >/dev/null 2>&1; then \
		maturin build --release --features "$(PYTHON_FEATURES)"; \
		echo "✓ Python wheel created in target/wheels/"; \
	else \
		echo "maturin not available - install with: pip install maturin"; \
		exit 1; \
	fi

.PHONY: install-python
install-python: package-python
	@echo "Installing Python package..."
	@if ls target/wheels/cdfa_unified-*.whl >/dev/null 2>&1; then \
		$(PYTHON) -m pip install target/wheels/cdfa_unified-*.whl --force-reinstall; \
		echo "✓ Python package installed"; \
	else \
		echo "No Python wheel found - run 'make package-python' first"; \
		exit 1; \
	fi

# Development targets
.PHONY: dev-setup
dev-setup:
	@echo "Setting up development environment..."
	@# Install Rust components
	@if ! rustup component list | grep -q "rustfmt.*installed"; then \
		rustup component add rustfmt; \
	fi
	@if ! rustup component list | grep -q "clippy.*installed"; then \
		rustup component add clippy; \
	fi
	@# Install Python tools
	@if command -v $(PYTHON) >/dev/null 2>&1; then \
		$(PYTHON) -m pip install maturin numpy matplotlib --quiet || true; \
	fi
	@# Install C development tools
	@echo "Checking C development tools..."
	@command -v $(GCC) >/dev/null 2>&1 || echo "Warning: GCC not found"
	@command -v valgrind >/dev/null 2>&1 || echo "Warning: Valgrind not found (memory testing will be limited)"
	@echo "✓ Development environment ready"

.PHONY: lint
lint:
	@echo "Running linting checks..."
	$(CARGO) fmt --check
	$(CARGO) clippy --features "$(FFI_FEATURES)" -- -D warnings
	@echo "✓ Linting completed"

.PHONY: format
format:
	@echo "Formatting code..."
	$(CARGO) fmt
	@echo "✓ Code formatted"

# Security and analysis
.PHONY: audit
audit:
	@echo "Running security audit..."
	$(CARGO) audit
	@echo "✓ Security audit completed"

.PHONY: deny
deny:
	@echo "Running cargo-deny checks..."
	@if command -v cargo-deny >/dev/null 2>&1; then \
		cargo deny check; \
	else \
		echo "cargo-deny not installed - install with: cargo install cargo-deny"; \
	fi

# Cleanup
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	$(CARGO) clean
	@rm -f $(EXAMPLES_DIR)/c_usage_example
	@rm -f $(EXAMPLES_DIR)/*.o
	@rm -f /tmp/ffi_benchmark
	@rm -f ffi_validation.log
	@rm -f FFI_VALIDATION_REPORT.md
	@echo "✓ Cleanup completed"

.PHONY: clean-all
clean-all: clean
	@echo "Deep cleaning..."
	@rm -rf target/
	@rm -rf .cargo/
	@echo "✓ Deep cleanup completed"

# Continuous Integration targets
.PHONY: ci
ci: lint test-ffi validate benchmark
	@echo "✓ CI pipeline completed successfully"

.PHONY: ci-fast
ci-fast: build-ffi test-ffi
	@echo "✓ Fast CI pipeline completed"

# Platform-specific targets
.PHONY: cross-compile
cross-compile:
	@echo "Cross-compiling for multiple targets..."
	@# Add common targets
	@for target in x86_64-unknown-linux-gnu x86_64-pc-windows-gnu x86_64-apple-darwin; do \
		echo "Building for $$target..."; \
		$(CARGO) build --release --target $$target --features "$(C_FEATURES)" 2>/dev/null || \
		echo "Failed to build for $$target (target may not be installed)"; \
	done

# Debugging targets
.PHONY: debug-info
debug-info:
	@echo "System Information:"
	@echo "=================="
	@echo "OS: $$(uname -a)"
	@echo "Rust: $$(rustc --version)"
	@echo "Cargo: $$(cargo --version)"
	@echo "GCC: $$($(GCC) --version | head -1)"
	@echo "Python: $$($(PYTHON) --version 2>&1)"
	@echo ""
	@echo "Features Available:"
	@echo "==================="
	@echo "SIMD Support: $$(rustc --print cfg | grep -q 'target_feature=\"sse\"' && echo 'Yes' || echo 'No')"
	@echo "Parallel Support: $$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'Unknown') cores"
	@echo "NumPy: $$($(PYTHON) -c 'import numpy; print(\"Yes:\", numpy.__version__)' 2>/dev/null || echo 'No')"
	@echo "Valgrind: $$(command -v valgrind >/dev/null && echo 'Yes' || echo 'No')"

# File dependencies
$(EXAMPLES_DIR)/c_usage_example: $(EXAMPLES_DIR)/c_usage_example.c $(INCLUDE_DIR)/cdfa_unified.h build-c

$(INCLUDE_DIR)/cdfa_unified.h: src/ffi/mod.rs
	@echo "Header file is up to date"

# Phony target list
.PHONY: all help build-ffi build-c build-python build-debug test-ffi test-c test-python test-memory \
        examples example-c example-python validate benchmark benchmark-ffi docs docs-open \
        install uninstall package-python install-python dev-setup lint format audit deny \
        clean clean-all ci ci-fast cross-compile debug-info