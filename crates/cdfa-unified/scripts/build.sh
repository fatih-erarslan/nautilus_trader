#!/bin/bash

# CDFA Unified Build Script
# Automated build system for development, testing, and production deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/target"
DIST_DIR="$PROJECT_ROOT/dist"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
PROFILE="release"
FEATURES="default"
TARGET=""
VERBOSE=""
CLEAN=false
DOCS=false
TESTS=false
BENCHMARKS=false
PYTHON=false
DOCKER=false
VALIDATE=false

# Help function
show_help() {
    cat << EOF
CDFA Unified Build Script

Usage: $0 [OPTIONS]

OPTIONS:
    -p, --profile PROFILE     Build profile (debug|release|bench) [default: release]
    -f, --features FEATURES   Cargo features to enable [default: default]
    -t, --target TARGET       Target triple for cross-compilation
    -c, --clean              Clean before building
    -d, --docs               Build documentation
    -T, --tests              Run tests after building
    -b, --benchmarks         Build and run benchmarks
    -P, --python             Build Python bindings
    -D, --docker             Build Docker image
    -v, --verbose            Verbose output
    -V, --validate           Run validation suite
    -h, --help               Show this help

EXAMPLES:
    $0                                          # Basic release build
    $0 -p debug -f "core,algorithms"           # Debug build with specific features
    $0 -c -d -T                                # Clean, build docs, run tests
    $0 -f "full-performance" -b                # Full performance build with benchmarks
    $0 -P -f "python"                          # Build Python bindings
    $0 -D -f "full-performance"                # Build Docker image

FEATURE PROFILES:
    default             Core functionality with common optimizations
    minimal             Only core features
    full-performance    All performance features (SIMD, parallel, GPU, ML)
    development         Features for development and testing
    python              Python integration features
    distributed         Distributed computing features

EOF
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -f|--features)
            FEATURES="$2"
            shift 2
            ;;
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -d|--docs)
            DOCS=true
            shift
            ;;
        -T|--tests)
            TESTS=true
            shift
            ;;
        -b|--benchmarks)
            BENCHMARKS=true
            shift
            ;;
        -P|--python)
            PYTHON=true
            shift
            ;;
        -D|--docker)
            DOCKER=true
            shift
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -V|--validate)
            VALIDATE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
check_environment() {
    log_info "Checking build environment..."
    
    # Check Rust toolchain
    if ! command -v rustc &> /dev/null; then
        log_error "Rust compiler not found. Please install Rust."
        exit 1
    fi
    
    local rust_version
    rust_version=$(rustc --version)
    log_info "Found Rust: $rust_version"
    
    # Check for required system dependencies
    if [[ "$FEATURES" == *"gpu"* ]]; then
        log_info "GPU features enabled - checking CUDA/ROCm..."
        if command -v nvcc &> /dev/null; then
            log_info "CUDA found: $(nvcc --version | grep 'release')"
        elif [[ -d "/opt/rocm" ]]; then
            log_info "ROCm found"
        else
            log_warning "GPU features enabled but no CUDA/ROCm found"
        fi
    fi
    
    # Check Python for bindings
    if [[ "$PYTHON" == true ]] || [[ "$FEATURES" == *"python"* ]]; then
        if command -v python3 &> /dev/null; then
            local python_version
            python_version=$(python3 --version)
            log_info "Found Python: $python_version"
            
            # Check for required Python packages
            if ! python3 -c "import numpy" &> /dev/null; then
                log_warning "NumPy not found - Python bindings may not work properly"
            fi
        else
            log_error "Python bindings requested but Python3 not found"
            exit 1
        fi
    fi
    
    # Check Docker
    if [[ "$DOCKER" == true ]]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker requested but not found"
            exit 1
        fi
        log_info "Found Docker: $(docker --version)"
    fi
}

# Clean build artifacts
clean_build() {
    if [[ "$CLEAN" == true ]]; then
        log_info "Cleaning build artifacts..."
        cargo clean $VERBOSE
        
        if [[ -d "$DIST_DIR" ]]; then
            rm -rf "$DIST_DIR"
        fi
        
        log_success "Clean completed"
    fi
}

# Configure build environment
configure_build() {
    log_info "Configuring build environment..."
    
    # Set target-specific environment variables
    if [[ -n "$TARGET" ]]; then
        export CARGO_BUILD_TARGET="$TARGET"
        log_info "Cross-compiling for target: $TARGET"
    fi
    
    # Optimize for current CPU if native build
    if [[ -z "$TARGET" && "$PROFILE" == "release" ]]; then
        export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
        log_info "Enabling native CPU optimizations"
    fi
    
    # Configure memory allocator
    if [[ "$FEATURES" == *"jemalloc"* ]]; then
        log_info "Using jemalloc allocator"
    elif [[ "$FEATURES" == *"mimalloc"* ]]; then
        log_info "Using mimalloc allocator"
    fi
    
    # Performance optimizations for release builds
    if [[ "$PROFILE" == "release" ]]; then
        export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-s"
        log_info "Enabling release optimizations"
    fi
}

# Build the project
build_project() {
    log_info "Building CDFA Unified (profile: $PROFILE, features: $FEATURES)..."
    
    local cargo_args=()
    cargo_args+=("build")
    cargo_args+=("--profile" "$PROFILE")
    cargo_args+=("--features" "$FEATURES")
    
    if [[ -n "$TARGET" ]]; then
        cargo_args+=("--target" "$TARGET")
    fi
    
    if [[ -n "$VERBOSE" ]]; then
        cargo_args+=("$VERBOSE")
    fi
    
    # Execute build
    cargo "${cargo_args[@]}"
    log_success "Build completed successfully"
}

# Build documentation
build_docs() {
    if [[ "$DOCS" == true ]]; then
        log_info "Building documentation..."
        
        local doc_args=()
        doc_args+=("doc")
        doc_args+=("--features" "$FEATURES")
        doc_args+=("--no-deps")
        doc_args+=("--document-private-items")
        
        if [[ -n "$VERBOSE" ]]; then
            doc_args+=("$VERBOSE")
        fi
        
        cargo "${doc_args[@]}"
        log_success "Documentation built successfully"
        log_info "Documentation available at: target/doc/cdfa_unified/index.html"
    fi
}

# Run tests
run_tests() {
    if [[ "$TESTS" == true ]]; then
        log_info "Running test suite..."
        
        local test_args=()
        test_args+=("test")
        test_args+=("--profile" "$PROFILE")
        test_args+=("--features" "$FEATURES")
        
        if [[ -n "$TARGET" ]]; then
            test_args+=("--target" "$TARGET")
        fi
        
        if [[ -n "$VERBOSE" ]]; then
            test_args+=("$VERBOSE")
        fi
        
        cargo "${test_args[@]}"
        log_success "All tests passed"
    fi
}

# Run benchmarks
run_benchmarks() {
    if [[ "$BENCHMARKS" == true ]]; then
        log_info "Running benchmarks..."
        
        local bench_args=()
        bench_args+=("bench")
        bench_args+=("--features" "$FEATURES,benchmarks")
        
        if [[ -n "$VERBOSE" ]]; then
            bench_args+=("$VERBOSE")
        fi
        
        cargo "${bench_args[@]}"
        log_success "Benchmarks completed"
    fi
}

# Build Python bindings
build_python() {
    if [[ "$PYTHON" == true ]]; then
        log_info "Building Python bindings..."
        
        # Ensure Python features are enabled
        local python_features="$FEATURES"
        if [[ "$python_features" != *"python"* ]]; then
            python_features="$python_features,python"
        fi
        
        # Build with maturin if available, otherwise use cargo
        if command -v maturin &> /dev/null; then
            maturin build --release --features "$python_features"
            log_success "Python bindings built with maturin"
        else
            cargo build --profile release --features "$python_features"
            log_success "Python bindings built with cargo"
            log_warning "Consider installing maturin for better Python integration"
        fi
    fi
}

# Build Docker image
build_docker() {
    if [[ "$DOCKER" == true ]]; then
        log_info "Building Docker image..."
        
        local image_tag="cdfa-unified:latest"
        
        docker build \
            --build-arg FEATURES="$FEATURES" \
            --build-arg PROFILE="$PROFILE" \
            -t "$image_tag" \
            "$PROJECT_ROOT"
        
        log_success "Docker image built: $image_tag"
    fi
}

# Run validation suite
run_validation() {
    if [[ "$VALIDATE" == true ]]; then
        log_info "Running validation suite..."
        
        # Run the validation script if it exists
        local validation_script="$SCRIPT_DIR/validate.sh"
        if [[ -f "$validation_script" ]]; then
            "$validation_script" --features "$FEATURES" --profile "$PROFILE"
        else
            log_warning "Validation script not found, running basic tests instead"
            cargo test --features "$FEATURES" --profile "$PROFILE"
        fi
        
        log_success "Validation completed"
    fi
}

# Create distribution package
create_distribution() {
    if [[ "$PROFILE" == "release" ]]; then
        log_info "Creating distribution package..."
        
        mkdir -p "$DIST_DIR"
        
        # Copy binary
        local binary_path="$BUILD_DIR/$PROFILE/cdfa-unified"
        if [[ -n "$TARGET" ]]; then
            binary_path="$BUILD_DIR/$TARGET/$PROFILE/cdfa-unified"
        fi
        
        if [[ -f "$binary_path" ]]; then
            cp "$binary_path" "$DIST_DIR/"
        fi
        
        # Copy library
        local lib_path="$BUILD_DIR/$PROFILE/libcdfa_unified.rlib"
        if [[ -n "$TARGET" ]]; then
            lib_path="$BUILD_DIR/$TARGET/$PROFILE/libcdfa_unified.rlib"
        fi
        
        if [[ -f "$lib_path" ]]; then
            cp "$lib_path" "$DIST_DIR/"
        fi
        
        # Copy dynamic library for FFI
        local dylib_path="$BUILD_DIR/$PROFILE/libcdfa_unified.so"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            dylib_path="$BUILD_DIR/$PROFILE/libcdfa_unified.dylib"
        elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
            dylib_path="$BUILD_DIR/$PROFILE/cdfa_unified.dll"
        fi
        
        if [[ -n "$TARGET" ]]; then
            dylib_path="${dylib_path/$PROFILE/$TARGET\/$PROFILE}"
        fi
        
        if [[ -f "$dylib_path" ]]; then
            cp "$dylib_path" "$DIST_DIR/"
        fi
        
        # Copy documentation if built
        if [[ -d "$BUILD_DIR/doc" ]]; then
            cp -r "$BUILD_DIR/doc" "$DIST_DIR/"
        fi
        
        # Create archive
        local archive_name="cdfa-unified-${TARGET:-native}-${PROFILE}.tar.gz"
        cd "$DIST_DIR"
        tar -czf "$archive_name" ./*
        cd - > /dev/null
        
        log_success "Distribution package created: $DIST_DIR/$archive_name"
    fi
}

# Print build summary
print_summary() {
    echo
    log_success "=== BUILD SUMMARY ==="
    log_info "Profile: $PROFILE"
    log_info "Features: $FEATURES"
    log_info "Target: ${TARGET:-native}"
    
    if [[ "$DOCS" == true ]]; then
        log_info "Documentation: Built"
    fi
    
    if [[ "$TESTS" == true ]]; then
        log_info "Tests: Passed"
    fi
    
    if [[ "$BENCHMARKS" == true ]]; then
        log_info "Benchmarks: Completed"
    fi
    
    if [[ "$PYTHON" == true ]]; then
        log_info "Python bindings: Built"
    fi
    
    if [[ "$DOCKER" == true ]]; then
        log_info "Docker image: Built"
    fi
    
    if [[ "$VALIDATE" == true ]]; then
        log_info "Validation: Passed"
    fi
    
    # Binary size information
    local binary_path="$BUILD_DIR/$PROFILE/libcdfa_unified.rlib"
    if [[ -n "$TARGET" ]]; then
        binary_path="$BUILD_DIR/$TARGET/$PROFILE/libcdfa_unified.rlib"
    fi
    
    if [[ -f "$binary_path" ]]; then
        local size
        size=$(du -h "$binary_path" | cut -f1)
        log_info "Library size: $size"
    fi
    
    echo
    log_success "Build completed successfully! 🚀"
}

# Main execution
main() {
    log_info "Starting CDFA Unified build process..."
    
    cd "$PROJECT_ROOT"
    
    check_environment
    clean_build
    configure_build
    build_project
    build_docs
    run_tests
    run_benchmarks
    build_python
    build_docker
    run_validation
    create_distribution
    print_summary
}

# Execute main function
main "$@"