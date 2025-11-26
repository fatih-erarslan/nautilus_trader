# Neural Trader v1 (Python Implementation)

This directory contains the original Python implementation of Neural Trader. This is the legacy version (v1) that is being superseded by the Rust implementation in `neural-trader-rust/`.

## ⚠️ Status: Maintenance Mode

This v1 Python implementation is in **maintenance mode**. New features and major improvements are being developed in the Rust version (`neural-trader-rust/`). This version is kept for:

- **Reference**: Historical implementation patterns and algorithms
- **Compatibility**: Existing integrations and deployments still using v1
- **Migration Support**: Side-by-side comparison during migration to Rust
- **Testing**: Validation and parity testing against the new Rust implementation

## Directory Structure

```
neural-trader-v1/
├── src/                    # Core Python source code
├── models/                 # ML models and checkpoints
├── model_management/       # Model versioning and lifecycle
├── monitoring/             # System monitoring and metrics
├── examples/               # Usage examples and demos
├── benchmarks/             # Performance benchmarks
├── config/                 # Configuration files
├── data/                   # Sample data and datasets
├── demo/                   # Demo applications
├── deployment/             # Deployment configurations
├── fly_deployment/         # Fly.io specific deployment
├── gpu_acceleration/       # GPU-accelerated components
├── tutorials/              # Learning materials
├── sql/                    # Database schemas and migrations
└── requirements.txt        # Python dependencies
```

## Migration to v2 (Rust)

The Rust implementation (`neural-trader-rust/`) offers:

- **10-100x Performance**: Native Rust performance with zero-copy operations
- **Type Safety**: Compile-time guarantees and memory safety
- **Concurrency**: Async/await with Tokio for high-throughput trading
- **NAPI Bindings**: Native Node.js integration for JavaScript/TypeScript
- **WASM Support**: Browser and edge deployment capabilities
- **Modern Architecture**: Clean separation of concerns with workspace structure

### Migration Path

1. **Assess dependencies**: Review your current v1 usage
2. **Check Rust parity**: Verify features exist in `neural-trader-rust/`
3. **Test in parallel**: Run both versions side-by-side
4. **Migrate gradually**: Move one component at a time
5. **Validate results**: Ensure numerical parity and behavior match

See `docs/migration-guide.md` (TODO) for detailed migration instructions.

## Python v1 Setup

### Prerequisites

- Python 3.8+
- pip or poetry
- Optional: CUDA for GPU acceleration

### Installation

```bash
cd neural-trader-v1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running v1

```bash
# Run demo
python src/main.py

# Run specific strategy
python src/strategies/momentum.py

# Start monitoring dashboard
python monitoring/dashboard.py
```

## Known Limitations (v1)

- **Performance**: Python GIL limits true parallelism
- **Memory**: Higher memory overhead compared to Rust
- **Type Safety**: Runtime type checking vs compile-time
- **Dependencies**: Heavy ML dependencies (TensorFlow, PyTorch)
- **Deployment**: Larger container sizes and slower cold starts

## Support

- **Bug Reports**: Critical bugs only for v1
- **New Features**: Development focused on Rust version
- **Questions**: Use GitHub Discussions or Issues
- **Migration Help**: See migration guide and examples

## Contributing

Contributions to v1 should be:
- **Bug fixes**: Critical production issues only
- **Documentation**: Improvements welcome
- **Migration tools**: Helpers for v1 → v2 migration

For new features, please contribute to `neural-trader-rust/` instead.

## License

Same as parent project (see root LICENSE file).

## Related Documentation

- [Rust Implementation](../neural-trader-rust/README.md)
- [Architecture Comparison](../docs/v1-vs-v2-architecture.md) (TODO)
- [Migration Guide](../docs/migration-guide.md) (TODO)
- [Performance Benchmarks](../docs/performance.md)
