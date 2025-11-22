# CWTS Ultra Neural Networks

High-performance neural network implementations optimized for trading applications, with specialized WASM support for browser deployment.

## Features

### 🚀 Performance Optimized
- **SIMD128 acceleration** for 3-4x speedup in supported browsers
- **Automatic scalar fallback** for compatibility
- **Zero-copy operations** where possible
- **Memory-aligned data structures**

### 🎯 Trading Focused
- **Real-time inference** (<200μs for typical layers)
- **Quantization support** (INT8/INT16) for mobile deployment
- **Streaming processing** for continuous market data
- **Low memory footprint** (<1MB for typical networks)

### 🌐 WebAssembly Ready
- **Browser compatible** with wasm-bindgen
- **Node.js support** for server-side inference
- **TypeScript definitions** for seamless integration
- **No external dependencies** in WASM runtime

## Modules

### `wasm_nn.rs`
Core WebAssembly neural network implementation with:
- **WasmNeuralLayer**: Single layer with SIMD acceleration
- **WasmNeuralNetwork**: Multi-layer network builder
- **WasmStreamingNN**: Real-time streaming processor
- **Quantization utilities** for memory optimization

### `simd_nn.rs`
Cross-platform SIMD neural networks for:
- x86_64 (AVX2, AVX-512)
- ARM64 (NEON)
- WebAssembly (SIMD128)

### `gpu_nn.rs`
GPU-accelerated neural networks:
- CUDA support
- Metal (macOS)
- Vulkan compute shaders
- WebGPU (future)

## Quick Start

### Rust Native Usage

```rust
use cwts_ultra::neural::wasm_nn::*;

// Create a simple classifier
let mut network = WasmNeuralNetwork::new();
network.add_layer(784, 128); // Input layer
network.add_layer(128, 64);  // Hidden layer
network.add_layer(64, 10);   // Output layer

// Make prediction
let input = vec![0.1; 784]; // Flattened 28x28 image
let activations = vec![0, 0, 1]; // ReLU, ReLU, Sigmoid
let prediction = network.predict(&input, &activations);
```

### JavaScript/WebAssembly Usage

```javascript
import { JSNeuralNetwork, JSClassifier } from './cwts_ultra_wasm.js';

// Create neural network
const network = new JSNeuralNetwork();
network.add_layer(10, 8, "relu");
network.add_layer(8, 3, "sigmoid");

// Make prediction
const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const output = network.predict(input);
console.log("Prediction:", output);

// Use pre-built classifier
const classifier = new JSClassifier(784, [128, 64], 10);
const imageData = new Float32Array(784);
const classId = classifier.predict_class(imageData);
const confidence = classifier.confidence(imageData);
```

### Trading Application

```javascript
import { JSTradingNN } from './cwts_ultra_wasm.js';

const tradingNN = new JSTradingNN();

// Technical indicators (20 values)
const indicators = new Float32Array([
    0.5, 0.3, 0.8, -0.2, 0.1, // RSI, MACD, etc.
    // ... more indicators
]);

// Market data (15 values)
const marketData = new Float32Array([
    1.2345, 1000, 500, 0.02, // price, volume, etc.
    // ... more market data
]);

// Position data (10 values)
const positionData = new Float32Array([
    0.1, 1000, 0.5, // position size, value, etc.
    // ... more position data
]);

const decision = tradingNN.make_decision(indicators, marketData, positionData);
console.log("Trading decision:", decision);
```

## Performance Characteristics

### Inference Speed
- **32-neuron layer**: <200μs forward pass
- **SIMD acceleration**: 3-4x speedup over scalar
- **Batch processing**: Linear scaling with batch size
- **Quantized models**: 20-40% faster inference

### Memory Usage
- **Float32 weights**: 4 bytes per parameter
- **INT8 quantization**: 1 byte per parameter (75% reduction)
- **INT16 quantization**: 2 bytes per parameter (50% reduction)
- **Overhead**: <128 bytes per layer structure

### Browser Compatibility
- **Chrome 91+**: Full SIMD128 support
- **Firefox 89+**: Full SIMD128 support  
- **Safari 14.1+**: Full SIMD128 support
- **Older browsers**: Automatic scalar fallback

## Activation Functions

All activation functions support both SIMD and scalar implementations:

- **ReLU**: `max(0, x)` - fastest, most common
- **Sigmoid**: `1 / (1 + exp(-x))` - output range [0, 1]
- **Tanh**: `tanh(x)` - output range [-1, 1] 
- **LeakyReLU**: `max(0.01x, x)` - prevents dead neurons
- **Swish**: `x / (1 + exp(-x))` - smooth, self-gating
- **GELU**: Gaussian Error Linear Unit - modern transformer activation

## Quantization

Reduce memory usage and increase inference speed:

```rust
// Enable INT8 quantization
layer.quantize(1, true); // mode=INT8, dynamic_range=true

// Network-wide quantization
network.quantize_network(1, true);
```

Quantization modes:
- **None (0)**: Full float32 precision
- **INT8 (1)**: 8-bit integers, ~4x memory reduction
- **INT16 (2)**: 16-bit integers, ~2x memory reduction  
- **Dynamic (3)**: Runtime-adaptive quantization

## Streaming Processing

For real-time applications:

```rust
let mut streaming = WasmStreamingNN::new(1024); // Buffer size
streaming.add_layer(1024, 512, 0); // ReLU
streaming.add_layer(512, 256, 1);  // Sigmoid

// Process continuous data
let results = streaming.process_stream(&new_data);
```

## Build Instructions

### Native Rust
```bash
cargo build --release --features simd
```

### WebAssembly
```bash
wasm-pack build --target web --out-dir pkg
```

### With optimization
```bash
wasm-pack build --release --target web -- --features simd
```

## Testing

Run comprehensive tests:
```bash
# Native tests
cargo test neural::wasm_nn

# WASM tests (requires wasm-pack)
wasm-pack test --headless --firefox
```

## Examples

See `examples/wasm_neural_demo.rs` for comprehensive usage examples including:
- Basic layer operations
- Multi-layer networks  
- Quantization comparison
- Streaming processing
- Performance benchmarking
- Classification tasks

## Performance Tuning

### For Maximum Speed
- Enable SIMD features: `--features simd`
- Use quantization for large models
- Batch process multiple inputs
- Prefer ReLU activation (fastest)

### For Minimum Size
- Use INT8 quantization
- Compile with `opt-level = "z"`
- Enable LTO: `lto = true`
- Strip debug info: `strip = true`

### For Battery Life (Mobile)
- Use quantized models
- Prefer smaller architectures
- Batch processing to reduce overhead
- Consider INT16 over INT8 for accuracy/speed balance

## Roadmap

- [ ] **WebGPU support** for GPU acceleration in browsers
- [ ] **INT4 quantization** for extreme memory reduction
- [ ] **Dynamic neural architecture search** for optimal models
- [ ] **Federated learning** for privacy-preserving training
- [ ] **ONNX model import** for external model compatibility
- [ ] **TensorFlow.js interop** for model conversion

## Contributing

Contributions welcome! Areas of focus:
- Additional activation functions
- More quantization schemes
- Platform-specific optimizations
- Better browser compatibility
- Trading-specific layers

## License

This neural network implementation is part of the CWTS Ultra trading system.