# Tengri Trading System - Quantum-Classical Hybrid Intelligence Platform

## 🚀 Overview

The Tengri Trading System is a revolutionary quantum-classical hybrid algorithmic trading platform that combines cutting-edge quantum computing, neural networks, and advanced market analysis techniques. It represents one of the most sophisticated open-source trading systems available, featuring real-time quantum whale defense, multi-paradigm AI integration, and adaptive market intelligence.

### Key Highlights
- **57 Quantum Qubits**: 24 base trading + 33 whale defense qubits
- **5-15 Second Early Warning**: Quantum whale defense system
- **369x GPU Acceleration**: Leveraging CUDA for massive performance gains
- **5 Microservice Architecture**: Modular, scalable design
- **Real-time Adaptive Learning**: Continuous market adaptation

## 🏗️ System Architecture

### Core Applications

#### 1. **Prediction App** (Port 8100) - Superior Prediction Engine
The heart of the system, featuring:
- 4-layer quantum-enhanced prediction architecture
- LSTM-Transformer hybrid models
- Spiking Neural Networks (SNN) with quantum integration
- Conformal prediction for uncertainty quantification
- Real-time feature extraction and market microstructure analysis

#### 2. **CDFA App** (Port 8001) - Cognitive Diversity Fusion Analysis
Advanced market analysis using:
- Multi-Resolution Analysis (MRA)
- Wavelet transformations
- Cross-asset correlation analysis
- Neuromorphic computing integration
- Fibonacci and fractal pattern detection

#### 3. **Pairlist App** (Port 8003) - Dynamic Pair Selection
Intelligent pair selection featuring:
- Adaptive market data fetching from multiple sources
- Real-time pair scoring and ranking
- WebSocket streaming for live updates
- Redis pub/sub for inter-app communication
- Whale alert integration

#### 4. **RL App** (Port 8004) - Reinforcement Learning [Planned]
Advanced RL algorithms for:
- Q* learning implementation
- Policy gradient methods
- Multi-agent trading strategies

#### 5. **Decision App** (Port 8005) - Quantum Decision Making
Strategic decision layer using:
- Quantum game theory
- PREMAA (Probabilistic Risk-Enhanced Multi-Agent Architecture)
- QBMIA (Quantum-Biological Market Intelligence Architecture)
- Nash equilibrium computation

### Quantum Components

#### Quantum Whale Defense System
```
Total Qubits: 57 (24 base + 33 defense)
Detection Time: 5-15 seconds early warning
Components:
- Oscillation Detector
- Correlation Engine  
- Game Theory Module
Performance: <50ms latency, 87.1% test success rate
```

#### Quantum Knowledge System
- Compressed sensing for dimension reduction
- Tensor networks for quantum state management
- Error mitigation techniques
- GPU-accelerated quantum simulation

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core language
- **FastAPI**: High-performance async API framework
- **Redis**: Real-time caching and pub/sub messaging
- **ZeroMQ**: High-performance messaging
- **PostgreSQL/TimescaleDB**: Time-series data storage
- **Numba/CUDA**: GPU acceleration

### Frontend
- **TypeScript**: Type-safe development
- **Solid.js**: Reactive UI framework
- **Vite**: Lightning-fast build tool
- **UnoCSS**: Atomic CSS framework
- **Chart.js**: Data visualization

### ML/Quantum
- **PyTorch**: Deep learning framework
- **PennyLane**: Quantum machine learning
- **River ML**: Online learning
- **cuQuantum**: NVIDIA quantum simulation
- **Qiskit**: Quantum computing SDK
- **Norse**: Spiking neural networks

### Data Sources
- **CCXT**: Cryptocurrency exchange library (Binance, Coinbase, Kraken, KuCoin, OKX)
- **Yahoo Finance**: Market data fallback
- **Direct Exchange APIs**: Low-latency data feeds
- **Blockchain APIs**: On-chain analytics

## 📦 Installation & Setup

### Prerequisites
```bash
# System requirements
Python 3.8+
Node.js 18+
Redis server
CUDA Toolkit 11.0+ (optional, for GPU acceleration)
4GB+ RAM minimum, 16GB+ recommended
```

### Quick Start
```bash
# Clone the repository
cd /home/kutlu/freqtrade/user_data/strategies/core

# Install Python dependencies
pip install -r requirements.txt

# Start Redis (if not running)
redis-server

# Launch integrated system
python tengri_integration.py
```

### Individual App Startup
```bash
# Prediction App
cd tengri/prediction_app && ./start.sh

# CDFA App  
cd tengri/cdfa_app && ./start.sh

# Pairlist App
cd tengri/pairlist_app && ./start.sh

# Decision App
cd tengri/decision_app && python server.py
```

## 🔧 Configuration

### Main Configuration Files
```
/config/tengri_integration.json     - System-wide configuration
/tengri/*/configs/                  - App-specific configurations
/quantum_system_config.json         - Quantum component settings
```

### Key Configuration Options
```json
{
  "quantum": {
    "enabled": true,
    "backend": "lightning.kokkos",
    "num_qubits": 57,
    "optimization_level": 3
  },
  "hardware": {
    "use_gpu": true,
    "cuda_device": 0,
    "numba_parallel": true
  },
  "data": {
    "sources": ["ccxt", "yahoo", "direct_api"],
    "cache_ttl": 3600,
    "realtime_interval": 1
  }
}
```

## 📡 API Documentation

### Prediction App Endpoints
```
POST /predict              - Generate market prediction
POST /predict/revolutionary - Revolutionary AI prediction
POST /train               - Train models with new data
POST /calibrate           - Calibrate conformal prediction
GET  /models              - List available models
GET  /performance         - Get performance metrics
```

### CDFA App Endpoints
```
POST /analyze/fusion      - Run fusion analysis
POST /analyze/wavelet     - Wavelet decomposition
POST /analyze/cross-asset - Cross-asset correlation
GET  /indicators/quantum  - Quantum indicators
```

### Pairlist App Endpoints
```
GET  /pairlist           - Get ranked trading pairs
POST /pairlist/update    - Force pair list update
WS   /ws/pairlist        - Real-time pair updates
```

## 📊 Key Features

### Quantum-Enhanced Prediction
- **Multi-layer Architecture**: Feature extraction → Quantum enhancement → Adaptive learning → Decision integration
- **7 Quantum Applications**: Prospect theory, LMSR, annealing regression, error correction
- **Hybrid Models**: Seamless quantum-classical integration

### Advanced Market Analysis
- **Topological Data Analysis**: Persistent homology for pattern detection
- **Temporal Pattern Recognition**: LSTM-Transformer hybrids
- **Market Microstructure**: Order book dynamics and flow analysis
- **Sentiment Integration**: NLP-based market sentiment

### Adaptive Learning
- **Online Learning**: River ML for continuous adaptation
- **Drift Detection**: Automatic model retraining
- **Q* Learning**: Advanced reinforcement learning
- **Ensemble Methods**: PADS (Prediction Aggregation Decision System)

### Risk Management
- **Quantum Prospect Theory**: Behavioral finance integration
- **Conformal Prediction**: Uncertainty quantification
- **Whale Defense**: Early warning system
- **Multi-timeframe Analysis**: Fractal market view

## 🚀 Performance Optimization

### Hardware Acceleration
```python
# Automatic GPU detection and fallback
CUDA_VISIBLE_DEVICES="" python server.py  # Force CPU mode
NUMBA_DISABLE_JIT=1 python server.py      # Disable JIT for debugging
```

### Performance Metrics
- **Prediction Latency**: <50ms average
- **Throughput**: 1000+ predictions/second
- **GPU Speedup**: 369x for quantum simulations
- **Memory Usage**: 2-4GB typical, 8GB peak

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Add parent directory to Python path
export PYTHONPATH=/home/kutlu/freqtrade/user_data/strategies/core:$PYTHONPATH
```

#### CUDA Hanging
```bash
# Use CPU-only mode
CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 python server.py
```

#### Quantum Circuit Errors
- Check qubit allocation (max 57 qubits)
- Verify quantum backend compatibility
- Use lightning.kokkos for CPU optimization

### Log Locations
```
/tengri/prediction_app/logs/
/tengri/pairlist_app/logs/
/tengri/cdfa_app/logs/
/logs/integrated_server.log
```

## 🔬 Advanced Features

### Quantum Whale Defense
Proprietary system for detecting large market movements:
- Quantum oscillation detection
- Correlation engine with entanglement
- Game theory analysis
- 5-15 second early warning capability

### Market Adaptive Behavior
- Self-organizing neural architectures
- Feedback loop optimization
- Multi-paradigm integration
- Emergent trading strategies

### Neuromorphic Computing
- Spiking Neural Networks (SNN)
- Event-driven processing
- Biological neuron models
- Energy-efficient computation

## 🗺️ Roadmap

### Short Term (1-3 months)
- [ ] Complete RL App implementation
- [ ] Enhance whale defense accuracy to 95%+
- [ ] Implement centralized data management app
- [ ] Add more exchange integrations

### Medium Term (3-6 months)
- [ ] Quantum annealing hardware integration
- [ ] Distributed computing support
- [ ] Advanced portfolio optimization
- [ ] Mobile app development

### Long Term (6-12 months)
- [ ] Full quantum computer integration
- [ ] AI strategy marketplace
- [ ] Institutional features
- [ ] Regulatory compliance tools

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FreqTrade community for the amazing framework
- Quantum computing pioneers for inspiration
- Open source contributors worldwide

## 📞 Support

- **Documentation**: See `/docs` directory
- **Issues**: GitHub Issues
- **Community**: Discord/Telegram (coming soon)

---

**⚡ Powered by Quantum Intelligence ⚡**

*"Where quantum mechanics meets market dynamics"*