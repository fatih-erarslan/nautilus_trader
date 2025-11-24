# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

The Tengri Trading System is a sophisticated quantum-classical hybrid algorithmic trading platform consisting of **5 microservice applications** with integrated **Quantum Whale Defense System**:

1. **Prediction App** - Superior Prediction Engine (Port 8040)
2. **CDFA App** - Cognitive Diversity Fusion Analysis (Port 8020)  
3. **Pairlist App** - Dynamic pair selection (Port 8030)
4. **Data App** - Centralized market data (Port 8010)
5. **RL App** - Reinforcement Learning (Port 8004)
6. **Decision App** - Quantum decision making (Port 8005)

### Quantum Whale Defense System (5-15 Second Early Warning)
- **Total Qubits**: 57 (24 base trading + 33 whale defense)
- **Detection Components**: Oscillation detector, correlation engine, game theory
- **Performance**: <50ms latency, 87.1% test success rate
- **Status**: Partially operational, C++/Cython optimization in progress

## Architecture

### Technology Stack
- **Backend**: Python 3.8+ with FastAPI
- **Frontend**: TypeScript with Solid.js, Vite, UnoCSS
- **ML/Quantum**: PyTorch, PennyLane, River ML, Numba JIT
- **Communication**: Redis pub/sub, ZeroMQ messaging, WebSockets
- **Hardware Acceleration**: CUDA/GPU with CPU fallbacks

### Communication Flow
```
Market Data → [CDFA/Prediction/RL Apps] → Decision App → FreqTrade
                        ↓
                Redis/ZeroMQ messaging
```

## Key Development Commands

### Starting Services
```bash
# Individual applications
cd tengri/prediction_app && ./start.sh      # Port 8100
cd tengri/pairlist_app && ./start.sh        # Port 8003
python tengri_integration.py                # Integrated system

# Stop services
cd tengri/prediction_app && ./stop.sh
cd tengri/pairlist_app && ./stop.sh
```

### Development Workflow
```bash
# Backend development
cd tengri/prediction_app
python server.py                            # Direct server start
python -m pytest tests/                     # Run tests
black prediction_app/                       # Code formatting
mypy prediction_app/                        # Type checking

# Frontend development  
cd tengri/prediction_app/frontend
npm run dev                                  # Development server
npm run build                               # Production build
npm run lint                                # ESLint
npm run type-check                          # TypeScript checking
```

### Testing and Quality
```bash
# Python testing
pytest tests/                               # All tests
pytest tests/test_specific.py              # Single test file

# Frontend testing
cd frontend && npm run lint                 # Linting
cd frontend && npm run type-check           # Type checking
```

## Critical Architecture Patterns

### Multi-Layer Prediction Engine
The Prediction App uses a 4-layer architecture:
1. **Feature Extraction**: TopologicalDataAnalyzer, TemporalPatternAnalyzer, MarketDetectors
2. **Quantum Enhancement**: QuantumProspectTheory (7 circuits), QERC, CerebellarSNN
3. **Adaptive Learning**: Q* Learning with RiverML, LSTM-Transformer, DriftDetector
4. **Decision Integration**: PADS ensemble, NarrativeForecaster, RiskManagement

### Hardware Acceleration Strategy
- **Numba JIT compilation** for performance-critical code
- **GPU acceleration** with automatic CPU fallback
- **Quantum device selection** (lightning.kokkos > lightning.qubit > default.qubit)
- **Thread pool execution** for non-blocking operations
- **Note**: Lightning.gpu backend deprecated due to GPU compatibility issues (CUDA 6.1 unsupported)

### Microservice Communication
- **HTTP APIs** for direct service communication
- **Redis pub/sub** for event broadcasting
- **ZeroMQ** for high-performance messaging
- **WebSockets** for real-time frontend updates

### Configuration Management
- Main config: `/config/tengri_integration.json`
- App-specific configs in each app's `configs/` directory
- Environment variables for sensitive data (API keys, tokens)
- Hardware detection and automatic optimization

## Development Guidelines

### Python Code Patterns
- Use **async/await** for all I/O operations
- Implement **graceful degradation** for optional dependencies
- Add **comprehensive logging** with appropriate levels
- Use **Numba JIT** decorators for performance-critical functions
- Handle **hardware acceleration failures** with CPU fallbacks

### Frontend Patterns (Solid.js)
- Use **createSignal/createEffect** for reactivity
- Wrap all store initialization in **createRoot()**
- Use **lazy loading** for route components
- Implement **error boundaries** for robust UIs
- Use **UnoCSS** for styling with atomic classes

### Error Handling
- **Hardware failures**: Automatic fallback to CPU/classical methods
- **Service unavailability**: Graceful degradation with fallback data
- **Import errors**: Optional dependency handling with warnings
- **Connection failures**: Retry logic with exponential backoff

### Configuration Access
```python
# Load system configuration
from config.loader import load_config
config = load_config('config/tengri_integration.json')

# App-specific config
config = load_config('configs/prediction_config.json')
```

### Service Health Checks
All services expose `/health` endpoints:
```python
# Check service health
async with httpx.AsyncClient() as client:
    response = await client.get("http://localhost:8100/health")
    health = response.json()
```

## File Structure Key Points

### Core Components
- `/tengri/` - Main application directory
- `/config/` - System-wide configuration files
- `/analyzers/` - Market analysis components
- `/detectors/` - Pattern detection algorithms
- `/models/` - ML model definitions and state files
- `/quantum_whale_detection_core.py` - Main whale defense system
- `/whale_defense_tests.py` - Test suite for whale detection
- `/quantum_knowledge_system/` - Integrated system documentation

### Application Structure (per app)
```
app_name/
├── server.py              # FastAPI server
├── core.py                # Business logic
├── client.py              # Client interface
├── start.sh / stop.sh     # Service management
├── configs/               # App configuration
├── frontend/              # Solid.js frontend
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── logs/                  # Application logs
└── requirements.txt       # Python dependencies
```

### Quantum Integration
- Use **PennyLane** for quantum circuits
- **Hardware manager** handles device selection
- **Quantum-classical hybrid** architectures throughout
- **7 distinct quantum applications** in prediction engine

### Performance Optimization
- **Numba compilation** with environment variable controls
- **GPU acceleration** detection and utilization
- **Caching strategies** with Redis
- **Asynchronous processing** with thread pools

## Common Debugging

### Environment Variables
```bash
# Disable JIT compilation for debugging
NUMBA_DISABLE_JIT=1 python server.py

# Disable CUDA for CPU-only execution  
CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 python server.py

# Enable debug logging
DEBUG=1 python server.py
```

### Quantum Whale Defense Testing
```bash
# Test whale detection system
python whale_defense_tests.py

# Run with CPU-only mode (recommended for older GPUs)
CUDA_VISIBLE_DEVICES="" python quantum_whale_detection_core.py
```

### Log Locations
- Prediction App: `tengri/prediction_app/logs/prediction_server.log`
- Pairlist App: `tengri/pairlist_app/logs/pairlist_server.log`
- Integration: `tengri_integration.log`
- Whale Defense: `whale_defense.log`

### Common Issues
- **Import errors**: Ensure parent directory in Python path
- **CUDA hanging**: Use CPU-only mode with environment variables
- **Quantum circuit wire errors**: Circuit trying to use more qubits than allocated
- **GPU compatibility**: CUDA 6.1 not supported by lightning.gpu, use lightning.kokkos
- **Redis connection**: Optional but recommended for performance
- **Frontend [object Object]**: Usually SolidJS reactivity issues, check createRoot() usage

## Integration Points

### FreqTrade Integration
- **Strategy files** in parent directory implement trading logic
- **Signal generation** through decision app API calls
- **Risk management** via quantum prospect theory
- **Performance tracking** and optimization feedback loops

### External Data Sources
- **Multi-exchange integration**: Binance, OKX, KuCoin, Coinbase
- **Yahoo Finance**: Primary fallback data source
- **Real-time streaming**: WebSocket connections to exchanges
- **Alternative data**: Sentiment, whale alerts, on-chain metrics

The system follows complex adaptive systems principles with self-organization, feedback loops, and multi-paradigm integration for sophisticated market analysis and trading decision-making.

## Quantum Whale Defense System Status

### Current Implementation
- **File**: `quantum_whale_detection_core.py`
- **Test Suite**: `whale_defense_tests.py`
- **Performance**: 87.1% test success rate, <50ms latency
- **Quantum Backend**: Lightning.kokkos (CPU-optimized for compatibility)

### Known Issues
- **Quantum Circuit Wire Allocation**: Some circuits attempt to use wires {8, 9, 10, 11} on 8-qubit devices
- **GPU Compatibility**: CUDA 6.1 unsupported by lightning.gpu backend
- **Detection Sensitivity**: May need further tuning for production use

### Development Status
- **Phase 1**: ✅ Basic integration complete
- **Phase 2**: ⚠️ GPU compatibility issues resolved with CPU fallback
- **Phase 3**: 🔄 C++/Cython optimization in progress
- **Phase 4**: ⏳ Production deployment pending

### Next Steps
1. Complete C++/Cython implementation for performance optimization
2. Resolve quantum circuit wire allocation issues
3. Extensive backtesting on historical whale events
4. Production deployment with reduced position sizing