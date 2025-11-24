# CLAUDE.md (Updated June 4, 2025)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

The Tengri Trading System is a sophisticated quantum-classical hybrid algorithmic trading platform consisting of **5 microservice applications** with integrated **Quantum Whale Defense System**:

1. **Data App** - Centralized market data management (Port 8002) 
2. **Prediction App** - Superior Prediction Engine (Port 8100)
3. **CDFA App** - Cognitive Diversity Fusion Analysis (Port 8001)  
4. **Pairlist App** - Dynamic pair selection (Port 8003)
5. **Decision App** - Quantum decision making (Port 8005)
6. **RL App** - Reinforcement Learning (Port 8004) [Coming Soon]

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
- **Hardware Acceleration**: CUDA/GPU with CPU fallbacks, C++ extensions

### Communication Flow
```
Market Data → Data App → [CDFA/Prediction/RL Apps] → Decision App → FreqTrade
                  ↓
           Redis/ZeroMQ messaging
```

## Key Development Commands

### Starting Services (Updated)
```bash
# Unified startup (RECOMMENDED)
cd /home/kutlu/freqtrade/user_data/strategies/core/tengri
./start_tengri.sh                           # Simple, reliable startup
./start_all.sh                              # With health checks

# Stop all services
./stop_all.sh                               # Stop everything
./stop_all.sh --clean-logs                  # Also clean logs

# Individual applications (if needed)
cd tengri/data_app && python server.py      # Port 8002 (start first!)
cd tengri/cdfa_app && python cdfa_server_with_frontend.py  # Port 8001
cd tengri/pairlist_app && ./start.sh        # Port 8003
cd tengri/prediction_app && ./start_prediction.sh  # Port 8100
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
npm run dev                                 # Development server
npm run build                               # Production build
npm run lint                                # ESLint
npm run type-check                          # TypeScript checking

# C++ Extensions (CDFA)
cd tengri/cdfa_app
./build_extensions.sh                       # Build with bundled script
# or
/home/kutlu/freqtrade/.venv/bin/python setup.py build_ext --inplace
```

### Testing and Quality
```bash
# Integration test
cd /home/kutlu/freqtrade/user_data/strategies/core/tengri
/home/kutlu/freqtrade/.venv/bin/python test_final_integration.py

# Python testing
pytest tests/                               # All tests
pytest tests/test_specific.py              # Single test file

# Frontend testing
cd frontend && npm run lint                 # Linting
cd frontend && npm run type-check           # Type checking
```

## Critical Architecture Patterns

### Service Dependencies (IMPORTANT)
Services MUST start in this order:
1. **Data App** - Provides market data to all other services
2. **CDFA App** - Depends on Data App for market data
3. **Pairlist App** - Can run independently but uses Data App
4. **Prediction App** - Depends on Data App via DataAppClient

### Multi-Layer Prediction Engine
The Prediction App uses a 4-layer architecture:
1. **Feature Extraction**: TopologicalDataAnalyzer, TemporalPatternAnalyzer, MarketDetectors
2. **Quantum Enhancement**: QuantumProspectTheory (7 circuits), QERC, CerebellarSNN
3. **Adaptive Learning**: Q* Learning with RiverML, LSTM-Transformer, DriftDetector
4. **Decision Integration**: PADS ensemble, NarrativeForecaster, RiskManagement

### Hardware Acceleration Strategy
- **C++ Extensions**: CDFA now has `generate_signal` method (100x speedup)
- **Numba JIT compilation** for performance-critical code
- **GPU acceleration** with automatic CPU fallback
- **Quantum device selection** (lightning.kokkos > lightning.qubit > default.qubit)
- **Thread pool execution** for non-blocking operations
- **Note**: Lightning.gpu backend deprecated due to GPU compatibility issues (CUDA 6.1 unsupported)

### Microservice Communication
- **HTTP APIs** for direct service communication
- **Redis pub/sub** for event broadcasting (optional but recommended)
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
- **C++ extensions** for CDFA (cdfa_core, wavelet_core)
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
- Unified logs: `tengri/logs/`
- Data App: `tengri/logs/data_app.log`
- CDFA App: `tengri/logs/cdfa_app.log`
- Prediction App: `tengri/prediction_app/logs/prediction_server.log`
- Pairlist App: `tengri/pairlist_app/logs/pairlist_server.log`
- Whale Defense: `whale_defense.log`

### Common Issues
- **Service startup order**: Always start Data App first!
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
- **Multi-exchange integration**: OKX (2,431 symbols), Bybit (2,573 symbols), GateIO (5,580 symbols)
- **Blocked exchanges**: Binance, Coinbase, Kraken (network restrictions)
- **Yahoo Finance**: Primary fallback data source
- **Real-time streaming**: WebSocket connections to exchanges
- **Alternative data**: Sentiment, whale alerts, on-chain metrics

The system follows complex adaptive systems principles with self-organization, feedback loops, and multi-paradigm integration for sophisticated market analysis and trading decision-making.

## Recent Updates (June 4, 2025)

### C++ Extension Enhancement
- Added `generate_signal` method to CDFA cdfa_core.cpp
- 100x performance improvement (0.01ms per call)
- Calculates SMA, RSI, VWAP indicators in C++

### Unified Start/Stop Scripts
- **start_tengri.sh**: Simple, reliable startup
- **start_all.sh**: Advanced with health monitoring
- **stop_all.sh**: Comprehensive shutdown

### Service Integration
- All mock/synthetic data removed
- Full Data App integration across all services
- CDFA frontend served directly from backend

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
- **Phase 2**: ✅ GPU compatibility resolved with CPU fallback
- **Phase 3**: ✅ C++/Cython optimization complete for CDFA
- **Phase 4**: ⏳ Production deployment pending

### Next Steps
1. Complete C++/Cython implementation for other components
2. Resolve quantum circuit wire allocation issues
3. Extensive backtesting on historical whale events
4. Production deployment with reduced position sizing
5. Add more apps (RL, Decision) to unified scripts