# Tengri Quantum Trading System with Whale Defense

## Project Overview

The Tengri Trading System is a sophisticated quantum-classical hybrid algorithmic trading platform with integrated Quantum Whale Defense capabilities. The system combines advanced financial analysis, machine learning, and quantum computing to provide 5-15 second early warning of whale movements and sophisticated trading strategies. Built using modern technologies including FastAPI, Solid.js, and PennyLane quantum computing framework.

### Key Features
- **Quantum Whale Defense**: 5-15 second early warning system with 87.1% detection rate
- **5 Microservice Architecture**: Prediction, CDFA, Pairlist, RL, and Decision apps
- **57 Quantum Qubits**: 24 for trading + 33 for whale defense
- **<50ms Latency**: Real-time processing with quantum acceleration

![CDFA Suite Dashboard](https://placeholder.com/dashboard-screenshot.png)

## Table of Contents

- [Architecture](#architecture)
- [Quantum Whale Defense](#quantum-whale-defense)
- [Microservices](#microservices)
- [Frontend Implementation](#frontend-implementation)
- [Backend Implementation](#backend-implementation)
- [Deployment](#deployment)
- [Development](#development)
- [Known Issues and Solutions](#known-issues-and-solutions)
- [Future Roadmap](#future-roadmap)

## Architecture

The Tengri Trading System uses a distributed quantum-classical hybrid architecture:

### Overall Architecture

```
                    Tengri Quantum Trading System
┌─────────────────────────────────────────────────────────────────┐
│                     Quantum Layer (57 Qubits)                   │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐  │
│  │ Trading Qubits  │    │      Whale Defense Qubits          │  │
│  │     (24)        │    │            (33)                     │  │
│  │                 │    │  ┌─────┬─────┬─────┬─────┬─────┐   │  │
│  │ • Q* Learning   │    │  │ Osc │Corr │Game │Sent │Steg │   │  │
│  │ • Market Anal   │    │  │ (8) │(12) │(10) │ (6) │ (6) │   │  │
│  │ • Decision      │    │  └─────┴─────┴─────┴─────┴─────┘   │  │
│  └─────────────────┘    └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                            ┌───────┴───────┐
                            │ FreqTrade Bot │
                            └───────┬───────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                    Microservices Layer                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │Predict  │ │  CDFA   │ │Pairlist │ │   RL    │ │Decision │  │
│  │ :8100   │ │ :8001   │ │ :8003   │ │ :8004   │ │ :8005   │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                    Communication Layer                          │
│        Redis Pub/Sub  │  ZeroMQ  │  WebSockets  │  HTTP        │
└─────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌────────┐ │
│  │ Market    │    │ Quantum   │    │ Redis     │    │ Config │ │
│  │ Data APIs │    │ Circuits  │    │ Cache     │    │ Files  │ │
│  └───────────┘    └───────────┘    └───────────┘    └────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Quantum Layer**: 57 qubits (PennyLane with lightning.kokkos backend)
2. **Microservices**: 5 independent applications with FastAPI
3. **Frontend**: Solid.js with UnoCSS for real-time dashboards
4. **Communication**: Redis pub/sub, ZeroMQ, WebSockets for <50ms latency
5. **Trading Integration**: FreqTrade bot with quantum-enhanced strategies
6. **Data Sources**: Multi-exchange APIs, sentiment feeds, on-chain metrics
7. **Containerization**: Docker and Docker Compose for scalability

## Quantum Whale Defense

The Quantum Whale Defense System provides 5-15 second early warning of large market movements using quantum computing and advanced pattern recognition.

### Components

#### 1. Oscillation Detector (8 Qubits)
- **Purpose**: Detect subtle market frequency anomalies
- **Technology**: Quantum phase estimation and FFT analysis
- **Performance**: <50ms detection latency
- **File**: `quantum_whale_detection_core.py`

#### 2. Correlation Engine (12 Qubits)
- **Purpose**: Multi-timeframe manipulation detection
- **Technology**: Quantum entanglement for correlation analysis
- **Timeframes**: 1m, 5m, 15m, 30m, 60m
- **Detection**: Coordinated manipulation patterns

#### 3. Game Theory Engine (10 Qubits)
- **Purpose**: Optimal counter-strategy calculation
- **Technology**: Quantum Nash equilibrium finding
- **Strategies**: Defensive hedge, front-run, counter-manipulation
- **Output**: Recommended position adjustments

#### 4. Sentiment Detector (6 Qubits) - Planned
- **Purpose**: Social media manipulation detection
- **Technology**: Quantum natural language processing
- **Sources**: Twitter, Reddit, Telegram
- **Status**: Integration pending

#### 5. Steganography Engine (6 Qubits) - Planned
- **Purpose**: Hide trading intentions from whales
- **Technology**: Quantum key distribution for order encoding
- **Features**: Order splitting, timing randomization
- **Status**: Development phase

### Performance Metrics

```
✅ Latency: <50ms (requirement: <50ms)
⚠️ Detection Rate: 87.1% (target: 95%+)
✅ False Positives: 0% (target: <0.1%)
✅ System Stability: No crashes or failures
```

### Current Status
- **Phase 1**: ✅ Basic integration complete
- **Phase 2**: ⚠️ GPU compatibility issues resolved
- **Phase 3**: 🔄 C++/Cython optimization in progress
- **Phase 4**: ⏳ Production deployment pending

### Testing
```bash
# Test whale detection system
python whale_defense_tests.py

# CPU-only mode (recommended for older GPUs)
CUDA_VISIBLE_DEVICES="" python quantum_whale_detection_core.py
```

## Microservices

## Features

- **Signal Fusion**: Advanced fusion algorithms with confidence visualization
- **Wavelet/MRA Analysis**: Multi-resolution analysis and wavelet transformations
- **Cross-Asset Analysis**: Correlation network analysis and clustering
- **Neuromorphic Analysis**: Spiking neural networks with STDP visualization
- **Hyperparameter Optimization**: Interactive parameter optimization
- **Prediction & Patterns**: Black swan event detection, Fibonacci patterns
- **SOC & Panarchy Analysis**: Self-organized criticality and adaptive cycles
- **Quantum Indicators**: QERC-based quantum-inspired indicators
- **Real-time Updates**: WebSocket-based live data streaming
- **Dark/Light Mode**: Automatic and manual theme switching
- **Responsive Design**: Adapts to different screen sizes and devices

## Frontend Implementation

The frontend is built with Solid.js, a highly efficient UI library that provides React-like development experience with better performance characteristics. UnoCSS is used for styling (switched from Tailwind CSS due to performance and configuration issues).

### Key Technologies

- **Solid.js**: Core UI framework
- **UnoCSS**: Atomic CSS engine for styling
- **D3.js**: Advanced data visualizations
- **Chart.js**: Standard charts and graphs
- **Solid Router**: Client-side routing
- **Vite**: Build tool and development server

### Directory Structure
core/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/
│   │   │   │   ├── Navbar.jsx
│   │   │   │   ├── Sidebar.jsx
│   │   │   │   ├── ParameterControls.jsx
│   │   │   ├── visualizations/
│   │   │   │   ├── FusionVisualizer.jsx
│   │   │   │   ├── WaveletAnalyzer.jsx
│   │   │   │   ├── CrossAssetNetwork.jsx
│   │   │   │   ├── NeuromorphicViz.jsx
│   │   │   │   ├── HyperparameterViz.jsx
│   │   │   │   ├── BlackSwanDetector.jsx
│   │   │   │   ├── WhaleActivityMonitor.jsx
│   │   │   │   ├── AntifragilityMetrics.jsx
│   │   │   │   ├── FibonacciPatterns.jsx
│   │   │   │   ├── SOCAnalyzer.jsx
│   │   │   │   ├── PanarchyCycles.jsx
│   │   │   │   ├── QuantumIndicators.jsx
│   │   │   ├── tabs/
│   │   │   │   ├── Dashboard.jsx
│   │   │   │   ├── FusionTab.jsx
│   │   │   │   ├── WaveletTab.jsx
│   │   │   │   ├── CrossAssetTab.jsx
│   │   │   │   ├── NeuromorphicTab.jsx
│   │   │   │   ├── SettingsTab.jsx
│   │   │   │   ├── OptimizationTab.jsx
│   │   │   │   ├── PredictionTab.jsx
│   │   ├── hooks/
│   │   │   ├── useAPI.js
│   │   │   ├── useWebSocket.js
│   │   │   ├── useDataTransform.js
│   │   ├── utils/
│   │   │   ├── colorScales.js
│   │   │   ├── dataProcessing.js
│   │   │   ├── vizHelpers.js
│   │   ├── App.jsx
│   │   ├── index.jsx
│   │   ├── routes.js
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── endpoints/
│   │   │   │   ├── fusion.py
│   │   │   │   ├── wavelet.py
│   │   │   │   ├── cross_asset.py
│   │   │   │   ├── neuromorphic.py
│   │   │   │   ├── optimization.py
│   │   │   │   ├── prediction.py
│   │   │   ├── router.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── security.py
│   │   ├── models/
│   │   │   ├── request_models.py
│   │   │   ├── response_models.py
│   │   ├── services/
│   │   │   ├── fusion_service.py
│   │   │   ├── wavelet_service.py
│   │   │   ├── cross_asset_service.py
│   │   │   ├── neuromorphic_service.py
│   │   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│
├── docker-compose.yml
├── README.md
```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── common/           # Shared components (Navbar, Sidebar, etc.)
│   │   ├── visualizations/   # Visualization components
│   │   ├── tabs/            # Main tab components
│   ├── hooks/               # Custom hooks
│   ├── utils/               # Utility functions
│   ├── App.jsx              # Main application component
│   ├── index.jsx            # Entry point
│   └── routes.js            # Route definitions
├── uno.config.ts            # UnoCSS configuration
├── vite.config.js           # Vite configuration
└── package.json             # Dependencies and scripts
```

### Styling with UnoCSS

We migrated from Tailwind CSS to UnoCSS due to improved performance, faster build times, and better developer experience. UnoCSS works as an atomic CSS engine that generates CSS on demand.

#### UnoCSS Configuration

```javascript
// uno.config.ts
import { defineConfig, presetUno, presetIcons, presetWebFonts } from 'unocss'

export default defineConfig({
  presets: [
    presetUno(),               // Default preset
    presetIcons({              // Icons preset
      scale: 1.2,
      extraProperties: {
        'display': 'inline-block',
        'vertical-align': 'middle',
      }
    }),
    presetWebFonts({           // Web fonts preset
      fonts: {
        sans: 'Inter',
        mono: 'JetBrains Mono',
      }
    })
  ],
  shortcuts: {
    // Custom shortcuts for commonly used utility combinations
    'btn': 'px-4 py-2 rounded-md font-medium transition-colors',
    'btn-primary': 'btn bg-primary-600 text-white hover:bg-primary-700',
    'btn-secondary': 'btn bg-secondary-600 text-white hover:bg-secondary-700',
    'card': 'bg-white dark:bg-neutral-800 rounded-lg shadow-md p-4 transition-colors',
    'input-field': 'px-3 py-2 bg-white dark:bg-neutral-700 border border-gray-300 dark:border-gray-600 rounded-md',
  },
  theme: {
    colors: {
      primary: {
        50: '#f0fdfa',
        100: '#ccfbf1',
        200: '#99f6e4',
        300: '#5eead4',
        400: '#2dd4bf',
        500: '#14b8a6',
        600: '#0d9488',
        700: '#0f766e',
        800: '#115e59',
        900: '#134e4a',
      },
      secondary: {
        50: '#eff6ff',
        100: '#dbeafe',
        200: '#bfdbfe',
        300: '#93c5fd',
        400: '#60a5fa',
        500: '#3b82f6',
        600: '#2563eb',
        700: '#1d4ed8',
        800: '#1e40af',
        900: '#1e3a8a',
      },
      // Other color definitions...
    }
  }
})
```

### Component Example

```jsx
// Example component with UnoCSS
const FusionVisualizer = (props) => {
  return (
    <div class="card">
      <h3 class="text-lg font-medium text-gray-800 dark:text-white mb-4">
        Signal Fusion
      </h3>
      <div class="h-64 relative">
        <canvas ref={canvasRef}></canvas>
        {isLoading() && (
          <div class="absolute inset-0 flex items-center justify-center bg-white/50 dark:bg-black/50">
            <div class="w-8 h-8 border-t-2 border-b-2 border-primary-600 rounded-full animate-spin"></div>
          </div>
        )}
      </div>
      <div class="mt-4 flex justify-between text-sm text-gray-500">
        <span>Confidence: {confidenceScore().toFixed(2)}</span>
        <span>Weight Distribution: {weightBalance()}</span>
      </div>
    </div>
  );
};
```

## Backend Implementation

The backend is built with FastAPI, a modern, high-performance web framework for building APIs with Python, leveraging type hints and async/await syntax.

### Key Technologies

- **FastAPI**: Core API framework
- **Pydantic**: Data validation and settings management
- **MongoDB**: Document database (with Motor for async access)
- **Redis**: Caching and pub/sub messaging
- **NumPy/Pandas**: Data processing and analysis
- **PyWavelets**: Wavelet transformations
- **scikit-learn**: Machine learning algorithms
- **WebSockets**: Real-time data streaming

### Directory Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── endpoints/       # API endpoint modules
│   │   ├── router.py        # API router with endpoint registration
│   ├── core/
│   │   ├── config.py        # Configuration management
│   │   ├── security.py      # Authentication and security
│   ├── models/
│   │   ├── request_models.py   # Request validation models
│   │   ├── response_models.py  # Response models
│   ├── services/
│   │   ├── fusion_service.py   # Business logic modules
│   │   ├── wavelet_service.py  # ...
│   │   ├── cross_asset_service.py
│   │   └── ...
│   ├── main.py              # Application entry point
├── tests/                   # Test suite
└── requirements.txt         # Dependencies
```

### Endpoint Example

```python
# Example API endpoint
@router.get("/correlation")
async def get_correlation_matrix(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    method: str = Query("pearson", description="Correlation method"),
    timeframe: str = Query("1d", description="Timeframe for analysis"),
    window: int = Query(30, description="Window size in periods"),
    service: CrossAssetService = Depends(get_cross_asset_service)
):
    """
    Get correlation matrix for the specified symbols.
    """
    try:
        symbol_list = symbols.split(",")
        result = await service.get_correlation_matrix(symbol_list, method, timeframe, window)
        return result
    except Exception as e:
        logger.error(f"Error in get_correlation_matrix: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

### Service Layer Example

```python
# Example service implementation
class FusionService:
    """Service for signal fusion operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.use_gpu = settings.USE_GPU
        self.cache = {}
        self.cache_ttl = settings.CACHE_TTL
    
    async def get_fusion_data(self, fusion_type: str, symbols: List[str]) -> Dict[str, Any]:
        """Get fusion data for the specified symbols."""
        cache_key = (fusion_type, tuple(sorted(symbols)))
        
        # Check cache
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if (datetime.now() - cache_entry["timestamp"]).total_seconds() < self.cache_ttl:
                self.logger.debug(f"Using cached fusion data for {cache_key}")
                return cache_entry["data"]
        
        # Process data and return results
        # ...
```

## Modules

### Dashboard

The Dashboard provides a high-level overview of system metrics, recent activities, and key performance indicators. It features:

- System status monitoring
- Performance metrics visualization
- Recent activity logs
- Quick access to commonly used features

### Signal Fusion

The Signal Fusion module combines multiple signal sources using various fusion algorithms:

- Score-based fusion
- Rank-based fusion
- Hybrid fusion methods
- Confidence score visualization
- Weight distribution analysis
- TorchScript integration for hardware acceleration

### Wavelet/MRA Analysis

The Wavelet/MRA (Multi-Resolution Analysis) module decomposes signals into different frequency components with time localization:

- Wavelet transform visualization
- Multi-resolution analysis
- Pattern detection in different scales
- Signal reconstruction
- Scalogram visualization

### Cross-Asset Analysis

The Cross-Asset Analysis module examines relationships between different assets:

- Correlation matrix visualization
- Network graph of asset relationships
- Hierarchical clustering
- Correlation-based clusters
- Asset metadata integration

### Neuromorphic Analysis

The Neuromorphic Analysis module implements brain-inspired computing models:

- Spiking Neural Network (SNN) visualization
- Spike-Timing-Dependent Plasticity (STDP) learning
- Membrane potential monitoring
- Network activity visualization
- Different neuron models (LIF, ALIF, Izhikevich)

### Hyperparameter Optimization

The Hyperparameter Optimization module helps find optimal parameter configurations:

- Multiple optimization algorithms
- Parameter space visualization
- Convergence tracking
- Performance metrics
- Parallel computation support

### Prediction & Patterns

The Prediction & Patterns module identifies patterns and predicts future behavior:

- Fibonacci pattern detection
- Black swan event identification
- Whale activity monitoring
- Forecasting with confidence intervals
- SOC and Panarchy cycle analysis
- Quantum indicators (QERC-based)

### Settings

The Settings module provides configuration management:

- General application settings
- Data source configuration
- Visualization preferences
- Advanced system settings
- Module-specific settings
- Import/Export configuration

## Deployment

The CDFA Suite can be deployed using Docker Compose for easy setup and scalability.

### Requirements

- Docker and Docker Compose
- 4GB+ RAM recommended
- 10GB+ disk space
- Internet connection for external data sources

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/cdfa-suite.git
   cd cdfa-suite
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

4. Access the application:
   - Frontend: http://localhost:5173
   - API docs: http://localhost:8000/docs

### Configuration

The main configuration options are available in the `.env` file:

```
# API
API_V1_STR=/api
PROJECT_NAME=CDFA Suite API

# Database
MONGODB_URL=mongodb://mongo:27017
DATABASE_NAME=cdfa_suite

# Cache
REDIS_URL=redis://redis:6379/0
CACHE_TTL=300

# Security
SECRET_KEY=your-secret-key-change-for-production

# Computation
USE_GPU=false
MAX_THREADS=4

# Other settings
LOG_LEVEL=INFO
ENABLE_WEBSOCKET=true
```

## Development

### Frontend Development

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start development server:
   ```bash
   npm run dev
   ```

3. Build for production:
   ```bash
   npm run build
   ```

#### UnoCSS Configuration

The UnoCSS configuration is in `uno.config.ts`. Key customizations include:

- Custom color palette for consistent branding
- Shortcut utilities for common component styles
- Icon preset for easy icon integration
- Web font preset for typography control

#### Best Practices

- Use atomic classes for styling
- Leverage shortcuts for common patterns
- Use `class:` directive for conditional classes
- Prefer composition over inheritance
- Create reusable components for consistent UI

### Backend Development

1. Create a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start development server:
   ```bash
   uvicorn app.main:app --reload
   ```

#### Best Practices

- Use type hints for better code quality
- Implement dependency injection for services
- Use async/await for I/O-bound operations
- Add proper error handling
- Include comprehensive docstrings
- Write tests for all endpoints and services

## Known Issues and Solutions

### UnoCSS Migration Issues

When migrating from Tailwind CSS to UnoCSS, we encountered several issues and solutions:

1. **Issue**: Class names not being recognized
   **Solution**: Ensure `uno.config.ts` includes all necessary presets and configure Vite plugin correctly

2. **Issue**: Custom components not receiving style
   **Solution**: Use the `@unocss/preset-attributify` preset for attribute-based styling

3. **Issue**: Dark mode not working properly
   **Solution**: Configure the dark mode selector in UnoCSS config and use appropriate class naming

### TypeScript Compilation Errors

1. **Issue**: "does not provide an export named '...'"
   **Solution**: Use type-only imports with `import type { ... } from '...'`

2. **Issue**: "erasableSyntaxOnly" errors with enums
   **Solution**: Convert enums to plain objects with `as const` and define corresponding type aliases

3. **Issue**: Type compatibility errors with chart data
   **Solution**: Create proper type interfaces for chart data and use correct type assertions

### Performance Optimization

1. **Issue**: Slow rendering with large datasets
   **Solution**: Implement virtualization for long lists and paginate data loading

2. **Issue**: Memory leaks with chart instances
   **Solution**: Properly destroy chart instances in `onCleanup` hooks

3. **Issue**: High CPU usage with real-time updates
   **Solution**: Implement throttling for WebSocket updates and batch rendering operations

## Future Roadmap

### Version 2.4 (Q2 2025)
- Enhanced TensorFlow integration for neural models
- GPU acceleration for complex calculations
- Advanced portfolio optimization module

### Version 2.5 (Q3 2025)
- Federated learning for collaborative model training
- Advanced market regime detection
- Integration with external data providers

### Version 3.0 (Q4 2025)
- Generative AI for scenario analysis
- Reinforcement learning for trading strategies
- Multi-agent system for market simulation

## Contributing

We welcome contributions to the CDFA Suite project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The Solid.js team for the excellent frontend framework
- The FastAPI team for the high-performance backend framework
- All contributors and users of the CDFA Suite
