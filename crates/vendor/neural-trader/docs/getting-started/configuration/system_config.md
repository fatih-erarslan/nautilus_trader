# System Configuration Guide

Comprehensive configuration guide for the AI News Trading Platform with Neural Forecasting capabilities.

## Overview

The AI News Trading Platform uses multiple configuration layers to provide flexibility and maintainability:

- **Environment Variables**: Runtime configuration and secrets
- **YAML Configuration Files**: Structured settings for modules
- **JSON Configuration**: MCP server and API settings
- **Command Line Arguments**: Override default settings

## Configuration Hierarchy

Configuration precedence (highest to lowest):

1. **Command Line Arguments**
2. **Environment Variables**
3. **Local Configuration Files** (`.env`, `config/*.yaml`)
4. **Default Configuration Files** (`config/defaults/`)
5. **Built-in Defaults**

## Environment Variables

### Core System Configuration

```bash
# System Environment
NODE_ENV=production                    # Environment: development, staging, production
CLAUDE_WORKING_DIR=/workspaces/ai-news-trader
CLAUDE_DEV_MODE=false                  # Enable development features
DEBUG=false                            # Enable debug logging

# Neural Forecasting Configuration
NEURAL_FORECAST_GPU=true               # Enable GPU acceleration
NEURAL_FORECAST_DEVICE=cuda            # Device: cuda, cpu, auto
NEURAL_FORECAST_BATCH_SIZE=32          # Default batch size
NEURAL_FORECAST_MAX_MEMORY=8           # Max GPU memory (GB)
NEURAL_FORECAST_CACHE=true             # Enable forecast caching
NEURAL_FORECAST_WORKERS=4              # Data loading workers

# MCP Server Configuration
MCP_SERVER_PORT=3000                   # MCP server port
MCP_SERVER_HOST=0.0.0.0               # MCP server host
MCP_NEURAL_ENABLED=true                # Enable neural forecasting tools
MCP_GPU_ACCELERATION=true              # Enable GPU acceleration
MCP_TIMEOUT=60                         # Request timeout (seconds)
MCP_WORKERS=8                          # Worker processes
MCP_DEBUG_LOGGING=false                # Enable debug logging

# Claude-Flow Configuration
CLAUDE_GPU_ENABLED=true                # Enable GPU support
CLAUDE_NEURAL_FORECASTING=true         # Enable neural forecasting
CLAUDE_MAX_AGENTS=8                    # Maximum concurrent agents
CLAUDE_MEMORY_ENABLED=true             # Enable memory system
CLAUDE_SWARM_MODE=distributed          # Default swarm mode

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/ai_news_trader
DATABASE_POOL_SIZE=20                  # Connection pool size
DATABASE_POOL_TIMEOUT=30               # Pool timeout (seconds)
DATABASE_SSL_MODE=prefer               # SSL mode: disable, allow, prefer, require

# Redis Configuration
REDIS_URL=redis://localhost:6379/0     # Redis connection URL
REDIS_POOL_SIZE=10                     # Connection pool size
REDIS_TIMEOUT=5                        # Command timeout (seconds)
CACHE_DEFAULT_TIMEOUT=3600             # Default cache timeout (seconds)

# API Keys and Secrets
ALPHA_VANTAGE_API_KEY=your_key_here    # Alpha Vantage API key
FINNHUB_API_KEY=your_key_here          # Finnhub API key
OPENAI_API_KEY=your_key_here           # OpenAI API key
NEWS_API_KEY=your_key_here             # News API key
POLYGON_API_KEY=your_key_here          # Polygon.io API key

# Security Configuration
SECRET_KEY=your_secret_key_here        # Application secret key
JWT_SECRET_KEY=your_jwt_secret_here    # JWT signing key
SECURITY_PASSWORD_SALT=salt_here       # Password salt
API_RATE_LIMIT=100                     # Requests per minute
API_RATE_WINDOW=60                     # Rate limit window (seconds)

# Logging Configuration
LOG_LEVEL=INFO                         # Log level: DEBUG, INFO, WARNING, ERROR
LOG_FILE=/var/log/ai-news-trader/app.log
ACCESS_LOG=/var/log/ai-news-trader/access.log
ERROR_LOG=/var/log/ai-news-trader/error.log
LOG_ROTATION=daily                     # Rotation: daily, weekly, monthly
LOG_MAX_SIZE=100MB                     # Max log file size
LOG_BACKUP_COUNT=30                    # Number of backup files

# Performance Configuration
MAX_WORKERS=16                         # Maximum worker processes
WORKER_TIMEOUT=300                     # Worker timeout (seconds)
KEEP_ALIVE=2                          # Keep-alive connections
MAX_REQUESTS=10000                     # Max requests per worker
MAX_REQUESTS_JITTER=100               # Request jitter

# Monitoring Configuration
ENABLE_METRICS=true                    # Enable metrics collection
METRICS_PORT=9090                      # Metrics server port
HEALTH_CHECK_PATH=/health              # Health check endpoint
MONITORING_INTERVAL=30                 # Monitoring interval (seconds)
```

### Development Environment

```bash
# Development-specific settings
NODE_ENV=development
DEBUG=true
CLAUDE_DEV_MODE=true
MCP_DEBUG_LOGGING=true
LOG_LEVEL=DEBUG

# Quick iteration settings
NEURAL_FORECAST_CACHE=false            # Disable caching for testing
NEURAL_FORECAST_MAX_EPOCHS=25          # Faster training
MCP_TIMEOUT=300                        # Longer timeout for debugging

# Development database
DATABASE_URL=sqlite:///ai_news_trader_dev.db
REDIS_URL=redis://localhost:6379/1     # Different Redis database
```

### Production Environment

```bash
# Production-specific settings
NODE_ENV=production
DEBUG=false
CLAUDE_DEV_MODE=false

# Production optimization
NEURAL_FORECAST_BATCH_SIZE=64          # Larger batch size
NEURAL_FORECAST_WORKERS=8              # More workers
MCP_WORKERS=16                         # More MCP workers
MAX_WORKERS=32                         # More application workers

# Production security
API_RATE_LIMIT=50                      # Stricter rate limiting
JWT_EXPIRATION=3600                    # Shorter JWT expiration
SECURITY_HEADERS=true                  # Enable security headers

# Production monitoring
ENABLE_METRICS=true
MONITORING_INTERVAL=10                 # More frequent monitoring
```

## Configuration Files

### Neural Forecasting Configuration

**File:** `config/neural_forecast.yaml`

```yaml
# Neural Forecasting Configuration
models:
  # NHITS Model Configuration
  nhits:
    input_size: 168                    # Input sequence length (1 week)
    horizon: 30                        # Forecast horizon (30 days)
    n_freq_downsample: [168, 24, 1]    # Multi-scale architecture
    stack_types: ["trend", "seasonality"]
    n_blocks: [1, 1]                   # Blocks per stack
    mlp_units: [[512, 512], [512, 512]] # MLP architecture
    interpolation_mode: "linear"        # Interpolation method
    pooling_mode: "MaxPool1d"          # Pooling method
    dropout: 0.1                       # Dropout rate
    activation: "ReLU"                 # Activation function
    
  # NBEATS Model Configuration
  nbeats:
    input_size: 168
    horizon: 30
    stack_types: ["trend", "seasonality"]
    n_blocks: [1, 1]
    mlp_units: [[512, 512], [512, 512]]
    share_weights_in_stack: false
    dropout: 0.1
    
  # NBEATSx Model Configuration (with exogenous variables)
  nbeatsx:
    input_size: 168
    horizon: 30
    stack_types: ["trend", "seasonality"]
    n_blocks: [1, 1]
    mlp_units: [[512, 512], [512, 512]]
    hist_exog_list: ["vix", "interest_rate", "oil_price"]
    futr_exog_list: ["vix", "interest_rate"]
    stat_exog_list: ["sector", "market_cap"]

# Training Configuration
training:
  max_epochs: 100                      # Maximum training epochs
  learning_rate: 0.001                 # Learning rate
  batch_size: 32                       # Batch size
  val_size: 0.2                       # Validation set size
  test_size: 0.1                      # Test set size
  early_stopping_patience: 10          # Early stopping patience
  reduce_lr_patience: 5                # Learning rate reduction patience
  min_lr: 1e-6                        # Minimum learning rate
  weight_decay: 1e-4                   # L2 regularization
  gradient_clip_val: 1.0               # Gradient clipping
  
  # Data loading
  num_workers: 4                       # Data loader workers
  pin_memory: true                     # Pin memory for GPU
  persistent_workers: true             # Keep workers alive
  drop_last: true                      # Drop incomplete batches
  
  # Validation
  val_check_steps: 50                  # Validation frequency
  log_every_n_steps: 10               # Logging frequency
  enable_progress_bar: true            # Show progress bar
  enable_model_summary: true           # Show model summary

# GPU Configuration
gpu:
  enabled: true                        # Enable GPU acceleration
  device_id: 0                        # GPU device ID
  memory_fraction: 0.8                # GPU memory fraction
  allow_growth: true                   # Allow memory growth
  mixed_precision: true                # Enable mixed precision (16-bit)
  compile_model: false                 # PyTorch 2.0 compilation
  
  # Multi-GPU settings
  strategy: "ddp"                      # Distributed strategy
  devices: [0]                        # GPU devices to use
  sync_batchnorm: true                # Sync batch normalization

# Data Processing
data:
  frequency: "D"                       # Data frequency
  missing_value_strategy: "interpolate" # forward_fill, backward_fill, interpolate, drop
  outlier_detection: true              # Enable outlier detection
  outlier_threshold: 3.0              # Z-score threshold
  normalize: true                      # Normalize data
  normalization_method: "robust"       # standard, minmax, robust
  
  # Feature engineering
  add_calendar_features: true          # Add day, month, year features
  add_lag_features: false             # Add lagged features
  lag_periods: [1, 7, 30]            # Lag periods to add
  add_rolling_features: false         # Add rolling statistics
  rolling_windows: [7, 14, 30]       # Rolling windows

# Forecasting
forecasting:
  default_horizon: 30                  # Default forecast horizon
  confidence_levels: [80, 95]         # Confidence intervals
  quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9] # Quantile forecasts
  enable_backtesting: true             # Enable backtesting
  backtest_windows: 5                  # Number of backtest windows
  backtest_step_size: 30              # Step size between windows
  
  # Post-processing
  apply_constraints: true              # Apply forecast constraints
  min_value: null                     # Minimum forecast value
  max_value: null                     # Maximum forecast value
  positive_only: false                # Force positive forecasts

# Model Management
model_management:
  save_best_model: true               # Save best model
  save_last_model: false              # Save last model
  model_checkpoint_frequency: 10       # Checkpoint frequency (epochs)
  max_models_to_keep: 5               # Maximum saved models
  model_versioning: true              # Enable model versioning
  
  # Model paths
  model_dir: "models/"                # Model directory
  checkpoint_dir: "checkpoints/"       # Checkpoint directory
  log_dir: "logs/"                    # Log directory
  
  # Model deployment
  auto_deploy: false                  # Auto-deploy best model
  deployment_threshold: 0.05          # MAPE threshold for deployment
  A_B_testing: true                   # Enable A/B testing

# Evaluation
evaluation:
  metrics: ["mae", "mape", "rmse", "mase", "smape"] # Evaluation metrics
  cross_validation: true              # Enable cross-validation
  cv_folds: 5                        # Cross-validation folds
  seasonal_evaluation: true           # Seasonal evaluation
  residual_analysis: true            # Residual analysis
  
  # Benchmark comparison
  compare_with_baselines: true        # Compare with baseline models
  baseline_models: ["naive", "seasonal_naive", "moving_average", "ets"]
  statistical_tests: true            # Statistical significance tests

# Optimization
optimization:
  hyperparameter_tuning: false       # Enable hyperparameter tuning
  tuning_method: "optuna"            # optuna, ray_tune, grid_search
  n_trials: 100                      # Number of optimization trials
  optimization_metric: "mape"        # Optimization metric
  
  # Search space
  search_space:
    learning_rate: [1e-5, 1e-2]      # Learning rate range
    batch_size: [16, 32, 64, 128]    # Batch size options
    hidden_size: [128, 256, 512, 1024] # Hidden size options
    dropout: [0.0, 0.5]              # Dropout range
    
  # Pruning
  enable_pruning: true               # Enable trial pruning
  pruning_warmup_steps: 10           # Warmup steps before pruning
  pruning_frequency: 5               # Pruning frequency

# Caching
caching:
  enabled: true                      # Enable caching
  cache_forecasts: true              # Cache forecast results
  cache_models: true                 # Cache trained models
  cache_data: true                   # Cache processed data
  cache_ttl: 3600                   # Cache TTL (seconds)
  max_cache_size: "2GB"             # Maximum cache size
  cache_backend: "redis"             # redis, memory, disk
  
# Monitoring
monitoring:
  enabled: true                      # Enable monitoring
  log_metrics: true                  # Log metrics
  track_experiments: true            # Track experiments
  experiment_tracker: "mlflow"       # mlflow, wandb, tensorboard
  
  # Alerts
  performance_alerts: true           # Performance degradation alerts
  accuracy_threshold: 0.1            # MAPE threshold for alerts
  latency_threshold: 1000            # Latency threshold (ms)
  memory_threshold: 0.9              # Memory usage threshold
```

### MCP Server Configuration

**File:** `config/mcp_server.json`

```json
{
  "server": {
    "name": "ai-news-trader-gpu",
    "version": "1.0.0",
    "description": "GPU-accelerated neural forecasting MCP server",
    "host": "0.0.0.0",
    "port": 3000,
    "workers": 8,
    "timeout": 60,
    "max_request_size": "10MB",
    "keepalive_timeout": 30,
    "ssl": {
      "enabled": false,
      "cert_file": null,
      "key_file": null,
      "ca_file": null
    }
  },
  
  "capabilities": {
    "tools": true,
    "resources": true,
    "prompts": true,
    "sampling": false
  },
  
  "neural_forecasting": {
    "enabled": true,
    "default_model": "nhits",
    "default_horizon": 30,
    "max_symbols": 50,
    "cache_forecasts": true,
    "cache_duration": 3600,
    "gpu_acceleration": true,
    "batch_processing": true,
    "max_batch_size": 100
  },
  
  "tools": {
    "ping": {
      "enabled": true,
      "description": "Health check and server status"
    },
    "list_strategies": {
      "enabled": true,
      "description": "List available trading strategies"
    },
    "get_strategy_info": {
      "enabled": true,
      "description": "Get detailed strategy information",
      "cache_duration": 1800
    },
    "quick_analysis": {
      "enabled": true,
      "description": "Quick market analysis with neural forecasting",
      "neural_enhancement": true,
      "gpu_acceleration": true,
      "cache_duration": 300,
      "rate_limit": "100/hour"
    },
    "simulate_trade": {
      "enabled": true,
      "description": "Simulate trading operations",
      "gpu_acceleration": true,
      "risk_checks": true,
      "rate_limit": "50/hour"
    },
    "get_portfolio_status": {
      "enabled": true,
      "description": "Get portfolio status and analytics",
      "include_analytics": true,
      "neural_forecasts": true,
      "cache_duration": 60
    },
    "analyze_news": {
      "enabled": true,
      "description": "AI sentiment analysis of market news",
      "sentiment_model": "enhanced",
      "gpu_acceleration": true,
      "max_lookback_hours": 168,
      "cache_duration": 900,
      "rate_limit": "200/hour"
    },
    "get_news_sentiment": {
      "enabled": true,
      "description": "Real-time news sentiment",
      "sources": ["financial_news", "social_media", "analyst_reports"],
      "cache_duration": 300
    },
    "run_backtest": {
      "enabled": true,
      "description": "Comprehensive historical backtesting",
      "gpu_acceleration": true,
      "include_costs": true,
      "max_period_years": 5,
      "cache_duration": 7200,
      "rate_limit": "10/hour"
    },
    "optimize_strategy": {
      "enabled": true,
      "description": "Strategy parameter optimization",
      "gpu_acceleration": true,
      "max_iterations": 1000,
      "parallel_processing": true,
      "cache_duration": 10800,
      "rate_limit": "5/hour"
    },
    "risk_analysis": {
      "enabled": true,
      "description": "Portfolio risk analysis",
      "monte_carlo": true,
      "gpu_acceleration": true,
      "max_portfolio_size": 100,
      "cache_duration": 1800
    },
    "execute_trade": {
      "enabled": false,
      "description": "Execute live trades",
      "paper_trading_only": true,
      "risk_controls": true,
      "approval_required": true,
      "rate_limit": "20/hour"
    },
    "performance_report": {
      "enabled": true,
      "description": "Generate performance reports",
      "include_benchmark": true,
      "neural_analysis": true,
      "cache_duration": 3600
    },
    "correlation_analysis": {
      "enabled": true,
      "description": "Asset correlation analysis",
      "gpu_acceleration": true,
      "max_symbols": 50,
      "cache_duration": 1800
    },
    "run_benchmark": {
      "enabled": true,
      "description": "System and strategy benchmarks",
      "gpu_acceleration": true,
      "system_benchmarks": true,
      "performance_benchmarks": true,
      "cache_duration": 3600,
      "rate_limit": "5/hour"
    }
  },
  
  "neural_models": {
    "preload": ["nhits", "nbeats"],
    "model_cache_size": "4GB",
    "optimization_level": "high",
    "auto_update": false,
    "version_check": true,
    
    "model_configs": {
      "nhits": {
        "input_size": 168,
        "horizon": 30,
        "batch_size": 32,
        "gpu_memory": "2GB"
      },
      "nbeats": {
        "input_size": 168,
        "horizon": 30,
        "batch_size": 24,
        "gpu_memory": "2.5GB"
      },
      "nbeatsx": {
        "input_size": 168,
        "horizon": 30,
        "batch_size": 16,
        "gpu_memory": "3GB",
        "exog_variables": true
      }
    }
  },
  
  "rate_limiting": {
    "enabled": true,
    "default_limit": "100/hour",
    "burst_limit": "20/minute",
    "storage": "redis",
    "key_function": "ip_and_user",
    
    "limits": {
      "quick_analysis": "200/hour",
      "neural_forecast": "50/hour",
      "backtest": "10/hour",
      "optimize": "5/hour",
      "execute_trade": "20/hour"
    }
  },
  
  "security": {
    "api_key_required": false,
    "jwt_required": false,
    "cors_enabled": true,
    "cors_origins": ["*"],
    "cors_methods": ["GET", "POST", "OPTIONS"],
    "cors_headers": ["Content-Type", "Authorization"],
    
    "request_validation": true,
    "response_validation": true,
    "input_sanitization": true,
    "sql_injection_protection": true,
    "xss_protection": true
  },
  
  "logging": {
    "level": "INFO",
    "format": "json",
    "file": "logs/mcp_server.log",
    "max_size": "100MB",
    "backup_count": 10,
    "rotation": "daily",
    
    "access_log": {
      "enabled": true,
      "file": "logs/mcp_access.log",
      "format": "combined"
    },
    
    "performance_log": {
      "enabled": true,
      "file": "logs/mcp_performance.log",
      "log_slow_requests": true,
      "slow_threshold_ms": 1000
    }
  },
  
  "monitoring": {
    "metrics_enabled": true,
    "metrics_port": 9090,
    "health_check_path": "/health",
    "ready_check_path": "/ready",
    "metrics_path": "/metrics",
    
    "custom_metrics": {
      "neural_forecast_latency": true,
      "gpu_utilization": true,
      "model_accuracy": true,
      "cache_hit_rate": true
    },
    
    "alerts": {
      "enabled": true,
      "webhook_url": null,
      "email_recipients": [],
      
      "conditions": {
        "high_latency": {
          "threshold_ms": 5000,
          "duration_minutes": 5
        },
        "low_accuracy": {
          "threshold_mape": 0.15,
          "duration_minutes": 30
        },
        "high_error_rate": {
          "threshold_percent": 5,
          "duration_minutes": 10
        }
      }
    }
  },
  
  "caching": {
    "enabled": true,
    "backend": "redis",
    "default_ttl": 3600,
    "max_size": "1GB",
    "compression": true,
    
    "cache_policies": {
      "forecasts": {
        "ttl": 1800,
        "invalidate_on_new_data": true
      },
      "strategy_info": {
        "ttl": 3600,
        "invalidate_on_update": true
      },
      "market_analysis": {
        "ttl": 300,
        "per_symbol": true
      }
    }
  },
  
  "database": {
    "url": "${DATABASE_URL}",
    "pool_size": 20,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "echo": false,
    "echo_pool": false,
    
    "retry": {
      "max_attempts": 3,
      "backoff_factor": 2,
      "max_backoff": 60
    }
  },
  
  "external_apis": {
    "alpha_vantage": {
      "api_key": "${ALPHA_VANTAGE_API_KEY}",
      "rate_limit": "5/minute",
      "timeout": 30,
      "retry_attempts": 3
    },
    "finnhub": {
      "api_key": "${FINNHUB_API_KEY}",
      "rate_limit": "60/minute",
      "timeout": 30,
      "retry_attempts": 3
    },
    "polygon": {
      "api_key": "${POLYGON_API_KEY}",
      "rate_limit": "100/minute",
      "timeout": 30,
      "retry_attempts": 3
    }
  }
}
```

### Claude-Flow Configuration

**File:** `.claude/config.yaml`

```yaml
# Claude-Flow Configuration
system:
  # Neural Forecasting
  neural_forecasting:
    enabled: true
    default_model: "nhits"
    gpu_acceleration: true
    max_memory_gb: 8
    preload_models: true
    batch_processing: true
    
  # MCP Server
  mcp_server:
    auto_start: true
    port: 3000
    host: "localhost"
    timeout: 60
    neural_tools: true
    gpu_tools: true
    
  # GPU Configuration
  gpu:
    enabled: true
    device_id: 0
    memory_growth: true
    mixed_precision: true
    auto_select_device: true
    
  # Performance
  performance:
    max_workers: 8
    worker_timeout: 300
    enable_caching: true
    cache_size: "1GB"

# Agent Configuration
agents:
  max_concurrent: 8
  default_timeout: 300
  memory_limit: "2GB"
  
  # Specialized Agents
  neural_forecaster:
    enabled: true
    gpu_access: true
    memory_limit: "4GB"
    model_access: ["nhits", "nbeats", "nbeatsx"]
    
  researcher:
    enabled: true
    internet_access: true
    search_engines: ["google", "bing", "arxiv"]
    
  coder:
    enabled: true
    code_execution: true
    file_access: true
    git_access: true
    
  analyst:
    enabled: true
    data_access: true
    neural_access: true
    visualization: true
    
  trader:
    enabled: true
    paper_trading: true
    live_trading: false
    risk_controls: true

# Memory System
memory:
  enabled: true
  storage_backend: "sqlite"  # sqlite, postgresql, redis
  max_entries: 10000
  compression: true
  encryption: false
  
  # Neural Context
  neural_context:
    enabled: true
    max_forecasts: 1000
    compression: true
    cache_duration: 3600
    auto_cleanup: true
    
  # Session Memory
  session_memory:
    enabled: true
    max_size: "100MB"
    persist_across_restarts: true

# Swarm Coordination
swarm:
  enabled: true
  max_agents: 10
  coordination_mode: "distributed"  # centralized, distributed, hierarchical
  load_balancing: true
  fault_tolerance: true
  
  # Communication
  communication:
    protocol: "websocket"
    encryption: false
    compression: true
    heartbeat_interval: 30
    
  # Resource Allocation
  resources:
    cpu_allocation: "fair"  # fair, priority, weighted
    memory_allocation: "dynamic"
    gpu_sharing: true

# SPARC Development Modes
sparc:
  enabled: true
  default_mode: "orchestrator"
  
  # Mode Configurations
  modes:
    orchestrator:
      neural_access: true
      agent_spawning: true
      memory_access: true
      
    neural_forecaster:
      gpu_priority: true
      model_access: "all"
      data_access: true
      
    researcher:
      internet_access: true
      paper_access: true
      data_collection: true
      
    coder:
      code_execution: true
      file_modification: true
      git_operations: true

# Workflow Management
workflows:
  enabled: true
  max_concurrent: 5
  schedule_enabled: true
  persistence: true
  
  # Default Workflows
  default_workflows:
    daily_analysis:
      enabled: false
      schedule: "0 9 * * 1-5"  # Weekdays at 9 AM
      timeout: 3600
      
    model_retraining:
      enabled: false
      schedule: "0 2 * * 0"    # Sundays at 2 AM
      timeout: 7200

# Security
security:
  authentication:
    enabled: false
    method: "jwt"  # jwt, api_key, oauth
    
  authorization:
    enabled: false
    rbac: false
    
  encryption:
    enabled: false
    algorithm: "AES-256"
    
  audit_logging:
    enabled: true
    log_file: "logs/audit.log"

# Logging
logging:
  level: "INFO"
  format: "structured"  # simple, structured, json
  file: "logs/claude-flow.log"
  max_size: "100MB"
  backup_count: 10
  rotation: "daily"
  
  # Component Logging
  components:
    neural_forecasting: "INFO"
    mcp_server: "INFO"
    agents: "INFO"
    swarm: "INFO"
    memory: "DEBUG"

# Monitoring
monitoring:
  enabled: true
  metrics_port: 9091
  dashboard_port: 8080
  real_time_updates: true
  
  # Metrics Collection
  metrics:
    system_metrics: true
    neural_metrics: true
    agent_metrics: true
    performance_metrics: true
    
  # Alerting
  alerts:
    enabled: true
    channels: ["log", "webhook"]
    thresholds:
      high_memory: 0.9
      high_cpu: 0.8
      agent_failure_rate: 0.1

# Development
development:
  debug_mode: false
  hot_reload: false
  profiling: false
  experiment_tracking: false
  
  # Testing
  testing:
    mock_external_apis: false
    test_data_path: "tests/data/"
    performance_testing: false

# Production
production:
  high_availability: false
  load_balancing: false
  auto_scaling: false
  disaster_recovery: false
  
  # Backup
  backup:
    enabled: false
    schedule: "0 3 * * *"    # Daily at 3 AM
    retention_days: 30
    compression: true

# Integration
integration:
  external_systems:
    trading_platforms: []
    data_providers: []
    notification_systems: []
    
  webhooks:
    enabled: false
    endpoints: []
    authentication: "none"
    
  apis:
    rest_api: true
    graphql_api: false
    websocket_api: true
```

## Configuration Management

### Configuration Validation

The system includes built-in configuration validation:

```python
# config/validator.py
def validate_config():
    """Validate system configuration"""
    
    # Check required environment variables
    required_vars = [
        'NEURAL_FORECAST_GPU',
        'MCP_SERVER_PORT',
        'DATABASE_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ConfigurationError(f"Missing required environment variables: {missing_vars}")
    
    # Validate GPU configuration
    if os.getenv('NEURAL_FORECAST_GPU', 'false').lower() == 'true':
        import torch
        if not torch.cuda.is_available():
            warnings.warn("GPU acceleration enabled but CUDA not available")
    
    # Validate port availability
    mcp_port = int(os.getenv('MCP_SERVER_PORT', 3000))
    if not is_port_available(mcp_port):
        raise ConfigurationError(f"Port {mcp_port} is not available")
```

### Configuration Templates

**Development Template** (`.env.development`):

```bash
# Development Configuration Template
NODE_ENV=development
DEBUG=true

# Neural Forecasting - Development Settings
NEURAL_FORECAST_GPU=true
NEURAL_FORECAST_BATCH_SIZE=16
NEURAL_FORECAST_MAX_EPOCHS=25
NEURAL_FORECAST_CACHE=false

# MCP Server - Development
MCP_SERVER_PORT=3000
MCP_DEBUG_LOGGING=true
MCP_TIMEOUT=300

# Database - Development
DATABASE_URL=sqlite:///ai_news_trader_dev.db
REDIS_URL=redis://localhost:6379/1

# API Keys - Development (use test keys)
ALPHA_VANTAGE_API_KEY=demo
FINNHUB_API_KEY=sandbox_key
```

**Production Template** (`.env.production`):

```bash
# Production Configuration Template
NODE_ENV=production
DEBUG=false

# Neural Forecasting - Production Settings
NEURAL_FORECAST_GPU=true
NEURAL_FORECAST_BATCH_SIZE=64
NEURAL_FORECAST_WORKERS=8
NEURAL_FORECAST_CACHE=true

# MCP Server - Production
MCP_SERVER_PORT=3000
MCP_WORKERS=16
MCP_TIMEOUT=60

# Database - Production
DATABASE_URL=postgresql://user:password@localhost:5432/ai_news_trader_prod
DATABASE_POOL_SIZE=20
REDIS_URL=redis://localhost:6379/0

# Security - Production
SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_here
API_RATE_LIMIT=50

# Monitoring - Production
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

### Configuration Migration

When upgrading, use the configuration migration tool:

```bash
# Migrate configuration from old version
python tools/migrate_config.py --from-version 1.0 --to-version 2.0

# Backup current configuration
python tools/backup_config.py --output config_backup_$(date +%Y%m%d).tar.gz

# Validate new configuration
python tools/validate_config.py --config-dir config/
```

## Best Practices

### Environment Management

1. **Use Environment-Specific Files**:
   - `.env.development`
   - `.env.staging` 
   - `.env.production`

2. **Never Commit Secrets**:
   - Add `.env*` to `.gitignore`
   - Use secret management systems in production

3. **Validate Early**:
   - Validate configuration at startup
   - Fail fast on invalid configuration

### Configuration Security

1. **Encrypt Sensitive Values**:
   ```bash
   # Use encrypted environment variables
   SECRET_KEY=$(echo "your_secret" | base64)
   ```

2. **Use Secret Management**:
   - AWS Secrets Manager
   - HashiCorp Vault
   - Kubernetes Secrets

3. **Rotate Secrets Regularly**:
   - API keys
   - Database passwords
   - JWT signing keys

### Performance Optimization

1. **GPU Configuration**:
   ```bash
   # Optimize for your GPU
   NEURAL_FORECAST_BATCH_SIZE=64    # RTX 3080
   NEURAL_FORECAST_BATCH_SIZE=128   # A100
   ```

2. **Memory Management**:
   ```bash
   # Prevent OOM errors
   NEURAL_FORECAST_MAX_MEMORY=8
   DATABASE_POOL_SIZE=20
   ```

3. **Caching Strategy**:
   ```bash
   # Production caching
   NEURAL_FORECAST_CACHE=true
   CACHE_DEFAULT_TIMEOUT=3600
   ```

## Troubleshooting

### Common Configuration Issues

1. **GPU Not Available**:
   ```bash
   # Check CUDA installation
   nvidia-smi
   
   # Fallback to CPU
   NEURAL_FORECAST_GPU=false
   ```

2. **Port Conflicts**:
   ```bash
   # Find process using port
   lsof -i :3000
   
   # Use different port
   MCP_SERVER_PORT=3001
   ```

3. **Memory Issues**:
   ```bash
   # Reduce batch size
   NEURAL_FORECAST_BATCH_SIZE=16
   
   # Reduce workers
   MCP_WORKERS=4
   ```

4. **Database Connection**:
   ```bash
   # Test connection
   python -c "
   import os
   from sqlalchemy import create_engine
   engine = create_engine(os.getenv('DATABASE_URL'))
   engine.connect()
   print('Database connection successful')
   "
   ```

### Configuration Debugging

```bash
# Debug configuration loading
python -c "
import os
from config import load_config
config = load_config()
print('Configuration loaded successfully')
print(f'Neural forecasting enabled: {config.neural_forecasting.enabled}')
print(f'GPU acceleration: {config.neural_forecasting.gpu_acceleration}')
"

# Check environment variables
env | grep -E "(NEURAL|MCP|CLAUDE)" | sort

# Validate configuration
python tools/validate_config.py --verbose
```

For more configuration help, see:
- [Troubleshooting Guide](../guides/troubleshooting.md)
- [Installation Guide](../guides/installation.md)
- [Deployment Guide](../guides/deployment.md)