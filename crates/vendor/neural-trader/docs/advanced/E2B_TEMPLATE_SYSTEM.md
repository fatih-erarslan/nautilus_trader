# E2B Template System Documentation

## Overview

The E2B Template System provides a comprehensive framework for creating, managing, and deploying reusable templates for E2B sandboxes. It supports various agent types including trading agents, Claude-Flow swarm orchestration, and Claude Code SPARC methodology implementations.

## Architecture

### Core Components

1. **Template Models** (`models.py`) - Data structures and enums
2. **Template Collections** - Pre-built templates for common use cases
3. **Template Builder** (`template_builder.py`) - Builder pattern for custom templates
4. **Template Registry** (`template_registry.py`) - Template storage and management
5. **Template Deployer** (`template_deployer.py`) - E2B sandbox deployment
6. **Template API** (`template_api.py`) - REST API endpoints

### Template Types

#### Base Templates
- **Python Base** - Python 3.10 environment with data science libraries
- **Node.js Base** - Node.js 20 environment with common packages
- **Trading Agent Base** - Specialized for financial trading applications

#### Claude-Flow Templates
- **Swarm Orchestrator** - Multi-agent coordination with mesh topology
- **Neural Agent** - GPU-accelerated pattern recognition and learning

#### Claude Code Templates
- **SPARC Developer** - Complete SPARC methodology implementation
- **Code Reviewer** - Automated code review with multiple linters

## Quick Start

### 1. Using Pre-built Templates

```python
from src.e2b_templates import TemplateRegistry, TemplateDeployer

# Initialize components
registry = TemplateRegistry()
deployer = TemplateDeployer()

# List available templates
templates = registry.list_templates()
print(f"Available templates: {len(templates)}")

# Deploy a template
deployment = await deployer.deploy_template("python_base")
sandbox_id = deployment["sandbox_id"]

# Execute template
result = await deployer.execute_template(
    sandbox_id,
    input_data={"message": "Hello from template!"}
)
```

### 2. Creating Custom Templates

```python
from src.e2b_templates import TemplateBuilder, TemplateType

# Create Python template
builder = TemplateBuilder.create_python_template(
    "My Custom Agent",
    "Custom trading agent with specific indicators"
)

# Customize requirements
builder.set_requirements(
    python_packages=["yfinance", "ta", "numpy"],
    memory_mb=2048
)

# Set custom script
builder.set_main_script("""
import yfinance as yf
import json
import sys

def main():
    config = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    symbol = config.get('symbol', 'AAPL')
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='1d')
    
    return {
        'symbol': symbol,
        'price': float(data['Close'].iloc[-1]),
        'volume': int(data['Volume'].iloc[-1])
    }

if __name__ == '__main__':
    result = main()
    print(json.dumps(result))
""")

# Build and register
template = builder.build()
registry.register_template("custom_agent", template)
```

### 3. Using the REST API

#### List Templates
```bash
curl http://localhost:8000/e2b/templates/registry/list
```

#### Search Templates
```bash
curl "http://localhost:8000/e2b/templates/registry/search?query=trading"
```

#### Deploy Template
```bash
curl -X POST http://localhost:8000/e2b/templates/deploy \
  -H "Content-Type: application/json" \
  -d '{"template_id": "python_base", "config": {"env_vars": {"DEBUG": "true"}}}'
```

#### Execute Template
```bash
curl -X POST http://localhost:8000/e2b/templates/deploy/sandbox_123/execute \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"symbol": "AAPL", "period": "1mo"}}'
```

## Template Configuration

### Template Structure

```python
@dataclass
class TemplateConfig:
    template_type: TemplateType
    metadata: TemplateMetadata
    requirements: TemplateRequirements
    files: TemplateFiles
    hooks: Optional[TemplateHooks] = None
    claude_flow: Optional[ClaudeFlowConfig] = None
    claude_code: Optional[ClaudeCodeConfig] = None
    trading_agent: Optional[TradingAgentConfig] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)
```

### Runtime Requirements

```python
@dataclass
class TemplateRequirements:
    runtime: RuntimeEnvironment = RuntimeEnvironment.PYTHON_3_10
    cpu_cores: int = 2
    memory_mb: int = 1024
    storage_gb: int = 1
    gpu_enabled: bool = False
    network_access: bool = True
    python_packages: List[str] = field(default_factory=list)
    node_packages: List[str] = field(default_factory=list)
    system_packages: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
```

### Template Files

```python
@dataclass
class TemplateFiles:
    main_script: str
    modules: Dict[str, str] = field(default_factory=dict)
    configs: Dict[str, str] = field(default_factory=dict)
    data: Dict[str, str] = field(default_factory=dict)
    scripts: Dict[str, str] = field(default_factory=dict)
```

## Advanced Features

### 1. Claude-Flow Integration

```python
# Create swarm orchestrator template
builder = TemplateBuilder.create_claude_flow_template(
    "Risk Management Swarm",
    agent_types=["researcher", "analyst", "risk_manager"]
)

# Configure swarm parameters
builder.set_claude_flow_config(
    swarm_topology="hierarchical",
    max_agents=5,
    coordination_mode="collaborative",
    enable_neural=True,
    enable_memory=True
)

template = builder.build()
```

### 2. Trading Agent Configuration

```python
builder = TemplateBuilder.create_trading_template(
    "Momentum Strategy",
    "momentum"
)

builder.set_trading_agent_config(
    strategy_type="momentum",
    symbols=["AAPL", "GOOGL", "MSFT"],
    risk_params={"max_position_size": 0.1, "stop_loss": 0.02},
    execution_mode="paper_trading",
    data_sources=["yfinance", "alpaca"],
    indicators=["RSI", "MACD", "SMA"],
    backtest_enabled=True
)
```

### 3. Lifecycle Hooks

```python
builder.set_hooks(
    pre_install="apt-get update && apt-get install -y curl",
    post_install="pip install --upgrade pip",
    pre_start="echo 'Starting application...'",
    health_check="curl -f http://localhost:8080/health || exit 1",
    cleanup="rm -rf /tmp/cache/*"
)
```

### 4. Template Scaling

```python
# Deploy multiple instances
deployments = await deployer.scale_template(
    "trading_momentum",
    instances=3,
    config={"symbols": ["AAPL", "GOOGL", "MSFT"]}
)

# Each instance gets different configuration
for i, deployment in enumerate(deployments):
    print(f"Instance {i}: {deployment['sandbox_id']}")
```

## Template Examples

### 1. Simple Python Data Processor

```python
builder = (TemplateBuilder()
    .set_type(TemplateType.PYTHON_BASE)
    .set_metadata("Data Processor", "Process CSV data")
    .set_requirements(python_packages=["pandas", "numpy"])
    .set_main_script("""
import pandas as pd
import json
import sys

def process_data(config):
    # Load data
    df = pd.read_csv(config.get('input_file', '/tmp/data.csv'))
    
    # Process
    result = {
        'rows': len(df),
        'columns': list(df.columns),
        'summary': df.describe().to_dict()
    }
    
    return result

if __name__ == '__main__':
    config = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    result = process_data(config)
    print(json.dumps(result, indent=2))
"""))

template = builder.build()
```

### 2. Trading Signal Generator

```python
builder = TemplateBuilder.create_trading_template(
    "RSI Signal Generator",
    "rsi_signals"
)

builder.set_main_script("""
import yfinance as yf
import pandas as pd
import numpy as np
import json
import sys

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_signals(symbol, config):
    # Fetch data
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=config.get('period', '1mo'))
    
    # Calculate RSI
    data['RSI'] = calculate_rsi(data['Close'])
    
    # Generate signals
    signals = []
    current_rsi = data['RSI'].iloc[-1]
    
    if current_rsi < 30:
        signals.append({
            'action': 'buy',
            'reason': 'RSI oversold',
            'strength': (30 - current_rsi) / 30
        })
    elif current_rsi > 70:
        signals.append({
            'action': 'sell',
            'reason': 'RSI overbought',
            'strength': (current_rsi - 70) / 30
        })
    
    return {
        'symbol': symbol,
        'current_price': float(data['Close'].iloc[-1]),
        'rsi': float(current_rsi),
        'signals': signals
    }

def main():
    config = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    symbols = config.get('symbols', ['AAPL'])
    
    results = {}
    for symbol in symbols:
        try:
            results[symbol] = generate_signals(symbol, config)
        except Exception as e:
            results[symbol] = {'error': str(e)}
    
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
""")
```

### 3. Claude-Flow Swarm Template

```python
builder = TemplateBuilder.create_claude_flow_template(
    "Research Swarm",
    agent_types=["researcher", "analyst", "synthesizer"]
)

builder.set_claude_flow_config(
    swarm_topology="mesh",
    max_agents=6,
    coordination_mode="collaborative",
    enable_neural=True,
    hooks={
        "pre-task": "npx claude-flow@alpha hooks pre-task",
        "post-task": "npx claude-flow@alpha hooks post-task"
    }
)
```

## Error Handling

### Common Issues

1. **Template Validation Errors**
```python
try:
    template = builder.build()
except ValueError as e:
    print(f"Template validation failed: {e}")
    errors = builder.validate()
    for error in errors:
        print(f"- {error}")
```

2. **Deployment Failures**
```python
try:
    deployment = await deployer.deploy_template("my_template")
except Exception as e:
    print(f"Deployment failed: {e}")
    # Check template exists
    template = registry.get_template("my_template")
    if not template:
        print("Template not found in registry")
```

3. **Execution Errors**
```python
result = await deployer.execute_template(sandbox_id, input_data=data)
if result["status"] == "error":
    print(f"Execution failed: {result['error']}")
    
    # Check health
    health = await deployer.health_check(sandbox_id)
    print(f"Sandbox health: {health}")
```

## Best Practices

### 1. Template Design
- Keep templates focused on single responsibilities
- Use meaningful names and descriptions
- Include comprehensive error handling
- Validate input data
- Provide clear output formats

### 2. Resource Management
- Set appropriate CPU and memory limits
- Use specific package versions
- Clean up temporary files
- Implement health checks

### 3. Security
- Validate all inputs
- Avoid hardcoded secrets
- Use environment variables for configuration
- Implement proper logging

### 4. Performance
- Cache dependencies when possible
- Use GPU acceleration for ML workloads
- Implement connection pooling
- Monitor resource usage

## Monitoring and Debugging

### 1. Deployment Status
```python
# Check deployment status
status = deployer.get_deployment_status(sandbox_id)
print(f"Status: {status['status']}")
print(f"Deployed: {status['deployed_at']}")

# Get deployment statistics
stats = deployer.get_deployment_stats()
print(f"Total deployments: {stats['total_deployments']}")
```

### 2. Health Monitoring
```python
# Regular health checks
health = await deployer.health_check(sandbox_id)
if health['status'] != 'healthy':
    print(f"Health issue: {health['message']}")
    
    # Attempt recovery
    await deployer.update_template(sandbox_id, template_id)
```

### 3. Performance Monitoring
```python
# Track execution times
import time
start_time = time.time()

result = await deployer.execute_template(sandbox_id, input_data=data)

execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.2f}s")
```

## API Reference

### Registry Endpoints
- `GET /e2b/templates/registry/list` - List templates
- `GET /e2b/templates/registry/search` - Search templates
- `GET /e2b/templates/registry/template/{id}` - Get template details
- `DELETE /e2b/templates/registry/{id}` - Delete template

### Builder Endpoints
- `POST /e2b/templates/builder/python` - Create Python template
- `POST /e2b/templates/builder/trading` - Create trading template
- `POST /e2b/templates/builder/claude-flow` - Create Claude-Flow template

### Deployment Endpoints
- `POST /e2b/templates/deploy` - Deploy template
- `POST /e2b/templates/deploy/{id}/execute` - Execute template
- `PUT /e2b/templates/deploy/{id}/update` - Update deployment
- `DELETE /e2b/templates/deploy/{id}` - Cleanup deployment
- `GET /e2b/templates/deploy/{id}/status` - Get deployment status
- `GET /e2b/templates/deploy/{id}/health` - Health check

## Integration Examples

### 1. With FastAPI Application
```python
from fastapi import FastAPI
from src.e2b_templates import template_router

app = FastAPI()
app.include_router(template_router)

# Now templates are available at /e2b/templates/*
```

### 2. With Background Tasks
```python
from fastapi import BackgroundTasks

async def deploy_and_run(template_id: str, config: dict):
    deployment = await deployer.deploy_template(template_id, config)
    result = await deployer.execute_template(
        deployment["sandbox_id"],
        input_data=config
    )
    return result

@app.post("/run-template")
async def run_template(template_id: str, config: dict, background_tasks: BackgroundTasks):
    background_tasks.add_task(deploy_and_run, template_id, config)
    return {"status": "started"}
```

### 3. With Streaming Results
```python
import asyncio
from fastapi.responses import StreamingResponse

async def stream_template_results(template_id: str, config: dict):
    deployment = await deployer.deploy_template(template_id, config)
    sandbox_id = deployment["sandbox_id"]
    
    while True:
        result = await deployer.execute_template(sandbox_id, input_data=config)
        yield f"data: {json.dumps(result)}\\n\\n"
        await asyncio.sleep(1)

@app.get("/stream-template")
async def stream_template(template_id: str):
    return StreamingResponse(
        stream_template_results(template_id, {}),
        media_type="text/plain"
    )
```

## Conclusion

The E2B Template System provides a powerful and flexible framework for creating, managing, and deploying sophisticated agent templates. It supports a wide range of use cases from simple data processing to complex multi-agent swarm orchestration, making it ideal for building scalable AI-driven applications.

For additional support and examples, refer to the test suite in `tests/test_e2b_templates.py` and the source code documentation.