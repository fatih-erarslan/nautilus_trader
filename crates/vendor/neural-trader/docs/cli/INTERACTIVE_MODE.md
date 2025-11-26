# Neural Trader - Interactive CLI Mode

## Overview

Neural Trader now includes a comprehensive interactive CLI mode with REPL (Read-Eval-Print-Loop), auto-completion, command history, and configuration management.

## Features

### 🎯 Interactive REPL Mode
- **Full-featured shell** with command history
- **Tab auto-completion** for commands, options, and values
- **Syntax highlighting** for better readability
- **Multi-line input** support with `\` continuation
- **Persistent history** (1000 entries saved across sessions)
- **Command suggestions** and interactive help

### ⚙️ Configuration Management
- **Interactive wizard** for guided setup
- **Get/Set/List** operations for config values
- **Import/Export** configurations
- **Validation** with Zod schemas
- **User and project-level** configuration
- **cosmiconfig** integration for flexible config files

### 📜 Command History
- **Search history** with pattern matching
- **Navigate** with up/down arrows
- **Export/Import** history
- **Statistics** and most-used commands

## Commands

### Starting Interactive Mode

```bash
# Start interactive mode
neural-trader interactive

# Or use short alias
neural-trader i

# Disable history
neural-trader interactive --no-history

# Disable colors
neural-trader interactive --no-color
```

### REPL Commands

Once in interactive mode, you can use:

```bash
# REPL control commands
.help                    # Show REPL help
.exit or .quit          # Exit REPL
.clear                  # Clear screen
.history [n]            # Show last n commands
.save <file>            # Save history to file
.load <file>            # Load history from file
.editor                 # Enter multi-line editor mode

# Neural Trader commands (all available)
version                 # Show version info
help [command]          # Show help
list                    # List packages
info <package>          # Package information
init <type>             # Initialize project
configure               # Run configuration wizard
config <cmd>            # Manage configuration
doctor                  # Health checks
test                    # Test bindings
```

### Multi-line Input

Use backslash `\` to continue commands on multiple lines:

```bash
neural-trader> init \
... trading \
... --template pairs-trading
```

### Configuration Commands

#### Interactive Configuration Wizard

```bash
# Run full configuration wizard
neural-trader configure

# Show advanced options
neural-trader configure --advanced

# Update existing configuration
neural-trader configure --update

# Reset to defaults
neural-trader configure --reset
```

#### Configuration Management

```bash
# Get configuration value
neural-trader config get trading.symbols
neural-trader config get trading.risk.maxPositionSize

# Set configuration value
neural-trader config set trading.risk.maxPositionSize 20000
neural-trader config set logging.level debug

# List all configuration
neural-trader config list

# List only project config
neural-trader config list --project

# List only user config
neural-trader config list --user

# Export configuration
neural-trader config export my-config.json
neural-trader config export my-config.yaml --yaml

# Import configuration
neural-trader config import my-config.json

# Merge with existing config
neural-trader config import my-config.json --merge

# Reset configuration
neural-trader config reset

# Show config file paths
neural-trader config path

# Validate configuration
neural-trader config validate
```

## Configuration Structure

### Project Configuration

Project configuration is stored in one of these files (searched in order):
- `.neuraltraderrc`
- `.neuraltraderrc.json`
- `.neuraltraderrc.yaml`
- `.neuraltraderrc.yml`
- `.neuraltraderrc.js`
- `neuraltrader.config.js`
- `config.json`
- `package.json` (in `neuraltrader` field)

### User Configuration

User-level preferences are stored in:
- **Linux/Mac**: `~/.config/neural-trader/config.json`
- **Windows**: `%APPDATA%/neural-trader/config.json`

### Configuration Schema

```json
{
  "name": "my-trading-project",
  "version": "1.0.0",
  "description": "My trading system",
  "trading": {
    "provider": {
      "name": "alpaca",
      "sandbox": true
    },
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "strategies": [
      {
        "name": "momentum",
        "type": "momentum",
        "parameters": {
          "lookback": 20,
          "threshold": 0.02
        },
        "enabled": true
      }
    ],
    "risk": {
      "maxPositionSize": 10000,
      "maxPortfolioRisk": 0.02,
      "stopLossPct": 0.05,
      "maxDrawdown": 0.15
    }
  },
  "neural": {
    "model": "lstm",
    "hiddenSize": 64,
    "learningRate": 0.001,
    "epochs": 100,
    "gpuAcceleration": true
  },
  "backtesting": {
    "startDate": "2023-01-01",
    "endDate": "2024-12-31",
    "initialCapital": 100000,
    "commission": 0.001,
    "slippage": 0.0005
  },
  "accounting": {
    "method": "HIFO",
    "currency": "USD",
    "taxYear": 2024,
    "washSaleTracking": true
  },
  "swarm": {
    "enabled": false,
    "topology": "mesh",
    "maxAgents": 5,
    "coordinationProtocol": "raft"
  },
  "logging": {
    "level": "info",
    "format": "pretty",
    "outputs": ["console"],
    "directory": "./logs"
  },
  "performance": {
    "cacheEnabled": true,
    "parallelExecution": true,
    "workerThreads": 4
  }
}
```

## Auto-Completion

Tab completion works for:

### Commands
```bash
neural-trader> ver<TAB>
neural-trader> version
```

### Options
```bash
neural-trader> config --<TAB>
--user  --project  --json  --yaml  --force
```

### Values
```bash
neural-trader> init <TAB>
trading  backtesting  portfolio  accounting  predictor
```

### File Paths
```bash
neural-trader> config export ./con<TAB>
neural-trader> config export ./config/
```

## Command History

### Navigation
- **↑ (Up Arrow)**: Previous command
- **↓ (Down Arrow)**: Next command
- **Ctrl+R**: Search history (coming soon)

### History Commands

```bash
# Show last 10 commands
.history

# Show last 20 commands
.history 20

# Save history to file
.save my-history.txt

# Load history from file
.load my-history.txt

# Clear all history
config history clear
```

### History File Location
- **Linux/Mac**: `~/.neural-trader-history`
- **Windows**: `%USERPROFILE%\.neural-trader-history`

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Tab` | Auto-complete |
| `↑` / `↓` | Navigate history |
| `Ctrl+C` | Cancel current input / Exit prompt |
| `Ctrl+D` | Exit REPL |
| `Ctrl+L` | Clear screen (some terminals) |
| `\` | Multi-line continuation |

## Examples

### Quick Start

```bash
# Start interactive mode
$ neural-trader interactive

neural-trader> help
# Shows all commands

neural-trader> configure
# Run interactive wizard

neural-trader> init trading
# Initialize trading project

neural-trader> config list
# Show current configuration

neural-trader> .exit
# Exit interactive mode
```

### Configuration Workflow

```bash
# Create configuration
$ neural-trader configure --advanced

# Check configuration
$ neural-trader config list

# Update a value
$ neural-trader config set trading.risk.maxPositionSize 20000

# Validate
$ neural-trader config validate

# Export for backup
$ neural-trader config export backup-config.json
```

### Multi-Project Setup

```bash
# Project 1: Momentum Trading
$ cd ~/projects/momentum-trader
$ neural-trader configure
# ... configure momentum strategy ...

# Project 2: Mean Reversion
$ cd ~/projects/mean-reversion
$ neural-trader configure
# ... configure mean reversion ...

# Each project has its own .neuraltraderrc
```

## Tips & Tricks

### 1. Use Tab Completion
Always press `Tab` to see available options:
```bash
neural-trader> config <TAB><TAB>
get  set  list  reset  export  import  path  validate
```

### 2. Quick Commands
Use command aliases in interactive mode:
```bash
version  → v
help     → h or ?
list     → ls
```

### 3. JSON Output
Pipe output as JSON for scripting:
```bash
$ neural-trader config list --json | jq '.project.trading.symbols'
["AAPL", "MSFT", "GOOGL"]
```

### 4. Configuration Inheritance
User config + Project config = Final config
```bash
# Set user default
$ neural-trader config set logging.level debug --user

# Project inherits but can override
$ neural-trader config set logging.level warn --project
```

### 5. Backup Configurations
```bash
# Export current config
$ neural-trader config export config-$(date +%Y%m%d).json

# Restore later
$ neural-trader config import config-20241117.json
```

## Troubleshooting

### Configuration Not Found
```bash
$ neural-trader config list
⚠ No project configuration found. Run "configure" to create one.

# Solution: Create configuration
$ neural-trader configure
```

### History Not Saving
Check file permissions:
```bash
# Linux/Mac
$ ls -l ~/.neural-trader-history
$ chmod 644 ~/.neural-trader-history

# Windows
$ icacls %USERPROFILE%\.neural-trader-history
```

### Validation Errors
```bash
$ neural-trader config validate
✗ Configuration validation failed:
  trading.risk.maxPositionSize: Must be positive

# Fix the issue
$ neural-trader config set trading.risk.maxPositionSize 10000
```

### REPL Not Starting
Ensure dependencies are installed:
```bash
$ npm install
$ npm ls inquirer cosmiconfig conf zod
```

## Advanced Usage

### Custom Configuration Location
```bash
# Set custom config file
$ NEURAL_TRADER_CONFIG=/path/to/config.json neural-trader interactive
```

### Programmatic Access
```javascript
const ConfigManager = require('neural-trader/src/cli/lib/config-manager');

const manager = new ConfigManager();
await manager.loadProjectConfig();

const symbols = manager.get('trading.symbols');
console.log(symbols); // ['AAPL', 'MSFT', 'GOOGL']
```

### Configuration Validation
```javascript
const { validateConfig } = require('neural-trader/src/cli/lib/config-schema');

try {
  const config = validateConfig(myConfig);
  console.log('Valid!');
} catch (error) {
  console.error('Invalid:', error.message);
}
```

## API Reference

See individual module documentation:
- [REPL](/home/user/neural-trader/src/cli/lib/repl.js)
- [ConfigManager](/home/user/neural-trader/src/cli/lib/config-manager.js)
- [HistoryManager](/home/user/neural-trader/src/cli/lib/history-manager.js)
- [AutoComplete](/home/user/neural-trader/src/cli/lib/auto-complete.js)
- [ConfigWizard](/home/user/neural-trader/src/cli/lib/config-wizard.js)

## Contributing

To add new commands to interactive mode:

1. Register command in AutoComplete
2. Add handler in `interactive.js`
3. Update help text
4. Add to this documentation

## Support

- GitHub Issues: https://github.com/ruvnet/neural-trader/issues
- Documentation: https://github.com/ruvnet/neural-trader/wiki
- Discord: https://discord.gg/neural-trader
