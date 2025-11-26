# Neural Trader Interactive CLI - Quick Start Guide

## 🚀 Get Started in 60 Seconds

### 1. Start Interactive Mode

```bash
neural-trader interactive
```

### 2. Run Configuration Wizard

```bash
neural-trader> configure
```

Follow the prompts to set up:
- Project information
- Trading provider and symbols
- Risk management parameters
- Optional modules (neural, backtesting, accounting, swarm)

### 3. Initialize Your Project

```bash
neural-trader> init trading
```

### 4. Check Your Setup

```bash
neural-trader> doctor
```

## 📖 Essential Commands

### Interactive Mode

```bash
# Start interactive shell
neural-trader interactive

# Inside the shell:
.help              # Show REPL help
help               # Show neural-trader help
.history           # Show command history
.exit              # Exit interactive mode
```

### Configuration

```bash
# Interactive wizard
neural-trader configure

# Quick config operations
neural-trader config get trading.symbols
neural-trader config set logging.level debug
neural-trader config list
neural-trader config export backup.json
```

### Navigation

```bash
# In interactive mode:
↑/↓               # Navigate history
Tab               # Auto-complete
\                 # Continue command on next line
Ctrl+C            # Cancel / Exit
```

## 🎯 Common Workflows

### First-Time Setup

```bash
# 1. Configure your project
neural-trader configure

# 2. Initialize trading system
neural-trader init trading

# 3. Verify setup
neural-trader doctor

# 4. Start trading
node src/main.js
```

### Update Configuration

```bash
# Interactive mode
neural-trader interactive
neural-trader> config set trading.risk.maxPositionSize 20000
neural-trader> config list

# Or directly
neural-trader config set trading.risk.maxPositionSize 20000
```

### Backup and Restore

```bash
# Backup
neural-trader config export backup-$(date +%Y%m%d).json

# Restore
neural-trader config import backup-20241117.json
```

## 📋 Configuration Structure

Your configuration includes:

- **Trading**: Provider, symbols, strategies, risk management
- **Neural**: Model type, learning parameters, GPU settings
- **Backtesting**: Date ranges, capital, optimization
- **Accounting**: Tax methods, wash sale tracking
- **Swarm**: Multi-agent coordination
- **Logging**: Levels, formats, outputs
- **Performance**: Caching, parallelization

## 💡 Pro Tips

1. **Use Tab Completion**: Press Tab to see available options
2. **Multi-line Commands**: End lines with `\` to continue
3. **Search History**: Use `.history` to find past commands
4. **Quick Edit**: `config set` infers types automatically
5. **Backup Often**: Export configs before major changes

## 🐛 Troubleshooting

### Can't Find Configuration
```bash
neural-trader config path  # Shows config file locations
neural-trader configure    # Creates new configuration
```

### History Not Working
```bash
ls -l ~/.neural-trader-history  # Check file exists
neural-trader interactive --no-history  # Disable temporarily
```

### Validation Errors
```bash
neural-trader config validate  # Check for errors
neural-trader config reset     # Reset to defaults
```

## 📚 Learn More

- **Full Documentation**: [INTERACTIVE_MODE.md](./INTERACTIVE_MODE.md)
- **Implementation Details**: [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)
- **Run Tests**: `node tests/cli/test-interactive-mode.js`

## 🆘 Get Help

```bash
# General help
neural-trader help

# Command-specific help
neural-trader config --help
neural-trader interactive --help

# In interactive mode
help [command]
.help
```

## 🎉 You're Ready!

Start building your trading system:

```bash
neural-trader interactive
neural-trader> configure
neural-trader> init trading
neural-trader> .exit
npm install
node src/main.js
```

Happy trading! 🚀📈
