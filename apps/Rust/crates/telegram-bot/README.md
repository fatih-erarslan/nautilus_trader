# Ximera Neural Trading Telegram Bot

A comprehensive Rust-based Telegram bot that provides full access to the Ximera Neural Trading Platform via Telegram messages.

## Features

### 🎯 Trading Operations
- Real-time price quotes and market data
- Portfolio status and performance tracking
- Strategy management and execution
- Top profitable pairs with AI scoring
- Trade execution (simulation mode for safety)

### 🧠 Neural AI & Machine Learning
- Neural model training initiation and monitoring
- AI-powered price forecasting (N-HiTS, N-BEATS, Temporal Fusion)
- Model performance metrics and comparisons
- Training progress tracking with real-time updates

### 💭 Sentiment Analysis
- TCN (Temporal Convolutional Network) sentiment processing
- Real-time news sentiment analysis
- Market sentiment scoring and trends
- News article analysis with impact scoring

### 🤖 MCP (Model Context Protocol) Integration
- Direct access to all 68 MCP commands
- Command execution with parameter support
- GPU-accelerated operations
- Real-time command results

### 🔮 Prediction Markets
- Prediction market data and analysis
- Portfolio tracking for prediction positions
- Market analysis with Kelly criterion calculations
- Arbitrage opportunity detection

### 🔔 Alerts & Notifications
- Customizable price alerts
- Automated market updates
- Strategy performance notifications
- Real-time trading signals

## Quick Start

### 1. Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 2. Clone and Setup
```bash
cd telegram-bot
cp .env.example .env
# Edit .env with your Telegram bot token
```

### 3. Build and Run
```bash
cargo build --release
cargo run
```

## Bot Commands

### Core Commands
- `/help` - Show all available commands
- `/status` - Platform status and health check

### Trading Commands
- `/price BTC/USDT` - Get current price for symbol
- `/portfolio` - Show portfolio status and performance
- `/strategies` - List active trading strategies
- `/trade buy BTC/USDT 0.1` - Execute trade (simulation)
- `/toppairs` - Show top profitable pairs with AI scores

### Neural/AI Commands
- `/train lstm BTC/USDT` - Start neural model training
- `/trainstatus` - Get training progress for all models
- `/forecast BTC/USDT 24h` - Generate neural price forecast
- `/models` - Show model performance metrics

### Sentiment Analysis
- `/sentiment BTC` - Get current sentiment analysis
- `/news BTC` - Analyze recent news sentiment
- `/tcnstatus` - TCN processing status and metrics

### MCP Commands
- `/mcp ping` - Execute any MCP command
- `/mcplist` - List all available MCP commands

### Prediction Markets
- `/prediction btc-50k-2024` - Get prediction market data
- `/predictions` - List trending prediction markets

### Analysis & Backtesting
- `/backtest momentum 30d` - Run strategy backtest
- `/risk` - Portfolio risk analysis
- `/performance` - Performance report

### Alerts & Subscriptions
- `/alert BTC/USDT 50000 above` - Set price alert
- `/alerts` - List active alerts
- `/subscribe prices` - Subscribe to price updates
- `/unsubscribe prices` - Unsubscribe from updates

## Architecture

### Service Layer
- **TradingService**: Handles all trading operations and market data
- **NeuralService**: Manages AI model training and forecasting
- **SentimentService**: Processes news and sentiment analysis
- **MCPService**: Interfaces with the MCP command system
- **PredictionService**: Manages prediction market operations

### Real-time Features
- WebSocket connections to Ximera backend
- Live price updates and alerts
- Training progress monitoring
- Market data streaming

### Safety Features
- Trading simulation mode (no real money by default)
- Rate limiting and user session management
- Admin user controls
- Command validation and parameter checking

## Configuration

### Environment Variables
```bash
TELOXIDE_TOKEN=your_telegram_bot_token
XIMERA_API_URL=http://localhost:8001
RUST_LOG=info
ENABLE_LIVE_TRADING=false  # Safety first!
```

### Telegram Bot Setup
1. Create a bot with @BotFather on Telegram
2. Get your bot token
3. Set the token in `.env` file
4. Run the bot

### Integration with Ximera
The bot connects to the Ximera Neural Trading Platform backend:
- REST API for command execution
- WebSocket for real-time updates
- MCP protocol for neural operations

## Usage Examples

### Get Market Data
```
/price BTC/USDT
```
Response:
```
💰 BTC/USDT
Price: $45,123.45
24h Change: +2.34%
Volume: $25.4M
Last Update: 14:30:25
```

### Start Neural Training
```
/train nhits BTC/USDT
```
Response:
```
🧠 Neural Training Started

Model: nhits
Symbol: BTC/USDT
Training ID: abc12345
Status: Initializing...

Use /trainstatus to monitor progress
```

### Execute MCP Command
```
/mcp neural_forecast BTC/USDT
```
Response:
```
🤖 MCP Command Executed
Command: neural_forecast
Status: ✅ Success

{
  "model": "nhits",
  "forecasts": [...],
  "confidence": 0.89
}
```

## Development

### Adding New Commands
1. Add command to `Command` enum in `main.rs`
2. Implement handler in `handle_command` function
3. Add service method if needed
4. Update help text

### Service Integration
Each service module (`trading.rs`, `neural.rs`, etc.) provides:
- Structured data types
- API integration methods
- Error handling
- Response formatting

## Security Considerations

- Bot tokens should be kept secure
- Live trading is disabled by default
- Rate limiting prevents abuse
- Admin controls for sensitive operations
- All trades are simulated unless explicitly enabled

## Dependencies

- **teloxide**: Telegram bot framework
- **tokio**: Async runtime
- **reqwest**: HTTP client for API calls
- **serde**: Serialization/deserialization
- **chrono**: Date/time handling
- **anyhow**: Error handling

## Monitoring

The bot provides extensive logging and monitoring:
- Command execution tracking
- Error logging and reporting
- Performance metrics
- User activity monitoring
- Service health checks

## License

Part of the Ximera Neural Trading Platform - All Rights Reserved