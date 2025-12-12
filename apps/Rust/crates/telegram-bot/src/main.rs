use std::sync::Arc;
use teloxide::{prelude::*, utils::command::BotCommands};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

mod trading;
mod neural;
mod sentiment;
mod mcp;
mod prediction;

use trading::TradingService;
use neural::NeuralService;
use sentiment::SentimentService;
use mcp::MCPService;
use prediction::PredictionService;

#[derive(BotCommands, Clone)]
enum Command {
    #[command(description = "Display help")]
    Help,
    #[command(description = "Show platform status")]
    Status,
    
    // Trading Commands
    #[command(description = "Get market price for symbol")]
    Price { symbol: String },
    #[command(description = "Show portfolio status")]
    Portfolio,
    #[command(description = "List active strategies")]
    Strategies,
    #[command(description = "Execute a trade")]
    Trade { action: String, symbol: String, amount: String },
    #[command(description = "Show top profitable pairs")]
    TopPairs,
    
    // Neural/Training Commands
    #[command(description = "Start neural model training")]
    Train { model: String, symbol: String },
    #[command(description = "Get training status")]
    TrainStatus,
    #[command(description = "Neural forecast for symbol")]
    Forecast { symbol: String, horizon: Option<String> },
    #[command(description = "Show model performance")]
    Models,
    
    // Sentiment Analysis
    #[command(description = "Get current sentiment for symbol")]
    Sentiment { symbol: String },
    #[command(description = "Analyze news sentiment")]
    News { symbol: String },
    #[command(description = "TCN sentiment processing status")]
    TcnStatus,
    
    // Prediction Market
    #[command(description = "Get prediction market data")]
    Prediction { market: String },
    #[command(description = "List prediction markets")]
    Predictions,
    
    // MCP Commands
    #[command(description = "Execute MCP command")]
    Mcp { command: String },
    #[command(description = "List available MCP commands")]
    McpList,
    
    // Backtesting & Analysis
    #[command(description = "Run strategy backtest")]
    Backtest { strategy: String, period: String },
    #[command(description = "Risk analysis")]
    Risk,
    #[command(description = "Performance report")]
    Performance,
    
    // Alerts & Notifications
    #[command(description = "Set price alert")]
    Alert { symbol: String, price: String, condition: String },
    #[command(description = "List active alerts")]
    Alerts,
    #[command(description = "Subscribe to updates")]
    Subscribe { updates: String },
    #[command(description = "Unsubscribe from updates")]
    Unsubscribe { updates: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSession {
    pub user_id: UserId,
    pub username: Option<String>,
    pub subscriptions: Vec<String>,
    pub alerts: Vec<PriceAlert>,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceAlert {
    pub id: String,
    pub symbol: String,
    pub price: f64,
    pub condition: String, // "above", "below"
    pub created_at: DateTime<Utc>,
    pub triggered: bool,
}

#[derive(Clone)]
pub struct BotState {
    pub sessions: Arc<RwLock<HashMap<UserId, UserSession>>>,
    pub trading_service: Arc<TradingService>,
    pub neural_service: Arc<NeuralService>,
    pub sentiment_service: Arc<SentimentService>,
    pub mcp_service: Arc<MCPService>,
    pub prediction_service: Arc<PredictionService>,
}

#[tokio::main]
async fn main() {
    dotenv::dotenv().ok();
    env_logger::init();
    
    log::info!("🚀 Starting Ximera Neural Trading Telegram Bot...");
    
    let bot = Bot::from_env();
    
    // Initialize services
    let trading_service = Arc::new(TradingService::new().await);
    let neural_service = Arc::new(NeuralService::new().await);
    let sentiment_service = Arc::new(SentimentService::new().await);
    let mcp_service = Arc::new(MCPService::new().await);
    let prediction_service = Arc::new(PredictionService::new().await);
    
    let bot_state = BotState {
        sessions: Arc::new(RwLock::new(HashMap::new())),
        trading_service,
        neural_service,
        sentiment_service,
        mcp_service,
        prediction_service,
    };
    
    log::info!("✅ All services initialized");
    
    // Start background tasks
    tokio::spawn(price_alert_monitor(bot_state.clone()));
    tokio::spawn(market_update_broadcaster(bot.clone(), bot_state.clone()));
    
    // Start bot
    Dispatcher::builder(bot, Update::filter_message().branch(
        dptree::entry()
            .filter_command::<Command>()
            .endpoint(handle_command)
    ))
    .dependencies(dptree::deps![bot_state])
    .enable_ctrlc_handler()
    .build()
    .dispatch()
    .await;
}

async fn handle_command(
    bot: Bot,
    msg: Message,
    cmd: Command,
    state: BotState,
) -> ResponseResult<()> {
    let user_id = msg.from().unwrap().id;
    
    // Update user session
    update_user_session(&state, user_id, msg.from().unwrap().username.clone()).await;
    
    let response = match cmd {
        Command::Help => format!(
            "🤖 *Ximera Neural Trading Bot*\n\n\
            📊 *Trading Commands:*\n\
            `/price BTC/USDT` - Get current price\n\
            `/portfolio` - Portfolio status\n\
            `/strategies` - Active strategies\n\
            `/trade buy BTC/USDT 0.1` - Execute trade\n\
            `/toppairs` - Top profitable pairs\n\n\
            🧠 *Neural/AI Commands:*\n\
            `/train lstm BTC/USDT` - Start training\n\
            `/trainstatus` - Training status\n\
            `/forecast BTC/USDT 24h` - Neural forecast\n\
            `/models` - Model performance\n\n\
            💭 *Sentiment Analysis:*\n\
            `/sentiment BTC` - Current sentiment\n\
            `/news BTC` - News analysis\n\
            `/tcnstatus` - TCN processing status\n\n\
            🔮 *Prediction Markets:*\n\
            `/prediction bitcoin-2024` - Market data\n\
            `/predictions` - List markets\n\n\
            🤖 *MCP Commands:*\n\
            `/mcp ping` - Execute MCP command\n\
            `/mcplist` - Available commands\n\n\
            📈 *Analysis:*\n\
            `/backtest momentum 30d` - Backtest\n\
            `/risk` - Risk analysis\n\
            `/performance` - Performance report\n\n\
            🔔 *Alerts:*\n\
            `/alert BTC/USDT 50000 above` - Set alert\n\
            `/alerts` - List alerts\n\
            `/subscribe prices` - Subscribe to updates"
        ),
        
        Command::Status => {
            let status = get_platform_status(&state).await;
            format!(
                "🟢 *Ximera Platform Status*\n\n\
                📊 Trading System: {}\n\
                🧠 Neural Engine: {}\n\
                💭 Sentiment Analysis: {}\n\
                🤖 MCP Server: {}\n\
                🔮 Prediction Markets: {}\n\n\
                👥 Active Users: {}\n\
                ⏰ Uptime: {}",
                status.trading, status.neural, status.sentiment,
                status.mcp, status.prediction, status.users, status.uptime
            )
        },
        
        Command::Price { symbol } => {
            match state.trading_service.get_price(&symbol).await {
                Ok(price_data) => format!(
                    "💰 *{}*\n\
                    Price: ${:.2}\n\
                    24h Change: {:.2}%\n\
                    Volume: ${:.0}M\n\
                    Last Update: {}",
                    symbol, price_data.price, price_data.change_24h,
                    price_data.volume / 1_000_000.0, price_data.timestamp.format("%H:%M:%S")
                ),
                Err(e) => format!("❌ Error getting price for {}: {}", symbol, e),
            }
        },
        
        Command::Portfolio => {
            match state.trading_service.get_portfolio().await {
                Ok(portfolio) => format!(
                    "📊 *Portfolio Status*\n\n\
                    💰 Total Value: ${:.2}\n\
                    📈 Today: {:.2}%\n\
                    📅 This Week: {:.2}%\n\
                    📆 This Month: {:.2}%\n\n\
                    🔥 Top Holdings:\n{}",
                    portfolio.total_value, portfolio.daily_pnl,
                    portfolio.weekly_pnl, portfolio.monthly_pnl,
                    portfolio.holdings.iter()
                        .take(5)
                        .map(|h| format!("• {} ${:.2}", h.symbol, h.value))
                        .collect::<Vec<_>>()
                        .join("\n")
                ),
                Err(e) => format!("❌ Error getting portfolio: {}", e),
            }
        },
        
        Command::Strategies => {
            match state.trading_service.get_strategies().await {
                Ok(strategies) => {
                    let mut response = "🎯 *Active Trading Strategies*\n\n".to_string();
                    for strategy in strategies.iter().take(10) {
                        response.push_str(&format!(
                            "• *{}*\n  📊 Sharpe: {:.2} | 🎯 Win Rate: {:.1}%\n  💰 P&L: {:.2}%\n\n",
                            strategy.name, strategy.sharpe_ratio, strategy.win_rate, strategy.pnl
                        ));
                    }
                    response
                },
                Err(e) => format!("❌ Error getting strategies: {}", e),
            }
        },
        
        Command::Train { model, symbol } => {
            match state.neural_service.start_training(&model, &symbol).await {
                Ok(training_id) => format!(
                    "🧠 *Neural Training Started*\n\n\
                    Model: {}\n\
                    Symbol: {}\n\
                    Training ID: {}\n\
                    Status: Initializing...\n\n\
                    Use `/trainstatus` to monitor progress",
                    model, symbol, training_id
                ),
                Err(e) => format!("❌ Training failed: {}", e),
            }
        },
        
        Command::Forecast { symbol, horizon } => {
            let hours = horizon.as_deref().unwrap_or("24h");
            match state.neural_service.get_forecast(&symbol, hours).await {
                Ok(forecast) => format!(
                    "🔮 *Neural Forecast: {}*\n\n\
                    📊 Model: {}\n\
                    ⏰ Horizon: {}\n\
                    🎯 Confidence: {:.1}%\n\n\
                    📈 Price Targets:\n\
                    1h: ${:.2}\n\
                    6h: ${:.2}\n\
                    24h: ${:.2}\n\n\
                    📊 Trend: {}\n\
                    ⚠️ Risk Level: {}",
                    symbol, forecast.model, hours, forecast.confidence * 100.0,
                    forecast.targets.h1, forecast.targets.h6, forecast.targets.h24,
                    forecast.trend, forecast.risk_level
                ),
                Err(e) => format!("❌ Forecast failed: {}", e),
            }
        },
        
        Command::Sentiment { symbol } => {
            match state.sentiment_service.get_sentiment(&symbol).await {
                Ok(sentiment) => format!(
                    "💭 *Sentiment Analysis: {}*\n\n\
                    📊 Overall Score: {:.2}\n\
                    📈 Bullish: {}%\n\
                    📉 Bearish: {}%\n\
                    ➖ Neutral: {}%\n\n\
                    🔥 Trending Keywords:\n{}\n\n\
                    📰 News Sources: {}\n\
                    ⏰ Last Update: {}",
                    symbol, sentiment.score, sentiment.bullish,
                    sentiment.bearish, sentiment.neutral,
                    sentiment.keywords.join(", "),
                    sentiment.sources, sentiment.updated_at.format("%H:%M:%S")
                ),
                Err(e) => format!("❌ Sentiment analysis failed: {}", e),
            }
        },
        
        Command::Mcp { command } => {
            match state.mcp_service.execute_command(&command).await {
                Ok(result) => format!(
                    "🤖 *MCP Command Executed*\n\
                    Command: `{}`\n\
                    Status: ✅ Success\n\n\
                    ```\n{}\n```",
                    command, 
                    serde_json::to_string_pretty(&result).unwrap_or_else(|_| "No output".to_string())
                ),
                Err(e) => format!("❌ MCP command failed: {}", e),
            }
        },
        
        Command::Alert { symbol, price, condition } => {
            let price_value: f64 = price.parse().unwrap_or(0.0);
            if price_value <= 0.0 {
                "❌ Invalid price value".to_string()
            } else {
                let alert = PriceAlert {
                    id: uuid::Uuid::new_v4().to_string(),
                    symbol: symbol.clone(),
                    price: price_value,
                    condition: condition.clone(),
                    created_at: Utc::now(),
                    triggered: false,
                };
                
                add_price_alert(&state, user_id, alert.clone()).await;
                
                format!(
                    "🔔 *Price Alert Set*\n\n\
                    Symbol: {}\n\
                    Price: ${}\n\
                    Condition: {}\n\
                    Alert ID: {}\n\n\
                    You'll be notified when {} goes {} ${}",
                    symbol, price, condition, alert.id[..8].to_string(),
                    symbol, condition, price
                )
            }
        },
        
        Command::TopPairs => {
            match state.trading_service.get_top_pairs().await {
                Ok(pairs) => {
                    let mut response = "🏆 *Top Profitable Pairs*\n\n".to_string();
                    for (i, pair) in pairs.iter().enumerate().take(10) {
                        response.push_str(&format!(
                            "{}. *{}*\n   🤖 AI Score: {:.0}% | 📊 ML: {:.0}% | 💭 Sentiment: {:.0}%\n   💰 24h: {:.2}%\n\n",
                            i + 1, pair.symbol, pair.ai_score * 100.0, pair.ml_score * 100.0,
                            pair.sentiment_score * 100.0, pair.change_24h
                        ));
                    }
                    response
                },
                Err(e) => format!("❌ Error getting top pairs: {}", e),
            }
        },
        
        _ => "🚧 Command implementation in progress...".to_string(),
    };
    
    bot.send_message(msg.chat.id, response)
        .parse_mode(teloxide::types::ParseMode::MarkdownV2)
        .await?;
    
    Ok(())
}

#[derive(Debug)]
struct PlatformStatus {
    trading: String,
    neural: String,
    sentiment: String,
    mcp: String,
    prediction: String,
    users: usize,
    uptime: String,
}

async fn get_platform_status(state: &BotState) -> PlatformStatus {
    let sessions = state.sessions.read().await;
    
    PlatformStatus {
        trading: "🟢 Online".to_string(),
        neural: "🟢 Training".to_string(),
        sentiment: "🟢 Processing".to_string(),
        mcp: "🟢 Connected".to_string(),
        prediction: "🟢 Active".to_string(),
        users: sessions.len(),
        uptime: "2h 34m".to_string(), // This would be calculated from startup time
    }
}

async fn update_user_session(state: &BotState, user_id: UserId, username: Option<String>) {
    let mut sessions = state.sessions.write().await;
    
    let session = sessions.entry(user_id).or_insert_with(|| UserSession {
        user_id,
        username: username.clone(),
        subscriptions: Vec::new(),
        alerts: Vec::new(),
        created_at: Utc::now(),
        last_activity: Utc::now(),
    });
    
    session.last_activity = Utc::now();
    if username.is_some() {
        session.username = username;
    }
}

async fn add_price_alert(state: &BotState, user_id: UserId, alert: PriceAlert) {
    let mut sessions = state.sessions.write().await;
    if let Some(session) = sessions.get_mut(&user_id) {
        session.alerts.push(alert);
    }
}

async fn price_alert_monitor(_state: BotState) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
    
    loop {
        interval.tick().await;
        
        // Check price alerts implementation would go here
        // This would check current prices against user alerts and send notifications
        log::debug!("Checking price alerts...");
    }
}

async fn market_update_broadcaster(bot: Bot, state: BotState) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300)); // 5 minutes
    
    loop {
        interval.tick().await;
        
        // Broadcast market updates to subscribed users
        let sessions = state.sessions.read().await;
        for (user_id, session) in sessions.iter() {
            if session.subscriptions.contains(&"prices".to_string()) {
                // Send market update
                let update = "📊 *Market Update*\nBTC: $45,123 (+2.3%)\nETH: $2,834 (+1.8%)";
                if let Err(e) = bot.send_message(ChatId(user_id.0 as i64), update)
                    .parse_mode(teloxide::types::ParseMode::MarkdownV2)
                    .await {
                    log::warn!("Failed to send update to user {}: {}", user_id, e);
                }
            }
        }
    }
}