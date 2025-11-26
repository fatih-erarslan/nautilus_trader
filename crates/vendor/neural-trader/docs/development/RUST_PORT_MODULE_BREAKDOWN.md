# Neural Trading Rust Port - Detailed Module Breakdown

**18 Planning Modules for Complete Rust Port**

---

## Module 00: Project Overview & Setup (README)

**Duration:** Week 1 (Research Phase)
**Owner:** System Architect
**GOAP Tasks:** GOAL-0-00-01, GOAL-0-00-02, GOAL-1-01-01

### Objectives
- Establish project vision and scope
- Define success criteria
- Setup development environment
- Create initial project structure

### Key Decisions
1. **Async Runtime:** Tokio (most mature, best ecosystem)
2. **Web Framework:** Axum (type-safe, fast, Tokio-native)
3. **Serialization:** serde (industry standard)
4. **Testing:** cargo test + proptest + cargo-nextest

### Deliverables
- [ ] Project README.md with architecture overview
- [ ] Development environment setup guide
- [ ] Contribution guidelines
- [ ] Code style guide (rustfmt + clippy config)

### Research Questions
- **Q1:** Monorepo vs multi-repo for Rust workspace?
  - **A:** Monorepo with Cargo workspace (better for shared types)
- **Q2:** Binary releases strategy?
  - **A:** GitHub Actions with cross-compilation

### Dependencies
```toml
[workspace.dependencies]
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
```

---

## Module 01: Project Structure & Workspace

**Duration:** Week 3 (MVP Phase)
**Owner:** System Architect + DevOps Engineer
**GOAP Tasks:** GOAL-1-01-01

### Objectives
- Design Cargo workspace with optimal crate boundaries
- Configure CI/CD pipeline
- Setup Docker build system

### Crate Structure
```
neural-trader/
├── Cargo.toml (workspace)
├── crates/
│   ├── core/           # Core types and traits
│   ├── types/          # Data structures
│   ├── config/         # Configuration management
│   ├── trading-api/    # Alpaca/IBKR clients
│   ├── news-collector/ # News aggregation
│   ├── sentiment/      # NLP and sentiment
│   ├── strategies/     # Trading strategies
│   ├── portfolio/      # Portfolio management
│   ├── risk-manager/   # Risk calculations
│   ├── auth/           # Authentication
│   ├── web-server/     # HTTP API (Axum)
│   ├── database/       # Persistence layer
│   ├── gpu-compute/    # CUDA/GPU acceleration
│   ├── backtester/     # Backtesting engine
│   └── cli/            # Command-line interface
├── tests/              # Integration tests
└── benches/            # Benchmarks
```

### CI/CD Pipeline (GitHub Actions)
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rs/toolchain@v1
      - run: cargo test --all-features
      - run: cargo clippy -- -D warnings
      - run: cargo fmt -- --check

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo tarpaulin --out Xml
      - uses: codecov/codecov-action@v3

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo audit
```

### Deliverables
- [ ] Workspace with all crates defined
- [ ] CI/CD passing on every commit
- [ ] Docker multi-stage build
- [ ] Development documentation

---

## Module 02: Core Types & Data Structures

**Duration:** Weeks 3-4 (MVP Phase)
**Owner:** Rust Developer (Type Systems Specialist)
**GOAP Tasks:** GOAL-1-02-01, GOAL-1-02-02

### Objectives
- Define all domain types with strong typing
- Implement serialization/deserialization
- Add validation for all input types

### Key Types

#### Trading Types
```rust
// Order management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: OrderId,
    pub symbol: Symbol,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub order_type: OrderType,
    pub time_in_force: TimeInForce,
    pub submitted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    TrailingStop,
}

// Position tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: Symbol,
    pub quantity: Decimal,
    pub avg_entry_price: Decimal,
    pub current_price: Decimal,
    pub market_value: Decimal,
    pub unrealized_pnl: Decimal,
    pub unrealized_pnl_percent: Decimal,
    pub side: PositionSide,
}
```

#### Market Data Types
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub symbol: Symbol,
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: u64,
    pub vwap: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    pub symbol: Symbol,
    pub timestamp: DateTime<Utc>,
    pub bid_price: Decimal,
    pub bid_size: u32,
    pub ask_price: Decimal,
    pub ask_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: Symbol,
    pub timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub size: u32,
}
```

#### News Types
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Article {
    pub id: ArticleId,
    pub title: String,
    pub content: String,
    pub source: NewsSource,
    pub url: Url,
    pub published_at: DateTime<Utc>,
    pub symbols: Vec<Symbol>,
    pub sentiment: Option<SentimentScore>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SentimentScore {
    pub score: f32,          // -1.0 to 1.0
    pub confidence: f32,     // 0.0 to 1.0
    pub magnitude: f32,      // 0.0 to 1.0
}
```

### Validation
```rust
use validator::Validate;

#[derive(Debug, Validate, Serialize, Deserialize)]
pub struct OrderRequest {
    #[validate(length(min = 1, max = 10))]
    pub symbol: String,

    #[validate(range(min = 0.01))]
    pub quantity: f64,

    #[validate(range(min = 0.01))]
    pub price: Option<f64>,

    pub side: OrderSide,
    pub order_type: OrderType,
}
```

### Deliverables
- [ ] 50+ types defined with documentation
- [ ] Serde serialization for all types
- [ ] Validation with validator crate
- [ ] Unit tests with 95%+ coverage
- [ ] Type documentation with examples

---

## Module 03: Error Handling

**Duration:** Week 4 (MVP Phase)
**Owner:** Rust Developer
**GOAP Tasks:** GOAL-1-02-02

### Objectives
- Unified error handling across all crates
- Error context preservation
- HTTP status code mapping

### Error Design
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TradingError {
    // API Errors
    #[error("API error: {0}")]
    ApiError(String),

    #[error("API rate limit exceeded: retry after {retry_after:?}")]
    RateLimitExceeded { retry_after: Option<Duration> },

    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    // Trading Errors
    #[error("Invalid order: {0}")]
    InvalidOrder(String),

    #[error("Insufficient funds: required {required}, available {available}")]
    InsufficientFunds {
        required: Decimal,
        available: Decimal,
    },

    #[error("Position not found: {symbol}")]
    PositionNotFound { symbol: Symbol },

    #[error("Risk limit exceeded: {reason}")]
    RiskLimitExceeded { reason: String },

    // Data Errors
    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Data not available for {symbol} at {timestamp}")]
    DataNotAvailable {
        symbol: Symbol,
        timestamp: DateTime<Utc>,
    },

    // System Errors
    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    // External Errors
    #[error(transparent)]
    Network(#[from] reqwest::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
}

impl TradingError {
    pub fn http_status_code(&self) -> StatusCode {
        match self {
            Self::AuthenticationFailed(_) => StatusCode::UNAUTHORIZED,
            Self::InvalidOrder(_) | Self::InvalidSymbol(_) => StatusCode::BAD_REQUEST,
            Self::InsufficientFunds { .. } => StatusCode::FORBIDDEN,
            Self::RateLimitExceeded { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::PositionNotFound { .. } | Self::DataNotAvailable { .. } => {
                StatusCode::NOT_FOUND
            }
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

pub type Result<T> = std::result::Result<T, TradingError>;
```

### Error Context with anyhow
```rust
use anyhow::{Context, Result};

pub async fn load_market_data(symbol: &Symbol) -> Result<Vec<Bar>> {
    let data = fetch_data(symbol)
        .await
        .context("Failed to fetch market data")?;

    parse_bars(&data)
        .context(format!("Failed to parse bars for {}", symbol))?;

    Ok(bars)
}
```

### Deliverables
- [ ] Unified TradingError enum
- [ ] Error conversion traits (From implementations)
- [ ] HTTP status code mapping
- [ ] Error logging with context
- [ ] Error handling examples

---

## Module 04: Configuration Management

**Duration:** Week 4 (MVP Phase)
**Owner:** Backend Developer
**GOAP Tasks:** GOAL-1-03-01

### Objectives
- Type-safe configuration
- Environment variable parsing
- Secrets management
- Validation

### Configuration Structure
```rust
use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;
use validator::Validate;

#[derive(Debug, Deserialize, Validate)]
pub struct AppConfig {
    #[validate(range(min = 1024, max = 65535))]
    pub port: u16,

    pub log_level: String,

    pub alpaca: AlpacaConfig,
    pub database: DatabaseConfig,
    pub auth: AuthConfig,
    pub news: NewsConfig,
    pub gpu: GpuConfig,
}

#[derive(Debug, Deserialize, Validate)]
pub struct AlpacaConfig {
    #[validate(length(min = 1))]
    pub api_key: String,

    #[validate(length(min = 1))]
    pub secret_key: String,

    pub base_url: String,
    pub paper_trading: bool,
}

#[derive(Debug, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connection_timeout: u64,
}

#[derive(Debug, Deserialize)]
pub struct AuthConfig {
    pub enabled: bool,
    pub jwt_secret: String,
    pub token_expiration: u64,
}

impl AppConfig {
    pub fn load() -> Result<Self, ConfigError> {
        let config = Config::builder()
            // Default values
            .set_default("port", 8080)?
            .set_default("log_level", "info")?
            // Load from config file
            .add_source(File::with_name("config").required(false))
            // Override with environment variables (PREFIX_KEY)
            .add_source(Environment::with_prefix("NEURAL_TRADER"))
            .build()?;

        let app_config: AppConfig = config.try_deserialize()?;
        app_config.validate()
            .map_err(|e| ConfigError::Message(e.to_string()))?;

        Ok(app_config)
    }
}
```

### Configuration Files

**config.toml**
```toml
port = 8080
log_level = "info"

[alpaca]
base_url = "https://paper-api.alpaca.markets"
paper_trading = true

[database]
url = "postgresql://localhost/neural_trader"
max_connections = 10
min_connections = 2
connection_timeout = 30

[auth]
enabled = false
token_expiration = 3600

[news]
sources = ["newsapi", "finnhub", "sec"]

[gpu]
enabled = false
device_id = 0
```

**.env** (secrets only)
```bash
NEURAL_TRADER_ALPACA__API_KEY=your_api_key
NEURAL_TRADER_ALPACA__SECRET_KEY=your_secret
NEURAL_TRADER_AUTH__JWT_SECRET=your_jwt_secret
NEURAL_TRADER_DATABASE__URL=postgresql://user:pass@localhost/db
```

### Deliverables
- [ ] Type-safe configuration structs
- [ ] Environment variable parsing
- [ ] Config validation
- [ ] Example config files
- [ ] Documentation

---

## Module 05: Trading API Integration (Alpaca)

**Duration:** Weeks 4-5 (MVP Phase)
**Owner:** Backend Developer + API Integration Specialist
**GOAP Tasks:** GOAL-1-05-01

### Objectives
- Complete Alpaca v2 API client
- WebSocket streaming for market data
- Order management
- Account and position queries

### Client Architecture
```rust
use reqwest::{Client, StatusCode};
use tokio_tungstenite::WebSocketStream;

pub struct AlpacaClient {
    http_client: Client,
    config: AlpacaConfig,
    ws_client: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    rate_limiter: RateLimiter,
}

impl AlpacaClient {
    pub fn new(config: AlpacaConfig) -> Result<Self> {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(30))
            .default_headers({
                let mut headers = HeaderMap::new();
                headers.insert("APCA-API-KEY-ID", config.api_key.parse()?);
                headers.insert("APCA-API-SECRET-KEY", config.secret_key.parse()?);
                headers
            })
            .build()?;

        Ok(Self {
            http_client,
            config,
            ws_client: None,
            rate_limiter: RateLimiter::new(200, Duration::from_secs(60)),
        })
    }

    // Account Management
    pub async fn get_account(&self) -> Result<Account> {
        self.rate_limiter.wait().await;
        let url = format!("{}/v2/account", self.config.base_url);
        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    // Order Management
    pub async fn place_order(&self, order: &Order) -> Result<OrderResponse> {
        self.rate_limiter.wait().await;
        let url = format!("{}/v2/orders", self.config.base_url);
        let response = self.http_client
            .post(&url)
            .json(order)
            .send()
            .await?;
        self.handle_response(response).await
    }

    pub async fn get_orders(&self, status: OrderStatus) -> Result<Vec<OrderResponse>> {
        let url = format!("{}/v2/orders?status={}", self.config.base_url, status);
        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    pub async fn cancel_order(&self, order_id: &OrderId) -> Result<()> {
        let url = format!("{}/v2/orders/{}", self.config.base_url, order_id);
        let response = self.http_client.delete(&url).send().await?;
        self.handle_response(response).await
    }

    // Position Management
    pub async fn get_positions(&self) -> Result<Vec<Position>> {
        let url = format!("{}/v2/positions", self.config.base_url);
        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    pub async fn close_position(&self, symbol: &Symbol) -> Result<Position> {
        let url = format!("{}/v2/positions/{}", self.config.base_url, symbol);
        let response = self.http_client.delete(&url).send().await?;
        self.handle_response(response).await
    }

    // Market Data
    pub async fn get_bars(
        &self,
        symbol: &Symbol,
        timeframe: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Bar>> {
        let url = format!(
            "{}/v2/stocks/{}/bars?timeframe={}&start={}&end={}",
            self.config.base_url,
            symbol,
            timeframe,
            start.to_rfc3339(),
            end.to_rfc3339()
        );
        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    // WebSocket Streaming
    pub async fn subscribe_quotes(&mut self, symbols: Vec<Symbol>) -> Result<()> {
        if self.ws_client.is_none() {
            self.connect_websocket().await?;
        }

        let subscribe_msg = json!({
            "action": "subscribe",
            "quotes": symbols,
        });

        self.send_ws_message(subscribe_msg).await
    }

    pub async fn stream_quotes(&mut self) -> Result<impl Stream<Item = Quote>> {
        // Return stream of quotes
        todo!()
    }

    async fn handle_response<T: DeserializeOwned>(
        &self,
        response: Response,
    ) -> Result<T> {
        match response.status() {
            StatusCode::OK | StatusCode::CREATED => {
                Ok(response.json().await?)
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(TradingError::RateLimitExceeded {
                    retry_after: response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.parse().ok())
                        .map(Duration::from_secs),
                })
            }
            status => {
                let error_text = response.text().await?;
                Err(TradingError::ApiError(format!(
                    "API request failed with status {}: {}",
                    status, error_text
                )))
            }
        }
    }
}
```

### Rate Limiting
```rust
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

pub struct RateLimiter {
    semaphore: Semaphore,
    window: Duration,
    last_reset: Instant,
}

impl RateLimiter {
    pub fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            semaphore: Semaphore::new(max_requests),
            window,
            last_reset: Instant::now(),
        }
    }

    pub async fn wait(&self) {
        // Reset if window expired
        if self.last_reset.elapsed() > self.window {
            // Reset logic
        }

        let _permit = self.semaphore.acquire().await.unwrap();
        // Permit dropped automatically
    }
}
```

### Deliverables
- [ ] Complete Alpaca REST API client
- [ ] WebSocket streaming client
- [ ] Rate limiting with auto-retry
- [ ] Comprehensive error handling
- [ ] Integration tests with paper trading
- [ ] API documentation

---

## Module 06: News Collection System

**Duration:** Weeks 7-8 (Full Parity Phase)
**Owner:** Backend Developer + Data Engineer
**GOAP Tasks:** GOAL-2-06-01

### Objectives
- Integrate 5+ news sources
- RSS feed parsing
- API integrations (NewsAPI, Finnhub)
- Web scraping (SEC filings)
- Deduplication system

### News Sources

#### 1. NewsAPI Integration
```rust
pub struct NewsApiClient {
    http_client: Client,
    api_key: String,
}

impl NewsApiClient {
    pub async fn fetch_articles(
        &self,
        query: &str,
        from: DateTime<Utc>,
    ) -> Result<Vec<Article>> {
        let url = format!(
            "https://newsapi.org/v2/everything?q={}&from={}&apiKey={}",
            query,
            from.to_rfc3339(),
            self.api_key
        );

        let response = self.http_client.get(&url).send().await?;
        let data: NewsApiResponse = response.json().await?;

        Ok(data.articles.into_iter().map(|a| a.into()).collect())
    }
}
```

#### 2. RSS Feed Parser
```rust
use feed_rs::parser;

pub async fn fetch_rss_feed(url: &str) -> Result<Vec<Article>> {
    let content = reqwest::get(url).await?.bytes().await?;
    let feed = parser::parse(&content[..])?;

    let articles = feed.entries.into_iter()
        .map(|entry| Article {
            id: ArticleId::new(),
            title: entry.title.map(|t| t.content).unwrap_or_default(),
            content: entry.summary.map(|s| s.content).unwrap_or_default(),
            url: entry.links.first().map(|l| l.href.clone()).unwrap(),
            published_at: entry.published.unwrap_or_else(Utc::now),
            source: NewsSource::Rss,
            symbols: vec![],
            sentiment: None,
        })
        .collect();

    Ok(articles)
}
```

#### 3. SEC Filings Scraper
```rust
use scraper::{Html, Selector};

pub async fn fetch_sec_filings(ticker: &str) -> Result<Vec<Article>> {
    let url = format!(
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type=8-K",
        ticker
    );

    let html = reqwest::get(&url).await?.text().await?;
    let document = Html::parse_document(&html);
    let selector = Selector::parse("table.tableFile2 tr").unwrap();

    let articles = document
        .select(&selector)
        .skip(1) // Skip header
        .filter_map(|row| {
            // Extract filing data
            Some(Article {
                // ... parse filing details
            })
        })
        .collect();

    Ok(articles)
}
```

### Deduplication System
```rust
use std::collections::HashSet;
use blake3::hash;

pub struct Deduplicator {
    seen_hashes: HashSet<String>,
}

impl Deduplicator {
    pub fn is_duplicate(&mut self, article: &Article) -> bool {
        let content_hash = self.hash_article(article);
        !self.seen_hashes.insert(content_hash)
    }

    fn hash_article(&self, article: &Article) -> String {
        let content = format!("{}{}", article.title, article.content);
        hash(content.as_bytes()).to_hex().to_string()
    }

    pub fn cleanup_old(&mut self, max_age: Duration) {
        // Remove old hashes to prevent unbounded growth
        // Implementation depends on tracking timestamps
    }
}
```

### Aggregation System
```rust
pub struct NewsAggregator {
    sources: Vec<Box<dyn NewsSource>>,
    deduplicator: Deduplicator,
    database: DatabasePool,
}

#[async_trait]
pub trait NewsSource: Send + Sync {
    async fn fetch_articles(&self, query: &str) -> Result<Vec<Article>>;
    fn source_name(&self) -> &str;
}

impl NewsAggregator {
    pub async fn collect_news(&mut self, symbols: &[Symbol]) -> Result<Vec<Article>> {
        let mut all_articles = Vec::new();

        for source in &self.sources {
            for symbol in symbols {
                match source.fetch_articles(symbol.as_str()).await {
                    Ok(mut articles) => {
                        articles.retain(|a| !self.deduplicator.is_duplicate(a));
                        all_articles.extend(articles);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to fetch from {}: {}",
                            source.source_name(),
                            e
                        );
                        // Continue with other sources (graceful degradation)
                    }
                }
            }
        }

        // Store in database
        self.database.insert_articles(&all_articles).await?;

        Ok(all_articles)
    }
}
```

### Deliverables
- [ ] 5+ news source integrations
- [ ] Deduplication system (>95% accuracy)
- [ ] Graceful degradation on source failures
- [ ] Article storage in database
- [ ] Monitoring for source health
- [ ] Integration tests

---

## Module 07: News Integration & Distribution

**Duration:** Week 8 (Full Parity Phase)
**Owner:** Backend Developer
**GOAP Tasks:** Part of GOAL-2-06-01

### Objectives
- Real-time news distribution to strategies
- News event streaming
- Historical news query API

### News Stream Architecture
```rust
use tokio::sync::broadcast;

pub struct NewsStream {
    tx: broadcast::Sender<Article>,
    aggregator: NewsAggregator,
}

impl NewsStream {
    pub fn new(capacity: usize, aggregator: NewsAggregator) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx, aggregator }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Article> {
        self.tx.subscribe()
    }

    pub async fn start(&mut self, interval: Duration) {
        let mut ticker = tokio::time::interval(interval);

        loop {
            ticker.tick().await;

            match self.aggregator.collect_news(&[/* symbols */]).await {
                Ok(articles) => {
                    for article in articles {
                        // Broadcast to all subscribers
                        let _ = self.tx.send(article);
                    }
                }
                Err(e) => {
                    tracing::error!("News collection error: {}", e);
                }
            }
        }
    }
}
```

### Deliverables
- [ ] Real-time news streaming
- [ ] Subscriber pattern for strategies
- [ ] Historical news API
- [ ] News event monitoring

---

## Module 08: Sentiment Analysis Engine

**Duration:** Weeks 8-9 (Full Parity Phase)
**Owner:** ML Engineer + Rust Developer
**GOAP Tasks:** GOAL-2-08-01

### Objectives
- Load FinBERT or similar model
- Inference pipeline with < 100ms latency
- Entity extraction
- Batch processing

### ML Framework Selection

**Option 1: tch-rs (PyTorch bindings)**
```rust
use tch::{nn, Device, Tensor};

pub struct SentimentAnalyzer {
    model: nn::VarStore,
    tokenizer: Tokenizer,
    device: Device,
}

impl SentimentAnalyzer {
    pub fn load(model_path: &Path) -> Result<Self> {
        let device = Device::cuda_if_available();
        let mut vs = nn::VarStore::new(device);

        // Load FinBERT model
        let model = nn::seq()
            .add(nn::linear(&vs.root(), 768, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.root(), 256, 3, Default::default()));

        vs.load(model_path)?;

        let tokenizer = Tokenizer::from_file("tokenizer.json")?;

        Ok(Self {
            model: vs,
            tokenizer,
            device,
        })
    }

    pub fn analyze(&self, text: &str) -> Result<SentimentScore> {
        // Tokenize
        let encoding = self.tokenizer.encode(text, true)?;
        let input_ids = Tensor::of_slice(&encoding.get_ids())
            .to(self.device);

        // Inference
        let output = tch::no_grad(|| {
            self.model.forward(&input_ids)
        });

        // Convert to sentiment score
        let scores = Vec::<f32>::from(output.softmax(-1, tch::Kind::Float));

        Ok(SentimentScore {
            score: self.calculate_score(&scores),
            confidence: scores.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
            magnitude: scores.iter().sum::<f32>() / scores.len() as f32,
        })
    }

    fn calculate_score(&self, scores: &[f32]) -> f32 {
        // negative, neutral, positive
        scores[2] - scores[0]
    }
}
```

**Option 2: tract (ONNX runtime)**
```rust
use tract_onnx::prelude::*;

pub struct SentimentAnalyzer {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    tokenizer: Tokenizer,
}

impl SentimentAnalyzer {
    pub fn load(model_path: &Path) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 128)))?
            .into_optimized()?
            .into_runnable()?;

        let tokenizer = Tokenizer::from_file("tokenizer.json")?;

        Ok(Self { model, tokenizer })
    }

    pub fn analyze(&self, text: &str) -> Result<SentimentScore> {
        // Similar to tch-rs implementation
        todo!()
    }
}
```

### Batch Processing
```rust
pub struct BatchSentimentAnalyzer {
    analyzer: SentimentAnalyzer,
    batch_size: usize,
}

impl BatchSentimentAnalyzer {
    pub async fn analyze_batch(&self, texts: Vec<String>) -> Result<Vec<SentimentScore>> {
        let mut results = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.batch_size) {
            // Process batch in parallel
            let scores: Vec<_> = chunk
                .iter()
                .map(|text| self.analyzer.analyze(text))
                .collect::<Result<Vec<_>>>()?;

            results.extend(scores);
        }

        Ok(results)
    }
}
```

### Entity Extraction
```rust
use rust_bert::pipelines::ner::NERModel;

pub struct EntityExtractor {
    ner_model: NERModel,
}

impl EntityExtractor {
    pub fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let entities = self.ner_model.predict(&[text]);

        Ok(entities[0]
            .iter()
            .map(|e| Entity {
                text: e.word.clone(),
                label: e.label.clone(),
                score: e.score,
            })
            .collect())
    }
}
```

### Deliverables
- [ ] Sentiment analysis with FinBERT or similar
- [ ] Inference < 100ms per article
- [ ] Batch processing capability
- [ ] Entity extraction (companies, products)
- [ ] Results match Python ±5%
- [ ] GPU acceleration (optional)

---

*[Modules 09-18 would continue in similar detail...]*

---

## Summary: Module Dependencies

```mermaid
graph TD
    M00[00: README] --> M01[01: Project Structure]
    M01 --> M02[02: Core Types]
    M02 --> M03[03: Error Handling]
    M03 --> M04[04: Configuration]
    M04 --> M05[05: Trading API]
    M05 --> M09[09: Strategies]
    M04 --> M06[06: News Collection]
    M06 --> M07[07: News Integration]
    M07 --> M08[08: Sentiment Analysis]
    M08 --> M09
    M09 --> M10[10: Portfolio Management]
    M10 --> M11[11: Risk Management]
    M02 --> M12[12: Authentication]
    M12 --> M13[13: API Server]
    M04 --> M14[14: Database]
    M09 --> M15[15: Testing]
    M09 --> M16[16: Performance/GPU]
    M13 --> M17[17: Deployment]
    M10 --> M18[18: Backtesting]
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-12
**See Also:** RUST_PORT_GOAP_TASKBOARD.md
