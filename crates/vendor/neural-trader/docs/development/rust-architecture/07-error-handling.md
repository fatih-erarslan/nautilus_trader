# Error Handling and Recovery Patterns

## Error Type Hierarchy

```rust
// src/error.rs
use thiserror::Error;

/// Top-level error type for the trading system
#[derive(Error, Debug)]
pub enum TradingError {
    #[error("Market data error: {0}")]
    MarketData(#[from] MarketDataError),

    #[error("Strategy error: {0}")]
    Strategy(#[from] StrategyError),

    #[error("Execution error: {0}")]
    Execution(#[from] ExecutionError),

    #[error("Risk management error: {0}")]
    Risk(#[from] RiskError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),

    #[error("Network error: {0}")]
    Network(#[from] NetworkError),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Market data specific errors
#[derive(Error, Debug)]
pub enum MarketDataError {
    #[error("Connection lost to {provider}")]
    ConnectionLost { provider: String },

    #[error("Invalid data format: {reason}")]
    InvalidFormat { reason: String },

    #[error("Symbol not found: {symbol}")]
    SymbolNotFound { symbol: String },

    #[error("Rate limit exceeded for {provider}")]
    RateLimitExceeded { provider: String },

    #[error("Stale data: last update {seconds_ago}s ago")]
    StaleData { seconds_ago: u64 },

    #[error("Data validation failed: {reason}")]
    ValidationFailed { reason: String },
}

/// Strategy execution errors
#[derive(Error, Debug)]
pub enum StrategyError {
    #[error("Strategy initialization failed: {reason}")]
    InitializationFailed { reason: String },

    #[error("Invalid parameters: {reason}")]
    InvalidParameters { reason: String },

    #[error("Strategy crashed: {reason}")]
    Crashed { reason: String },

    #[error("Insufficient data: need {required}, have {available}")]
    InsufficientData { required: usize, available: usize },

    #[error("Strategy timeout after {seconds}s")]
    Timeout { seconds: u64 },
}

/// Order execution errors
#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("Order rejected: {reason}")]
    OrderRejected { reason: String },

    #[error("Insufficient funds: need {required}, have {available}")]
    InsufficientFunds { required: String, available: String },

    #[error("Market closed for {symbol}")]
    MarketClosed { symbol: String },

    #[error("Exchange error: {exchange} - {message}")]
    ExchangeError { exchange: String, message: String },

    #[error("Order not found: {order_id}")]
    OrderNotFound { order_id: String },

    #[error("Order timeout: {order_id}")]
    OrderTimeout { order_id: String },

    #[error("Duplicate order: {client_order_id}")]
    DuplicateOrder { client_order_id: String },
}

/// Risk management errors
#[derive(Error, Debug)]
pub enum RiskError {
    #[error("Position limit exceeded: {current} > {limit}")]
    PositionLimitExceeded { current: String, limit: String },

    #[error("Portfolio VaR exceeded: {var} > {limit}")]
    VarLimitExceeded { var: f64, limit: f64 },

    #[error("Daily loss limit reached: {loss}")]
    DailyLossLimit { loss: String },

    #[error("Concentration risk: {symbol} is {percentage}% of portfolio")]
    ConcentrationRisk { symbol: String, percentage: f64 },

    #[error("Leverage limit exceeded: {leverage}x > {max}x")]
    LeverageExceeded { leverage: f64, max: f64 },

    #[error("Circuit breaker triggered: {reason}")]
    CircuitBreaker { reason: String },
}

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Missing required field: {field}")]
    MissingField { field: String },

    #[error("Invalid value for {field}: {reason}")]
    InvalidValue { field: String, reason: String },

    #[error("Configuration file not found: {path}")]
    FileNotFound { path: String },

    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Database errors
#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection failed: {reason}")]
    ConnectionFailed { reason: String },

    #[error("Query failed: {query} - {reason}")]
    QueryFailed { query: String, reason: String },

    #[error("Record not found: {id}")]
    RecordNotFound { id: String },

    #[error("Constraint violation: {constraint}")]
    ConstraintViolation { constraint: String },

    #[error("Transaction failed: {reason}")]
    TransactionFailed { reason: String },
}

/// Network errors
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Connection timeout: {endpoint}")]
    ConnectionTimeout { endpoint: String },

    #[error("DNS resolution failed: {hostname}")]
    DnsFailure { hostname: String },

    #[error("HTTP error {status}: {message}")]
    HttpError { status: u16, message: String },

    #[error("WebSocket closed: {reason}")]
    WebSocketClosed { reason: String },

    #[error("TLS error: {reason}")]
    TlsError { reason: String },
}
```

## Error Context and Enrichment

```rust
// src/error_context.rs
use std::backtrace::Backtrace;
use std::fmt;

/// Rich error context with metadata
#[derive(Debug)]
pub struct ErrorContext {
    pub error: TradingError,
    pub context: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub backtrace: Backtrace,
    pub timestamp: DateTime<Utc>,
}

impl ErrorContext {
    pub fn new(error: TradingError) -> Self {
        Self {
            error,
            context: Vec::new(),
            metadata: HashMap::new(),
            backtrace: Backtrace::capture(),
            timestamp: Utc::now(),
        }
    }

    pub fn add_context(mut self, context: impl Into<String>) -> Self {
        self.context.push(context.into());
        self
    }

    pub fn add_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)?;

        if !self.context.is_empty() {
            write!(f, "\nContext:")?;
            for ctx in &self.context {
                write!(f, "\n  - {}", ctx)?;
            }
        }

        if !self.metadata.is_empty() {
            write!(f, "\nMetadata:")?;
            for (key, value) in &self.metadata {
                write!(f, "\n  {} = {}", key, value)?;
            }
        }

        Ok(())
    }
}

/// Result type with rich context
pub type ContextResult<T> = Result<T, ErrorContext>;

/// Extension trait for adding context to Results
pub trait ResultExt<T> {
    fn context(self, context: impl Into<String>) -> ContextResult<T>;
    fn with_context<F>(self, f: F) -> ContextResult<T>
    where
        F: FnOnce() -> String;
}

impl<T, E: Into<TradingError>> ResultExt<T> for Result<T, E> {
    fn context(self, context: impl Into<String>) -> ContextResult<T> {
        self.map_err(|e| {
            ErrorContext::new(e.into()).add_context(context)
        })
    }

    fn with_context<F>(self, f: F) -> ContextResult<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            ErrorContext::new(e.into()).add_context(f())
        })
    }
}

// Usage example
async fn fetch_market_data(symbol: &Symbol) -> ContextResult<MarketData> {
    let provider = get_provider()
        .context("Failed to get market data provider")?;

    provider
        .get_quote(symbol)
        .await
        .context("Failed to fetch quote")
        .map_err(|e| e.add_metadata("symbol", symbol.to_string()))?;

    Ok(market_data)
}
```

## Circuit Breaker Pattern

```rust
// src/circuit_breaker.rs
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,      // Normal operation
    Open,        // Failing, reject requests
    HalfOpen,    // Testing if recovered
}

pub struct CircuitBreaker {
    state: Arc<Mutex<CircuitBreakerState>>,
    config: CircuitBreakerConfig,
}

struct CircuitBreakerState {
    state: CircuitState,
    failure_count: usize,
    success_count: usize,
    last_failure_time: Option<Instant>,
    last_state_change: Instant,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,      // Open after N failures
    pub success_threshold: usize,      // Close after N successes in HalfOpen
    pub timeout: Duration,             // Time before trying HalfOpen
    pub half_open_max_calls: usize,   // Max concurrent calls in HalfOpen
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_max_calls: 3,
        }
    }
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(CircuitBreakerState {
                state: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_time: None,
                last_state_change: Instant::now(),
            })),
            config,
        }
    }

    pub async fn call<F, T, E>(&self, f: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> std::pin::Pin<Box<dyn Future<Output = Result<T, E>> + Send>>,
    {
        // Check if we should allow the call
        {
            let mut state = self.state.lock();

            match state.state {
                CircuitState::Open => {
                    // Check if timeout expired
                    if let Some(last_failure) = state.last_failure_time {
                        if last_failure.elapsed() >= self.config.timeout {
                            state.state = CircuitState::HalfOpen;
                            state.success_count = 0;
                            state.last_state_change = Instant::now();
                        } else {
                            return Err(CircuitBreakerError::Open);
                        }
                    }
                }
                CircuitState::HalfOpen => {
                    if state.success_count >= self.config.half_open_max_calls {
                        return Err(CircuitBreakerError::TooManyCalls);
                    }
                }
                CircuitState::Closed => {}
            }
        }

        // Execute the function
        let result = f().await;

        // Update state based on result
        {
            let mut state = self.state.lock();

            match result {
                Ok(_) => {
                    state.failure_count = 0;
                    state.last_failure_time = None;

                    if state.state == CircuitState::HalfOpen {
                        state.success_count += 1;

                        if state.success_count >= self.config.success_threshold {
                            state.state = CircuitState::Closed;
                            state.last_state_change = Instant::now();
                            tracing::info!("Circuit breaker closed");
                        }
                    }
                }
                Err(_) => {
                    state.failure_count += 1;
                    state.last_failure_time = Some(Instant::now());

                    if state.failure_count >= self.config.failure_threshold {
                        state.state = CircuitState::Open;
                        state.last_state_change = Instant::now();
                        tracing::warn!(
                            "Circuit breaker opened after {} failures",
                            state.failure_count
                        );
                    }
                }
            }
        }

        result.map_err(CircuitBreakerError::Inner)
    }

    pub fn state(&self) -> CircuitState {
        self.state.lock().state
    }

    pub fn reset(&self) {
        let mut state = self.state.lock();
        state.state = CircuitState::Closed;
        state.failure_count = 0;
        state.success_count = 0;
        state.last_failure_time = None;
        state.last_state_change = Instant::now();
    }
}

#[derive(Error, Debug)]
pub enum CircuitBreakerError<E> {
    #[error("Circuit breaker is open")]
    Open,

    #[error("Too many concurrent calls in half-open state")]
    TooManyCalls,

    #[error("Inner error: {0}")]
    Inner(E),
}

// Usage example
async fn fetch_with_circuit_breaker(
    breaker: &CircuitBreaker,
    symbol: &Symbol,
) -> Result<Quote, CircuitBreakerError<MarketDataError>> {
    breaker.call(|| Box::pin(async {
        get_provider()?.get_quote(symbol).await
    })).await
}
```

## Retry Strategy

```rust
// src/retry.rs
use tokio::time::{sleep, Duration};
use tracing::warn;

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: usize,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub backoff_multiplier: f64,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

pub async fn retry_with_backoff<F, T, E, Fut>(
    config: &RetryConfig,
    mut f: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut attempt = 0;
    let mut backoff = config.initial_backoff;

    loop {
        attempt += 1;

        match f().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt >= config.max_attempts {
                    return Err(e);
                }

                warn!(
                    attempt = attempt,
                    max_attempts = config.max_attempts,
                    error = %e,
                    "Retry attempt failed, backing off"
                );

                // Calculate backoff with optional jitter
                let sleep_duration = if config.jitter {
                    let jitter = fastrand::f64() * 0.3; // ±30% jitter
                    Duration::from_secs_f64(
                        backoff.as_secs_f64() * (1.0 + jitter - 0.15)
                    )
                } else {
                    backoff
                };

                sleep(sleep_duration).await;

                // Increase backoff for next attempt
                backoff = Duration::from_secs_f64(
                    (backoff.as_secs_f64() * config.backoff_multiplier)
                        .min(config.max_backoff.as_secs_f64())
                );
            }
        }
    }
}

// Conditional retry (only retry transient errors)
pub async fn retry_if<F, T, E, Fut, P>(
    config: &RetryConfig,
    f: F,
    should_retry: P,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    P: Fn(&E) -> bool,
    E: std::fmt::Display,
{
    let mut attempt = 0;
    let mut backoff = config.initial_backoff;
    let mut f = f;

    loop {
        attempt += 1;

        match f().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if !should_retry(&e) {
                    return Err(e);
                }

                if attempt >= config.max_attempts {
                    return Err(e);
                }

                warn!(
                    attempt = attempt,
                    "Retrying after error: {}",
                    e
                );

                sleep(backoff).await;
                backoff = (backoff * config.backoff_multiplier as u32)
                    .min(config.max_backoff);
            }
        }
    }
}

// Usage example
async fn fetch_quote_with_retry(symbol: &Symbol) -> Result<Quote, MarketDataError> {
    let config = RetryConfig::default();

    retry_if(
        &config,
        || async { get_provider()?.get_quote(symbol).await },
        |e| matches!(e, MarketDataError::ConnectionLost { .. } | MarketDataError::RateLimitExceeded { .. })
    ).await
}
```

## Panic Recovery

```rust
// src/panic_handler.rs
use std::panic::{catch_unwind, AssertUnwindSafe};
use tracing::error;

/// Catch panics in async tasks and convert to Result
pub async fn catch_panic<F, T>(f: F) -> Result<T, String>
where
    F: Future<Output = T> + std::panic::UnwindSafe,
{
    match catch_unwind(AssertUnwindSafe(|| {
        // Block on the future
        tokio::runtime::Handle::current().block_on(f)
    })) {
        Ok(result) => Ok(result),
        Err(panic) => {
            let msg = if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic".to_string()
            };

            error!("Panic caught: {}", msg);
            Err(msg)
        }
    }
}

/// Supervisor pattern for strategy actors
pub struct StrategySupervisor {
    strategies: HashMap<String, StrategyHandle>,
}

struct StrategyHandle {
    sender: mpsc::Sender<StrategyMessage>,
    task_handle: JoinHandle<()>,
}

impl StrategySupervisor {
    pub fn spawn_strategy(&mut self, id: String, strategy: Box<dyn Strategy>) {
        let (tx, rx) = mpsc::channel(1000);

        let task_handle = tokio::spawn(async move {
            let mut actor = StrategyActor::new(strategy, rx);

            loop {
                match catch_panic(actor.run()).await {
                    Ok(()) => {
                        // Normal exit
                        break;
                    }
                    Err(panic_msg) => {
                        error!(
                            strategy_id = %id,
                            "Strategy panicked: {}",
                            panic_msg
                        );

                        // Restart strategy after delay
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        actor.reset();
                        continue;
                    }
                }
            }
        });

        self.strategies.insert(id, StrategyHandle {
            sender: tx,
            task_handle,
        });
    }
}
```

## Graceful Shutdown

```rust
// src/shutdown.rs
use tokio::sync::broadcast;
use tokio::signal;

pub struct ShutdownCoordinator {
    tx: broadcast::Sender<()>,
}

impl ShutdownCoordinator {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1);
        Self { tx }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<()> {
        self.tx.subscribe()
    }

    pub async fn wait_for_signal(&self) {
        let ctrl_c = async {
            signal::ctrl_c()
                .await
                .expect("Failed to install CTRL+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("Failed to install SIGTERM handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }

        tracing::info!("Shutdown signal received");
        let _ = self.tx.send(());
    }
}

pub async fn graceful_shutdown<F>(
    coordinator: &ShutdownCoordinator,
    cleanup: F,
) where
    F: Future<Output = ()>,
{
    let mut shutdown_rx = coordinator.subscribe();

    tokio::select! {
        _ = shutdown_rx.recv() => {
            tracing::info!("Starting graceful shutdown");

            // Run cleanup with timeout
            match tokio::time::timeout(Duration::from_secs(30), cleanup).await {
                Ok(_) => {
                    tracing::info!("Graceful shutdown complete");
                }
                Err(_) => {
                    tracing::warn!("Shutdown timeout, forcing exit");
                }
            }
        }
    }
}

// Usage example
#[tokio::main]
async fn main() {
    let coordinator = ShutdownCoordinator::new();

    // Spawn shutdown handler
    let shutdown_handle = {
        let coordinator = coordinator.clone();
        tokio::spawn(async move {
            coordinator.wait_for_signal().await;
        })
    };

    // Run application
    let app_handle = tokio::spawn(async move {
        // Application logic...
    });

    // Wait for shutdown
    graceful_shutdown(&coordinator, async {
        // Stop accepting new orders
        stop_order_intake().await;

        // Cancel open orders
        cancel_all_orders().await;

        // Close positions
        close_all_positions().await;

        // Flush logs
        flush_logs().await;

        // Disconnect from exchanges
        disconnect_all().await;
    }).await;
}
```

---

**Next:** [08-observability-security.md](./08-observability-security.md)
