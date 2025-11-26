# Observability Hooks and Security Boundaries

## Observability Architecture

### Three Pillars of Observability

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application                              │
└───────┬─────────────────────┬─────────────────────┬─────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│   Metrics    │      │    Traces    │     │     Logs     │
│  (Prometheus)│      │ (OpenTelemetry)     │  (Structured)│
└──────┬───────┘      └──────┬───────┘     └──────┬───────┘
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │  Export Pipeline │
                   └──────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Prometheus  │     │    Jaeger    │     │ Elasticsearch│
│   Server     │     │    Server    │     │   / Loki     │
└──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │   Grafana    │
                      │  (Dashboards)│
                      └──────────────┘
```

## Structured Logging with Tracing

### Configuration

```rust
// src/observability/logging.rs
use tracing::{Level, Subscriber};
use tracing_subscriber::{
    fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry,
};
use tracing_appender::{non_blocking, rolling};

pub fn init_logging() -> Result<(), Box<dyn std::error::Error>> {
    // Rolling file appender (daily rotation)
    let file_appender = rolling::daily("logs", "neural-trader.log");
    let (non_blocking_file, _guard) = non_blocking(file_appender);

    // Environment filter (from RUST_LOG env var)
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // JSON formatter for structured logs
    let json_layer = fmt::layer()
        .json()
        .with_writer(non_blocking_file)
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true);

    // Human-readable console output
    let console_layer = fmt::layer()
        .pretty()
        .with_writer(std::io::stdout)
        .with_target(false);

    // Combine layers
    Registry::default()
        .with(env_filter)
        .with(json_layer)
        .with(console_layer)
        .init();

    Ok(())
}
```

### Instrumentation

```rust
// src/observability/instrumentation.rs
use tracing::{debug, error, info, instrument, span, warn, Level};
use uuid::Uuid;

/// Instrument async functions with automatic span creation
#[instrument(
    skip(provider, symbol),
    fields(
        symbol = %symbol,
        request_id = %Uuid::new_v4(),
    )
)]
pub async fn fetch_market_data(
    provider: &MarketDataProvider,
    symbol: &Symbol,
) -> Result<Quote> {
    info!("Fetching market data");

    let quote = provider.get_quote(symbol).await?;

    debug!(
        bid = %quote.bid,
        ask = %quote.ask,
        "Quote received"
    );

    Ok(quote)
}

/// Manual span creation for fine-grained control
pub async fn process_signal(signal: Signal) -> Result<Order> {
    let span = span!(
        Level::INFO,
        "process_signal",
        signal_id = %signal.id,
        symbol = %signal.symbol,
        side = ?signal.side,
    );

    let _enter = span.enter();

    info!("Processing trading signal");

    // Risk check
    let risk_span = span!(Level::DEBUG, "risk_check");
    let _risk_enter = risk_span.enter();
    let risk_result = check_risk(&signal).await?;
    drop(_risk_enter);

    if !risk_result.approved {
        warn!(
            reason = %risk_result.reason,
            "Signal rejected by risk management"
        );
        return Err(RiskError::SignalRejected {
            reason: risk_result.reason,
        }.into());
    }

    // Create order
    let order_span = span!(Level::DEBUG, "create_order");
    let _order_enter = order_span.enter();
    let order = create_order_from_signal(&signal).await?;
    drop(_order_enter);

    info!(
        order_id = %order.id,
        "Order created successfully"
    );

    Ok(order)
}

/// Structured logging for events
pub fn log_order_filled(fill: &Fill) {
    info!(
        order_id = %fill.order_id,
        symbol = %fill.symbol,
        price = %fill.price,
        quantity = %fill.quantity,
        side = ?fill.side,
        commission = %fill.commission,
        "Order filled"
    );
}

/// Error logging with context
pub fn log_execution_error(error: &ExecutionError, context: &OrderContext) {
    error!(
        error = %error,
        order_id = %context.order_id,
        symbol = %context.symbol,
        retry_count = context.retry_count,
        "Order execution failed"
    );
}
```

## Metrics with Prometheus

### Metric Definitions

```rust
// src/observability/metrics.rs
use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec,
    IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Registry,
};
use once_cell::sync::Lazy;

/// Global Prometheus registry
pub static REGISTRY: Lazy<Registry> = Lazy::new(Registry::new);

/// Counter: Orders submitted (total count)
pub static ORDERS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    let counter = IntCounterVec::new(
        prometheus::opts!(
            "orders_total",
            "Total number of orders submitted"
        ),
        &["symbol", "side", "order_type"],
    )
    .unwrap();

    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

/// Counter: Orders filled
pub static ORDERS_FILLED: Lazy<IntCounterVec> = Lazy::new(|| {
    let counter = IntCounterVec::new(
        prometheus::opts!(
            "orders_filled_total",
            "Total number of orders filled"
        ),
        &["symbol", "side"],
    )
    .unwrap();

    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

/// Counter: Orders rejected
pub static ORDERS_REJECTED: Lazy<IntCounterVec> = Lazy::new(|| {
    let counter = IntCounterVec::new(
        prometheus::opts!(
            "orders_rejected_total",
            "Total number of orders rejected"
        ),
        &["symbol", "reason"],
    )
    .unwrap();

    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

/// Histogram: Order latency
pub static ORDER_LATENCY: Lazy<HistogramVec> = Lazy::new(|| {
    let histogram = HistogramVec::new(
        HistogramOpts::new(
            "order_latency_seconds",
            "Order submission latency in seconds"
        )
        .buckets(vec![0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]),
        &["symbol", "order_type"],
    )
    .unwrap();

    REGISTRY.register(Box::new(histogram.clone())).unwrap();
    histogram
});

/// Histogram: Signal generation latency
pub static SIGNAL_LATENCY: Lazy<HistogramVec> = Lazy::new(|| {
    let histogram = HistogramVec::new(
        HistogramOpts::new(
            "signal_generation_latency_seconds",
            "Signal generation latency in seconds"
        )
        .buckets(vec![0.001, 0.01, 0.05, 0.1, 0.5, 1.0]),
        &["strategy", "symbol"],
    )
    .unwrap();

    REGISTRY.register(Box::new(histogram.clone())).unwrap();
    histogram
});

/// Gauge: Active positions
pub static ACTIVE_POSITIONS: Lazy<IntGaugeVec> = Lazy::new(|| {
    let gauge = IntGaugeVec::new(
        prometheus::opts!(
            "active_positions",
            "Number of active positions"
        ),
        &["symbol", "side"],
    )
    .unwrap();

    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

/// Gauge: Portfolio value
pub static PORTFOLIO_VALUE: Lazy<Gauge> = Lazy::new(|| {
    let gauge = Gauge::new(
        "portfolio_value_usd",
        "Total portfolio value in USD"
    )
    .unwrap();

    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

/// Gauge: Unrealized PnL
pub static UNREALIZED_PNL: Lazy<GaugeVec> = Lazy::new(|| {
    let gauge = GaugeVec::new(
        prometheus::opts!(
            "unrealized_pnl_usd",
            "Unrealized profit/loss in USD"
        ),
        &["symbol"],
    )
    .unwrap();

    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

/// Histogram: Market data event processing time
pub static MARKET_DATA_LATENCY: Lazy<Histogram> = Lazy::new(|| {
    let histogram = Histogram::with_opts(
        HistogramOpts::new(
            "market_data_latency_seconds",
            "Market data event processing latency"
        )
        .buckets(vec![0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]),
    )
    .unwrap();

    REGISTRY.register(Box::new(histogram.clone())).unwrap();
    histogram
});

/// Counter: Total PnL
pub static REALIZED_PNL: Lazy<Counter> = Lazy::new(|| {
    let counter = Counter::new(
        "realized_pnl_usd_total",
        "Total realized profit/loss in USD"
    )
    .unwrap();

    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

/// Metrics HTTP endpoint
pub async fn metrics_handler() -> Result<String, Box<dyn std::error::Error>> {
    use prometheus::Encoder;

    let encoder = prometheus::TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();

    encoder.encode(&metric_families, &mut buffer)?;

    Ok(String::from_utf8(buffer)?)
}
```

### Metric Recording

```rust
// src/observability/recorder.rs
use std::time::Instant;

pub struct MetricsRecorder;

impl MetricsRecorder {
    pub fn record_order_submitted(order: &Order) {
        ORDERS_TOTAL
            .with_label_values(&[
                &order.symbol.to_string(),
                &format!("{:?}", order.side),
                &format!("{:?}", order.order_type),
            ])
            .inc();
    }

    pub fn record_order_filled(fill: &Fill) {
        ORDERS_FILLED
            .with_label_values(&[
                &fill.symbol.to_string(),
                &format!("{:?}", fill.side),
            ])
            .inc();

        // Update realized PnL
        if let Some(pnl) = fill.realized_pnl {
            REALIZED_PNL.inc_by(pnl as f64);
        }
    }

    pub fn record_order_rejected(symbol: &Symbol, reason: &str) {
        ORDERS_REJECTED
            .with_label_values(&[&symbol.to_string(), reason])
            .inc();
    }

    pub fn record_order_latency(symbol: &Symbol, order_type: &str, duration: Duration) {
        ORDER_LATENCY
            .with_label_values(&[&symbol.to_string(), order_type])
            .observe(duration.as_secs_f64());
    }

    pub fn update_portfolio_value(value: Decimal) {
        PORTFOLIO_VALUE.set(value.to_f64().unwrap());
    }

    pub fn update_position(symbol: &Symbol, side: Side, quantity: Decimal) {
        let qty = quantity.to_f64().unwrap() as i64;
        ACTIVE_POSITIONS
            .with_label_values(&[&symbol.to_string(), &format!("{:?}", side)])
            .set(qty);
    }

    pub fn update_unrealized_pnl(symbol: &Symbol, pnl: Decimal) {
        UNREALIZED_PNL
            .with_label_values(&[&symbol.to_string()])
            .set(pnl.to_f64().unwrap());
    }
}

/// Timer helper for automatic duration recording
pub struct TimedOperation {
    start: Instant,
    operation: String,
}

impl TimedOperation {
    pub fn start(operation: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.into(),
        }
    }

    pub fn record_histogram(self, histogram: &Histogram) {
        histogram.observe(self.start.elapsed().as_secs_f64());
    }

    pub fn record_histogram_vec(self, histogram: &HistogramVec, labels: &[&str]) {
        histogram
            .with_label_values(labels)
            .observe(self.start.elapsed().as_secs_f64());
    }
}

// Usage
async fn submit_order(order: Order) -> Result<OrderId> {
    let timer = TimedOperation::start("submit_order");

    MetricsRecorder::record_order_submitted(&order);

    let result = execute_order_submission(&order).await?;

    timer.record_histogram_vec(
        &ORDER_LATENCY,
        &[&order.symbol.to_string(), "market"],
    );

    Ok(result)
}
```

## Distributed Tracing with OpenTelemetry

```rust
// src/observability/tracing.rs
use opentelemetry::{
    global, sdk::{trace, Resource},
    trace::{Tracer, TracerProvider},
    KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use tracing_subscriber::{layer::SubscriberExt, Registry};

pub fn init_tracing() -> Result<(), Box<dyn std::error::Error>> {
    // Configure OpenTelemetry OTLP exporter
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint("http://localhost:4317"),
        )
        .with_trace_config(
            trace::config().with_resource(Resource::new(vec![
                KeyValue::new("service.name", "neural-trader"),
                KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
            ])),
        )
        .install_batch(opentelemetry::runtime::Tokio)?;

    // Create tracing layer
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

    // Combine with other layers
    let subscriber = Registry::default()
        .with(telemetry)
        .with(tracing_subscriber::fmt::layer());

    tracing::subscriber::set_global_default(subscriber)?;

    Ok(())
}

/// Custom spans for distributed tracing
#[instrument(skip_all, fields(
    trace_id = %Uuid::new_v4(),
    symbol = %signal.symbol,
))]
pub async fn execute_trading_workflow(signal: Signal) -> Result<()> {
    // This span will be part of a distributed trace

    // Fetch market data (could be from remote service)
    let market_data = fetch_market_data(&signal.symbol).await?;

    // Generate order (local processing)
    let order = create_order(&signal, &market_data).await?;

    // Submit to exchange (remote service)
    let order_id = submit_order_to_exchange(&order).await?;

    // Wait for fill (async event)
    wait_for_fill(&order_id).await?;

    Ok(())
}
```

## Security Architecture

### Principle: Defense in Depth

```
┌─────────────────────────────────────────────────────────────────┐
│                      Security Layers                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Layer 1: Network Security (TLS, Firewall)                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Layer 2: Authentication (API Keys, JWT, mTLS)             │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Layer 3: Authorization (RBAC, Policy Engine)              │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Layer 4: Input Validation (Type System, Sanitization)     │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Layer 5: Privilege Separation (Sandboxing, Capabilities)  │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Layer 6: Audit Logging (Immutable Logs, Signatures)       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Secrets Management

```rust
// src/security/secrets.rs
use secrecy::{ExposeSecret, Secret};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Credentials {
    #[serde(deserialize_with = "deserialize_secret")]
    pub api_key: Secret<String>,

    #[serde(deserialize_with = "deserialize_secret")]
    pub secret_key: Secret<String>,
}

impl Credentials {
    pub fn from_env() -> Result<Self, ConfigError> {
        let api_key = std::env::var("API_KEY")
            .map_err(|_| ConfigError::MissingField {
                field: "API_KEY".to_string(),
            })?;

        let secret_key = std::env::var("SECRET_KEY")
            .map_err(|_| ConfigError::MissingField {
                field: "SECRET_KEY".to_string(),
            })?;

        Ok(Self {
            api_key: Secret::new(api_key),
            secret_key: Secret::new(secret_key),
        })
    }

    /// Use the secret (logs redacted)
    pub fn sign_request(&self, message: &str) -> String {
        let key = self.secret_key.expose_secret();
        // HMAC signing...
        sign_hmac_sha256(message, key)
    }
}

// Secrets are not printed in logs
impl std::fmt::Display for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Credentials {{ api_key: [REDACTED], secret_key: [REDACTED] }}")
    }
}

fn deserialize_secret<'de, D>(deserializer: D) -> Result<Secret<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(Secret::new(s))
}
```

### Authentication and Authorization

```rust
// src/security/auth.rs
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,         // Subject (user ID)
    pub exp: usize,          // Expiration timestamp
    pub iat: usize,          // Issued at timestamp
    pub roles: Vec<String>,  // User roles
}

pub struct AuthService {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
}

impl AuthService {
    pub fn new(secret: &[u8]) -> Self {
        Self {
            encoding_key: EncodingKey::from_secret(secret),
            decoding_key: DecodingKey::from_secret(secret),
        }
    }

    pub fn create_token(&self, user_id: &str, roles: Vec<String>) -> Result<String> {
        let now = chrono::Utc::now().timestamp() as usize;
        let claims = Claims {
            sub: user_id.to_string(),
            exp: now + 3600,  // 1 hour
            iat: now,
            roles,
        };

        encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(|e| AuthError::TokenCreationFailed { reason: e.to_string() })
    }

    pub fn validate_token(&self, token: &str) -> Result<Claims> {
        let validation = Validation::new(Algorithm::HS256);

        decode::<Claims>(token, &self.decoding_key, &validation)
            .map(|data| data.claims)
            .map_err(|e| AuthError::InvalidToken { reason: e.to_string() })
    }

    pub fn check_permission(&self, claims: &Claims, required_role: &str) -> bool {
        claims.roles.iter().any(|role| role == required_role)
    }
}

/// Middleware for authentication
pub async fn auth_middleware<B>(
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    // Extract bearer token
    let token = req
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // Validate token
    let auth_service = req.extensions().get::<AuthService>().unwrap();
    let claims = auth_service
        .validate_token(token)
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    // Add claims to request extensions
    req.extensions_mut().insert(claims);

    Ok(next.run(req).await)
}
```

### Input Validation

```rust
// src/security/validation.rs
use validator::{Validate, ValidationError};

#[derive(Debug, Validate, Deserialize)]
pub struct OrderRequest {
    #[validate(length(min = 1, max = 10))]
    pub symbol: String,

    #[validate(custom = "validate_decimal_positive")]
    pub quantity: String,

    #[validate(custom = "validate_decimal_positive")]
    pub price: Option<String>,

    #[validate(range(min = 1.0, max = 100.0))]
    pub stop_loss_percent: Option<f64>,
}

fn validate_decimal_positive(value: &str) -> Result<(), ValidationError> {
    match Decimal::from_str(value) {
        Ok(d) if d > Decimal::ZERO => Ok(()),
        _ => Err(ValidationError::new("must be positive")),
    }
}

pub fn sanitize_symbol(symbol: &str) -> String {
    // Remove non-alphanumeric characters
    symbol
        .chars()
        .filter(|c| c.is_alphanumeric())
        .take(10)
        .collect()
}
```

### Audit Logging

```rust
// src/security/audit.rs
use sha2::{Digest, Sha256};

#[derive(Debug, Serialize)]
pub struct AuditEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub user_id: Option<String>,
    pub action: String,
    pub resource: String,
    pub result: AuditResult,
    pub metadata: HashMap<String, String>,
    pub signature: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum AuditEventType {
    OrderSubmitted,
    OrderFilled,
    OrderCancelled,
    ConfigChanged,
    AuthenticationFailed,
    AuthorizationDenied,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum AuditResult {
    Success,
    Failure { reason: String },
}

pub struct AuditLogger {
    signing_key: Secret<String>,
}

impl AuditLogger {
    pub fn new(signing_key: Secret<String>) -> Self {
        Self { signing_key }
    }

    pub async fn log_event(&self, event: AuditEvent) {
        // Sign event
        let mut signed_event = event;
        signed_event.signature = self.sign_event(&signed_event);

        // Write to append-only log
        self.write_to_log(&signed_event).await;

        // Send to SIEM
        self.send_to_siem(&signed_event).await;
    }

    fn sign_event(&self, event: &AuditEvent) -> String {
        let payload = serde_json::to_string(event).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(payload);
        hasher.update(self.signing_key.expose_secret());
        format!("{:x}", hasher.finalize())
    }

    async fn write_to_log(&self, event: &AuditEvent) {
        // Append to immutable log file
        let json = serde_json::to_string(event).unwrap();
        // Write with O_APPEND flag to prevent overwrites
    }

    async fn send_to_siem(&self, event: &AuditEvent) {
        // Send to security information and event management system
    }
}
```

### Rate Limiting

```rust
// src/security/rate_limit.rs
use governor::{Quota, RateLimiter as GovernorRateLimiter};
use std::num::NonZeroU32;

pub struct RateLimiter {
    limiter: GovernorRateLimiter<String, DashMap<String, InMemoryState>, DefaultClock>,
}

impl RateLimiter {
    pub fn new(requests_per_minute: u32) -> Self {
        let quota = Quota::per_minute(NonZeroU32::new(requests_per_minute).unwrap());
        Self {
            limiter: GovernorRateLimiter::keyed(quota),
        }
    }

    pub async fn check(&self, key: &str) -> Result<(), RateLimitError> {
        self.limiter
            .check_key(&key.to_string())
            .map_err(|_| RateLimitError::Exceeded)?;

        Ok(())
    }
}

// Usage in middleware
pub async fn rate_limit_middleware(
    req: Request<Body>,
    next: Next<Body>,
) -> Result<Response, StatusCode> {
    let user_id = req.extensions().get::<Claims>()
        .map(|c| c.sub.clone())
        .unwrap_or_else(|| "anonymous".to_string());

    let rate_limiter = req.extensions().get::<RateLimiter>().unwrap();

    rate_limiter
        .check(&user_id)
        .await
        .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;

    Ok(next.run(req).await)
}
```

---

**Next:** [09-build-configuration.md](./09-build-configuration.md)
