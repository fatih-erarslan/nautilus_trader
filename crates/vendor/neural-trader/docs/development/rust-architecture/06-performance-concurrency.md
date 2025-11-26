# Performance Targets and Concurrency Model

## Performance Targets

### Latency Requirements

#### P99 Latency Targets (99th percentile)

| Operation | Target | Max Acceptable | Current Python |
|-----------|--------|----------------|----------------|
| Market data event ingestion | <1ms | <5ms | ~50ms |
| DataFrame construction (1K rows) | <5ms | <10ms | ~100ms |
| Feature extraction (20 indicators) | <10ms | <50ms | ~200ms |
| Signal generation (1 symbol) | <50ms | <200ms | ~500ms |
| Risk validation | <10ms | <50ms | ~100ms |
| Order submission | <100ms | <500ms | ~1s |
| End-to-end (data→order) | <200ms | <1s | ~2s |

**Target Improvement:** 10x faster than Python implementation

#### P50 Latency Targets (median)

| Operation | Target |
|-----------|--------|
| Market data ingestion | <0.5ms |
| DataFrame construction | <2ms |
| Feature extraction | <5ms |
| Signal generation | <20ms |
| Order execution | <50ms |

### Throughput Requirements

| Metric | Baseline | Peak | Scaling Target |
|--------|----------|------|----------------|
| Market data events/sec | 10,000 | 100,000 | 1,000,000 |
| DataFrames/sec | 100 | 1,000 | 10,000 |
| Feature calculations/sec | 1,000 | 10,000 | 100,000 |
| Signal evaluations/sec | 100 | 1,000 | 10,000 |
| Order submissions/sec | 10 | 100 | 1,000 |

### Resource Utilization

#### Memory

| Component | Per Instance | 100 Instances | Target |
|-----------|-------------|---------------|--------|
| Market data buffer | 10MB | 1GB | Keep <2GB |
| Feature cache | 50MB | 5GB | Keep <10GB |
| Strategy state | 5MB | 500MB | Keep <1GB |
| AgentDB connection | 1MB | 100MB | Keep <200MB |
| **Total baseline** | **66MB** | **6.6GB** | **<15GB** |

**Python comparison:** 5-10x memory reduction

#### CPU

| Load Profile | Target CPU % | Acceptable | Current Python |
|--------------|-------------|-----------|----------------|
| Idle | <5% | <10% | ~20% |
| Normal trading | 20-40% | <60% | ~80% |
| Peak (100 strategies) | 60-80% | <95% | 100% (maxed) |

**Target:** <50% CPU under normal load, <80% at peak

#### Disk I/O

| Operation | Rate | Burst |
|-----------|------|-------|
| Market data writes | 10MB/s | 100MB/s |
| Feature reads/writes | 5MB/s | 50MB/s |
| Log writes | 1MB/s | 10MB/s |

### Efficiency Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Zero-copy operations | >95% | % data transfers without memcpy |
| Lock contention | <5% | % time waiting on locks |
| Async task utilization | >90% | % time tasks are runnable |
| Cache hit rate | >80% | % feature lookups from cache |
| GC overhead | 0% | No garbage collection in Rust |

---

## Concurrency Model

### Tokio Runtime Configuration

```rust
// src/runtime.rs
use tokio::runtime::{Builder, Runtime};
use std::time::Duration;

pub fn create_runtime() -> Runtime {
    Builder::new_multi_thread()
        .worker_threads(num_cpus::get())        // One thread per CPU core
        .max_blocking_threads(512)              // Pool for blocking operations
        .thread_name("nt-worker")
        .thread_stack_size(4 * 1024 * 1024)    // 4MB stack
        .enable_all()                           // Enable I/O and time drivers
        .build()
        .expect("Failed to create Tokio runtime")
}

pub fn create_custom_runtime(config: RuntimeConfig) -> Runtime {
    let mut builder = Builder::new_multi_thread();

    builder
        .worker_threads(config.worker_threads)
        .max_blocking_threads(config.max_blocking_threads)
        .thread_keep_alive(Duration::from_secs(60))
        .global_queue_interval(31)              // Check global queue every 31 tasks
        .event_interval(61);                    // Poll for I/O every 61 polls

    if config.enable_time {
        builder.enable_time();
    }

    if config.enable_io {
        builder.enable_io();
    }

    builder.build().expect("Failed to create runtime")
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub worker_threads: usize,
    pub max_blocking_threads: usize,
    pub enable_time: bool,
    pub enable_io: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            max_blocking_threads: 512,
            enable_time: true,
            enable_io: true,
        }
    }
}
```

### Threading Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tokio Work-Stealing Runtime                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Worker 0 │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  ...  │
│  │ (Core 0) │  │ (Core 1) │  │ (Core 2) │  │ (Core 3) │       │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘       │
│        │             │             │             │             │
│        └─────────────┴─────────────┴─────────────┘             │
│                      │                                          │
│              Global Task Queue                                  │
└─────────────────────┴───────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬──────────────┐
        │             │             │              │
        ▼             ▼             ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Market Data  │ │ Strategy │ │ Signal   │ │ Order        │
│ Tasks        │ │ Tasks    │ │ Tasks    │ │ Execution    │
└──────────────┘ └──────────┘ └──────────┘ └──────────────┘

Separate thread pool for blocking operations:
┌─────────────────────────────────────────────────────────────────┐
│                   Blocking Thread Pool (512 max)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ Thread 0 │  │ Thread 1 │  │ Thread 2 │  ...                │
│  │ (Disk I/O│  │ (Crypto) │  │ (CPU-    │                     │
│  │  ops)    │  │          │  │  heavy)  │                     │
│  └──────────┘  └──────────┘  └──────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### Task Prioritization

```rust
// src/scheduling.rs
use tokio::task::JoinHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Critical = 0,    // Order execution, risk checks
    High = 1,        // Signal generation, market data
    Medium = 2,      // Feature calculation
    Low = 3,         // Logging, metrics
    Background = 4,  // Cleanup, maintenance
}

pub struct PriorityTask<T> {
    priority: TaskPriority,
    handle: JoinHandle<T>,
}

impl<T> PriorityTask<T> {
    pub fn spawn_with_priority<F>(priority: TaskPriority, task: F) -> Self
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        // Tokio doesn't have built-in priority scheduling,
        // but we can use task_local storage and custom executor
        let handle = tokio::spawn(async move {
            // Set task priority in thread-local
            TASK_PRIORITY.scope(priority, task).await
        });

        Self { priority, handle }
    }

    pub async fn await_result(self) -> Result<T, tokio::task::JoinError> {
        self.handle.await
    }
}

tokio::task_local! {
    static TASK_PRIORITY: TaskPriority;
}

// Usage
let critical_task = PriorityTask::spawn_with_priority(
    TaskPriority::Critical,
    async {
        submit_order(order).await
    }
);

let background_task = PriorityTask::spawn_with_priority(
    TaskPriority::Background,
    async {
        cleanup_old_logs().await
    }
);
```

### Actor Pattern for Strategies

Each strategy runs as an independent actor with its own mailbox:

```rust
// src/actors/strategy_actor.rs
use tokio::sync::mpsc;

pub struct StrategyActor {
    strategy: Box<dyn Strategy>,
    mailbox: mpsc::Receiver<StrategyMessage>,
    signal_tx: mpsc::Sender<Signal>,
}

pub enum StrategyMessage {
    MarketData(MarketData),
    UpdateConfig(StrategyConfig),
    Stop,
}

impl StrategyActor {
    pub fn spawn(
        strategy: Box<dyn Strategy>,
        signal_tx: mpsc::Sender<Signal>,
    ) -> mpsc::Sender<StrategyMessage> {
        let (tx, rx) = mpsc::channel(1000);

        let mut actor = Self {
            strategy,
            mailbox: rx,
            signal_tx,
        };

        tokio::spawn(async move {
            actor.run().await;
        });

        tx
    }

    async fn run(&mut self) {
        while let Some(message) = self.mailbox.recv().await {
            match message {
                StrategyMessage::MarketData(data) => {
                    match self.strategy.on_market_data(data).await {
                        Ok(signals) => {
                            for signal in signals {
                                let _ = self.signal_tx.send(signal).await;
                            }
                        }
                        Err(e) => {
                            tracing::error!("Strategy error: {}", e);
                        }
                    }
                }
                StrategyMessage::UpdateConfig(config) => {
                    if let Err(e) = self.strategy.initialize(config).await {
                        tracing::error!("Config update failed: {}", e);
                    }
                }
                StrategyMessage::Stop => {
                    break;
                }
            }
        }
    }
}

// Supervisor pattern
pub struct StrategySupervisor {
    strategies: HashMap<String, mpsc::Sender<StrategyMessage>>,
    signal_tx: mpsc::Sender<Signal>,
}

impl StrategySupervisor {
    pub fn spawn_strategy(&mut self, id: String, strategy: Box<dyn Strategy>) {
        let tx = StrategyActor::spawn(strategy, self.signal_tx.clone());
        self.strategies.insert(id, tx);
    }

    pub async fn broadcast_market_data(&self, data: MarketData) {
        for tx in self.strategies.values() {
            let _ = tx.send(StrategyMessage::MarketData(data.clone())).await;
        }
    }

    pub async fn stop_strategy(&mut self, id: &str) {
        if let Some(tx) = self.strategies.remove(id) {
            let _ = tx.send(StrategyMessage::Stop).await;
        }
    }
}
```

### Channel-Based Communication

```rust
// src/channels.rs
use tokio::sync::{mpsc, broadcast, watch};

pub struct ChannelConfig {
    // MPSC for point-to-point (single consumer)
    pub market_data_buffer: usize,      // 10,000
    pub signal_buffer: usize,            // 1,000
    pub order_buffer: usize,             // 100

    // Broadcast for pub-sub (multiple consumers)
    pub event_bus_buffer: usize,         // 1,000

    // Watch for state updates (always latest value)
    // No buffer needed, watch holds single value
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            market_data_buffer: 10_000,
            signal_buffer: 1_000,
            order_buffer: 100,
            event_bus_buffer: 1_000,
        }
    }
}

pub struct Channels {
    // Market data: many producers (WebSocket handlers), many consumers (strategies)
    pub market_data_tx: broadcast::Sender<MarketData>,
    pub market_data_rx: broadcast::Receiver<MarketData>,

    // Signals: many producers (strategies), one consumer (risk manager)
    pub signal_tx: mpsc::Sender<Signal>,
    pub signal_rx: mpsc::Receiver<Signal>,

    // Orders: one producer (order manager), one consumer (executor)
    pub order_tx: mpsc::Sender<Order>,
    pub order_rx: mpsc::Receiver<Order>,

    // Portfolio state: one producer, many consumers
    pub portfolio_state_tx: watch::Sender<PortfolioState>,
    pub portfolio_state_rx: watch::Receiver<PortfolioState>,
}

impl Channels {
    pub fn new(config: ChannelConfig) -> Self {
        let (market_data_tx, market_data_rx) =
            broadcast::channel(config.market_data_buffer);

        let (signal_tx, signal_rx) =
            mpsc::channel(config.signal_buffer);

        let (order_tx, order_rx) =
            mpsc::channel(config.order_buffer);

        let (portfolio_state_tx, portfolio_state_rx) =
            watch::channel(PortfolioState::default());

        Self {
            market_data_tx,
            market_data_rx,
            signal_tx,
            signal_rx,
            order_tx,
            order_rx,
            portfolio_state_tx,
            portfolio_state_rx,
        }
    }
}
```

### Lock-Free Data Structures

```rust
// src/lockfree.rs
use crossbeam::queue::{ArrayQueue, SegQueue};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;

/// Lock-free position cache
pub struct PositionCache {
    // DashMap uses sharded locking (low contention)
    positions: DashMap<Symbol, Position>,
}

impl PositionCache {
    pub fn new() -> Self {
        Self {
            positions: DashMap::new(),
        }
    }

    pub fn update(&self, symbol: Symbol, position: Position) {
        self.positions.insert(symbol, position);
    }

    pub fn get(&self, symbol: &Symbol) -> Option<Position> {
        self.positions.get(symbol).map(|p| p.clone())
    }

    pub fn remove(&self, symbol: &Symbol) -> Option<Position> {
        self.positions.remove(symbol).map(|(_, p)| p)
    }
}

/// Lock-free order queue (bounded)
pub struct OrderQueue {
    queue: Arc<ArrayQueue<Order>>,
}

impl OrderQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(ArrayQueue::new(capacity)),
        }
    }

    pub fn try_push(&self, order: Order) -> Result<(), Order> {
        self.queue.push(order)
    }

    pub fn try_pop(&self) -> Option<Order> {
        self.queue.pop()
    }
}

/// Lock-free event log (unbounded)
pub struct EventLog {
    log: Arc<SegQueue<Event>>,
}

impl EventLog {
    pub fn new() -> Self {
        Self {
            log: Arc::new(SegQueue::new()),
        }
    }

    pub fn append(&self, event: Event) {
        self.log.push(event);
    }

    pub fn drain(&self) -> Vec<Event> {
        let mut events = Vec::new();
        while let Some(event) = self.log.pop() {
            events.push(event);
        }
        events
    }
}
```

### Backpressure Handling

```rust
// src/backpressure.rs
use tokio::sync::mpsc;
use tokio::time::{timeout, Duration};

pub struct BackpressureConfig {
    pub buffer_high_watermark: f32,  // 0.8 = 80% full
    pub buffer_low_watermark: f32,   // 0.3 = 30% full
    pub slow_down_factor: f32,       // 0.5 = 50% speed
}

pub struct BackpressureControl {
    config: BackpressureConfig,
    current_rate: Arc<RwLock<f32>>,  // 0.0 to 1.0
}

impl BackpressureControl {
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            config,
            current_rate: Arc::new(RwLock::new(1.0)),
        }
    }

    pub async fn send_with_backpressure<T>(
        &self,
        tx: &mpsc::Sender<T>,
        item: T,
    ) -> Result<(), mpsc::error::SendError<T>> {
        // Check buffer capacity
        let capacity = tx.capacity();
        let available = tx.max_capacity() - capacity;
        let utilization = available as f32 / tx.max_capacity() as f32;

        // Adjust rate based on utilization
        let mut rate = self.current_rate.write();
        if utilization > self.config.buffer_high_watermark {
            *rate *= self.config.slow_down_factor;
        } else if utilization < self.config.buffer_low_watermark {
            *rate = (*rate * 1.5).min(1.0);
        }
        drop(rate);

        // Apply rate limit
        let rate = *self.current_rate.read();
        if rate < 1.0 {
            let delay = Duration::from_secs_f32((1.0 - rate) * 0.1);
            tokio::time::sleep(delay).await;
        }

        // Send with timeout
        match timeout(Duration::from_secs(5), tx.send(item)).await {
            Ok(result) => result,
            Err(_) => {
                tracing::warn!("Send timeout - channel blocked");
                Err(mpsc::error::SendError(/* can't recover item */))
            }
        }
    }

    pub fn current_rate(&self) -> f32 {
        *self.current_rate.read()
    }
}
```

### SIMD Acceleration

```rust
// src/simd.rs
use std::simd::prelude::*;

/// SIMD-accelerated moving average calculation
pub fn calculate_sma_simd(prices: &[f32], window: usize) -> Vec<f32> {
    if prices.len() < window {
        return Vec::new();
    }

    let mut results = Vec::with_capacity(prices.len() - window + 1);
    let window_f32 = window as f32;

    // Process 8 values at a time with SIMD
    const LANES: usize = 8;
    let chunks = (prices.len() - window + 1) / LANES;

    for i in 0..chunks {
        let base_idx = i * LANES;
        let mut sums = f32x8::splat(0.0);

        // Calculate window sum for 8 positions simultaneously
        for j in 0..window {
            let idx = base_idx + j;
            let values = f32x8::from_slice(&prices[idx..idx + LANES]);
            sums += values;
        }

        let averages = sums / f32x8::splat(window_f32);
        results.extend_from_slice(averages.as_array());
    }

    // Handle remaining elements
    for i in (chunks * LANES)..(prices.len() - window + 1) {
        let sum: f32 = prices[i..i + window].iter().sum();
        results.push(sum / window_f32);
    }

    results
}

/// SIMD dot product for feature vectors
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    const LANES: usize = 8;
    let chunks = a.len() / LANES;
    let mut sum = f32x8::splat(0.0);

    for i in 0..chunks {
        let idx = i * LANES;
        let a_vec = f32x8::from_slice(&a[idx..idx + LANES]);
        let b_vec = f32x8::from_slice(&b[idx..idx + LANES]);
        sum += a_vec * b_vec;
    }

    let mut result: f32 = sum.as_array().iter().sum();

    // Handle remaining elements
    for i in (chunks * LANES)..a.len() {
        result += a[i] * b[i];
    }

    result
}
```

### CPU Pinning (Linux)

```rust
// src/cpu_affinity.rs
#[cfg(target_os = "linux")]
pub fn pin_thread_to_core(core: usize) -> std::io::Result<()> {
    use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};

    unsafe {
        let mut cpuset: cpu_set_t = std::mem::zeroed();
        CPU_ZERO(&mut cpuset);
        CPU_SET(core, &mut cpuset);

        let result = sched_setaffinity(
            0,  // Current thread
            std::mem::size_of::<cpu_set_t>(),
            &cpuset,
        );

        if result == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

pub fn configure_latency_sensitive_threads() {
    // Pin critical threads to specific cores
    std::thread::spawn(|| {
        pin_thread_to_core(0).expect("Failed to pin market data thread");
        market_data_loop();
    });

    std::thread::spawn(|| {
        pin_thread_to_core(1).expect("Failed to pin order execution thread");
        order_execution_loop();
    });
}
```

---

## Performance Monitoring

```rust
// src/monitoring.rs
use std::time::Instant;
use prometheus::{Histogram, HistogramVec};

lazy_static! {
    static ref OPERATION_DURATION: HistogramVec = HistogramVec::new(
        histogram_opts!(
            "operation_duration_seconds",
            "Duration of operations",
            vec![0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        ),
        &["operation"]
    ).unwrap();
}

pub struct PerformanceTimer {
    operation: String,
    start: Instant,
}

impl PerformanceTimer {
    pub fn start(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            start: Instant::now(),
        }
    }
}

impl Drop for PerformanceTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_secs_f64();
        OPERATION_DURATION
            .with_label_values(&[&self.operation])
            .observe(duration);
    }
}

// Usage
async fn process_signal(signal: Signal) -> Result<()> {
    let _timer = PerformanceTimer::start("signal_processing");
    // Processing logic...
    Ok(())
}
```

---

**Next:** [07-error-handling.md](./07-error-handling.md)
