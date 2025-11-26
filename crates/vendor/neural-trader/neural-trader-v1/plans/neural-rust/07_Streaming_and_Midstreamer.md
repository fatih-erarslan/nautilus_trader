# Neural Trading Rust Port - Event Streaming and Midstreamer Architecture

**Version:** 1.0.0
**Date:** 2025-11-12
**Status:** Design Complete
**Cross-References:** [Architecture](03_Architecture.md) | [Strategies](06_Strategy_and_Sublinear_Solvers.md) | [Memory](05_Memory_and_AgentDB.md)

---

## Table of Contents

1. [Complete Event Streaming Graph](#complete-event-streaming-graph)
2. [Midstreamer Operators and Backpressure](#midstreamer-operators-and-backpressure)
3. [Latency Budgets Per Stage](#latency-budgets-per-stage)
4. [Recovery and Replay Mechanisms](#recovery-and-replay-mechanisms)
5. [Observable Hooks for Monitoring](#observable-hooks-for-monitoring)
6. [Integration with Tokio Channels](#integration-with-tokio-channels)
7. [WebSocket Handling Patterns](#websocket-handling-patterns)
8. [Event Sourcing Patterns](#event-sourcing-patterns)

---

## Complete Event Streaming Graph

### End-to-End Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    Event Streaming Architecture                   │
└──────────────────────────────────────────────────────────────────┘

External Sources          Ingestion Layer       Processing Pipeline
─────────────────         ───────────────       ───────────────────
                                │
┌─────────────┐                 │
│ Alpaca WS   │────WebSocket────┤
└─────────────┘                 │
                                ▼
┌─────────────┐           ┌──────────┐
│ Polygon API │──REST─────│  Ticks   │
└─────────────┘           │ (Raw)    │
                          └────┬─────┘
┌─────────────┐                │
│ NewsAPI     │──────RSS────┐  │
└─────────────┘             │  │
                            ▼  ▼
                       ┌────────────┐
                       │ Event Bus  │
                       │ (Tokio)    │
                       └─────┬──────┘
                             │
                 ┌───────────┼───────────┐
                 │           │           │
                 ▼           ▼           ▼
          ┌──────────┐ ┌──────────┐ ┌──────────┐
          │ Tick     │ │ News     │ │ Quote    │
          │ Stream   │ │ Stream   │ │ Stream   │
          └────┬─────┘ └────┬─────┘ └────┬─────┘
               │            │            │
               │    ┌───────┴────────┐   │
               └────►                ◄───┘
                    │  Aggregator    │
                    │  (Windowing)   │
                    └────────┬───────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Bar Builder     │
                    │ (1m/5m/1h/1d)   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Feature         │
                    │ Extractor       │
                    │ (SMA/RSI/MACD)  │
                    └────────┬────────┘
                             │
                 ┌───────────┼───────────┐
                 │           │           │
                 ▼           ▼           ▼
          ┌──────────┐ ┌──────────┐ ┌──────────┐
          │Strategy 1│ │Strategy 2│ │Strategy 3│
          │(Momentum)│ │(Mirror)  │ │(Mean Rev)│
          └────┬─────┘ └────┬─────┘ └────┬─────┘
               │            │            │
               └────────────┼────────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Signal           │
                   │ Aggregator       │
                   │ (Dedup/Fusion)   │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Risk             │
                   │ Manager          │
                   │ (Validate)       │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Order            │
                   │ Manager          │
                   │ (Execute)        │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Broker API       │
                   │ (Alpaca)         │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Fill             │
                   │ Tracker          │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Portfolio        │
                   │ Updater          │
                   └────────┬─────────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
                ▼           ▼           ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐
         │ AgentDB  │ │ Session  │ │ Metrics  │
         │ Storage  │ │ Memory   │ │ Export   │
         └──────────┘ └──────────┘ └──────────┘

Performance Targets:
─────────────────────
Market Data → Bar:       <1ms (p99)
Bar → Features:          <10ms (p95)
Features → Signal:       <50ms (p95)
Signal → Risk Check:     <5ms (p99)
Risk Check → Order:      <10ms (p99)
Order → Broker:          <100ms (external)
Fill → Portfolio Update: <2ms (p99)
──────────────────────────────────────
TOTAL END-TO-END:        <178ms (p95)
```

---

## Midstreamer Operators and Backpressure

### Core Stream Operators

```rust
use tokio::sync::mpsc;
use futures::Stream;

/// Stream operator trait
pub trait StreamOperator<I, O>: Send + Sync {
    fn process(&mut self, input: I) -> Vec<O>;
}

/// Map operator: Transform each item
pub struct MapOperator<I, O, F>
where
    F: Fn(I) -> O + Send + Sync,
{
    f: F,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F> MapOperator<I, O, F>
where
    F: Fn(I) -> O + Send + Sync,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

impl<I, O, F> StreamOperator<I, O> for MapOperator<I, O, F>
where
    I: Send,
    O: Send,
    F: Fn(I) -> O + Send + Sync,
{
    fn process(&mut self, input: I) -> Vec<O> {
        vec![(self.f)(input)]
    }
}

/// Filter operator: Keep only matching items
pub struct FilterOperator<I, F>
where
    F: Fn(&I) -> bool + Send + Sync,
{
    predicate: F,
    _phantom: PhantomData<I>,
}

impl<I, F> StreamOperator<I, I> for FilterOperator<I, F>
where
    I: Send + Clone,
    F: Fn(&I) -> bool + Send + Sync,
{
    fn process(&mut self, input: I) -> Vec<I> {
        if (self.predicate)(&input) {
            vec![input]
        } else {
            vec![]
        }
    }
}

/// FlatMap operator: One-to-many transformation
pub struct FlatMapOperator<I, O, F>
where
    F: Fn(I) -> Vec<O> + Send + Sync,
{
    f: F,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F> StreamOperator<I, O> for FlatMapOperator<I, O, F>
where
    I: Send,
    O: Send,
    F: Fn(I) -> Vec<O> + Send + Sync,
{
    fn process(&mut self, input: I) -> Vec<O> {
        (self.f)(input)
    }
}

/// Window operator: Aggregate over time window
pub struct WindowOperator<I> {
    window: Vec<I>,
    window_size: usize,
    window_duration: Duration,
    last_emit: Instant,
}

impl<I: Clone> WindowOperator<I> {
    pub fn new(window_size: usize, window_duration: Duration) -> Self {
        Self {
            window: Vec::with_capacity(window_size),
            window_size,
            window_duration,
            last_emit: Instant::now(),
        }
    }

    pub fn add(&mut self, item: I) -> Option<Vec<I>> {
        self.window.push(item);

        // Emit if window is full or duration expired
        if self.window.len() >= self.window_size
            || self.last_emit.elapsed() >= self.window_duration
        {
            let result = Some(self.window.clone());
            self.window.clear();
            self.last_emit = Instant::now();
            result
        } else {
            None
        }
    }
}
```

---

### Backpressure Handling

```rust
pub struct StreamPipeline<I, O> {
    input_rx: mpsc::Receiver<I>,
    output_tx: mpsc::Sender<O>,
    operators: Vec<Box<dyn StreamOperator<I, O>>>,
    buffer_size: usize,
    backpressure_strategy: BackpressureStrategy,
}

pub enum BackpressureStrategy {
    /// Block sender until buffer has space
    Block,

    /// Drop oldest items when buffer is full
    DropOldest,

    /// Drop newest items when buffer is full
    DropNewest,

    /// Sample: Keep only every Nth item when under pressure
    Sample(usize),
}

impl<I, O> StreamPipeline<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    pub async fn run(mut self) {
        let mut buffer = VecDeque::with_capacity(self.buffer_size);

        while let Some(input) = self.input_rx.recv().await {
            // Check buffer capacity
            if buffer.len() >= self.buffer_size {
                match self.backpressure_strategy {
                    BackpressureStrategy::Block => {
                        // Wait for buffer to drain
                        while buffer.len() >= self.buffer_size {
                            tokio::task::yield_now().await;
                        }
                    }
                    BackpressureStrategy::DropOldest => {
                        buffer.pop_front();
                    }
                    BackpressureStrategy::DropNewest => {
                        continue; // Skip this item
                    }
                    BackpressureStrategy::Sample(n) => {
                        // Keep only every Nth item
                        if buffer.len() % n != 0 {
                            continue;
                        }
                    }
                }
            }

            // Process through operators
            let mut outputs = vec![input];

            for operator in &mut self.operators {
                outputs = outputs
                    .into_iter()
                    .flat_map(|item| operator.process(item))
                    .collect();
            }

            // Send outputs
            for output in outputs {
                if self.output_tx.send(output).await.is_err() {
                    return; // Receiver dropped
                }
            }
        }
    }
}
```

---

### Usage Example

```rust
// Create pipeline
let (tick_tx, tick_rx) = mpsc::channel(1000);
let (signal_tx, signal_rx) = mpsc::channel(100);

let pipeline = StreamPipeline::new(tick_rx, signal_tx)
    // Map ticks to bars
    .map(|tick| aggregate_to_bar(tick))
    // Filter out low-volume bars
    .filter(|bar| bar.volume > 1000)
    // Extract features
    .map(|bar| extract_features(bar))
    // Generate signals
    .flat_map(|features| generate_signals(features))
    // Filter high-confidence signals
    .filter(|signal| signal.confidence > 0.7)
    .with_buffer_size(1000)
    .with_backpressure(BackpressureStrategy::DropOldest);

// Run pipeline
tokio::spawn(pipeline.run());

// Send ticks
tick_tx.send(tick).await?;

// Receive signals
while let Some(signal) = signal_rx.recv().await {
    println!("Signal: {:?}", signal);
}
```

---

## Latency Budgets Per Stage

### Detailed Latency Budget

| Stage | Operation | Target (p50) | Target (p99) | Critical | Budget % |
|-------|-----------|-------------|-------------|----------|----------|
| **Ingestion** | WebSocket → Tick parse | 0.1ms | 0.5ms | ✅ Yes | 0.3% |
| | Tick → Event bus | 0.05ms | 0.2ms | ✅ Yes | 0.1% |
| **Aggregation** | Tick aggregation (1000 ticks) | 0.5ms | 2ms | ✅ Yes | 1.1% |
| | Bar construction | 0.3ms | 1ms | ✅ Yes | 0.6% |
| **Features** | Technical indicators | 5ms | 15ms | ✅ Yes | 8.4% |
| | Feature vector assembly | 1ms | 3ms | No | 1.7% |
| **Strategies** | Momentum strategy | 10ms | 30ms | ✅ Yes | 16.9% |
| | Mean reversion | 8ms | 25ms | No | 14.0% |
| | Neural sentiment | 50ms | 150ms | No | 84.3% |
| **Signal Fusion** | Deduplication | 1ms | 3ms | No | 1.7% |
| | Ensemble voting | 2ms | 5ms | No | 2.8% |
| **Risk** | Risk validation | 2ms | 8ms | ✅ Yes | 4.5% |
| | Position sizing | 1ms | 2ms | ✅ Yes | 1.1% |
| **Execution** | Order creation | 1ms | 3ms | ✅ Yes | 1.7% |
| | Submit to broker | 5ms | 20ms | ✅ Yes | 11.2% |
| | Broker ACK | 50ms | 200ms | No | 112.4% |
| **Settlement** | Fill processing | 1ms | 5ms | No | 2.8% |
| | Portfolio update | 0.5ms | 2ms | ✅ Yes | 1.1% |
| | AgentDB store | 2ms | 10ms | No | 5.6% |
| **TOTAL** | End-to-end | **140ms** | **486ms** | | **272%** |

**Notes:**
- Budget % = (Target p99 / Total p50) × 100
- Critical path: Stages marked "Yes" are blocking
- Broker ACK is external and variable

---

### Latency Monitoring

```rust
pub struct LatencyTracker {
    stage_timings: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
}

impl LatencyTracker {
    pub fn start_stage(&self, stage: &str) -> StageTimer {
        StageTimer {
            stage: stage.to_string(),
            start: Instant::now(),
            tracker: self.clone(),
        }
    }

    pub fn record(&self, stage: &str, duration: Duration) {
        let mut timings = self.stage_timings.lock().unwrap();
        timings.entry(stage.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }

    pub fn report(&self) -> LatencyReport {
        let timings = self.stage_timings.lock().unwrap();
        let mut report = LatencyReport::default();

        for (stage, durations) in timings.iter() {
            let mut sorted = durations.clone();
            sorted.sort();

            let p50 = percentile(&sorted, 50.0);
            let p95 = percentile(&sorted, 95.0);
            let p99 = percentile(&sorted, 99.0);

            report.stages.push(StageLatency {
                stage: stage.clone(),
                p50,
                p95,
                p99,
                count: sorted.len(),
            });
        }

        report
    }
}

pub struct StageTimer {
    stage: String,
    start: Instant,
    tracker: LatencyTracker,
}

impl Drop for StageTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.tracker.record(&self.stage, duration);
    }
}

// Usage
let tracker = LatencyTracker::new();

{
    let _timer = tracker.start_stage("feature_extraction");
    extract_features(&bars)?;
} // Timer automatically records on drop

// Generate report
let report = tracker.report();
println!("{:#?}", report);
```

---

## Recovery and Replay Mechanisms

### Event Sourcing for Replay

```rust
pub struct EventLog {
    storage: Arc<dyn EventStorage>,
}

#[async_trait]
pub trait EventStorage: Send + Sync {
    async fn append(&self, event: Event) -> Result<EventId>;
    async fn read_from(&self, offset: EventId) -> Result<Vec<Event>>;
    async fn read_range(&self, start: EventId, end: EventId) -> Result<Vec<Event>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub timestamp: DateTime<Utc>,
    pub event_type: EventType,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    TickReceived,
    BarCreated,
    SignalGenerated,
    OrderPlaced,
    OrderFilled,
    PositionUpdated,
}

impl EventLog {
    /// Append event to log
    pub async fn append(&self, event_type: EventType, payload: impl Serialize) -> Result<EventId> {
        let event = Event {
            id: EventId::new(),
            timestamp: Utc::now(),
            event_type,
            payload: serde_json::to_value(payload)?,
        };

        self.storage.append(event).await
    }

    /// Replay events from offset
    pub async fn replay_from(&self, offset: EventId) -> Result<ReplayStream> {
        let events = self.storage.read_from(offset).await?;

        Ok(ReplayStream {
            events: events.into_iter(),
            current: 0,
        })
    }

    /// Replay events for time range
    pub async fn replay_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<ReplayStream> {
        // Find event IDs for time range
        let start_id = self.find_event_by_time(start).await?;
        let end_id = self.find_event_by_time(end).await?;

        let events = self.storage.read_range(start_id, end_id).await?;

        Ok(ReplayStream {
            events: events.into_iter(),
            current: 0,
        })
    }
}

pub struct ReplayStream {
    events: std::vec::IntoIter<Event>,
    current: usize,
}

impl ReplayStream {
    pub fn next(&mut self) -> Option<Event> {
        self.current += 1;
        self.events.next()
    }

    pub fn progress(&self) -> f64 {
        self.current as f64 / (self.events.len() + self.current) as f64
    }
}
```

---

### Checkpoint and Recovery

```rust
pub struct CheckpointManager {
    storage: Arc<dyn CheckpointStorage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: CheckpointId,
    pub timestamp: DateTime<Utc>,
    pub event_offset: EventId,
    pub state_snapshot: StateSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub portfolio: Portfolio,
    pub positions: HashMap<String, Position>,
    pub pending_orders: Vec<Order>,
    pub strategy_states: HashMap<String, serde_json::Value>,
}

impl CheckpointManager {
    /// Create checkpoint
    pub async fn create_checkpoint(
        &self,
        event_offset: EventId,
        state: StateSnapshot,
    ) -> Result<CheckpointId> {
        let checkpoint = Checkpoint {
            id: CheckpointId::new(),
            timestamp: Utc::now(),
            event_offset,
            state_snapshot: state,
        };

        self.storage.save(checkpoint).await
    }

    /// Restore from latest checkpoint
    pub async fn restore_latest(&self) -> Result<(Checkpoint, ReplayStream)> {
        let checkpoint = self.storage.load_latest().await?;
        let replay_stream = self.event_log
            .replay_from(checkpoint.event_offset)
            .await?;

        Ok((checkpoint, replay_stream))
    }

    /// Recover system state
    pub async fn recover(&self) -> Result<TradingSystemState> {
        let (checkpoint, mut replay) = self.restore_latest().await?;

        // Restore base state
        let mut state = TradingSystemState::from_snapshot(checkpoint.state_snapshot);

        // Replay events since checkpoint
        while let Some(event) = replay.next() {
            state.apply_event(event)?;
        }

        Ok(state)
    }
}

// Usage
let checkpoint_mgr = CheckpointManager::new(storage);

// Create checkpoint every hour
tokio::spawn(async move {
    loop {
        tokio::time::sleep(Duration::from_hours(1)).await;

        let state = get_current_state().await;
        checkpoint_mgr.create_checkpoint(current_event_id, state).await?;
    }
});

// On crash recovery
if let Ok(state) = checkpoint_mgr.recover().await {
    println!("Recovered state from checkpoint");
    resume_trading(state).await?;
}
```

---

## Observable Hooks for Monitoring

### Observable Stream Pattern

```rust
pub struct Observable<T> {
    observers: Arc<Mutex<Vec<Box<dyn Observer<T>>>>>,
}

pub trait Observer<T>: Send + Sync {
    fn on_next(&mut self, value: &T);
    fn on_error(&mut self, error: &Error);
    fn on_complete(&mut self);
}

impl<T: Clone> Observable<T> {
    pub fn new() -> Self {
        Self {
            observers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn subscribe<O: Observer<T> + 'static>(&self, observer: O) {
        self.observers.lock().unwrap().push(Box::new(observer));
    }

    pub fn emit(&self, value: T) {
        let observers = self.observers.lock().unwrap();
        for observer in observers.iter() {
            observer.on_next(&value);
        }
    }

    pub fn error(&self, error: Error) {
        let observers = self.observers.lock().unwrap();
        for observer in observers.iter() {
            observer.on_error(&error);
        }
    }
}

// Logging observer
pub struct LoggingObserver;

impl<T: Debug> Observer<T> for LoggingObserver {
    fn on_next(&mut self, value: &T) {
        tracing::info!("Event: {:?}", value);
    }

    fn on_error(&mut self, error: &Error) {
        tracing::error!("Error: {:?}", error);
    }

    fn on_complete(&mut self) {
        tracing::info!("Stream completed");
    }
}

// Metrics observer
pub struct MetricsObserver {
    counter: prometheus::Counter,
}

impl<T> Observer<T> for MetricsObserver {
    fn on_next(&mut self, _value: &T) {
        self.counter.inc();
    }

    fn on_error(&mut self, _error: &Error) {
        // Record error metric
    }

    fn on_complete(&mut self) {
        // Record completion
    }
}

// Usage
let tick_stream = Observable::<Tick>::new();

tick_stream.subscribe(LoggingObserver);
tick_stream.subscribe(MetricsObserver {
    counter: prometheus::register_counter!("ticks_received", "Total ticks received").unwrap(),
});

// Emit events
tick_stream.emit(tick);
```

---

## Integration with Tokio Channels

### Channel Types and Use Cases

```rust
// 1. mpsc: Multi-producer, single-consumer
// Use for: Main data pipeline, guaranteed ordering
let (tx, mut rx) = mpsc::channel::<Tick>(1000);

tokio::spawn(async move {
    while let Some(tick) = rx.recv().await {
        process_tick(tick).await;
    }
});

// 2. broadcast: Multi-producer, multi-consumer
// Use for: State updates, notifications
let (tx, _) = broadcast::channel::<PortfolioUpdate>(100);

let mut rx1 = tx.subscribe();
let mut rx2 = tx.subscribe();

tokio::spawn(async move {
    while let Ok(update) = rx1.recv().await {
        log_update(update);
    }
});

tokio::spawn(async move {
    while let Ok(update) = rx2.recv().await {
        send_to_ui(update);
    }
});

// 3. watch: Single-producer, multi-consumer (latest value only)
// Use for: Configuration updates, system state
let (tx, mut rx) = watch::channel(SystemState::Running);

tokio::spawn(async move {
    while rx.changed().await.is_ok() {
        let state = *rx.borrow();
        handle_state_change(state).await;
    }
});

// 4. oneshot: Single-use channel
// Use for: Request-response patterns
let (tx, rx) = oneshot::channel::<OrderResponse>();

tokio::spawn(async move {
    let response = place_order(order).await;
    tx.send(response).ok();
});

let response = rx.await?;
```

---

## WebSocket Handling Patterns

### Robust WebSocket Connection

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};

pub struct WebSocketClient {
    url: String,
    reconnect_delay: Duration,
    max_reconnect_attempts: usize,
}

impl WebSocketClient {
    pub async fn connect_with_retry(&self) -> Result<WebSocketStream> {
        let mut attempts = 0;

        loop {
            match connect_async(&self.url).await {
                Ok((ws_stream, _)) => {
                    tracing::info!("WebSocket connected");
                    return Ok(ws_stream);
                }
                Err(e) if attempts < self.max_reconnect_attempts => {
                    attempts += 1;
                    tracing::warn!(
                        "WebSocket connection failed (attempt {}/{}): {}",
                        attempts,
                        self.max_reconnect_attempts,
                        e
                    );

                    tokio::time::sleep(self.reconnect_delay).await;
                }
                Err(e) => {
                    return Err(e.into());
                }
            }
        }
    }

    pub async fn run_with_reconnect<F>(
        &self,
        mut handler: F,
    ) -> Result<()>
    where
        F: FnMut(Message) -> Result<()>,
    {
        loop {
            let mut ws_stream = self.connect_with_retry().await?;

            // Read messages until error
            loop {
                match ws_stream.next().await {
                    Some(Ok(msg)) => {
                        if let Err(e) = handler(msg) {
                            tracing::error!("Handler error: {}", e);
                        }
                    }
                    Some(Err(e)) => {
                        tracing::error!("WebSocket error: {}", e);
                        break; // Reconnect
                    }
                    None => {
                        tracing::info!("WebSocket closed");
                        break; // Reconnect
                    }
                }
            }

            tokio::time::sleep(self.reconnect_delay).await;
        }
    }
}

// Usage
let ws_client = WebSocketClient {
    url: "wss://stream.alpaca.markets/v2/sip".to_string(),
    reconnect_delay: Duration::from_secs(5),
    max_reconnect_attempts: 10,
};

ws_client.run_with_reconnect(|msg| {
    let tick = parse_tick_message(msg)?;
    tick_tx.send(tick).await?;
    Ok(())
}).await?;
```

---

## Event Sourcing Patterns

### Complete Event Sourcing System

```rust
pub struct EventSourcedSystem {
    event_log: Arc<EventLog>,
    checkpoint_mgr: Arc<CheckpointManager>,
    current_state: Arc<RwLock<TradingSystemState>>,
}

impl EventSourcedSystem {
    /// Process command and append event
    pub async fn process_command(&self, command: Command) -> Result<Event> {
        // 1. Validate command against current state
        let state = self.current_state.read().await;
        command.validate(&state)?;
        drop(state);

        // 2. Generate event from command
        let event = self.command_to_event(command)?;

        // 3. Append event to log
        let event_id = self.event_log.append(event.event_type, &event.payload).await?;

        // 4. Apply event to current state
        let mut state = self.current_state.write().await;
        state.apply_event(event.clone())?;

        // 5. Emit event to observers
        self.emit_event(&event);

        Ok(event)
    }

    /// Rebuild state from events
    pub async fn rebuild_state(&self) -> Result<TradingSystemState> {
        let mut state = TradingSystemState::default();

        // Get all events
        let events = self.event_log.storage.read_from(EventId::zero()).await?;

        // Apply events sequentially
        for event in events {
            state.apply_event(event)?;
        }

        Ok(state)
    }

    /// Get current state
    pub async fn get_state(&self) -> TradingSystemState {
        self.current_state.read().await.clone()
    }
}

#[derive(Debug, Clone)]
pub enum Command {
    PlaceOrder(PlaceOrderCommand),
    CancelOrder(CancelOrderCommand),
    UpdatePosition(UpdatePositionCommand),
}

#[derive(Debug, Clone)]
pub struct PlaceOrderCommand {
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: u32,
    pub order_type: OrderType,
}

impl Command {
    fn validate(&self, state: &TradingSystemState) -> Result<()> {
        match self {
            Command::PlaceOrder(cmd) => {
                // Check sufficient capital
                let cost = cmd.quantity as f64 * get_current_price(&cmd.symbol)?;
                if state.portfolio.cash < Decimal::from_f64(cost).unwrap() {
                    return Err(Error::InsufficientCapital);
                }

                // Check risk limits
                state.risk_manager.validate_order(cmd)?;

                Ok(())
            }
            _ => Ok(()),
        }
    }
}

// Usage
let system = EventSourcedSystem::new(event_log, checkpoint_mgr);

// Place order
let command = Command::PlaceOrder(PlaceOrderCommand {
    symbol: "AAPL".to_string(),
    side: OrderSide::Buy,
    quantity: 100,
    order_type: OrderType::Market,
});

let event = system.process_command(command).await?;
println!("Order placed: {:?}", event);

// Recover from crash
let recovered_state = system.rebuild_state().await?;
println!("Recovered state: {:?}", recovered_state);
```

---

## Cross-References

- **Architecture:** [03_Architecture.md](03_Architecture.md)
- **Strategies:** [06_Strategy_and_Sublinear_Solvers.md](06_Strategy_and_Sublinear_Solvers.md)
- **Memory & AgentDB:** [05_Memory_and_AgentDB.md](05_Memory_and_AgentDB.md)
- **Parity Requirements:** [02_Parity_Requirements.md](02_Parity_Requirements.md)

---

**Document Status:** ✅ Complete
**Last Updated:** 2025-11-12
**Next Review:** Phase 5 (Week 11)
**Owner:** Backend Developer + Systems Engineer
