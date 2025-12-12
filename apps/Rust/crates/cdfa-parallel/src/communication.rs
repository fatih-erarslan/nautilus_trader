//! Communication patterns for parallel CDFA components
//!
//! Provides efficient message passing, result aggregation, and error propagation
//! mechanisms for parallel processing contexts.

use crossbeam::channel::{bounded, unbounded, Receiver, Select, Sender, TryRecvError};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cdfa_core::error::{Error, Result};
use cdfa_core::types::{AnalysisResult, Signal};

/// Message types for inter-component communication
#[derive(Debug, Clone)]
pub enum Message {
    /// Signal to be processed
    Signal(Signal),
    
    /// Analysis result
    Result(AnalysisResult),
    
    /// Error occurred
    Error(String),
    
    /// Control message
    Control(ControlMessage),
    
    /// Heartbeat with timestamp
    Heartbeat(u64),
    
    /// Statistics update
    Stats(ComponentStats),
}

/// Control messages for component coordination
#[derive(Debug, Clone)]
pub enum ControlMessage {
    /// Start processing
    Start,
    
    /// Stop processing
    Stop,
    
    /// Pause processing
    Pause,
    
    /// Resume processing
    Resume,
    
    /// Flush buffers
    Flush,
    
    /// Update configuration
    UpdateConfig(Vec<u8>),
    
    /// Request statistics
    RequestStats,
}

/// Component statistics
#[derive(Debug, Clone)]
pub struct ComponentStats {
    pub component_id: String,
    pub messages_processed: u64,
    pub errors: u64,
    pub avg_latency_ns: u64,
    pub queue_depth: usize,
}

/// Message router for multi-component systems
pub struct MessageRouter {
    /// Component channels
    channels: DashMap<String, ComponentChannel>,
    
    /// Broadcast channels for pub/sub
    broadcast_channels: DashMap<String, BroadcastChannel>,
    
    /// Router statistics
    stats: Arc<RouterStats>,
    
    /// Router configuration
    config: RouterConfig,
}

/// Component channel pair
struct ComponentChannel {
    sender: Sender<Message>,
    receiver: Receiver<Message>,
}

/// Broadcast channel for pub/sub pattern
struct BroadcastChannel {
    subscribers: Vec<Sender<Message>>,
    last_message: Option<Message>,
}

/// Router statistics
#[derive(Default)]
struct RouterStats {
    messages_routed: AtomicU64,
    routing_errors: AtomicU64,
    broadcasts_sent: AtomicU64,
    avg_routing_time_ns: AtomicU64,
}

/// Router configuration
#[derive(Clone)]
pub struct RouterConfig {
    /// Default channel capacity
    pub channel_capacity: usize,
    
    /// Enable message persistence
    pub enable_persistence: bool,
    
    /// Message TTL in seconds
    pub message_ttl: u64,
    
    /// Maximum subscribers per broadcast channel
    pub max_subscribers: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 1000,
            enable_persistence: false,
            message_ttl: 60,
            max_subscribers: 100,
        }
    }
}

impl MessageRouter {
    /// Creates a new message router
    pub fn new(config: RouterConfig) -> Self {
        Self {
            channels: DashMap::new(),
            broadcast_channels: DashMap::new(),
            stats: Arc::new(RouterStats::default()),
            config,
        }
    }
    
    /// Registers a new component
    pub fn register_component(&self, component_id: String) -> (Sender<Message>, Receiver<Message>) {
        let (tx, rx) = bounded(self.config.channel_capacity);
        
        self.channels.insert(component_id.clone(), ComponentChannel {
            sender: tx.clone(),
            receiver: rx.clone(),
        });
        
        (tx, rx)
    }
    
    /// Sends a message to a specific component
    pub fn send_to(&self, component_id: &str, message: Message) -> Result<()> {
        let start = Instant::now();
        
        if let Some(channel) = self.channels.get(component_id) {
            channel.sender.send(message)
                .map_err(|_| Error::ChannelClosed)?;
            
            let routing_time = start.elapsed().as_nanos() as u64;
            self.update_routing_stats(routing_time);
            
            Ok(())
        } else {
            self.stats.routing_errors.fetch_add(1, Ordering::Relaxed);
            Err(Error::ComponentNotFound(component_id.to_string()))
        }
    }
    
    /// Broadcasts a message to all subscribers of a topic
    pub fn broadcast(&self, topic: &str, message: Message) -> Result<()> {
        if let Some(mut channel) = self.broadcast_channels.get_mut(topic) {
            let mut failed = Vec::new();
            
            for (i, subscriber) in channel.subscribers.iter().enumerate() {
                if subscriber.send(message.clone()).is_err() {
                    failed.push(i);
                }
            }
            
            // Remove failed subscribers
            for &i in failed.iter().rev() {
                channel.subscribers.swap_remove(i);
            }
            
            channel.last_message = Some(message);
            self.stats.broadcasts_sent.fetch_add(1, Ordering::Relaxed);
            
            Ok(())
        } else {
            Err(Error::TopicNotFound(topic.to_string()))
        }
    }
    
    /// Subscribes to a broadcast topic
    pub fn subscribe(&self, topic: &str) -> Receiver<Message> {
        let (tx, rx) = bounded(self.config.channel_capacity);
        
        self.broadcast_channels
            .entry(topic.to_string())
            .or_insert_with(|| BroadcastChannel {
                subscribers: Vec::new(),
                last_message: None,
            })
            .subscribers
            .push(tx);
        
        rx
    }
    
    /// Updates routing statistics
    fn update_routing_stats(&self, routing_time_ns: u64) {
        self.stats.messages_routed.fetch_add(1, Ordering::Relaxed);
        
        // Update average using exponential moving average
        let count = self.stats.messages_routed.load(Ordering::Relaxed);
        let current_avg = self.stats.avg_routing_time_ns.load(Ordering::Relaxed);
        let new_avg = if count == 1 {
            routing_time_ns
        } else {
            (current_avg * (count - 1) + routing_time_ns) / count
        };
        
        self.stats.avg_routing_time_ns.store(new_avg, Ordering::Relaxed);
    }
    
    /// Gets router statistics
    pub fn stats(&self) -> RouterStatsSnapshot {
        RouterStatsSnapshot {
            messages_routed: self.stats.messages_routed.load(Ordering::Relaxed),
            routing_errors: self.stats.routing_errors.load(Ordering::Relaxed),
            broadcasts_sent: self.stats.broadcasts_sent.load(Ordering::Relaxed),
            avg_routing_time_ns: self.stats.avg_routing_time_ns.load(Ordering::Relaxed),
            active_components: self.channels.len(),
            active_topics: self.broadcast_channels.len(),
        }
    }
}

/// Snapshot of router statistics
#[derive(Debug, Clone)]
pub struct RouterStatsSnapshot {
    pub messages_routed: u64,
    pub routing_errors: u64,
    pub broadcasts_sent: u64,
    pub avg_routing_time_ns: u64,
    pub active_components: usize,
    pub active_topics: usize,
}

/// Result aggregator with timeout support
pub struct ResultAggregator {
    /// Expected number of results
    expected_count: usize,
    
    /// Timeout duration
    timeout: Duration,
    
    /// Results channel
    receiver: Receiver<AnalysisResult>,
    
    /// Aggregated results
    results: Arc<Mutex<Vec<AnalysisResult>>>,
    
    /// Completion flag
    completed: Arc<AtomicBool>,
    
    /// Error count
    error_count: AtomicUsize,
}

impl ResultAggregator {
    /// Creates a new result aggregator
    pub fn new(expected_count: usize, timeout: Duration) -> (Self, Sender<AnalysisResult>) {
        let (tx, rx) = bounded(expected_count);
        
        (
            Self {
                expected_count,
                timeout,
                receiver: rx,
                results: Arc::new(Mutex::new(Vec::with_capacity(expected_count))),
                completed: Arc::new(AtomicBool::new(false)),
                error_count: AtomicUsize::new(0),
            },
            tx,
        )
    }
    
    /// Waits for all results or timeout
    pub fn wait_for_results(&self) -> Result<Vec<AnalysisResult>> {
        let start = Instant::now();
        let mut received = 0;
        
        while received < self.expected_count && start.elapsed() < self.timeout {
            match self.receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(result) => {
                    self.results.lock().push(result);
                    received += 1;
                }
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => continue,
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }
        
        self.completed.store(true, Ordering::Release);
        
        let results = self.results.lock().clone();
        if results.len() < self.expected_count {
            Err(Error::PartialResults {
                expected: self.expected_count,
                received: results.len(),
            })
        } else {
            Ok(results)
        }
    }
    
    /// Tries to get results without blocking
    pub fn try_get_results(&self) -> Option<Vec<AnalysisResult>> {
        if self.completed.load(Ordering::Acquire) {
            Some(self.results.lock().clone())
        } else {
            None
        }
    }
    
    /// Records an error
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Gets the number of errors recorded
    pub fn error_count(&self) -> usize {
        self.error_count.load(Ordering::Relaxed)
    }
}

/// Error propagation handler for parallel contexts
pub struct ErrorPropagator {
    /// Error channels for each component
    error_channels: DashMap<String, Sender<Error>>,
    
    /// Global error handler
    global_handler: Option<Arc<dyn Fn(Error) + Send + Sync>>,
    
    /// Error statistics
    error_stats: Arc<ErrorStats>,
    
    /// Circuit breaker state
    circuit_breaker: Arc<CircuitBreaker>,
}

/// Error statistics
#[derive(Default)]
struct ErrorStats {
    total_errors: AtomicU64,
    errors_by_type: RwLock<HashMap<String, u64>>,
    last_error_time: RwLock<Option<Instant>>,
}

/// Circuit breaker for error handling
struct CircuitBreaker {
    /// Failure threshold
    failure_threshold: u32,
    
    /// Reset timeout
    reset_timeout: Duration,
    
    /// Current failure count
    failure_count: AtomicU32,
    
    /// Circuit state
    state: AtomicU8, // 0: Closed, 1: Open, 2: Half-Open
    
    /// Last failure time
    last_failure: RwLock<Option<Instant>>,
}

impl CircuitBreaker {
    const CLOSED: u8 = 0;
    const OPEN: u8 = 1;
    const HALF_OPEN: u8 = 2;
    
    fn new(failure_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            reset_timeout,
            failure_count: AtomicU32::new(0),
            state: AtomicU8::new(Self::CLOSED),
            last_failure: RwLock::new(None),
        }
    }
    
    fn record_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        self.state.store(Self::CLOSED, Ordering::Relaxed);
    }
    
    fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_failure.write() = Some(Instant::now());
        
        if count >= self.failure_threshold {
            self.state.store(Self::OPEN, Ordering::Relaxed);
        }
    }
    
    fn is_open(&self) -> bool {
        let state = self.state.load(Ordering::Relaxed);
        if state == Self::OPEN {
            // Check if we should transition to half-open
            if let Some(last_failure) = *self.last_failure.read() {
                if last_failure.elapsed() > self.reset_timeout {
                    self.state.store(Self::HALF_OPEN, Ordering::Relaxed);
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
}

impl ErrorPropagator {
    /// Creates a new error propagator
    pub fn new(failure_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            error_channels: DashMap::new(),
            global_handler: None,
            error_stats: Arc::new(ErrorStats::default()),
            circuit_breaker: Arc::new(CircuitBreaker::new(failure_threshold, reset_timeout)),
        }
    }
    
    /// Sets a global error handler
    pub fn set_global_handler<F>(&mut self, handler: F)
    where
        F: Fn(Error) + Send + Sync + 'static,
    {
        self.global_handler = Some(Arc::new(handler));
    }
    
    /// Registers an error channel for a component
    pub fn register_component(&self, component_id: String) -> Receiver<Error> {
        let (tx, rx) = bounded(100);
        self.error_channels.insert(component_id, tx);
        rx
    }
    
    /// Propagates an error to appropriate handlers
    pub fn propagate(&self, component_id: &str, error: Error) -> Result<()> {
        // Check circuit breaker
        if self.circuit_breaker.is_open() {
            return Err(Error::CircuitOpen);
        }
        
        // Update statistics
        self.error_stats.total_errors.fetch_add(1, Ordering::Relaxed);
        {
            let mut errors_by_type = self.error_stats.errors_by_type.write();
            let error_type = format!("{:?}", error);
            *errors_by_type.entry(error_type).or_insert(0) += 1;
        }
        *self.error_stats.last_error_time.write() = Some(Instant::now());
        
        // Send to component channel
        if let Some(channel) = self.error_channels.get(component_id) {
            if channel.send(error.clone()).is_err() {
                self.circuit_breaker.record_failure();
                return Err(Error::ChannelClosed);
            }
        }
        
        // Call global handler
        if let Some(handler) = &self.global_handler {
            handler(error);
        }
        
        self.circuit_breaker.record_success();
        Ok(())
    }
    
    /// Gets error statistics
    pub fn stats(&self) -> ErrorStatsSnapshot {
        ErrorStatsSnapshot {
            total_errors: self.error_stats.total_errors.load(Ordering::Relaxed),
            errors_by_type: self.error_stats.errors_by_type.read().clone(),
            last_error_time: *self.error_stats.last_error_time.read(),
            circuit_breaker_state: match self.circuit_breaker.state.load(Ordering::Relaxed) {
                CircuitBreaker::CLOSED => "closed",
                CircuitBreaker::OPEN => "open",
                CircuitBreaker::HALF_OPEN => "half-open",
                _ => "unknown",
            },
        }
    }
}

/// Snapshot of error statistics
#[derive(Debug, Clone)]
pub struct ErrorStatsSnapshot {
    pub total_errors: u64,
    pub errors_by_type: HashMap<String, u64>,
    pub last_error_time: Option<Instant>,
    pub circuit_breaker_state: &'static str,
}

/// Message multiplexer for handling multiple channels
pub struct MessageMultiplexer {
    /// Channels to multiplex
    receivers: Vec<Receiver<Message>>,
    
    /// Select operation for efficient multiplexing
    select: Select<'static>,
    
    /// Channel priorities
    priorities: Vec<u32>,
}

impl MessageMultiplexer {
    /// Creates a new message multiplexer
    pub fn new() -> Self {
        Self {
            receivers: Vec::new(),
            select: Select::new(),
            priorities: Vec::new(),
        }
    }
    
    /// Adds a channel to multiplex
    pub fn add_channel(&mut self, receiver: Receiver<Message>, priority: u32) {
        self.receivers.push(receiver);
        self.priorities.push(priority);
    }
    
    /// Receives the next message from any channel
    pub fn recv(&mut self) -> Result<(usize, Message)> {
        // Build select operation
        for receiver in &self.receivers {
            self.select.recv(receiver);
        }
        
        // Wait for a message
        let oper = self.select.select();
        let index = oper.index();
        
        match oper.recv(&self.receivers[index]) {
            Ok(msg) => Ok((index, msg)),
            Err(_) => Err(Error::ChannelClosed),
        }
    }
    
    /// Tries to receive without blocking
    pub fn try_recv(&self) -> Option<(usize, Message)> {
        // Try channels in priority order
        let mut indexed_priorities: Vec<(usize, u32)> = self.priorities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        indexed_priorities.sort_by_key(|&(_, p)| std::cmp::Reverse(p));
        
        for (index, _) in indexed_priorities {
            if let Ok(msg) = self.receivers[index].try_recv() {
                return Some((index, msg));
            }
        }
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cdfa_core::types::SignalId;
    
    #[test]
    fn test_message_router() {
        let router = MessageRouter::new(RouterConfig::default());
        
        // Register components
        let (tx1, rx1) = router.register_component("comp1".to_string());
        let (tx2, rx2) = router.register_component("comp2".to_string());
        
        // Send message
        let signal = Signal::new(SignalId(1), 1000, vec![1.0, 2.0, 3.0]);
        router.send_to("comp1", Message::Signal(signal.clone())).unwrap();
        
        // Receive message
        match rx1.recv().unwrap() {
            Message::Signal(s) => assert_eq!(s.id, signal.id),
            _ => panic!("Wrong message type"),
        }
        
        // Check stats
        let stats = router.stats();
        assert_eq!(stats.messages_routed, 1);
        assert_eq!(stats.active_components, 2);
    }
    
    #[test]
    fn test_result_aggregator() {
        let (aggregator, sender) = ResultAggregator::new(3, Duration::from_secs(1));
        
        // Send results
        for i in 0..3 {
            let result = AnalysisResult::new(format!("analyzer{}", i), 0.5 + i as f64 * 0.1, 0.9);
            sender.send(result).unwrap();
        }
        
        // Wait for results
        let results = aggregator.wait_for_results().unwrap();
        assert_eq!(results.len(), 3);
    }
    
    #[test]
    fn test_error_propagator() {
        let mut propagator = ErrorPropagator::new(3, Duration::from_secs(1));
        
        // Set global handler
        let error_count = Arc::new(AtomicU32::new(0));
        let counter = Arc::clone(&error_count);
        propagator.set_global_handler(move |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        
        // Register component
        let rx = propagator.register_component("comp1".to_string());
        
        // Propagate error
        propagator.propagate("comp1", Error::Internal).unwrap();
        
        // Check error was received
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Error::Internal) => {},
            _ => panic!("Expected Internal error"),
        }
        
        // Check global handler was called
        assert_eq!(error_count.load(Ordering::Relaxed), 1);
        
        // Check stats
        let stats = propagator.stats();
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.circuit_breaker_state, "closed");
    }
}