//! Event system for TENGRI trading strategy
//! 
//! Provides event-driven architecture for handling market data,
//! neural network outputs, and trading decisions in real-time.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::types::{TradingSignal, PriceData, TradeData, OrderBookData};

/// Maximum events in queue before dropping old events
const MAX_EVENT_QUEUE_SIZE: usize = 100000;

/// Event types supported by the system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    /// Market data events
    MarketData,
    /// Trading signal events
    TradingSignal,
    /// Neural network events
    Neural,
    /// Order execution events
    OrderExecution,
    /// System events
    System,
    /// Risk management events
    RiskManagement,
}

/// Base event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    /// Unique event identifier
    pub id: Uuid,
    /// Event type
    pub event_type: EventType,
    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Event source
    pub source: String,
    /// Event priority (0=low, 5=normal, 10=high)
    pub priority: u8,
    /// Event payload
    pub payload: EventPayload,
    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// Event payload variants
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EventPayload {
    /// Market price data
    PriceUpdate {
        data: PriceData,
    },
    /// Trade execution data
    TradeExecution {
        data: TradeData,
    },
    /// Order book update
    OrderBookUpdate {
        data: OrderBookData,
    },
    /// Trading signal generated
    TradingSignal {
        signal: TradingSignal,
    },
    /// Neural network output
    NeuralOutput {
        model_id: String,
        predictions: Vec<f64>,
        confidence: f64,
    },
    /// System status update
    SystemStatus {
        component: String,
        status: String,
        metrics: HashMap<String, f64>,
    },
    /// Error occurred
    Error {
        error_type: String,
        message: String,
        details: Option<serde_json::Value>,
    },
}

/// Neural event specific structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralEvent {
    /// Neural model identifier
    pub model_id: String,
    /// Event subtype
    pub event_subtype: NeuralEventType,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Event data
    pub data: serde_json::Value,
}

/// Neural event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NeuralEventType {
    /// Model prediction generated
    Prediction,
    /// Model training completed
    TrainingComplete,
    /// Model evaluation results
    Evaluation,
    /// Spike activity detected
    SpikeActivity,
    /// Avalanche detected in spike swarm
    Avalanche,
    /// Synchrony event
    Synchrony,
}

/// Event filter for selective processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFilter {
    /// Event types to include
    pub event_types: Option<Vec<EventType>>,
    /// Event sources to include
    pub sources: Option<Vec<String>>,
    /// Minimum priority level
    pub min_priority: Option<u8>,
    /// Time window for events
    pub time_window: Option<Duration>,
    /// Custom filter expression
    pub custom_filter: Option<String>,
}

/// Event processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventStats {
    /// Total events processed
    pub total_processed: u64,
    /// Events by type
    pub by_type: HashMap<EventType, u64>,
    /// Processing latency statistics
    pub avg_latency_ms: f64,
    /// Current queue size
    pub queue_size: usize,
    /// Dropped events count
    pub dropped_events: u64,
    /// Last update timestamp
    pub last_update: chrono::DateTime<chrono::Utc>,
}

/// Event queue for managing event flow
pub struct EventQueue {
    /// Internal queue storage
    queue: Arc<RwLock<VecDeque<EventEnvelope>>>,
    /// Event statistics
    stats: Arc<RwLock<EventStats>>,
    /// Event filters
    filters: Vec<EventFilter>,
    /// Maximum queue size
    max_size: usize,
}

/// Event processor trait
pub trait EventProcessor: Send + Sync {
    /// Process a single event
    fn process_event(&self, event: &EventEnvelope) -> Result<()>;
    
    /// Get processor name
    fn name(&self) -> &str;
    
    /// Check if processor can handle event type
    fn can_handle(&self, event_type: &EventType) -> bool;
    
    /// Get processing priority (higher = processed first)
    fn priority(&self) -> u8 { 5 }
}

impl EventQueue {
    /// Create a new event queue
    pub fn new() -> Self {
        Self {
            queue: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            stats: Arc::new(RwLock::new(EventStats {
                total_processed: 0,
                by_type: HashMap::new(),
                avg_latency_ms: 0.0,
                queue_size: 0,
                dropped_events: 0,
                last_update: chrono::Utc::now(),
            })),
            filters: Vec::new(),
            max_size: MAX_EVENT_QUEUE_SIZE,
        }
    }
    
    /// Create event queue with custom size
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            max_size: capacity,
            ..Self::new()
        }
    }
    
    /// Add event to queue
    pub fn push(&self, event: EventEnvelope) -> Result<()> {
        // Apply filters
        if !self.passes_filters(&event) {
            debug!("Event filtered out: {}", event.id);
            return Ok(());
        }
        
        let mut queue = self.queue.write().unwrap();
        
        // Check queue capacity
        if queue.len() >= self.max_size {
            // Drop oldest event
            if let Some(dropped) = queue.pop_front() {
                warn!("Dropped event due to queue full: {}", dropped.id);
                let mut stats = self.stats.write().unwrap();
                stats.dropped_events += 1;
            }
        }
        
        // Insert based on priority (higher priority first)
        let insert_pos = queue
            .iter()
            .position(|e| e.priority < event.priority)
            .unwrap_or(queue.len());
        
        queue.insert(insert_pos, event);
        
        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.queue_size = queue.len();
            stats.last_update = chrono::Utc::now();
        }
        
        Ok(())
    }
    
    /// Get next event from queue
    pub fn pop(&self) -> Option<EventEnvelope> {
        let mut queue = self.queue.write().unwrap();
        let event = queue.pop_front();
        
        if event.is_some() {
            let mut stats = self.stats.write().unwrap();
            stats.queue_size = queue.len();
        }
        
        event
    }
    
    /// Get current queue size
    pub fn size(&self) -> usize {
        self.queue.read().unwrap().len()
    }
    
    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.read().unwrap().is_empty()
    }
    
    /// Add event filter
    pub fn add_filter(&mut self, filter: EventFilter) {
        self.filters.push(filter);
    }
    
    /// Clear all filters
    pub fn clear_filters(&mut self) {
        self.filters.clear();
    }
    
    /// Get processing statistics
    pub fn get_stats(&self) -> EventStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Check if event passes all filters
    fn passes_filters(&self, event: &EventEnvelope) -> bool {
        for filter in &self.filters {
            if !self.event_matches_filter(event, filter) {
                return false;
            }
        }
        true
    }
    
    /// Check if single event matches filter
    fn event_matches_filter(&self, event: &EventEnvelope, filter: &EventFilter) -> bool {
        // Check event type
        if let Some(ref allowed_types) = filter.event_types {
            if !allowed_types.contains(&event.event_type) {
                return false;
            }
        }
        
        // Check source
        if let Some(ref allowed_sources) = filter.sources {
            if !allowed_sources.contains(&event.source) {
                return false;
            }
        }
        
        // Check priority
        if let Some(min_priority) = filter.min_priority {
            if event.priority < min_priority {
                return false;
            }
        }
        
        // Check time window
        if let Some(time_window) = filter.time_window {
            let age = chrono::Utc::now() - event.timestamp;
            if age > chrono::Duration::from_std(time_window).unwrap_or_default() {
                return false;
            }
        }
        
        true
    }
    
    /// Clear all events from queue
    pub fn clear(&self) {
        let mut queue = self.queue.write().unwrap();
        queue.clear();
        
        let mut stats = self.stats.write().unwrap();
        stats.queue_size = 0;
        stats.last_update = chrono::Utc::now();
    }
}

impl Default for EventFilter {
    fn default() -> Self {
        Self {
            event_types: None,
            sources: None,
            min_priority: None,
            time_window: None,
            custom_filter: None,
        }
    }
}

impl EventEnvelope {
    /// Create new event envelope
    pub fn new(
        event_type: EventType,
        source: String,
        payload: EventPayload,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            event_type,
            timestamp: chrono::Utc::now(),
            source,
            priority: 5, // Default priority
            payload,
            metadata: HashMap::new(),
        }
    }
    
    /// Create high priority event
    pub fn high_priority(
        event_type: EventType,
        source: String,
        payload: EventPayload,
    ) -> Self {
        let mut event = Self::new(event_type, source, payload);
        event.priority = 10;
        event
    }
    
    /// Create low priority event
    pub fn low_priority(
        event_type: EventType,
        source: String,
        payload: EventPayload,
    ) -> Self {
        let mut event = Self::new(event_type, source, payload);
        event.priority = 1;
        event
    }
    
    /// Add metadata to event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Get event age
    pub fn age(&self) -> chrono::Duration {
        chrono::Utc::now() - self.timestamp
    }
    
    /// Check if event is expired
    pub fn is_expired(&self, max_age: Duration) -> bool {
        let age = self.age();
        age > chrono::Duration::from_std(max_age).unwrap_or_default()
    }
}

/// Event manager coordinates event flow
pub struct EventManager {
    /// Event queue
    queue: EventQueue,
    /// Registered processors
    processors: Vec<Box<dyn EventProcessor>>,
    /// Processing statistics
    processing_stats: HashMap<String, u64>,
    /// Running state
    is_running: Arc<RwLock<bool>>,
}

impl EventManager {
    /// Create new event manager
    pub fn new() -> Self {
        Self {
            queue: EventQueue::new(),
            processors: Vec::new(),
            processing_stats: HashMap::new(),
            is_running: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Register event processor
    pub fn register_processor(&mut self, processor: Box<dyn EventProcessor>) {
        info!("Registering event processor: {}", processor.name());
        self.processors.push(processor);
    }
    
    /// Submit event for processing
    pub fn submit(&self, event: EventEnvelope) -> Result<()> {
        debug!("Submitting event: {} type={:?}", event.id, event.event_type);
        self.queue.push(event)
    }
    
    /// Process all queued events
    pub fn process_all(&mut self) -> Result<usize> {
        let mut processed_count = 0;
        
        while let Some(event) = self.queue.pop() {
            self.process_event(&event)?;
            processed_count += 1;
        }
        
        Ok(processed_count)
    }
    
    /// Process single event
    fn process_event(&mut self, event: &EventEnvelope) -> Result<()> {
        let start_time = Instant::now();
        
        debug!("Processing event: {} type={:?}", event.id, event.event_type);
        
        // Find compatible processors
        let mut processed_by = Vec::new();
        
        for processor in &self.processors {
            if processor.can_handle(&event.event_type) {
                match processor.process_event(event) {
                    Ok(()) => {
                        processed_by.push(processor.name().to_string());
                        *self.processing_stats.entry(processor.name().to_string())
                            .or_insert(0) += 1;
                    }
                    Err(e) => {
                        error!("Processor {} failed for event {}: {}", 
                               processor.name(), event.id, e);
                    }
                }
            }
        }
        
        let processing_time = start_time.elapsed();
        
        if processed_by.is_empty() {
            warn!("No processors handled event: {} type={:?}", 
                  event.id, event.event_type);
        } else {
            debug!("Event {} processed by: {:?} in {:?}", 
                   event.id, processed_by, processing_time);
        }
        
        // Update latency statistics
        {
            let mut stats = self.queue.stats.write().unwrap();
            stats.total_processed += 1;
            *stats.by_type.entry(event.event_type.clone()).or_insert(0) += 1;
            
            // Update running average latency
            let latency_ms = processing_time.as_secs_f64() * 1000.0;
            stats.avg_latency_ms = (stats.avg_latency_ms * 0.9) + (latency_ms * 0.1);
            stats.last_update = chrono::Utc::now();
        }
        
        Ok(())
    }
    
    /// Get processing statistics
    pub fn get_stats(&self) -> (EventStats, HashMap<String, u64>) {
        (self.queue.get_stats(), self.processing_stats.clone())
    }
    
    /// Start event processing loop
    pub fn start_processing_loop(&mut self) -> Result<()> {
        {
            let mut running = self.is_running.write().unwrap();
            if *running {
                return Err(anyhow::anyhow!("Event manager already running"));
            }
            *running = true;
        }
        
        info!("Starting event processing loop");
        
        while *self.is_running.read().unwrap() {
            if self.queue.is_empty() {
                std::thread::sleep(Duration::from_millis(1)); // Short sleep when idle
                continue;
            }
            
            match self.process_all() {
                Ok(count) => {
                    if count > 0 {
                        debug!("Processed {} events", count);
                    }
                }
                Err(e) => {
                    error!("Error processing events: {}", e);
                }
            }
        }
        
        info!("Event processing loop stopped");
        Ok(())
    }
    
    /// Stop event processing loop
    pub fn stop(&self) {
        info!("Stopping event manager");
        let mut running = self.is_running.write().unwrap();
        *running = false;
    }
    
    /// Get queue size
    pub fn queue_size(&self) -> usize {
        self.queue.size()
    }
}

/// Convenience functions for creating common events
impl EventPayload {
    /// Create price update event
    pub fn price_update(price_data: PriceData) -> Self {
        Self::PriceUpdate { data: price_data }
    }
    
    /// Create trading signal event
    pub fn trading_signal(signal: TradingSignal) -> Self {
        Self::TradingSignal { signal }
    }
    
    /// Create neural output event
    pub fn neural_output(model_id: String, predictions: Vec<f64>, confidence: f64) -> Self {
        Self::NeuralOutput { model_id, predictions, confidence }
    }
    
    /// Create system status event
    pub fn system_status(component: String, status: String, metrics: HashMap<String, f64>) -> Self {
        Self::SystemStatus { component, status, metrics }
    }
    
    /// Create error event
    pub fn error(error_type: String, message: String, details: Option<serde_json::Value>) -> Self {
        Self::Error { error_type, message, details }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct TestProcessor {
        name: String,
        handled_types: Vec<EventType>,
    }
    
    impl TestProcessor {
        fn new(name: String, handled_types: Vec<EventType>) -> Self {
            Self { name, handled_types }
        }
    }
    
    impl EventProcessor for TestProcessor {
        fn process_event(&self, event: &EventEnvelope) -> Result<()> {
            println!("TestProcessor {} processing event {}", self.name, event.id);
            Ok(())
        }
        
        fn name(&self) -> &str {
            &self.name
        }
        
        fn can_handle(&self, event_type: &EventType) -> bool {
            self.handled_types.contains(event_type)
        }
    }
    
    #[test]
    fn test_event_queue_basic() {
        let queue = EventQueue::new();
        assert_eq!(queue.size(), 0);
        assert!(queue.is_empty());
        
        let event = EventEnvelope::new(
            EventType::MarketData,
            "test".to_string(),
            EventPayload::system_status(
                "test_component".to_string(),
                "running".to_string(),
                HashMap::new()
            ),
        );
        
        queue.push(event).unwrap();
        assert_eq!(queue.size(), 1);
        assert!(!queue.is_empty());
        
        let popped = queue.pop();
        assert!(popped.is_some());
        assert_eq!(queue.size(), 0);
    }
    
    #[test]
    fn test_event_priority_ordering() {
        let queue = EventQueue::new();
        
        // Add events with different priorities
        let low_event = EventEnvelope::low_priority(
            EventType::MarketData,
            "test".to_string(),
            EventPayload::system_status("test".to_string(), "status".to_string(), HashMap::new()),
        );
        
        let high_event = EventEnvelope::high_priority(
            EventType::TradingSignal,
            "test".to_string(),
            EventPayload::system_status("test".to_string(), "status".to_string(), HashMap::new()),
        );
        
        let normal_event = EventEnvelope::new(
            EventType::Neural,
            "test".to_string(),
            EventPayload::system_status("test".to_string(), "status".to_string(), HashMap::new()),
        );
        
        // Add in order: low, high, normal
        queue.push(low_event).unwrap();
        queue.push(high_event).unwrap();
        queue.push(normal_event).unwrap();
        
        // Should pop in priority order: high, normal, low
        let first = queue.pop().unwrap();
        assert_eq!(first.priority, 10); // High priority
        
        let second = queue.pop().unwrap();
        assert_eq!(second.priority, 5); // Normal priority
        
        let third = queue.pop().unwrap();
        assert_eq!(third.priority, 1); // Low priority
    }
    
    #[test]
    fn test_event_manager() {
        let mut manager = EventManager::new();
        
        // Register test processors
        manager.register_processor(Box::new(TestProcessor::new(
            "market_processor".to_string(),
            vec![EventType::MarketData],
        )));
        
        manager.register_processor(Box::new(TestProcessor::new(
            "signal_processor".to_string(),
            vec![EventType::TradingSignal],
        )));
        
        // Submit test events
        let market_event = EventEnvelope::new(
            EventType::MarketData,
            "test".to_string(),
            EventPayload::system_status("test".to_string(), "status".to_string(), HashMap::new()),
        );
        
        let signal_event = EventEnvelope::new(
            EventType::TradingSignal,
            "test".to_string(),
            EventPayload::system_status("test".to_string(), "status".to_string(), HashMap::new()),
        );
        
        manager.submit(market_event).unwrap();
        manager.submit(signal_event).unwrap();
        
        // Process all events
        let processed = manager.process_all().unwrap();
        assert_eq!(processed, 2);
        
        // Check stats
        let (queue_stats, processing_stats) = manager.get_stats();
        assert_eq!(queue_stats.total_processed, 2);
        assert_eq!(processing_stats.get("market_processor"), Some(&1));
        assert_eq!(processing_stats.get("signal_processor"), Some(&1));
    }
}