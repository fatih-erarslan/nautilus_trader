//! # Event-Driven Neural Processing Queue
//!
//! This module implements an efficient event queue for temporal neural processing.
//! Events are processed in chronological order using a binary heap for optimal
//! performance in sparse spiking networks.
//!
//! ## Features
//!
//! - Priority queue with temporal ordering
//! - Event batching for parallel processing
//! - Asynchronous message passing
//! - Event filtering and routing
//! - Low-latency spike delivery
//! - Memory-efficient sparse representation

use crate::neuromorphic::spiking_neuron::SpikeEvent;
use crate::{TengriError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::{Ordering, Reverse};
use std::time::Instant;

/// Priority levels for neural events
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventPriority {
    /// Critical system events (highest priority)
    Critical = 0,
    
    /// High priority spikes (e.g., error signals)
    High = 1,
    
    /// Normal spike events
    Normal = 2,
    
    /// Low priority events (e.g., background activity)
    Low = 3,
    
    /// Maintenance events (lowest priority)
    Maintenance = 4,
}

impl Default for EventPriority {
    fn default() -> Self {
        EventPriority::Normal
    }
}

/// Types of neural events that can be processed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuralEvent {
    /// Spike from a neuron
    Spike {
        /// The spike event data
        event: SpikeEvent,
        
        /// Target neuron IDs to receive this spike
        targets: Vec<usize>,
        
        /// Event priority
        priority: EventPriority,
    },
    
    /// Synaptic plasticity update
    Plasticity {
        /// Synapse ID to update
        synapse_id: usize,
        
        /// Weight change to apply
        weight_delta: f64,
        
        /// Update timestamp
        timestamp_ms: f64,
        
        /// Priority level
        priority: EventPriority,
    },
    
    /// External stimulus injection
    Stimulus {
        /// Target neuron ID
        neuron_id: usize,
        
        /// Current magnitude in pA
        current_pa: f64,
        
        /// Stimulus duration in ms
        duration_ms: f64,
        
        /// Start timestamp
        timestamp_ms: f64,
        
        /// Priority level
        priority: EventPriority,
    },
    
    /// Network reconfiguration event
    Reconfiguration {
        /// Type of reconfiguration
        config_type: String,
        
        /// Configuration data
        data: Vec<u8>,
        
        /// Timestamp
        timestamp_ms: f64,
        
        /// Priority level
        priority: EventPriority,
    },
    
    /// System maintenance event
    Maintenance {
        /// Maintenance type
        maintenance_type: String,
        
        /// Optional data payload
        data: Option<Vec<u8>>,
        
        /// Timestamp
        timestamp_ms: f64,
        
        /// Priority level
        priority: EventPriority,
    },
}

impl NeuralEvent {
    /// Get the timestamp of this event
    pub fn timestamp_ms(&self) -> f64 {
        match self {
            NeuralEvent::Spike { event, .. } => event.timestamp_ms,
            NeuralEvent::Plasticity { timestamp_ms, .. } => *timestamp_ms,
            NeuralEvent::Stimulus { timestamp_ms, .. } => *timestamp_ms,
            NeuralEvent::Reconfiguration { timestamp_ms, .. } => *timestamp_ms,
            NeuralEvent::Maintenance { timestamp_ms, .. } => *timestamp_ms,
        }
    }
    
    /// Get the priority of this event
    pub fn priority(&self) -> EventPriority {
        match self {
            NeuralEvent::Spike { priority, .. } => *priority,
            NeuralEvent::Plasticity { priority, .. } => *priority,
            NeuralEvent::Stimulus { priority, .. } => *priority,
            NeuralEvent::Reconfiguration { priority, .. } => *priority,
            NeuralEvent::Maintenance { priority, .. } => *priority,
        }
    }
    
    /// Create a spike event
    pub fn spike(event: SpikeEvent, targets: Vec<usize>, priority: EventPriority) -> Self {
        NeuralEvent::Spike { event, targets, priority }
    }
    
    /// Create a plasticity event
    pub fn plasticity(synapse_id: usize, weight_delta: f64, timestamp_ms: f64,
                     priority: EventPriority) -> Self {
        NeuralEvent::Plasticity { synapse_id, weight_delta, timestamp_ms, priority }
    }
    
    /// Create a stimulus event
    pub fn stimulus(neuron_id: usize, current_pa: f64, duration_ms: f64,
                   timestamp_ms: f64, priority: EventPriority) -> Self {
        NeuralEvent::Stimulus { neuron_id, current_pa, duration_ms, timestamp_ms, priority }
    }
    
    /// Check if this event targets a specific neuron
    pub fn targets_neuron(&self, neuron_id: usize) -> bool {
        match self {
            NeuralEvent::Spike { targets, .. } => targets.contains(&neuron_id),
            NeuralEvent::Stimulus { neuron_id: target, .. } => *target == neuron_id,
            _ => false,
        }
    }
    
    /// Get estimated processing cost for this event type
    pub fn processing_cost(&self) -> u64 {
        match self {
            NeuralEvent::Spike { targets, .. } => targets.len() as u64 * 10,
            NeuralEvent::Plasticity { .. } => 50,
            NeuralEvent::Stimulus { .. } => 20,
            NeuralEvent::Reconfiguration { data, .. } => data.len() as u64,
            NeuralEvent::Maintenance { .. } => 100,
        }
    }
}

/// Wrapper for events in the priority queue
#[derive(Debug, Clone)]
struct QueuedEvent {
    /// The neural event
    event: NeuralEvent,
    
    /// Insertion order for tie-breaking
    insertion_order: u64,
}

impl QueuedEvent {
    fn new(event: NeuralEvent, insertion_order: u64) -> Self {
        Self { event, insertion_order }
    }
}

impl PartialEq for QueuedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.event.timestamp_ms() == other.event.timestamp_ms() &&
        self.event.priority() == other.event.priority() &&
        self.insertion_order == other.insertion_order
    }
}

impl Eq for QueuedEvent {}

impl Ord for QueuedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: timestamp (earlier events first)
        let time_cmp = other.event.timestamp_ms()
            .partial_cmp(&self.event.timestamp_ms())
            .unwrap_or(Ordering::Equal);
            
        if time_cmp != Ordering::Equal {
            return time_cmp;
        }
        
        // Secondary: priority (higher priority first)  
        let priority_cmp = self.event.priority().cmp(&other.event.priority());
        if priority_cmp != Ordering::Equal {
            return priority_cmp;
        }
        
        // Tertiary: insertion order (FIFO for ties)
        self.insertion_order.cmp(&other.insertion_order)
    }
}

impl PartialOrd for QueuedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Event filter for selective processing
#[derive(Debug, Clone)]
pub enum EventFilter {
    /// Accept all events
    None,
    
    /// Filter by event type
    EventType(Vec<String>),
    
    /// Filter by priority level
    Priority(EventPriority),
    
    /// Filter by timestamp range
    TimeRange { start_ms: f64, end_ms: f64 },
    
    /// Filter by neuron IDs
    NeuronIds(Vec<usize>),
    
    /// Filter by synapse IDs
    SynapseIds(Vec<usize>),
    
    /// Custom filter function
    Custom(fn(&NeuralEvent) -> bool),
}

impl EventFilter {
    /// Check if event passes this filter
    pub fn passes(&self, event: &NeuralEvent) -> bool {
        match self {
            EventFilter::None => true,
            
            EventFilter::EventType(types) => {
                let event_type = match event {
                    NeuralEvent::Spike { .. } => "spike",
                    NeuralEvent::Plasticity { .. } => "plasticity", 
                    NeuralEvent::Stimulus { .. } => "stimulus",
                    NeuralEvent::Reconfiguration { .. } => "reconfiguration",
                    NeuralEvent::Maintenance { .. } => "maintenance",
                };
                types.contains(&event_type.to_string())
            }
            
            EventFilter::Priority(min_priority) => {
                event.priority() <= *min_priority
            }
            
            EventFilter::TimeRange { start_ms, end_ms } => {
                let timestamp = event.timestamp_ms();
                timestamp >= *start_ms && timestamp <= *end_ms
            }
            
            EventFilter::NeuronIds(ids) => {
                match event {
                    NeuralEvent::Spike { event, targets, .. } => {
                        ids.contains(&event.neuron_id) || 
                        targets.iter().any(|&id| ids.contains(&id))
                    }
                    NeuralEvent::Stimulus { neuron_id, .. } => ids.contains(neuron_id),
                    _ => false,
                }
            }
            
            EventFilter::SynapseIds(ids) => {
                match event {
                    NeuralEvent::Plasticity { synapse_id, .. } => ids.contains(synapse_id),
                    _ => false,
                }
            }
            
            EventFilter::Custom(filter_fn) => filter_fn(event),
        }
    }
}

/// Statistics for event queue performance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventQueueStats {
    /// Total events processed
    pub total_events_processed: u64,
    
    /// Events currently in queue
    pub events_in_queue: usize,
    
    /// Maximum queue size reached
    pub max_queue_size: usize,
    
    /// Average processing latency in microseconds
    pub avg_latency_us: f64,
    
    /// Events processed per second
    pub throughput_eps: f64,
    
    /// Total processing time
    pub total_processing_time_ms: f64,
    
    /// Number of events dropped due to filtering
    pub events_dropped: u64,
    
    /// Number of events batched
    pub events_batched: u64,
    
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
}

/// High-performance event queue for neural processing
#[derive(Debug)]
pub struct EventQueue {
    /// Priority queue for events (min-heap by timestamp)
    queue: BinaryHeap<QueuedEvent>,
    
    /// Event filters
    filters: Vec<EventFilter>,
    
    /// Current simulation time
    current_time_ms: f64,
    
    /// Event insertion counter for ordering
    insertion_counter: u64,
    
    /// Performance statistics
    stats: EventQueueStats,
    
    /// Maximum queue size (for memory management)
    max_queue_size: usize,
    
    /// Event batching configuration
    batch_size: usize,
    
    /// Batch timeout in milliseconds
    batch_timeout_ms: f64,
    
    /// Pending batch events
    batch_buffer: VecDeque<NeuralEvent>,
    
    /// Last batch processing time
    last_batch_time_ms: f64,
    
    /// Event routing table
    routing_table: HashMap<usize, Vec<usize>>, // neuron_id -> target_neurons
    
    /// Performance timing
    last_stats_update: Instant,
}

impl EventQueue {
    /// Create a new event queue
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            filters: Vec::new(),
            current_time_ms: 0.0,
            insertion_counter: 0,
            stats: EventQueueStats::default(),
            max_queue_size: 1_000_000, // 1M events max
            batch_size: 100,
            batch_timeout_ms: 1.0, // 1ms batch timeout
            batch_buffer: VecDeque::new(),
            last_batch_time_ms: 0.0,
            routing_table: HashMap::new(),
            last_stats_update: Instant::now(),
        }
    }
    
    /// Create event queue with custom configuration
    pub fn with_config(max_queue_size: usize, batch_size: usize, 
                      batch_timeout_ms: f64) -> Self {
        Self {
            max_queue_size,
            batch_size,
            batch_timeout_ms,
            ..Self::new()
        }
    }
    
    /// Add an event to the queue
    pub fn push_event(&mut self, event: NeuralEvent) -> Result<()> {
        // Check queue size limits
        if self.queue.len() >= self.max_queue_size {
            return Err(TengriError::Strategy(
                "Event queue is full".to_string()
            ));
        }
        
        // Apply filters
        if !self.passes_filters(&event) {
            self.stats.events_dropped += 1;
            return Ok(()); // Event filtered out
        }
        
        // Create queued event
        let queued_event = QueuedEvent::new(event, self.insertion_counter);
        self.insertion_counter += 1;
        
        // Add to queue
        self.queue.push(queued_event);
        
        // Update statistics
        self.stats.events_in_queue = self.queue.len();
        self.stats.max_queue_size = self.stats.max_queue_size.max(self.queue.len());
        
        Ok(())
    }
    
    /// Add multiple events at once
    pub fn push_events(&mut self, events: Vec<NeuralEvent>) -> Result<usize> {
        let mut added_count = 0;
        
        for event in events {
            if self.push_event(event).is_ok() {
                added_count += 1;
            }
        }
        
        Ok(added_count)
    }
    
    /// Get the next event to process (if ready)
    pub fn pop_event(&mut self) -> Option<NeuralEvent> {
        let start_time = Instant::now();
        
        // Check if next event is ready for processing
        if let Some(queued_event) = self.queue.peek() {
            if queued_event.event.timestamp_ms() <= self.current_time_ms {
                let event = self.queue.pop().unwrap().event;
                
                // Update statistics
                self.stats.total_events_processed += 1;
                self.stats.events_in_queue = self.queue.len();
                
                // Update latency tracking
                let processing_time = start_time.elapsed();
                let latency_us = processing_time.as_micros() as f64;
                self.update_latency_stats(latency_us);
                
                return Some(event);
            }
        }
        
        None
    }
    
    /// Get multiple events as a batch
    pub fn pop_batch(&mut self, max_count: usize) -> Vec<NeuralEvent> {
        let mut events = Vec::with_capacity(max_count);
        
        while events.len() < max_count {
            if let Some(event) = self.pop_event() {
                events.push(event);
            } else {
                break;
            }
        }
        
        if !events.is_empty() {
            self.stats.events_batched += events.len() as u64;
        }
        
        events
    }
    
    /// Process events up to a specific timestamp
    pub fn process_until(&mut self, target_time_ms: f64) -> Vec<NeuralEvent> {
        self.current_time_ms = target_time_ms;
        
        let mut processed_events = Vec::new();
        
        while let Some(event) = self.pop_event() {
            processed_events.push(event);
        }
        
        processed_events
    }
    
    /// Advance simulation time
    pub fn advance_time(&mut self, dt_ms: f64) {
        self.current_time_ms += dt_ms;
    }
    
    /// Set current simulation time
    pub fn set_time(&mut self, time_ms: f64) {
        self.current_time_ms = time_ms;
    }
    
    /// Get current simulation time
    pub fn current_time_ms(&self) -> f64 {
        self.current_time_ms
    }
    
    /// Add an event filter
    pub fn add_filter(&mut self, filter: EventFilter) {
        self.filters.push(filter);
    }
    
    /// Clear all filters
    pub fn clear_filters(&mut self) {
        self.filters.clear();
    }
    
    /// Check if event passes all filters
    fn passes_filters(&self, event: &NeuralEvent) -> bool {
        if self.filters.is_empty() {
            return true;
        }
        
        self.filters.iter().all(|filter| filter.passes(event))
    }
    
    /// Get number of events in queue
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    
    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
    
    /// Peek at next event without removing it
    pub fn peek(&self) -> Option<&NeuralEvent> {
        self.queue.peek().map(|queued| &queued.event)
    }
    
    /// Get timestamp of next event
    pub fn next_event_time(&self) -> Option<f64> {
        self.peek().map(|event| event.timestamp_ms())
    }
    
    /// Clear all events from queue
    pub fn clear(&mut self) {
        self.queue.clear();
        self.batch_buffer.clear();
        self.stats.events_in_queue = 0;
    }
    
    /// Get performance statistics
    pub fn stats(&self) -> &EventQueueStats {
        &self.stats
    }
    
    /// Update performance statistics
    pub fn update_stats(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_stats_update).as_secs_f64();
        
        if elapsed > 1.0 { // Update stats every second
            // Calculate throughput
            let events_per_second = self.stats.total_events_processed as f64 / 
                                   (self.stats.total_processing_time_ms / 1000.0);
            self.stats.throughput_eps = events_per_second;
            
            // Estimate memory usage
            let base_size = std::mem::size_of::<Self>();
            let queue_size = self.queue.capacity() * std::mem::size_of::<QueuedEvent>();
            let batch_size = self.batch_buffer.capacity() * std::mem::size_of::<NeuralEvent>();
            self.stats.memory_usage_bytes = base_size + queue_size + batch_size;
            
            self.last_stats_update = now;
        }
    }
    
    /// Update latency statistics  
    fn update_latency_stats(&mut self, latency_us: f64) {
        let total_events = self.stats.total_events_processed as f64;
        
        if total_events == 1.0 {
            self.stats.avg_latency_us = latency_us;
        } else {
            // Running average
            let alpha = 0.1; // Smoothing factor
            self.stats.avg_latency_us = self.stats.avg_latency_us * (1.0 - alpha) + 
                                       latency_us * alpha;
        }
    }
    
    /// Add routing for spike events
    pub fn add_routing(&mut self, source_neuron: usize, target_neurons: Vec<usize>) {
        self.routing_table.insert(source_neuron, target_neurons);
    }
    
    /// Get routing targets for a neuron
    pub fn get_routing_targets(&self, neuron_id: usize) -> Option<&Vec<usize>> {
        self.routing_table.get(&neuron_id)
    }
    
    /// Remove routing for a neuron
    pub fn remove_routing(&mut self, neuron_id: usize) {
        self.routing_table.remove(&neuron_id);
    }
    
    /// Create spike event with automatic routing
    pub fn create_routed_spike(&self, spike: SpikeEvent, 
                               priority: EventPriority) -> Option<NeuralEvent> {
        if let Some(targets) = self.get_routing_targets(spike.neuron_id) {
            Some(NeuralEvent::spike(spike, targets.clone(), priority))
        } else {
            None
        }
    }
    
    /// Process batch buffer if ready
    pub fn process_batch_buffer(&mut self) -> Vec<NeuralEvent> {
        let should_flush = self.batch_buffer.len() >= self.batch_size ||
                          (self.current_time_ms - self.last_batch_time_ms) >= self.batch_timeout_ms;
        
        if should_flush && !self.batch_buffer.is_empty() {
            let events: Vec<_> = self.batch_buffer.drain(..).collect();
            self.last_batch_time_ms = self.current_time_ms;
            self.stats.events_batched += events.len() as u64;
            return events;
        }
        
        Vec::new()
    }
    
    /// Add event to batch buffer
    pub fn add_to_batch(&mut self, event: NeuralEvent) {
        self.batch_buffer.push_back(event);
    }
    
    /// Get queue capacity
    pub fn capacity(&self) -> usize {
        self.max_queue_size
    }
    
    /// Get utilization percentage
    pub fn utilization(&self) -> f64 {
        (self.queue.len() as f64 / self.max_queue_size as f64) * 100.0
    }
    
    /// Estimate processing load
    pub fn estimated_load(&self) -> u64 {
        self.queue.iter()
            .map(|queued| queued.event.processing_cost())
            .sum()
    }
    
    /// Compact queue by removing old processed events from routing table
    pub fn compact(&mut self) {
        // Remove stale routing entries (this is a simplified example)
        // In practice, you'd want more sophisticated cleanup logic
        if self.routing_table.len() > 10000 {
            self.routing_table.retain(|_, targets| !targets.is_empty());
        }
    }
}

impl Default for EventQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Event queue manager for multiple queues
#[derive(Debug)]
pub struct EventQueueManager {
    /// Multiple priority queues
    queues: HashMap<EventPriority, EventQueue>,
    
    /// Global simulation time
    global_time_ms: f64,
    
    /// Load balancing configuration
    load_balancing_enabled: bool,
    
    /// Queue selection strategy
    selection_strategy: QueueSelectionStrategy,
}

/// Strategy for selecting which queue to use
#[derive(Debug, Clone, Copy)]
pub enum QueueSelectionStrategy {
    /// Round-robin selection
    RoundRobin,
    
    /// Select queue with lowest load
    LoadBalanced,
    
    /// Select by priority only
    PriorityOnly,
    
    /// Random selection
    Random,
}

impl EventQueueManager {
    /// Create a new queue manager
    pub fn new() -> Self {
        let mut queues = HashMap::new();
        
        // Create queues for each priority level
        for priority in [
            EventPriority::Critical,
            EventPriority::High,
            EventPriority::Normal,
            EventPriority::Low,
            EventPriority::Maintenance,
        ] {
            queues.insert(priority, EventQueue::new());
        }
        
        Self {
            queues,
            global_time_ms: 0.0,
            load_balancing_enabled: true,
            selection_strategy: QueueSelectionStrategy::PriorityOnly,
        }
    }
    
    /// Add event to appropriate queue
    pub fn push_event(&mut self, event: NeuralEvent) -> Result<()> {
        let priority = event.priority();
        
        if let Some(queue) = self.queues.get_mut(&priority) {
            queue.push_event(event)
        } else {
            Err(TengriError::Strategy(
                format!("No queue for priority {:?}", priority)
            ))
        }
    }
    
    /// Process events from all queues
    pub fn process_all(&mut self, target_time_ms: f64) -> HashMap<EventPriority, Vec<NeuralEvent>> {
        self.global_time_ms = target_time_ms;
        
        let mut all_events = HashMap::new();
        
        // Process in priority order
        for priority in [
            EventPriority::Critical,
            EventPriority::High,
            EventPriority::Normal,
            EventPriority::Low,
            EventPriority::Maintenance,
        ] {
            if let Some(queue) = self.queues.get_mut(&priority) {
                queue.set_time(target_time_ms);
                let events = queue.process_until(target_time_ms);
                if !events.is_empty() {
                    all_events.insert(priority, events);
                }
            }
        }
        
        all_events
    }
    
    /// Get total events across all queues
    pub fn total_events(&self) -> usize {
        self.queues.values().map(|q| q.len()).sum()
    }
    
    /// Get queue statistics for all priorities
    pub fn all_stats(&self) -> HashMap<EventPriority, EventQueueStats> {
        self.queues.iter()
            .map(|(&priority, queue)| (priority, queue.stats().clone()))
            .collect()
    }
    
    /// Clear all queues
    pub fn clear_all(&mut self) {
        for queue in self.queues.values_mut() {
            queue.clear();
        }
    }
}

impl Default for EventQueueManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuromorphic::spiking_neuron::SpikeEvent;
    
    #[test]
    fn test_event_creation() {
        let spike_event = SpikeEvent::new(1, 10.0, -55.0, 45.0);
        let neural_event = NeuralEvent::spike(spike_event, vec![2, 3], EventPriority::High);
        
        assert_eq!(neural_event.timestamp_ms(), 10.0);
        assert_eq!(neural_event.priority(), EventPriority::High);
        assert!(neural_event.targets_neuron(2));
        assert!(neural_event.targets_neuron(3));
        assert!(!neural_event.targets_neuron(4));
    }
    
    #[test]
    fn test_event_queue_basic() {
        let mut queue = EventQueue::new();
        
        // Add events with different timestamps
        let event1 = NeuralEvent::plasticity(1, 0.1, 5.0, EventPriority::Normal);
        let event2 = NeuralEvent::plasticity(2, -0.05, 3.0, EventPriority::Normal);
        let event3 = NeuralEvent::plasticity(3, 0.2, 7.0, EventPriority::Normal);
        
        queue.push_event(event1).unwrap();
        queue.push_event(event2).unwrap();
        queue.push_event(event3).unwrap();
        
        assert_eq!(queue.len(), 3);
        
        // Events should be processed in temporal order
        queue.set_time(10.0);
        
        let processed = queue.process_until(10.0);
        assert_eq!(processed.len(), 3);
        
        // First event should be timestamp 3.0
        assert_eq!(processed[0].timestamp_ms(), 3.0);
        assert_eq!(processed[1].timestamp_ms(), 5.0);
        assert_eq!(processed[2].timestamp_ms(), 7.0);
    }
    
    #[test]
    fn test_priority_ordering() {
        let mut queue = EventQueue::new();
        
        // Add events with same timestamp but different priorities
        let low_event = NeuralEvent::plasticity(1, 0.1, 5.0, EventPriority::Low);
        let high_event = NeuralEvent::plasticity(2, 0.1, 5.0, EventPriority::High);
        let critical_event = NeuralEvent::plasticity(3, 0.1, 5.0, EventPriority::Critical);
        
        // Add in random order
        queue.push_event(low_event).unwrap();
        queue.push_event(high_event).unwrap();
        queue.push_event(critical_event).unwrap();
        
        queue.set_time(10.0);
        let processed = queue.process_until(10.0);
        
        // Should be processed in priority order (Critical, High, Low)
        assert_eq!(processed[0].priority(), EventPriority::Critical);
        assert_eq!(processed[1].priority(), EventPriority::High);
        assert_eq!(processed[2].priority(), EventPriority::Low);
    }
    
    #[test]
    fn test_event_filtering() {
        let mut queue = EventQueue::new();
        
        // Add priority filter
        queue.add_filter(EventFilter::Priority(EventPriority::Normal));
        
        // Add events with different priorities
        let low_event = NeuralEvent::plasticity(1, 0.1, 5.0, EventPriority::Low);
        let normal_event = NeuralEvent::plasticity(2, 0.1, 5.0, EventPriority::Normal);
        let high_event = NeuralEvent::plasticity(3, 0.1, 5.0, EventPriority::High);
        
        queue.push_event(low_event).unwrap();
        queue.push_event(normal_event).unwrap();
        queue.push_event(high_event).unwrap(); // Should be filtered out
        
        queue.set_time(10.0);
        let processed = queue.process_until(10.0);
        
        // Only events with priority <= Normal should pass
        assert_eq!(processed.len(), 2);
        assert!(queue.stats().events_dropped > 0);
    }
    
    #[test]
    fn test_time_range_filter() {
        let mut queue = EventQueue::new();
        
        // Add time range filter
        queue.add_filter(EventFilter::TimeRange { start_ms: 5.0, end_ms: 15.0 });
        
        let early_event = NeuralEvent::plasticity(1, 0.1, 3.0, EventPriority::Normal);
        let valid_event = NeuralEvent::plasticity(2, 0.1, 10.0, EventPriority::Normal);
        let late_event = NeuralEvent::plasticity(3, 0.1, 20.0, EventPriority::Normal);
        
        queue.push_event(early_event).unwrap();
        queue.push_event(valid_event).unwrap();
        queue.push_event(late_event).unwrap();
        
        queue.set_time(25.0);
        let processed = queue.process_until(25.0);
        
        // Only event in time range should pass
        assert_eq!(processed.len(), 1);
        assert_eq!(processed[0].timestamp_ms(), 10.0);
    }
    
    #[test]
    fn test_routing_table() {
        let mut queue = EventQueue::new();
        
        // Set up routing
        queue.add_routing(1, vec![2, 3, 4]);
        queue.add_routing(2, vec![5, 6]);
        
        // Test routing lookup
        let targets = queue.get_routing_targets(1).unwrap();
        assert_eq!(*targets, vec![2, 3, 4]);
        
        // Test spike event creation with routing
        let spike = SpikeEvent::new(1, 10.0, -55.0, 45.0);
        let routed_event = queue.create_routed_spike(spike, EventPriority::Normal);
        
        assert!(routed_event.is_some());
        if let Some(NeuralEvent::Spike { targets, .. }) = routed_event {
            assert_eq!(targets, vec![2, 3, 4]);
        }
    }
    
    #[test]
    fn test_batch_processing() {
        let mut queue = EventQueue::with_config(10000, 3, 1.0); // Batch size 3
        
        // Add events to batch buffer
        for i in 0..5 {
            let event = NeuralEvent::plasticity(i, 0.1, i as f64, EventPriority::Normal);
            queue.add_to_batch(event);
        }
        
        // Process batch (should return 3 events due to batch size limit)
        let batch = queue.process_batch_buffer();
        assert_eq!(batch.len(), 3);
        
        // Process remaining events
        let remaining = queue.process_batch_buffer();
        assert_eq!(remaining.len(), 2);
    }
    
    #[test] 
    fn test_queue_manager() {
        let mut manager = EventQueueManager::new();
        
        // Add events with different priorities
        let critical_event = NeuralEvent::plasticity(1, 0.1, 5.0, EventPriority::Critical);
        let normal_event = NeuralEvent::plasticity(2, 0.1, 5.0, EventPriority::Normal);
        let low_event = NeuralEvent::plasticity(3, 0.1, 5.0, EventPriority::Low);
        
        manager.push_event(critical_event).unwrap();
        manager.push_event(normal_event).unwrap();
        manager.push_event(low_event).unwrap();
        
        assert_eq!(manager.total_events(), 3);
        
        // Process all events
        let all_events = manager.process_all(10.0);
        
        assert_eq!(all_events.len(), 3); // Three priority levels with events
        assert!(all_events.contains_key(&EventPriority::Critical));
        assert!(all_events.contains_key(&EventPriority::Normal));
        assert!(all_events.contains_key(&EventPriority::Low));
    }
    
    #[test]
    fn test_performance_stats() {
        let mut queue = EventQueue::new();
        
        // Add and process some events
        for i in 0..100 {
            let event = NeuralEvent::plasticity(i, 0.1, i as f64, EventPriority::Normal);
            queue.push_event(event).unwrap();
        }
        
        queue.set_time(200.0);
        let processed = queue.process_until(200.0);
        
        assert_eq!(processed.len(), 100);
        
        let stats = queue.stats();
        assert_eq!(stats.total_events_processed, 100);
        assert_eq!(stats.events_in_queue, 0);
        assert!(stats.max_queue_size > 0);
    }
    
    #[test]
    fn test_queue_capacity_limit() {
        let mut queue = EventQueue::with_config(5, 10, 1.0); // Max 5 events
        
        // Fill queue to capacity
        for i in 0..5 {
            let event = NeuralEvent::plasticity(i, 0.1, i as f64, EventPriority::Normal);
            assert!(queue.push_event(event).is_ok());
        }
        
        // Next event should fail
        let overflow_event = NeuralEvent::plasticity(6, 0.1, 6.0, EventPriority::Normal);
        assert!(queue.push_event(overflow_event).is_err());
    }
    
    #[test]
    fn test_neuron_filter() {
        let mut queue = EventQueue::new();
        
        // Filter for specific neurons
        queue.add_filter(EventFilter::NeuronIds(vec![1, 3, 5]));
        
        let spike1 = SpikeEvent::new(1, 10.0, -55.0, 45.0);
        let spike2 = SpikeEvent::new(2, 11.0, -55.0, 45.0);
        let spike3 = SpikeEvent::new(3, 12.0, -55.0, 45.0);
        
        let event1 = NeuralEvent::spike(spike1, vec![2], EventPriority::Normal);
        let event2 = NeuralEvent::spike(spike2, vec![3], EventPriority::Normal);
        let event3 = NeuralEvent::spike(spike3, vec![4], EventPriority::Normal);
        
        queue.push_event(event1).unwrap();
        queue.push_event(event2).unwrap();
        queue.push_event(event3).unwrap();
        
        queue.set_time(20.0);
        let processed = queue.process_until(20.0);
        
        // Only events from neurons 1 and 3 should pass
        assert_eq!(processed.len(), 2);
    }
}