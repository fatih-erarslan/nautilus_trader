//! Message Ordering Service
//! 
//! Ensures causal ordering and deterministic message delivery for consensus protocols.

use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

use crate::{config::ConsensusConfig, error::Result};
use super::ByzantineMessage;

/// Message ordering service with vector clocks and causal ordering
#[derive(Debug)]
pub struct MessageOrderingService {
    config: ConsensusConfig,
    vector_clocks: Arc<RwLock<HashMap<Uuid, VectorClock>>>,
    message_queue: Arc<RwLock<BTreeMap<u64, OrderedMessage>>>,
    sequence_number: Arc<RwLock<u64>>,
}

/// Vector clock for causal ordering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorClock {
    pub node_id: Uuid,
    pub clocks: HashMap<Uuid, u64>,
    pub last_update: Instant,
}

/// Ordered message with causality information
#[derive(Debug, Clone)]
pub struct OrderedMessage {
    pub sequence: u64,
    pub sender: Uuid,
    pub message: ByzantineMessage,
    pub vector_clock: VectorClock,
    pub timestamp: Instant,
    pub delivered: bool,
}

impl MessageOrderingService {
    pub async fn new(config: &ConsensusConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            vector_clocks: Arc::new(RwLock::new(HashMap::new())),
            message_queue: Arc::new(RwLock::new(BTreeMap::new())),
            sequence_number: Arc::new(RwLock::new(0)),
        })
    }
    
    /// Order message for delivery
    pub async fn order_message(&self, sender: Uuid, message: ByzantineMessage) -> Result<u64> {
        let mut seq_num = self.sequence_number.write().await;
        *seq_num += 1;
        let sequence = *seq_num;
        
        // Update vector clock
        let mut clocks = self.vector_clocks.write().await;
        let clock = clocks.entry(sender).or_insert_with(|| VectorClock {
            node_id: sender,
            clocks: HashMap::new(),
            last_update: Instant::now(),
        });
        
        *clock.clocks.entry(sender).or_insert(0) += 1;
        clock.last_update = Instant::now();
        
        // Add to message queue
        let ordered_msg = OrderedMessage {
            sequence,
            sender,
            message,
            vector_clock: clock.clone(),
            timestamp: Instant::now(),
            delivered: false,
        };
        
        let mut queue = self.message_queue.write().await;
        queue.insert(sequence, ordered_msg);
        
        Ok(sequence)
    }
}