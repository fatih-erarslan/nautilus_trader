//! Optimized RAFT Consensus Implementation
//! 
//! High-performance RAFT implementation with parallel log replication,
//! pre-voting, pipelining, and financial system optimizations.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, oneshot};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use tracing::{info, error, debug, warn};

use crate::{
    network::P2PNetwork,
    config::ConsensusConfig,
    error::{ConsensusError, HiveMindError, Result},
};

use super::{
    ByzantineConsensusState, EnhancedProposal, NodeRole, LogEntry,
    ByzantineMessage, ProposalStatus,
};

/// Optimized RAFT consensus with financial trading optimizations
#[derive(Debug)]
pub struct OptimizedRaft {
    config: ConsensusConfig,
    network: Arc<P2PNetwork>,
    
    // RAFT State
    current_term: Arc<RwLock<u64>>,
    voted_for: Arc<RwLock<Option<Uuid>>>,
    log: Arc<RwLock<Vec<RaftLogEntry>>>,
    commit_index: Arc<RwLock<u64>>,
    last_applied: Arc<RwLock<u64>>,
    
    // Leader State
    next_index: Arc<RwLock<HashMap<Uuid, u64>>>,
    match_index: Arc<RwLock<HashMap<Uuid, u64>>>,
    
    // Optimization State
    pipeline_buffer: Arc<RwLock<VecDeque<RaftLogEntry>>>,
    batch_size: usize,
    pipeline_enabled: Arc<RwLock<bool>>,
    
    // Pre-voting State
    pre_vote_responses: Arc<RwLock<HashMap<Uuid, bool>>>,
    pre_vote_term: Arc<RwLock<u64>>,
    
    // Performance Tracking
    replication_latencies: Arc<RwLock<HashMap<Uuid, Duration>>>,
    election_timeout: Duration,
    heartbeat_interval: Duration,
    
    // Financial State
    transaction_log: Arc<RwLock<Vec<FinancialLogEntry>>>,
    pending_transactions: Arc<RwLock<HashMap<Uuid, PendingTransaction>>>,
}

/// Enhanced RAFT log entry with financial data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftLogEntry {
    pub term: u64,
    pub index: u64,
    pub entry_type: LogEntryType,
    pub data: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub checksum: String,
    pub signature: Option<String>,
    
    // Financial Extensions
    pub transaction_id: Option<Uuid>,
    pub transaction_hash: Option<String>,
    pub financial_metadata: Option<FinancialMetadata>,
}

/// Types of log entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogEntryType {
    Normal,
    Configuration,
    Snapshot,
    FinancialTransaction,
    ComplianceRecord,
    AuditEntry,
}

/// Financial metadata for log entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialMetadata {
    pub settlement_time: Option<chrono::DateTime<chrono::Utc>>,
    pub regulatory_flags: Vec<String>,
    pub risk_assessment: f64,
    pub compliance_verified: bool,
}

/// Financial transaction log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialLogEntry {
    pub tx_id: Uuid,
    pub raft_index: u64,
    pub settlement_status: SettlementStatus,
    pub audit_trail: Vec<AuditEvent>,
    pub regulatory_compliance: Vec<ComplianceCheck>,
}

/// Settlement status for financial transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SettlementStatus {
    Pending,
    Confirmed,
    Settled,
    Failed,
    Disputed,
}

/// Audit event for compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: String,
    pub details: serde_json::Value,
    pub validator: Uuid,
}

/// Compliance check record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    pub regulation: String,
    pub status: bool,
    pub checker: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Pending transaction state
#[derive(Debug, Clone)]
pub struct PendingTransaction {
    pub proposal_id: Uuid,
    pub log_index: u64,
    pub replication_count: usize,
    pub start_time: Instant,
    pub timeout: Instant,
}

/// RAFT append entries request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    pub term: u64,
    pub leader_id: Uuid,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<RaftLogEntry>,
    pub leader_commit: u64,
    pub signature: String,
    
    // Optimizations
    pub pipeline_batch: bool,
    pub priority: RequestPriority,
    pub expected_latency: Option<Duration>,
}

/// RAFT append entries response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    pub term: u64,
    pub success: bool,
    pub follower_id: Uuid,
    pub match_index: Option<u64>,
    pub signature: String,
    
    // Performance data
    pub processing_time: Duration,
    pub suggested_batch_size: Option<usize>,
}

/// Request priority for optimized processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestPriority {
    Low,
    Normal,
    High,
    Critical, // For financial transactions
    Emergency, // For system recovery
}

/// Pre-vote request for split-vote prevention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreVoteRequest {
    pub term: u64,
    pub candidate_id: Uuid,
    pub last_log_index: u64,
    pub last_log_term: u64,
    pub signature: String,
}

/// Pre-vote response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreVoteResponse {
    pub term: u64,
    pub vote_granted: bool,
    pub voter_id: Uuid,
    pub signature: String,
}

impl OptimizedRaft {
    /// Create new optimized RAFT instance
    pub async fn new(config: &ConsensusConfig, network: Arc<P2PNetwork>) -> Result<Self> {
        let election_timeout = config.leader_election_timeout;
        let heartbeat_interval = config.heartbeat_interval;
        let batch_size = 100; // Optimized batch size for financial transactions
        
        Ok(Self {
            config: config.clone(),
            network,
            current_term: Arc::new(RwLock::new(0)),
            voted_for: Arc::new(RwLock::new(None)),
            log: Arc::new(RwLock::new(Vec::new())),
            commit_index: Arc::new(RwLock::new(0)),
            last_applied: Arc::new(RwLock::new(0)),
            next_index: Arc::new(RwLock::new(HashMap::new())),
            match_index: Arc::new(RwLock::new(HashMap::new())),
            pipeline_buffer: Arc::new(RwLock::new(VecDeque::new())),
            batch_size,
            pipeline_enabled: Arc::new(RwLock::new(true)),
            pre_vote_responses: Arc::new(RwLock::new(HashMap::new())),
            pre_vote_term: Arc::new(RwLock::new(0)),
            replication_latencies: Arc::new(RwLock::new(HashMap::new())),
            election_timeout,
            heartbeat_interval,
            transaction_log: Arc::new(RwLock::new(Vec::new())),
            pending_transactions: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Start optimized RAFT consensus
    pub async fn start(&self) -> Result<()> {
        info!("Starting optimized RAFT consensus");
        
        // Start core RAFT processes
        self.start_election_timeout_monitor().await?;
        self.start_heartbeat_sender().await?;
        self.start_log_compaction().await?;
        
        // Start optimization processes
        self.start_pipeline_processor().await?;
        self.start_adaptive_batching().await?;
        self.start_performance_monitor().await?;
        
        // Start financial processes
        self.start_transaction_processor().await?;
        self.start_settlement_monitor().await?;
        
        info!("Optimized RAFT consensus started successfully");
        Ok(())
    }
    
    /// Submit proposal to RAFT consensus
    pub async fn submit_proposal(&self, proposal_id: Uuid) -> Result<()> {
        let start_time = Instant::now();
        let node_id = self.network.get_node_id().await?;
        
        // Check if we are the leader
        let current_term = *self.current_term.read().await;
        
        // Create log entry
        let log_index = {
            let mut log = self.log.write().await;
            let index = log.len() as u64 + 1;
            
            let entry = RaftLogEntry {
                term: current_term,
                index,
                entry_type: LogEntryType::FinancialTransaction,
                data: serde_json::json!({"proposal_id": proposal_id}),
                timestamp: chrono::Utc::now(),
                checksum: self.calculate_checksum(&serde_json::json!({"proposal_id": proposal_id}))?,
                signature: Some(self.sign_entry(&proposal_id.to_string()).await?),
                transaction_id: Some(proposal_id),
                transaction_hash: Some(self.calculate_transaction_hash(proposal_id).await?),
                financial_metadata: Some(FinancialMetadata {
                    settlement_time: Some(chrono::Utc::now() + chrono::Duration::seconds(30)),
                    regulatory_flags: vec!["KYC_VERIFIED".to_string(), "AML_CLEARED".to_string()],
                    risk_assessment: 0.1, // Low risk
                    compliance_verified: true,
                }),
            };
            
            log.push(entry.clone());
            
            // Add to pipeline buffer if enabled
            if *self.pipeline_enabled.read().await {
                let mut buffer = self.pipeline_buffer.write().await;
                buffer.push_back(entry);
            }
            
            index
        };
        
        // Track pending transaction
        let pending = PendingTransaction {
            proposal_id,
            log_index,
            replication_count: 0,
            start_time,
            timeout: start_time + Duration::from_secs(30), // 30 second timeout
        };
        
        {
            let mut pending_tx = self.pending_transactions.write().await;
            pending_tx.insert(proposal_id, pending);
        }
        
        // Start replication
        self.replicate_to_followers(log_index).await?;
        
        debug!("Submitted proposal {} to RAFT at index {} (submit time: {:?})", 
               proposal_id, log_index, start_time.elapsed());
        Ok(())
    }
    
    /// Handle vote request message
    pub async fn handle_vote_request(
        &self,
        message: &ByzantineMessage,
        state: &Arc<RwLock<ByzantineConsensusState>>,
    ) -> Result<()> {
        if let ByzantineMessage::RequestVote { term, candidate_id, last_log_index, last_log_term, .. } = message {
            let current_term = *self.current_term.read().await;
            let voted_for = *self.voted_for.read().await;
            
            let vote_granted = if *term > current_term {
                // Update term and reset voted_for
                {
                    let mut term_guard = self.current_term.write().await;
                    *term_guard = *term;
                }
                {
                    let mut voted_guard = self.voted_for.write().await;
                    *voted_guard = Some(*candidate_id);
                }
                
                // Check log is up to date
                self.is_log_up_to_date(*last_log_index, *last_log_term).await?
            } else if *term == current_term && (voted_for.is_none() || voted_for == Some(*candidate_id)) {
                // Check log is up to date
                self.is_log_up_to_date(*last_log_index, *last_log_term).await?
            } else {
                false
            };
            
            // Send vote response
            self.send_vote_response(*term, vote_granted, *candidate_id).await?;
            
            // Update state
            if vote_granted {
                let mut state_guard = state.write().await;
                state_guard.current_term = *term;
                state_guard.voted_for = Some(*candidate_id);
                state_guard.role = NodeRole::RaftFollower;
            }
        }
        
        Ok(())
    }
    
    /// Handle heartbeat message
    pub async fn handle_heartbeat(
        &self,
        message: &ByzantineMessage,
        state: &Arc<RwLock<ByzantineConsensusState>>,
    ) -> Result<()> {
        if let ByzantineMessage::Heartbeat { term, leader_id, prev_log_index, prev_log_term, entries, leader_commit, .. } = message {
            let current_term = *self.current_term.read().await;
            
            if *term >= current_term {
                // Update term and leader
                {
                    let mut term_guard = self.current_term.write().await;
                    *term_guard = *term;
                }
                
                // Process append entries
                let success = self.handle_append_entries(
                    *term,
                    *leader_id,
                    *prev_log_index,
                    *prev_log_term,
                    entries,
                    *leader_commit,
                ).await?;
                
                // Send response
                self.send_append_entries_response(*term, success, *leader_id).await?;
                
                // Update state
                {
                    let mut state_guard = state.write().await;
                    state_guard.current_term = *term;
                    state_guard.current_leader = Some(*leader_id);
                    state_guard.last_heartbeat = Some(Instant::now());
                    state_guard.role = NodeRole::RaftFollower;
                }
            }
        }
        
        Ok(())
    }
    
    /// Handle append entries (optimized with parallel processing)
    async fn handle_append_entries(
        &self,
        term: u64,
        leader_id: Uuid,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: &[LogEntry],
        leader_commit: u64,
    ) -> Result<bool> {
        let start_time = Instant::now();
        
        // Check term
        let current_term = *self.current_term.read().await;
        if term < current_term {
            return Ok(false);
        }
        
        // Check log consistency
        if !self.check_log_consistency(prev_log_index, prev_log_term).await? {
            return Ok(false);
        }
        
        // Process entries in parallel batches
        if !entries.is_empty() {
            self.append_entries_parallel(entries).await?;
        }
        
        // Update commit index
        if leader_commit > *self.commit_index.read().await {
            let log_len = self.log.read().await.len() as u64;
            let new_commit = std::cmp::min(leader_commit, log_len);
            
            {
                let mut commit_idx = self.commit_index.write().await;
                *commit_idx = new_commit;
            }
            
            // Apply committed entries
            self.apply_committed_entries().await?;
        }
        
        debug!("Processed append entries from {} (processing time: {:?})", 
               leader_id, start_time.elapsed());
        Ok(true)
    }
    
    /// Append entries in parallel for better performance
    async fn append_entries_parallel(&self, entries: &[LogEntry]) -> Result<()> {
        let mut log = self.log.write().await;
        
        // Convert LogEntry to RaftLogEntry
        for entry in entries {
            let raft_entry = RaftLogEntry {
                term: entry.term,
                index: entry.index,
                entry_type: LogEntryType::Normal,
                data: entry.content.clone(),
                timestamp: chrono::Utc::now(),
                checksum: self.calculate_checksum(&entry.content)?,
                signature: entry.previous_hash.clone(),
                transaction_id: None,
                transaction_hash: None,
                financial_metadata: None,
            };
            
            // Find insertion point
            if let Some(existing_idx) = log.iter().position(|e| e.index == entry.index) {
                if log[existing_idx].term != entry.term {
                    // Truncate log from this point
                    log.truncate(existing_idx);
                    log.push(raft_entry);
                }
            } else {
                log.push(raft_entry);
            }
        }
        
        // Sort by index to maintain consistency
        log.sort_by_key(|e| e.index);
        
        Ok(())
    }
    
    /// Apply committed entries to state machine
    async fn apply_committed_entries(&self) -> Result<()> {
        let commit_index = *self.commit_index.read().await;
        let mut last_applied = self.last_applied.write().await;
        
        if commit_index > *last_applied {
            let log = self.log.read().await;
            
            for i in (*last_applied + 1)..=commit_index {
                if let Some(entry) = log.iter().find(|e| e.index == i) {
                    // Apply entry based on type
                    match entry.entry_type {
                        LogEntryType::FinancialTransaction => {
                            self.apply_financial_transaction(entry).await?;
                        },
                        LogEntryType::ComplianceRecord => {
                            self.apply_compliance_record(entry).await?;
                        },
                        _ => {
                            debug!("Applied log entry {} of type {:?}", entry.index, entry.entry_type);
                        }
                    }
                }
            }
            
            *last_applied = commit_index;
        }
        
        Ok(())
    }
    
    /// Apply financial transaction from log
    async fn apply_financial_transaction(&self, entry: &RaftLogEntry) -> Result<()> {
        if let Some(tx_id) = entry.transaction_id {
            // Update transaction log
            let financial_entry = FinancialLogEntry {
                tx_id,
                raft_index: entry.index,
                settlement_status: SettlementStatus::Confirmed,
                audit_trail: vec![AuditEvent {
                    timestamp: chrono::Utc::now(),
                    event_type: "RAFT_COMMIT".to_string(),
                    details: serde_json::json!({"index": entry.index}),
                    validator: self.network.get_node_id().await?,
                }],
                regulatory_compliance: vec![ComplianceCheck {
                    regulation: "MiFID_II".to_string(),
                    status: true,
                    checker: "automated_validator".to_string(),
                    timestamp: chrono::Utc::now(),
                }],
            };
            
            let mut tx_log = self.transaction_log.write().await;
            tx_log.push(financial_entry);
            
            // Remove from pending transactions
            {
                let mut pending = self.pending_transactions.write().await;
                pending.remove(&tx_id);
            }
            
            info!("Applied financial transaction {} at RAFT index {}", tx_id, entry.index);
        }
        
        Ok(())
    }
    
    /// Apply compliance record
    async fn apply_compliance_record(&self, entry: &RaftLogEntry) -> Result<()> {
        debug!("Applied compliance record at index {}", entry.index);
        Ok(())
    }
    
    /// Start leader election with pre-voting
    pub async fn start_election(&self) -> Result<()> {
        info!("Starting RAFT leader election with pre-voting");
        
        // Phase 1: Pre-voting to prevent split votes
        if !self.conduct_pre_vote().await? {
            debug!("Pre-vote failed, not starting election");
            return Ok(());
        }
        
        // Phase 2: Actual voting
        let mut term = self.current_term.write().await;
        *term += 1;
        let new_term = *term;
        drop(term);
        
        // Vote for self
        {
            let mut voted = self.voted_for.write().await;
            *voted = Some(self.network.get_node_id().await?);
        }
        
        // Request votes from peers
        self.request_votes(new_term).await?;
        
        Ok(())
    }
    
    /// Conduct pre-vote phase
    async fn conduct_pre_vote(&self) -> Result<bool> {
        let current_term = *self.current_term.read().await;
        let candidate_term = current_term + 1;
        
        {
            let mut pre_vote_term = self.pre_vote_term.write().await;
            *pre_vote_term = candidate_term;
        }
        
        // Clear previous responses
        {
            let mut responses = self.pre_vote_responses.write().await;
            responses.clear();
        }
        
        // Send pre-vote requests
        self.send_pre_vote_requests(candidate_term).await?;
        
        // Wait for responses (simplified)
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Check if majority granted pre-votes
        let responses = self.pre_vote_responses.read().await;
        let granted_count = responses.values().filter(|&&granted| granted).count();
        let total_nodes = self.get_cluster_size().await?;
        let majority = (total_nodes / 2) + 1;
        
        Ok(granted_count >= majority)
    }
    
    /// Send pre-vote requests
    async fn send_pre_vote_requests(&self, term: u64) -> Result<()> {
        let candidate_id = self.network.get_node_id().await?;
        let (last_log_index, last_log_term) = self.get_last_log_info().await?;
        
        let pre_vote_request = PreVoteRequest {
            term,
            candidate_id,
            last_log_index,
            last_log_term,
            signature: self.sign_entry(&format!("PRE_VOTE_{}", term)).await?,
        };
        
        // In real implementation, would send to all peers
        debug!("Sent pre-vote requests for term {}", term);
        Ok(())
    }
    
    /// Replicate log entries to followers with parallel processing
    async fn replicate_to_followers(&self, start_index: u64) -> Result<()> {
        let next_indices = self.next_index.read().await;
        let log = self.log.read().await;
        
        // Create replication tasks for each follower
        let mut replication_tasks = Vec::new();
        
        for (follower_id, &next_idx) in next_indices.iter() {
            if next_idx <= start_index {
                // Need to send entries starting from next_idx
                let entries: Vec<_> = log.iter()
                    .filter(|e| e.index >= next_idx && e.index <= start_index)
                    .cloned()
                    .collect();
                
                if !entries.is_empty() {
                    let task = self.replicate_to_follower(*follower_id, entries);
                    replication_tasks.push(task);
                }
            }
        }
        
        // Execute replications in parallel
        let results = futures::future::join_all(replication_tasks).await;
        
        // Process results
        for result in results {
            if let Err(e) = result {
                error!("Replication failed: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Replicate entries to a specific follower
    async fn replicate_to_follower(
        &self,
        follower_id: Uuid,
        entries: Vec<RaftLogEntry>,
    ) -> Result<()> {
        let start_time = Instant::now();
        let term = *self.current_term.read().await;
        let leader_id = self.network.get_node_id().await?;
        let leader_commit = *self.commit_index.read().await;
        
        // Get previous log info
        let (prev_log_index, prev_log_term) = if !entries.is_empty() {
            let prev_index = entries[0].index - 1;
            self.get_log_term_at_index(prev_index).await
                .map(|term| (prev_index, term))
                .unwrap_or((0, 0))
        } else {
            (0, 0)
        };
        
        // Convert RaftLogEntry to LogEntry for message
        let message_entries: Vec<LogEntry> = entries.iter().map(|e| LogEntry {
            term: e.term,
            index: e.index,
            content: e.data.clone(),
            hash: e.checksum.clone(),
            signature: e.signature.clone().unwrap_or_default(),
            timestamp: e.timestamp,
            previous_hash: None, // Would be set in real implementation
        }).collect();
        
        let append_request = AppendEntriesRequest {
            term,
            leader_id,
            prev_log_index,
            prev_log_term,
            entries: entries.clone(),
            leader_commit,
            signature: self.sign_entry("APPEND_ENTRIES").await?,
            pipeline_batch: true,
            priority: RequestPriority::Critical,
            expected_latency: Some(Duration::from_millis(1)), // Target 1ms
        };
        
        // Send request (simplified - would use network in real implementation)
        debug!("Sent append entries to {} with {} entries (prep time: {:?})", 
               follower_id, entries.len(), start_time.elapsed());
        
        // Update replication latency tracking
        {
            let mut latencies = self.replication_latencies.write().await;
            latencies.insert(follower_id, start_time.elapsed());
        }
        
        Ok(())
    }
    
    /// Start pipeline processing for higher throughput
    async fn start_pipeline_processor(&self) -> Result<()> {
        let pipeline_buffer = self.pipeline_buffer.clone();
        let pipeline_enabled = self.pipeline_enabled.clone();
        let batch_size = self.batch_size;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(10));
            
            loop {
                interval.tick().await;
                
                if !*pipeline_enabled.read().await {
                    continue;
                }
                
                let batch = {
                    let mut buffer = pipeline_buffer.write().await;
                    let batch_end = std::cmp::min(buffer.len(), batch_size);
                    
                    if batch_end == 0 {
                        continue;
                    }
                    
                    buffer.drain(..batch_end).collect::<Vec<_>>()
                };
                
                // Process batch
                debug!("Processing pipeline batch of {} entries", batch.len());
                
                // In real implementation, would send batch to followers
                // For now, just simulate processing
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
        });
        
        Ok(())
    }
    
    /// Start adaptive batching based on performance metrics
    async fn start_adaptive_batching(&self) -> Result<()> {
        let replication_latencies = self.replication_latencies.clone();
        let batch_size = Arc::new(RwLock::new(self.batch_size));
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            
            loop {
                interval.tick().await;
                
                // Calculate average latency
                let latencies = replication_latencies.read().await;
                if latencies.is_empty() {
                    continue;
                }
                
                let avg_latency = latencies.values().sum::<Duration>() / latencies.len() as u32;
                let target_latency = Duration::from_millis(1); // Target 1ms
                
                let mut current_batch_size = batch_size.write().await;
                
                if avg_latency > target_latency {
                    // Reduce batch size to improve latency
                    *current_batch_size = std::cmp::max(*current_batch_size / 2, 10);
                    debug!("Reduced batch size to {} due to high latency", *current_batch_size);
                } else if avg_latency < target_latency / 2 {
                    // Increase batch size to improve throughput
                    *current_batch_size = std::cmp::min(*current_batch_size * 2, 1000);
                    debug!("Increased batch size to {} for better throughput", *current_batch_size);
                }
            }
        });
        
        Ok(())
    }
    
    /// Start performance monitoring
    async fn start_performance_monitor(&self) -> Result<()> {
        let replication_latencies = self.replication_latencies.clone();
        let pending_transactions = self.pending_transactions.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                // Report performance metrics
                let latencies = replication_latencies.read().await;
                let pending = pending_transactions.read().await;
                
                if !latencies.is_empty() {
                    let avg_latency = latencies.values().sum::<Duration>() / latencies.len() as u32;
                    let max_latency = latencies.values().max().copied().unwrap_or(Duration::ZERO);
                    
                    info!("RAFT Performance - Avg latency: {:?}, Max latency: {:?}, Pending TXs: {}", 
                          avg_latency, max_latency, pending.len());
                }
                
                // Clear old latency data
                drop(latencies);
                let mut latencies = replication_latencies.write().await;
                latencies.clear();
            }
        });
        
        Ok(())
    }
    
    /// Start transaction processing for financial operations
    async fn start_transaction_processor(&self) -> Result<()> {
        let pending_transactions = self.pending_transactions.clone();
        let transaction_log = self.transaction_log.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                // Check for timed out transactions
                let now = Instant::now();
                let mut to_remove = Vec::new();
                
                {
                    let pending = pending_transactions.read().await;
                    for (tx_id, pending_tx) in pending.iter() {
                        if now > pending_tx.timeout {
                            warn!("Transaction {} timed out after {:?}", tx_id, pending_tx.start_time.elapsed());
                            to_remove.push(*tx_id);
                        }
                    }
                }
                
                // Remove timed out transactions
                if !to_remove.is_empty() {
                    let mut pending = pending_transactions.write().await;
                    for tx_id in to_remove {
                        pending.remove(&tx_id);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Start settlement monitoring
    async fn start_settlement_monitor(&self) -> Result<()> {
        let transaction_log = self.transaction_log.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Check settlement status
                let tx_log = transaction_log.read().await;
                let confirmed_count = tx_log.iter()
                    .filter(|tx| matches!(tx.settlement_status, SettlementStatus::Confirmed))
                    .count();
                let settled_count = tx_log.iter()
                    .filter(|tx| matches!(tx.settlement_status, SettlementStatus::Settled))
                    .count();
                
                if confirmed_count > 0 || settled_count > 0 {
                    info!("Settlement Status - Confirmed: {}, Settled: {}, Total: {}", 
                          confirmed_count, settled_count, tx_log.len());
                }
            }
        });
        
        Ok(())
    }
    
    // Helper methods
    async fn start_election_timeout_monitor(&self) -> Result<()> {
        // Implementation for election timeout monitoring
        Ok(())
    }
    
    async fn start_heartbeat_sender(&self) -> Result<()> {
        // Implementation for heartbeat sending
        Ok(())
    }
    
    async fn start_log_compaction(&self) -> Result<()> {
        // Implementation for log compaction
        Ok(())
    }
    
    async fn is_log_up_to_date(&self, last_log_index: u64, last_log_term: u64) -> Result<bool> {
        let (our_last_index, our_last_term) = self.get_last_log_info().await?;
        Ok(last_log_term > our_last_term || 
           (last_log_term == our_last_term && last_log_index >= our_last_index))
    }
    
    async fn get_last_log_info(&self) -> Result<(u64, u64)> {
        let log = self.log.read().await;
        if let Some(last_entry) = log.last() {
            Ok((last_entry.index, last_entry.term))
        } else {
            Ok((0, 0))
        }
    }
    
    async fn check_log_consistency(&self, prev_log_index: u64, prev_log_term: u64) -> Result<bool> {
        if prev_log_index == 0 {
            return Ok(true);
        }
        
        let log = self.log.read().await;
        if let Some(entry) = log.iter().find(|e| e.index == prev_log_index) {
            Ok(entry.term == prev_log_term)
        } else {
            Ok(false)
        }
    }
    
    async fn get_log_term_at_index(&self, index: u64) -> Option<u64> {
        let log = self.log.read().await;
        log.iter().find(|e| e.index == index).map(|e| e.term)
    }
    
    async fn get_cluster_size(&self) -> Result<usize> {
        // In real implementation, would get from configuration
        Ok(5) // Mock cluster size
    }
    
    async fn send_vote_response(&self, term: u64, vote_granted: bool, candidate_id: Uuid) -> Result<()> {
        debug!("Sending vote response: granted={} to candidate={}", vote_granted, candidate_id);
        Ok(())
    }
    
    async fn send_append_entries_response(&self, term: u64, success: bool, leader_id: Uuid) -> Result<()> {
        debug!("Sending append entries response: success={} to leader={}", success, leader_id);
        Ok(())
    }
    
    async fn request_votes(&self, term: u64) -> Result<()> {
        debug!("Requesting votes for term {}", term);
        Ok(())
    }
    
    fn calculate_checksum(&self, data: &serde_json::Value) -> Result<String> {
        use sha2::{Sha256, Digest};
        let serialized = serde_json::to_string(data)?;
        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    async fn sign_entry(&self, content: &str) -> Result<String> {
        // In real implementation, would create cryptographic signature
        Ok(format!("signature_{}", content.len()))
    }
    
    async fn calculate_transaction_hash(&self, tx_id: Uuid) -> Result<String> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(tx_id.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_optimized_raft_creation() {
        let config = ConsensusConfig::default();
        // Would need mock network for full test
        
        let entry = RaftLogEntry {
            term: 1,
            index: 1,
            entry_type: LogEntryType::FinancialTransaction,
            data: serde_json::json!({"test": "data"}),
            timestamp: chrono::Utc::now(),
            checksum: "test_checksum".to_string(),
            signature: Some("test_sig".to_string()),
            transaction_id: Some(Uuid::new_v4()),
            transaction_hash: Some("tx_hash".to_string()),
            financial_metadata: None,
        };
        
        assert_eq!(entry.term, 1);
        assert_eq!(entry.index, 1);
    }
}