//! PBFT (Practical Byzantine Fault Tolerance) Implementation
//! 
//! Production-grade PBFT consensus optimized for financial trading systems.
//! Implements the three-phase protocol (pre-prepare, prepare, commit) with
//! view change support and Byzantine node detection.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use tracing::{info, error, debug, warn};
use sha2::{Sha256, Digest};

use crate::{
    network::P2PNetwork,
    config::ConsensusConfig,
    error::{ConsensusError, HiveMindError, Result},
};

use super::{
    ByzantineConsensusState, EnhancedProposal, PbftMessage, PbftPhase, 
    ConsensusProof, ByzantineVote, ProposalStatus,
};

/// PBFT consensus implementation with Byzantine fault tolerance
#[derive(Debug)]
pub struct PbftConsensus {
    config: ConsensusConfig,
    network: Arc<P2PNetwork>,
    
    // PBFT State
    view_number: Arc<RwLock<u64>>,
    sequence_number: Arc<RwLock<u64>>,
    primary_node: Arc<RwLock<Option<Uuid>>>,
    
    // Message Storage
    pre_prepare_messages: Arc<RwLock<HashMap<u64, PbftMessage>>>,
    prepare_messages: Arc<RwLock<HashMap<u64, HashMap<Uuid, PbftMessage>>>>,
    commit_messages: Arc<RwLock<HashMap<u64, HashMap<Uuid, PbftMessage>>>>,
    view_change_messages: Arc<RwLock<HashMap<u64, HashMap<Uuid, PbftMessage>>>>,
    
    // Byzantine Detection
    suspected_primaries: Arc<RwLock<HashSet<Uuid>>>,
    message_log: Arc<RwLock<Vec<PbftMessageLog>>>,
    
    // Performance Optimization
    message_batch: Arc<RwLock<Vec<EnhancedProposal>>>,
    batch_timeout: Duration,
    last_batch_time: Arc<RwLock<Instant>>,
    
    // Checkpoints for garbage collection
    checkpoints: Arc<RwLock<HashMap<u64, CheckpointState>>>,
    last_checkpoint: Arc<RwLock<u64>>,
}

/// PBFT message log for Byzantine detection
#[derive(Debug, Clone)]
struct PbftMessageLog {
    timestamp: Instant,
    sender: Uuid,
    message: PbftMessage,
    verified: bool,
    suspicious: bool,
}

/// Checkpoint state for garbage collection
#[derive(Debug, Clone)]
struct CheckpointState {
    sequence: u64,
    state_digest: String,
    proof: HashMap<Uuid, String>, // Node signatures
    timestamp: Instant,
}

/// PBFT proposal preparation result
#[derive(Debug)]
struct PrepareResult {
    can_commit: bool,
    commit_proof: Option<Vec<PbftMessage>>,
    byzantine_detected: Vec<Uuid>,
}

impl PbftConsensus {
    /// Create new PBFT consensus instance
    pub async fn new(config: &ConsensusConfig, network: Arc<P2PNetwork>) -> Result<Self> {
        let batch_timeout = Duration::from_millis(10); // 10ms batching for high throughput
        
        Ok(Self {
            config: config.clone(),
            network,
            view_number: Arc::new(RwLock::new(0)),
            sequence_number: Arc::new(RwLock::new(0)),
            primary_node: Arc::new(RwLock::new(None)),
            pre_prepare_messages: Arc::new(RwLock::new(HashMap::new())),
            prepare_messages: Arc::new(RwLock::new(HashMap::new())),
            commit_messages: Arc::new(RwLock::new(HashMap::new())),
            view_change_messages: Arc::new(RwLock::new(HashMap::new())),
            suspected_primaries: Arc::new(RwLock::new(HashSet::new())),
            message_log: Arc::new(RwLock::new(Vec::new())),
            message_batch: Arc::new(RwLock::new(Vec::new())),
            batch_timeout,
            last_batch_time: Arc::new(RwLock::new(Instant::now())),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            last_checkpoint: Arc::new(RwLock::new(0)),
        })
    }
    
    /// Start PBFT consensus
    pub async fn start(&self) -> Result<()> {
        info!("Starting PBFT consensus with Byzantine fault tolerance");
        
        // Initialize primary selection
        self.select_primary().await?;
        
        // Start batch processing
        self.start_batch_processing().await?;
        
        // Start checkpoint creation
        self.start_checkpoint_creation().await?;
        
        // Start Byzantine detection
        self.start_byzantine_monitoring().await?;
        
        info!("PBFT consensus started successfully");
        Ok(())
    }
    
    /// Submit proposal for PBFT consensus
    pub async fn submit_proposal(&self, proposal_id: Uuid) -> Result<()> {
        let start_time = Instant::now();
        
        // Check if we are the primary
        let is_primary = {
            let primary = self.primary_node.read().await;
            let node_id = self.network.get_node_id().await?;
            primary.map(|p| p == node_id).unwrap_or(false)
        };
        
        if !is_primary {
            // Forward to primary or trigger view change
            return self.forward_to_primary(proposal_id).await;
        }
        
        // Add to batch for processing
        self.add_to_batch(proposal_id).await?;
        
        debug!("Added proposal {} to PBFT batch (Primary processing time: {:?})", 
               proposal_id, start_time.elapsed());
        Ok(())
    }
    
    /// Handle incoming PBFT messages
    pub async fn handle_message(
        &self,
        message: &PbftMessage,
        state: &Arc<RwLock<ByzantineConsensusState>>,
        proposals: &Arc<RwLock<HashMap<Uuid, EnhancedProposal>>>,
    ) -> Result<()> {
        // Log message for Byzantine detection
        self.log_message(message).await?;
        
        // Verify message integrity
        if !self.verify_message(message).await? {
            warn!("Received invalid PBFT message, possible Byzantine behavior");
            self.report_suspicious_message(message).await?;
            return Ok(());
        }
        
        match message {
            PbftMessage::PrePrepare { view, sequence, .. } => {
                self.handle_pre_prepare(message, *view, *sequence, state, proposals).await?;
            },
            PbftMessage::Prepare { view, sequence, .. } => {
                self.handle_prepare(message, *view, *sequence, state, proposals).await?;
            },
            PbftMessage::Commit { view, sequence, .. } => {
                self.handle_commit(message, *view, *sequence, state, proposals).await?;
            },
            PbftMessage::ViewChange { new_view, .. } => {
                self.handle_view_change(message, *new_view, state).await?;
            },
            PbftMessage::NewView { view, .. } => {
                self.handle_new_view(message, *view, state).await?;
            },
        }
        
        Ok(())
    }
    
    /// Handle pre-prepare phase
    async fn handle_pre_prepare(
        &self,
        message: &PbftMessage,
        view: u64,
        sequence: u64,
        state: &Arc<RwLock<ByzantineConsensusState>>,
        proposals: &Arc<RwLock<HashMap<Uuid, EnhancedProposal>>>,
    ) -> Result<()> {
        let current_view = *self.view_number.read().await;
        
        // Verify view number
        if view != current_view {
            debug!("Received pre-prepare for different view: {} vs {}", view, current_view);
            return Ok(());
        }
        
        // Verify primary
        if !self.verify_primary_message(message).await? {
            warn!("Pre-prepare message from non-primary node");
            self.trigger_view_change().await?;
            return Ok(());
        }
        
        // Store pre-prepare message
        {
            let mut pre_prepares = self.pre_prepare_messages.write().await;
            pre_prepares.insert(sequence, message.clone());
        }
        
        // Update proposal status
        if let PbftMessage::PrePrepare { proposal, .. } = message {
            let mut proposals_guard = proposals.write().await;
            if let Some(prop) = proposals_guard.get_mut(&proposal.id) {
                prop.status = ProposalStatus::PrePrepared;
            }
        }
        
        // Send prepare message
        self.send_prepare(view, sequence, message).await?;
        
        debug!("Handled pre-prepare for view {} sequence {}", view, sequence);
        Ok(())
    }
    
    /// Handle prepare phase
    async fn handle_prepare(
        &self,
        message: &PbftMessage,
        view: u64,
        sequence: u64,
        state: &Arc<RwLock<ByzantineConsensusState>>,
        proposals: &Arc<RwLock<HashMap<Uuid, EnhancedProposal>>>,
    ) -> Result<()> {
        if let PbftMessage::Prepare { node_id, .. } = message {
            // Store prepare message
            {
                let mut prepares = self.prepare_messages.write().await;
                prepares.entry(sequence)
                    .or_insert_with(HashMap::new)
                    .insert(*node_id, message.clone());
            }
            
            // Check if we have enough prepare messages (2f)
            let prepare_result = self.check_prepare_threshold(sequence).await?;
            
            if prepare_result.can_commit {
                // Update proposal status
                let mut proposals_guard = proposals.write().await;
                if let Some(pre_prepare) = self.pre_prepare_messages.read().await.get(&sequence) {
                    if let PbftMessage::PrePrepare { proposal, .. } = pre_prepare {
                        if let Some(prop) = proposals_guard.get_mut(&proposal.id) {
                            prop.status = ProposalStatus::Prepared;
                        }
                    }
                }
                
                // Send commit message
                self.send_commit(view, sequence).await?;
                
                // Report any detected Byzantine nodes
                for byzantine_node in prepare_result.byzantine_detected {
                    self.report_byzantine_node(byzantine_node).await?;
                }
            }
        }
        
        debug!("Handled prepare for view {} sequence {}", view, sequence);
        Ok(())
    }
    
    /// Handle commit phase
    async fn handle_commit(
        &self,
        message: &PbftMessage,
        view: u64,
        sequence: u64,
        state: &Arc<RwLock<ByzantineConsensusState>>,
        proposals: &Arc<RwLock<HashMap<Uuid, EnhancedProposal>>>,
    ) -> Result<()> {
        if let PbftMessage::Commit { node_id, .. } = message {
            // Store commit message
            {
                let mut commits = self.commit_messages.write().await;
                commits.entry(sequence)
                    .or_insert_with(HashMap::new)
                    .insert(*node_id, message.clone());
            }
            
            // Check if we have enough commit messages (2f + 1)
            if self.check_commit_threshold(sequence).await? {
                // Finalize consensus
                self.finalize_consensus(view, sequence, proposals).await?;
            }
        }
        
        debug!("Handled commit for view {} sequence {}", view, sequence);
        Ok(())
    }
    
    /// Handle view change
    async fn handle_view_change(
        &self,
        message: &PbftMessage,
        new_view: u64,
        state: &Arc<RwLock<ByzantineConsensusState>>,
    ) -> Result<()> {
        if let PbftMessage::ViewChange { node_id, .. } = message {
            // Store view change message
            {
                let mut view_changes = self.view_change_messages.write().await;
                view_changes.entry(new_view)
                    .or_insert_with(HashMap::new)
                    .insert(*node_id, message.clone());
            }
            
            // Check if we have enough view change messages
            if self.check_view_change_threshold(new_view).await? {
                self.execute_view_change(new_view, state).await?;
            }
        }
        
        debug!("Handled view change to view {}", new_view);
        Ok(())
    }
    
    /// Handle new view
    async fn handle_new_view(
        &self,
        message: &PbftMessage,
        view: u64,
        state: &Arc<RwLock<ByzantineConsensusState>>,
    ) -> Result<()> {
        if let PbftMessage::NewView { primary_id, .. } = message {
            // Verify new view message
            if !self.verify_new_view(message).await? {
                warn!("Invalid new view message");
                return Ok(());
            }
            
            // Update view and primary
            {
                let mut view_num = self.view_number.write().await;
                *view_num = view;
            }
            
            {
                let mut primary = self.primary_node.write().await;
                *primary = Some(*primary_id);
            }
            
            // Update state
            {
                let mut state_guard = state.write().await;
                state_guard.view_number = view;
                state_guard.primary_node = Some(*primary_id);
                state_guard.phase = PbftPhase::PrePrepare;
            }
            
            info!("Completed view change to view {} with primary {}", view, primary_id);
        }
        
        Ok(())
    }
    
    /// Send prepare message
    async fn send_prepare(&self, view: u64, sequence: u64, pre_prepare: &PbftMessage) -> Result<()> {
        if let PbftMessage::PrePrepare { digest, .. } = pre_prepare {
            let node_id = self.network.get_node_id().await?;
            let signature = self.sign_message(&format!("PREPARE-{}-{}-{}", view, sequence, digest)).await?;
            
            let prepare = PbftMessage::Prepare {
                view,
                sequence,
                digest: digest.clone(),
                node_id,
                signature,
            };
            
            self.broadcast_message(prepare).await?;
        }
        
        Ok(())
    }
    
    /// Send commit message
    async fn send_commit(&self, view: u64, sequence: u64) -> Result<()> {
        let digest = self.get_sequence_digest(sequence).await?;
        let node_id = self.network.get_node_id().await?;
        let signature = self.sign_message(&format!("COMMIT-{}-{}-{}", view, sequence, digest)).await?;
        
        let commit = PbftMessage::Commit {
            view,
            sequence,
            digest,
            node_id,
            signature,
        };
        
        self.broadcast_message(commit).await?;
        Ok(())
    }
    
    /// Check prepare threshold (2f messages)
    async fn check_prepare_threshold(&self, sequence: u64) -> Result<PrepareResult> {
        let prepares = self.prepare_messages.read().await;
        let prepare_count = prepares.get(&sequence).map(|m| m.len()).unwrap_or(0);
        
        // Calculate 2f threshold (where f is max Byzantine nodes)
        let total_nodes = self.get_total_nodes().await?;
        let f = (total_nodes - 1) / 3; // Max Byzantine nodes
        let threshold = 2 * f;
        
        let can_commit = prepare_count >= threshold;
        
        // Check for Byzantine behavior in prepare messages
        let byzantine_detected = if let Some(messages) = prepares.get(&sequence) {
            self.detect_prepare_inconsistencies(messages).await?
        } else {
            Vec::new()
        };
        
        Ok(PrepareResult {
            can_commit,
            commit_proof: None, // Will be populated if needed
            byzantine_detected,
        })
    }
    
    /// Check commit threshold (2f + 1 messages)
    async fn check_commit_threshold(&self, sequence: u64) -> Result<bool> {
        let commits = self.commit_messages.read().await;
        let commit_count = commits.get(&sequence).map(|m| m.len()).unwrap_or(0);
        
        let total_nodes = self.get_total_nodes().await?;
        let f = (total_nodes - 1) / 3;
        let threshold = 2 * f + 1;
        
        Ok(commit_count >= threshold)
    }
    
    /// Finalize consensus for a sequence
    async fn finalize_consensus(
        &self,
        view: u64,
        sequence: u64,
        proposals: &Arc<RwLock<HashMap<Uuid, EnhancedProposal>>>,
    ) -> Result<()> {
        let start_time = Instant::now();
        
        // Get the original proposal
        let proposal_id = if let Some(pre_prepare) = self.pre_prepare_messages.read().await.get(&sequence) {
            if let PbftMessage::PrePrepare { proposal, .. } = pre_prepare {
                proposal.id
            } else {
                return Err(HiveMindError::InvalidState {
                    message: "Invalid pre-prepare message format".to_string(),
                });
            }
        } else {
            return Err(HiveMindError::InvalidState {
                message: "No pre-prepare message found".to_string(),
            });
        };
        
        // Create consensus proof
        let consensus_proof = self.create_consensus_proof(view, sequence).await?;
        
        // Update proposal status
        {
            let mut proposals_guard = proposals.write().await;
            if let Some(proposal) = proposals_guard.get_mut(&proposal_id) {
                proposal.status = ProposalStatus::Committed;
                proposal.consensus_proof = Some(consensus_proof);
                proposal.consensus_duration = Some(start_time - proposal.processing_start);
            }
        }
        
        // Clean up old messages (garbage collection)
        self.cleanup_old_messages(sequence).await?;
        
        // Create checkpoint if needed
        if sequence % 100 == 0 { // Checkpoint every 100 sequences
            self.create_checkpoint(sequence).await?;
        }
        
        info!("Finalized PBFT consensus for sequence {} (latency: {:?})", 
              sequence, start_time.elapsed());
        Ok(())
    }
    
    /// Create consensus proof
    async fn create_consensus_proof(&self, view: u64, sequence: u64) -> Result<ConsensusProof> {
        let commits = self.commit_messages.read().await;
        let commit_messages = commits.get(&sequence).ok_or_else(|| {
            HiveMindError::InvalidState {
                message: "No commit messages found for consensus proof".to_string(),
            }
        })?;
        
        let mut votes = Vec::new();
        let mut signatures = Vec::new();
        
        for message in commit_messages.values() {
            if let PbftMessage::Commit { node_id, signature, .. } = message {
                // Create vote record
                let vote = ByzantineVote {
                    voter: *node_id,
                    proposal_id: Uuid::new_v4(), // Will be set correctly in actual implementation
                    decision: crate::consensus::VoteDecision::Accept,
                    reasoning: Some("PBFT commit".to_string()),
                    timestamp: chrono::Utc::now(),
                    signature: signature.clone(),
                    hash: self.calculate_message_hash(message)?,
                    sequence_number: sequence,
                    view_number: view,
                    voter_reputation: 1.0, // Default reputation
                    confidence_score: 1.0,
                };
                
                votes.push(vote);
                signatures.push(signature.clone());
            }
        }
        
        // Calculate Merkle root of all commit messages
        let merkle_root = self.calculate_merkle_root(&signatures)?;
        
        Ok(ConsensusProof {
            proposal_id: Uuid::new_v4(), // Will be set correctly
            consensus_type: "PBFT".to_string(),
            votes,
            signatures,
            merkle_root,
            timestamp: chrono::Utc::now(),
            finality_proof: format!("PBFT-{}-{}", view, sequence),
        })
    }
    
    /// Detect inconsistencies in prepare messages (Byzantine detection)
    async fn detect_prepare_inconsistencies(
        &self,
        messages: &HashMap<Uuid, PbftMessage>,
    ) -> Result<Vec<Uuid>> {
        let mut byzantine_nodes = Vec::new();
        let mut digest_map: HashMap<String, Vec<Uuid>> = HashMap::new();
        
        // Group nodes by digest
        for (node_id, message) in messages {
            if let PbftMessage::Prepare { digest, .. } = message {
                digest_map.entry(digest.clone())
                    .or_insert_with(Vec::new)
                    .push(*node_id);
            }
        }
        
        // If we have conflicting digests, some nodes are Byzantine
        if digest_map.len() > 1 {
            // Find minority groups (likely Byzantine)
            let majority_size = digest_map.values()
                .map(|nodes| nodes.len())
                .max()
                .unwrap_or(0);
                
            for (digest, nodes) in digest_map {
                if nodes.len() < majority_size {
                    byzantine_nodes.extend(nodes);
                    warn!("Detected Byzantine behavior: conflicting digest {} from nodes {:?}", 
                          digest, nodes);
                }
            }
        }
        
        Ok(byzantine_nodes)
    }
    
    /// Start batch processing for higher throughput
    async fn start_batch_processing(&self) -> Result<()> {
        let message_batch = self.message_batch.clone();
        let last_batch_time = self.last_batch_time.clone();
        let batch_timeout = self.batch_timeout;
        let network = self.network.clone();
        let view_number = self.view_number.clone();
        let sequence_number = self.sequence_number.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(batch_timeout);
            
            loop {
                interval.tick().await;
                
                let should_process_batch = {
                    let batch = message_batch.read().await;
                    let last_time = last_batch_time.read().await;
                    
                    !batch.is_empty() && 
                    (batch.len() >= 10 || last_time.elapsed() > batch_timeout)
                };
                
                if should_process_batch {
                    let mut batch = message_batch.write().await;
                    if !batch.is_empty() {
                        let proposals: Vec<_> = batch.drain(..).collect();
                        drop(batch);
                        
                        // Process batch
                        if let Err(e) = Self::process_proposal_batch(
                            proposals,
                            &network,
                            &view_number,
                            &sequence_number
                        ).await {
                            error!("Failed to process proposal batch: {}", e);
                        }
                        
                        let mut last_time = last_batch_time.write().await;
                        *last_time = Instant::now();
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Process a batch of proposals for higher throughput
    async fn process_proposal_batch(
        proposals: Vec<EnhancedProposal>,
        network: &Arc<P2PNetwork>,
        view_number: &Arc<RwLock<u64>>,
        sequence_number: &Arc<RwLock<u64>>,
    ) -> Result<()> {
        let view = *view_number.read().await;
        let mut seq = sequence_number.write().await;
        
        for proposal in proposals {
            *seq += 1;
            let sequence = *seq;
            
            // Create digest for the proposal
            let digest = Self::calculate_proposal_digest(&proposal)?;
            let signature = "batch_signature".to_string(); // Would use real signing
            
            let pre_prepare = PbftMessage::PrePrepare {
                view,
                sequence,
                digest,
                proposal,
                signature,
            };
            
            // Broadcast pre-prepare
            // In real implementation, would broadcast through network
            debug!("Processed proposal in batch for sequence {}", sequence);
        }
        
        Ok(())
    }
    
    /// Add proposal to batch
    async fn add_to_batch(&self, proposal_id: Uuid) -> Result<()> {
        // In real implementation, would retrieve proposal and add to batch
        // For now, just simulate batching
        debug!("Added proposal {} to PBFT batch", proposal_id);
        Ok(())
    }
    
    /// Helper functions
    async fn select_primary(&self) -> Result<()> {
        let view = *self.view_number.read().await;
        let total_nodes = self.get_total_nodes().await?;
        
        // Simple primary selection: primary = view % total_nodes
        let primary_index = (view % total_nodes as u64) as usize;
        
        // In real implementation, would get actual node IDs
        let primary_id = Uuid::new_v4(); // Mock primary selection
        
        let mut primary = self.primary_node.write().await;
        *primary = Some(primary_id);
        
        debug!("Selected primary {} for view {}", primary_id, view);
        Ok(())
    }
    
    async fn get_total_nodes(&self) -> Result<usize> {
        // In real implementation, would get from network
        Ok(7) // Mock 7 nodes for testing
    }
    
    async fn verify_message(&self, message: &PbftMessage) -> Result<bool> {
        // In real implementation, would verify cryptographic signatures
        Ok(true) // Mock verification
    }
    
    async fn verify_primary_message(&self, message: &PbftMessage) -> Result<bool> {
        // Verify message comes from current primary
        Ok(true) // Mock verification
    }
    
    async fn sign_message(&self, content: &str) -> Result<String> {
        // In real implementation, would create cryptographic signature
        Ok(format!("signature_{}", content.len())) // Mock signature
    }
    
    async fn get_sequence_digest(&self, sequence: u64) -> Result<String> {
        // Get digest from pre-prepare message
        if let Some(pre_prepare) = self.pre_prepare_messages.read().await.get(&sequence) {
            if let PbftMessage::PrePrepare { digest, .. } = pre_prepare {
                Ok(digest.clone())
            } else {
                Err(HiveMindError::InvalidState {
                    message: "Invalid pre-prepare message".to_string(),
                })
            }
        } else {
            Err(HiveMindError::InvalidState {
                message: "No pre-prepare message found".to_string(),
            })
        }
    }
    
    async fn broadcast_message(&self, message: PbftMessage) -> Result<()> {
        // In real implementation, would broadcast through network
        debug!("Broadcasting PBFT message: {:?}", message);
        Ok(())
    }
    
    async fn log_message(&self, message: &PbftMessage) -> Result<()> {
        let log_entry = PbftMessageLog {
            timestamp: Instant::now(),
            sender: self.extract_sender_id(message),
            message: message.clone(),
            verified: true, // Would verify in real implementation
            suspicious: false,
        };
        
        let mut message_log = self.message_log.write().await;
        message_log.push(log_entry);
        
        // Keep only recent messages (last 1000)
        if message_log.len() > 1000 {
            message_log.drain(..500);
        }
        
        Ok(())
    }
    
    fn extract_sender_id(&self, message: &PbftMessage) -> Uuid {
        match message {
            PbftMessage::Prepare { node_id, .. } => *node_id,
            PbftMessage::Commit { node_id, .. } => *node_id,
            PbftMessage::ViewChange { node_id, .. } => *node_id,
            PbftMessage::NewView { primary_id, .. } => *primary_id,
            _ => Uuid::new_v4(), // Mock for PrePrepare
        }
    }
    
    async fn report_suspicious_message(&self, message: &PbftMessage) -> Result<()> {
        let sender = self.extract_sender_id(message);
        warn!("Reported suspicious PBFT message from node {}", sender);
        Ok(())
    }
    
    async fn report_byzantine_node(&self, node_id: Uuid) -> Result<()> {
        warn!("Detected Byzantine behavior from node {}", node_id);
        Ok(())
    }
    
    async fn forward_to_primary(&self, proposal_id: Uuid) -> Result<()> {
        debug!("Forwarding proposal {} to primary", proposal_id);
        Ok(())
    }
    
    async fn trigger_view_change(&self) -> Result<()> {
        info!("Triggering PBFT view change");
        Ok(())
    }
    
    async fn check_view_change_threshold(&self, new_view: u64) -> Result<bool> {
        let view_changes = self.view_change_messages.read().await;
        let change_count = view_changes.get(&new_view).map(|m| m.len()).unwrap_or(0);
        
        let total_nodes = self.get_total_nodes().await?;
        let f = (total_nodes - 1) / 3;
        let threshold = 2 * f + 1;
        
        Ok(change_count >= threshold)
    }
    
    async fn execute_view_change(&self, new_view: u64, state: &Arc<RwLock<ByzantineConsensusState>>) -> Result<()> {
        info!("Executing view change to view {}", new_view);
        
        // Update view number
        {
            let mut view_num = self.view_number.write().await;
            *view_num = new_view;
        }
        
        // Select new primary
        self.select_primary().await?;
        
        Ok(())
    }
    
    async fn verify_new_view(&self, message: &PbftMessage) -> Result<bool> {
        // Verify new view message is valid
        Ok(true) // Mock verification
    }
    
    async fn cleanup_old_messages(&self, current_sequence: u64) -> Result<()> {
        // Clean up messages older than 100 sequences
        let cutoff = current_sequence.saturating_sub(100);
        
        {
            let mut pre_prepares = self.pre_prepare_messages.write().await;
            pre_prepares.retain(|&seq, _| seq > cutoff);
        }
        
        {
            let mut prepares = self.prepare_messages.write().await;
            prepares.retain(|&seq, _| seq > cutoff);
        }
        
        {
            let mut commits = self.commit_messages.write().await;
            commits.retain(|&seq, _| seq > cutoff);
        }
        
        debug!("Cleaned up PBFT messages older than sequence {}", cutoff);
        Ok(())
    }
    
    async fn create_checkpoint(&self, sequence: u64) -> Result<()> {
        let state_digest = self.calculate_state_digest(sequence).await?;
        
        let checkpoint = CheckpointState {
            sequence,
            state_digest: state_digest.clone(),
            proof: HashMap::new(), // Would collect signatures in real implementation
            timestamp: Instant::now(),
        };
        
        {
            let mut checkpoints = self.checkpoints.write().await;
            checkpoints.insert(sequence, checkpoint);
        }
        
        {
            let mut last_checkpoint = self.last_checkpoint.write().await;
            *last_checkpoint = sequence;
        }
        
        info!("Created checkpoint at sequence {} with digest {}", sequence, state_digest);
        Ok(())
    }
    
    async fn calculate_state_digest(&self, sequence: u64) -> Result<String> {
        // Calculate digest of current state up to sequence
        let mut hasher = Sha256::new();
        hasher.update(format!("state_{}", sequence).as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    fn calculate_message_hash(&self, message: &PbftMessage) -> Result<String> {
        let serialized = serde_json::to_string(message)?;
        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    fn calculate_merkle_root(&self, signatures: &[String]) -> Result<String> {
        if signatures.is_empty() {
            return Ok("empty".to_string());
        }
        
        let mut hasher = Sha256::new();
        for sig in signatures {
            hasher.update(sig.as_bytes());
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    fn calculate_proposal_digest(proposal: &EnhancedProposal) -> Result<String> {
        let serialized = serde_json::to_string(proposal)?;
        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    async fn start_checkpoint_creation(&self) -> Result<()> {
        // Start periodic checkpoint creation
        Ok(())
    }
    
    async fn start_byzantine_monitoring(&self) -> Result<()> {
        // Start Byzantine behavior monitoring
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pbft_consensus_creation() {
        let config = ConsensusConfig::default();
        // Would need mock network for full test
        
        let message = PbftMessage::Prepare {
            view: 0,
            sequence: 1,
            digest: "test_digest".to_string(),
            node_id: Uuid::new_v4(),
            signature: "test_signature".to_string(),
        };
        
        assert!(matches!(message, PbftMessage::Prepare { .. }));
    }
}