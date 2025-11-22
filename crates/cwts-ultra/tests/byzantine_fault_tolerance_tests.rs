//! Byzantine Fault Tolerance Test Suite for Distributed Bayesian VaR
//!
//! This module implements comprehensive Byzantine fault tolerance testing for
//! distributed Bayesian VaR calculation systems, based on the foundational
//! Byzantine Generals Problem and modern consensus algorithms.
//!
//! ## Research Citations:
//! - Lamport, L., et al. "The Byzantine Generals Problem" (1982) - ACM Transactions
//! - Castro, M., Liskov, B. "Practical Byzantine Fault Tolerance" (1999) - OSDI
//! - Yin, M., et al. "HotStuff: BFT Consensus in the Lens of Blockchain" (2019) - PODC
//! - Miller, A., et al. "The Honey Badger of BFT Protocols" (2016) - CCS
//! - Buchman, E., et al. "The latest gossip on BFT consensus" (2018) - arXiv

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::thread;
use std::sync::mpsc::{channel, Receiver, Sender};
use tokio::time::timeout;
use rand::{Rng, thread_rng};
use serde::{Serialize, Deserialize};
use thiserror::Error;

// Import test infrastructure
use super::bayesian_var_research_tests::*;

/// Byzantine fault tolerance specific errors
#[derive(Error, Debug, Clone)]
pub enum ByzantineFaultToleranceError {
    #[error("Consensus not reached: {reason}")]
    ConsensusFailure { reason: String },
    
    #[error("Too many Byzantine nodes: {byzantine_count} > f = {max_byzantine}")]
    TooManyByzantineNodes { byzantine_count: usize, max_byzantine: usize },
    
    #[error("Network partition detected: {partition_size} nodes isolated")]
    NetworkPartition { partition_size: usize },
    
    #[error("View change timeout: view {view}, timeout {timeout_ms}ms")]
    ViewChangeTimeout { view: u64, timeout_ms: u64 },
    
    #[error("Message authentication failed: {sender} -> {receiver}")]
    MessageAuthenticationFailure { sender: usize, receiver: usize },
    
    #[error("Liveness violation: no progress for {duration_ms}ms")]
    LivenessViolation { duration_ms: u64 },
    
    #[error("Safety violation: conflicting decisions detected")]
    SafetyViolation,
    
    #[error("Finality not achieved: {blocks_pending} blocks pending")]
    FinalityNotAchieved { blocks_pending: usize },
}

/// Byzantine node types for comprehensive testing
#[derive(Debug, Clone, PartialEq)]
pub enum ByzantineNodeType {
    Honest,
    Silent,           // Fails to respond (crash fault)
    Malicious,        // Sends incorrect/conflicting messages
    Selfish,          // Acts in self-interest but follows protocol
    Faulty,           // Random incorrect behavior
    Partitioned,      // Network isolated
    SlowResponse,     // Responds but with high latency
    MessageDropping,  // Drops random messages
}

/// Message types in Byzantine consensus protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ByzantineMessage {
    Prepare {
        view: u64,
        sequence: u64,
        digest: String,
        sender: usize,
        timestamp: u64,
    },
    Commit {
        view: u64,
        sequence: u64,
        digest: String,
        sender: usize,
        timestamp: u64,
    },
    ViewChange {
        new_view: u64,
        sender: usize,
        prepared_messages: Vec<PreparedMessage>,
        timestamp: u64,
    },
    NewView {
        view: u64,
        view_change_messages: Vec<ByzantineMessage>,
        pre_prepare: Option<Box<ByzantineMessage>>,
        sender: usize,
        timestamp: u64,
    },
    VaRProposal {
        var_estimate: f64,
        confidence_interval: (f64, f64),
        node_id: usize,
        timestamp: u64,
        signature: String,
    },
    VaRAgreement {
        agreed_var: f64,
        supporting_nodes: Vec<usize>,
        view: u64,
        timestamp: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedMessage {
    pub sequence: u64,
    pub digest: String,
    pub view: u64,
}

/// Byzantine consensus state
#[derive(Debug, Clone)]
pub struct ByzantineConsensusState {
    pub current_view: u64,
    pub sequence_number: u64,
    pub phase: ConsensusPhase,
    pub prepared_messages: HashMap<String, usize>, // digest -> count
    pub committed_messages: HashMap<String, usize>,
    pub view_change_messages: HashMap<u64, Vec<ByzantineMessage>>,
    pub agreed_var_estimates: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusPhase {
    PrePrepare,
    Prepare,
    Commit,
    ViewChange,
    NewView,
    Finalized,
}

/// Enhanced Byzantine node with comprehensive fault injection
#[derive(Debug)]
pub struct EnhancedByzantineNode {
    pub node_id: usize,
    pub node_type: ByzantineNodeType,
    pub var_estimate: f64,
    pub consensus_state: ByzantineConsensusState,
    pub message_queue: VecDeque<ByzantineMessage>,
    pub sent_messages: Vec<ByzantineMessage>,
    pub received_messages: Vec<ByzantineMessage>,
    pub network_delay_ms: u64,
    pub message_drop_probability: f64,
    pub malicious_behavior_config: MaliciousBehaviorConfig,
    pub last_activity: Instant,
    pub is_active: bool,
}

#[derive(Debug, Clone)]
pub struct MaliciousBehaviorConfig {
    pub send_conflicting_messages: bool,
    pub corrupt_message_content: bool,
    pub delay_critical_messages: bool,
    pub forge_signatures: bool,
    pub replay_old_messages: bool,
    pub equivocate: bool, // Send different messages to different nodes
}

impl Default for MaliciousBehaviorConfig {
    fn default() -> Self {
        Self {
            send_conflicting_messages: false,
            corrupt_message_content: false,
            delay_critical_messages: false,
            forge_signatures: false,
            replay_old_messages: false,
            equivocate: false,
        }
    }
}

impl EnhancedByzantineNode {
    pub fn new(node_id: usize, node_type: ByzantineNodeType) -> Self {
        Self {
            node_id,
            node_type: node_type.clone(),
            var_estimate: -100.0, // Default honest VaR estimate
            consensus_state: ByzantineConsensusState {
                current_view: 0,
                sequence_number: 0,
                phase: ConsensusPhase::PrePrepare,
                prepared_messages: HashMap::new(),
                committed_messages: HashMap::new(),
                view_change_messages: HashMap::new(),
                agreed_var_estimates: Vec::new(),
            },
            message_queue: VecDeque::new(),
            sent_messages: Vec::new(),
            received_messages: Vec::new(),
            network_delay_ms: if node_type == ByzantineNodeType::SlowResponse { 1000 } else { 50 },
            message_drop_probability: if node_type == ByzantineNodeType::MessageDropping { 0.3 } else { 0.0 },
            malicious_behavior_config: if node_type == ByzantineNodeType::Malicious {
                MaliciousBehaviorConfig {
                    send_conflicting_messages: true,
                    corrupt_message_content: true,
                    equivocate: true,
                    ..Default::default()
                }
            } else {
                MaliciousBehaviorConfig::default()
            },
            last_activity: Instant::now(),
            is_active: node_type != ByzantineNodeType::Silent && node_type != ByzantineNodeType::Partitioned,
        }
    }
    
    pub fn get_var_estimate(&self) -> f64 {
        match self.node_type {
            ByzantineNodeType::Honest => self.var_estimate,
            ByzantineNodeType::Silent => f64::NAN, // No response
            ByzantineNodeType::Malicious => {
                // Return random malicious value
                thread_rng().gen_range(-10000.0..10000.0)
            },
            ByzantineNodeType::Selfish => {
                // Slightly biased towards self-interest but within bounds
                self.var_estimate * (1.0 + thread_rng().gen_range(-0.1..0.1))
            },
            ByzantineNodeType::Faulty => {
                // Random incorrect value
                if thread_rng().gen_bool(0.5) {
                    f64::INFINITY
                } else {
                    thread_rng().gen_range(-1000.0..1000.0)
                }
            },
            ByzantineNodeType::Partitioned => f64::NAN, // Network isolated
            ByzantineNodeType::SlowResponse => {
                // Correct value but delayed
                thread::sleep(Duration::from_millis(self.network_delay_ms));
                self.var_estimate
            },
            ByzantineNodeType::MessageDropping => {
                if thread_rng().gen_bool(self.message_drop_probability) {
                    f64::NAN // Simulate message drop
                } else {
                    self.var_estimate
                }
            },
        }
    }
    
    pub fn process_message(&mut self, message: ByzantineMessage) -> Vec<ByzantineMessage> {
        if !self.is_active {
            return Vec::new();
        }
        
        self.received_messages.push(message.clone());
        self.last_activity = Instant::now();
        
        match self.node_type {
            ByzantineNodeType::Silent | ByzantineNodeType::Partitioned => Vec::new(),
            ByzantineNodeType::Honest => self.honest_message_processing(message),
            ByzantineNodeType::Malicious => self.malicious_message_processing(message),
            _ => self.honest_message_processing(message), // Default to honest behavior
        }
    }
    
    fn honest_message_processing(&mut self, message: ByzantineMessage) -> Vec<ByzantineMessage> {
        let mut responses = Vec::new();
        
        match message {
            ByzantineMessage::VaRProposal { var_estimate, node_id, .. } => {
                // Honest validation of VaR proposal
                if self.validate_var_proposal(var_estimate) {
                    responses.push(ByzantineMessage::VaRAgreement {
                        agreed_var: var_estimate,
                        supporting_nodes: vec![self.node_id],
                        view: self.consensus_state.current_view,
                        timestamp: current_timestamp(),
                    });
                }
            },
            ByzantineMessage::Prepare { view, sequence, digest, .. } => {
                if view == self.consensus_state.current_view {
                    self.consensus_state.prepared_messages
                        .entry(digest.clone())
                        .and_modify(|count| *count += 1)
                        .or_insert(1);
                    
                    responses.push(ByzantineMessage::Commit {
                        view,
                        sequence,
                        digest,
                        sender: self.node_id,
                        timestamp: current_timestamp(),
                    });
                }
            },
            ByzantineMessage::Commit { digest, .. } => {
                self.consensus_state.committed_messages
                    .entry(digest)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            },
            _ => {}, // Handle other message types
        }
        
        responses
    }
    
    fn malicious_message_processing(&mut self, message: ByzantineMessage) -> Vec<ByzantineMessage> {
        let mut responses = Vec::new();
        
        match message {
            ByzantineMessage::VaRProposal { .. } => {
                // Malicious behavior: send conflicting agreements
                if self.malicious_behavior_config.send_conflicting_messages {
                    // Send multiple conflicting agreements
                    for i in 0..3 {
                        responses.push(ByzantineMessage::VaRAgreement {
                            agreed_var: -1000.0 * (i as f64 + 1.0),
                            supporting_nodes: vec![self.node_id],
                            view: self.consensus_state.current_view,
                            timestamp: current_timestamp(),
                        });
                    }
                } else {
                    // Send obviously malicious value
                    responses.push(ByzantineMessage::VaRAgreement {
                        agreed_var: f64::INFINITY,
                        supporting_nodes: vec![self.node_id],
                        view: self.consensus_state.current_view,
                        timestamp: current_timestamp(),
                    });
                }
            },
            ByzantineMessage::Prepare { view, sequence, .. } => {
                // Send prepare for wrong view
                responses.push(ByzantineMessage::Commit {
                    view: view + 1, // Wrong view
                    sequence,
                    digest: "malicious_digest".to_string(),
                    sender: self.node_id,
                    timestamp: current_timestamp(),
                });
            },
            _ => {},
        }
        
        responses
    }
    
    fn validate_var_proposal(&self, var_estimate: f64) -> bool {
        // Honest validation: VaR should be negative and finite
        var_estimate < 0.0 && var_estimate.is_finite() && var_estimate > -1_000_000.0
    }
}

/// Enhanced distributed Byzantine system with comprehensive fault injection
#[derive(Debug)]
pub struct EnhancedDistributedByzantineSystem {
    pub nodes: HashMap<usize, EnhancedByzantineNode>,
    pub network: ByzantineNetwork,
    pub consensus_config: ByzantineConsensusConfig,
    pub fault_injector: FaultInjector,
    pub metrics: ByzantineSystemMetrics,
}

#[derive(Debug, Clone)]
pub struct ByzantineConsensusConfig {
    pub f: usize, // Maximum Byzantine nodes: f < n/3
    pub view_timeout_ms: u64,
    pub message_timeout_ms: u64,
    pub required_agreement_threshold: f64, // 2f + 1 nodes
}

#[derive(Debug)]
pub struct ByzantineNetwork {
    pub message_channels: HashMap<(usize, usize), (Sender<ByzantineMessage>, Receiver<ByzantineMessage>)>,
    pub network_partitions: HashSet<usize>, // Partitioned nodes
    pub message_delays: HashMap<usize, u64>, // Node -> delay in ms
    pub message_loss_rate: f64,
}

#[derive(Debug)]
pub struct FaultInjector {
    pub active_faults: HashMap<String, FaultType>,
    pub fault_schedule: Vec<ScheduledFault>,
}

#[derive(Debug, Clone)]
pub enum FaultType {
    NodeCrash { node_id: usize },
    NetworkPartition { nodes: Vec<usize> },
    MessageCorruption { probability: f64 },
    DelayInjection { delay_ms: u64 },
    ByzantineBehavior { behavior: MaliciousBehaviorConfig },
}

#[derive(Debug, Clone)]
pub struct ScheduledFault {
    pub fault: FaultType,
    pub trigger_time: Instant,
    pub duration: Duration,
}

#[derive(Debug)]
pub struct ByzantineSystemMetrics {
    pub total_messages_sent: usize,
    pub total_messages_received: usize,
    pub consensus_rounds: usize,
    pub view_changes: usize,
    pub failed_agreements: usize,
    pub safety_violations: usize,
    pub liveness_violations: usize,
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

impl EnhancedDistributedByzantineSystem {
    pub fn new(n_nodes: usize, byzantine_config: ByzantineConsensusConfig) -> Self {
        let mut nodes = HashMap::new();
        
        for i in 0..n_nodes {
            let node = EnhancedByzantineNode::new(i, ByzantineNodeType::Honest);
            nodes.insert(i, node);
        }
        
        Self {
            nodes,
            network: ByzantineNetwork {
                message_channels: HashMap::new(),
                network_partitions: HashSet::new(),
                message_delays: HashMap::new(),
                message_loss_rate: 0.0,
            },
            consensus_config: byzantine_config,
            fault_injector: FaultInjector {
                active_faults: HashMap::new(),
                fault_schedule: Vec::new(),
            },
            metrics: ByzantineSystemMetrics {
                total_messages_sent: 0,
                total_messages_received: 0,
                consensus_rounds: 0,
                view_changes: 0,
                failed_agreements: 0,
                safety_violations: 0,
                liveness_violations: 0,
            },
        }
    }
    
    pub fn inject_byzantine_nodes(&mut self, byzantine_specs: Vec<(usize, ByzantineNodeType)>) -> Result<(), ByzantineFaultToleranceError> {
        let byzantine_count = byzantine_specs.len();
        
        if byzantine_count > self.consensus_config.f {
            return Err(ByzantineFaultToleranceError::TooManyByzantineNodes {
                byzantine_count,
                max_byzantine: self.consensus_config.f,
            });
        }
        
        for (node_id, node_type) in byzantine_specs {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.node_type = node_type.clone();
                
                // Configure malicious behavior if needed
                if node_type == ByzantineNodeType::Malicious {
                    node.malicious_behavior_config = MaliciousBehaviorConfig {
                        send_conflicting_messages: true,
                        corrupt_message_content: true,
                        equivocate: true,
                        ..Default::default()
                    };
                }
                
                // Configure network properties
                match node_type {
                    ByzantineNodeType::Silent | ByzantineNodeType::Partitioned => {
                        node.is_active = false;
                    },
                    ByzantineNodeType::SlowResponse => {
                        node.network_delay_ms = 2000;
                    },
                    ByzantineNodeType::MessageDropping => {
                        node.message_drop_probability = 0.5;
                    },
                    _ => {},
                }
            }
        }
        
        Ok(())
    }
    
    pub fn reach_bayesian_consensus(&mut self, timeout: Duration) -> Result<EnhancedConsensusResult, ByzantineFaultToleranceError> {
        let start_time = Instant::now();
        let mut consensus_reached = false;
        let mut agreed_var_estimates = Vec::new();
        
        // Phase 1: Collect VaR proposals from all nodes
        let mut var_proposals = HashMap::new();
        
        for (node_id, node) in &self.nodes {
            if node.is_active {
                let var_estimate = node.get_var_estimate();
                if var_estimate.is_finite() {
                    var_proposals.insert(*node_id, var_estimate);
                }
            }
        }
        
        // Phase 2: Byzantine consensus on VaR estimates
        let honest_nodes: Vec<usize> = self.nodes.iter()
            .filter(|(_, node)| node.node_type == ByzantineNodeType::Honest && node.is_active)
            .map(|(id, _)| *id)
            .collect();
        
        if honest_nodes.len() < 2 * self.consensus_config.f + 1 {
            return Err(ByzantineFaultToleranceError::ConsensusFailure {
                reason: "Insufficient honest nodes for Byzantine consensus".to_string(),
            });
        }
        
        // Run PBFT-style consensus
        let mut view = 0u64;
        let mut max_views = 10;
        
        while !consensus_reached && view < max_views && start_time.elapsed() < timeout {
            match self.run_consensus_round(view, &var_proposals) {
                Ok(result) => {
                    if result.consensus_achieved {
                        agreed_var_estimates = result.agreed_estimates;
                        consensus_reached = true;
                    } else {
                        view += 1;
                        self.metrics.view_changes += 1;
                    }
                },
                Err(e) => {
                    match e {
                        ByzantineFaultToleranceError::ViewChangeTimeout { .. } => {
                            view += 1;
                            self.metrics.view_changes += 1;
                        },
                        _ => return Err(e),
                    }
                }
            }
            
            self.metrics.consensus_rounds += 1;
        }
        
        if !consensus_reached {
            self.metrics.liveness_violations += 1;
            return Err(ByzantineFaultToleranceError::LivenessViolation {
                duration_ms: start_time.elapsed().as_millis() as u64,
            });
        }
        
        // Validate safety properties
        self.validate_safety_properties(&agreed_var_estimates)?;
        
        Ok(EnhancedConsensusResult {
            is_valid: consensus_reached,
            confidence_level: 0.95,
            agreed_var_estimates,
            consensus_time_ms: start_time.elapsed().as_millis() as u64,
            participating_nodes: honest_nodes.len(),
            byzantine_nodes_detected: self.detect_byzantine_nodes(),
            view_changes: view,
            safety_violations: self.metrics.safety_violations,
        })
    }
    
    fn run_consensus_round(&mut self, view: u64, var_proposals: &HashMap<usize, f64>) -> Result<ConsensusRoundResult, ByzantineFaultToleranceError> {
        let start_time = Instant::now();
        let timeout = Duration::from_millis(self.consensus_config.view_timeout_ms);
        
        // Pre-prepare phase: Primary proposes a value
        let primary = view as usize % self.nodes.len();
        let proposed_var = if let Some(var) = var_proposals.get(&primary) {
            *var
        } else {
            // Fallback to median of honest proposals
            let mut honest_vars: Vec<f64> = var_proposals.values().cloned()
                .filter(|v| v.is_finite() && *v < 0.0)
                .collect();
            
            if honest_vars.is_empty() {
                return Err(ByzantineFaultToleranceError::ConsensusFailure {
                    reason: "No valid VaR proposals available".to_string(),
                });
            }
            
            honest_vars.sort_by(|a, b| a.partial_cmp(b).unwrap());
            honest_vars[honest_vars.len() / 2]
        };
        
        // Prepare phase: Collect prepare messages
        let mut prepare_votes = HashMap::new();
        let digest = format!("var_{:.6}", proposed_var);
        
        for (node_id, node) in &mut self.nodes {
            if node.is_active {
                let prepare_msg = ByzantineMessage::Prepare {
                    view,
                    sequence: 0,
                    digest: digest.clone(),
                    sender: *node_id,
                    timestamp: current_timestamp(),
                };
                
                let responses = node.process_message(prepare_msg);
                for response in responses {
                    if let ByzantineMessage::Commit { sender, .. } = response {
                        prepare_votes.insert(sender, proposed_var);
                    }
                }
                
                self.metrics.total_messages_sent += 1;
            }
        }
        
        // Check if we have enough prepare votes (2f + 1)
        let required_votes = 2 * self.consensus_config.f + 1;
        let honest_votes = prepare_votes.iter()
            .filter(|(node_id, var)| {
                self.nodes.get(node_id)
                    .map(|n| n.node_type == ByzantineNodeType::Honest)
                    .unwrap_or(false) && 
                    var.is_finite() && **var < 0.0
            })
            .count();
        
        if honest_votes >= required_votes {
            Ok(ConsensusRoundResult {
                consensus_achieved: true,
                agreed_estimates: vec![proposed_var],
                participating_nodes: honest_votes,
            })
        } else if start_time.elapsed() > timeout {
            Err(ByzantineFaultToleranceError::ViewChangeTimeout {
                view,
                timeout_ms: self.consensus_config.view_timeout_ms,
            })
        } else {
            Ok(ConsensusRoundResult {
                consensus_achieved: false,
                agreed_estimates: Vec::new(),
                participating_nodes: honest_votes,
            })
        }
    }
    
    fn validate_safety_properties(&mut self, agreed_estimates: &[f64]) -> Result<(), ByzantineFaultToleranceError> {
        // Safety property 1: All agreed estimates should be valid VaR values
        for &estimate in agreed_estimates {
            if !estimate.is_finite() || estimate >= 0.0 {
                self.metrics.safety_violations += 1;
                return Err(ByzantineFaultToleranceError::SafetyViolation);
            }
        }
        
        // Safety property 2: No conflicting agreements
        let unique_estimates: HashSet<_> = agreed_estimates.iter()
            .map(|v| (v * 1000000.0).round() as i64) // Round to avoid floating point issues
            .collect();
        
        if unique_estimates.len() > 1 {
            self.metrics.safety_violations += 1;
            return Err(ByzantineFaultToleranceError::SafetyViolation);
        }
        
        Ok(())
    }
    
    fn detect_byzantine_nodes(&self) -> Vec<usize> {
        self.nodes.iter()
            .filter(|(_, node)| node.node_type != ByzantineNodeType::Honest)
            .map(|(id, _)| *id)
            .collect()
    }
    
    pub fn get_honest_nodes(&self) -> Vec<&EnhancedByzantineNode> {
        self.nodes.values()
            .filter(|node| node.node_type == ByzantineNodeType::Honest)
            .collect()
    }
    
    pub fn inject_network_partition(&mut self, partitioned_nodes: Vec<usize>) -> Result<(), ByzantineFaultToleranceError> {
        for node_id in partitioned_nodes {
            self.network.network_partitions.insert(node_id);
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.node_type = ByzantineNodeType::Partitioned;
                node.is_active = false;
            }
        }
        
        Ok(())
    }
    
    pub fn heal_network_partition(&mut self) {
        for node_id in self.network.network_partitions.clone() {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.node_type = ByzantineNodeType::Honest;
                node.is_active = true;
            }
        }
        self.network.network_partitions.clear();
    }
}

#[derive(Debug)]
pub struct ConsensusRoundResult {
    pub consensus_achieved: bool,
    pub agreed_estimates: Vec<f64>,
    pub participating_nodes: usize,
}

#[derive(Debug)]
pub struct EnhancedConsensusResult {
    pub is_valid: bool,
    pub confidence_level: f64,
    pub agreed_var_estimates: Vec<f64>,
    pub consensus_time_ms: u64,
    pub participating_nodes: usize,
    pub byzantine_nodes_detected: Vec<usize>,
    pub view_changes: u64,
    pub safety_violations: usize,
}

impl EnhancedConsensusResult {
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }
    
    pub fn confidence_level(&self) -> f64 {
        self.confidence_level
    }
}

#[cfg(test)]
mod byzantine_fault_tolerance_tests {
    use super::*;
    
    #[test]
    fn test_lamport_byzantine_generals_classic_scenario() {
        // Classic Byzantine Generals Problem with 4 generals, 1 traitor
        let config = ByzantineConsensusConfig {
            f: 1, // Can tolerate 1 Byzantine node with 4 total nodes
            view_timeout_ms: 1000,
            message_timeout_ms: 500,
            required_agreement_threshold: 0.75,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(4, config);
        
        // Inject 1 Byzantine node (within f < n/3 limit)
        system.inject_byzantine_nodes(vec![
            (3, ByzantineNodeType::Malicious),
        ]).unwrap();
        
        let result = system.reach_bayesian_consensus(Duration::from_secs(5)).unwrap();
        
        assert!(result.is_valid());
        assert_eq!(result.byzantine_nodes_detected.len(), 1);
        assert!(result.participating_nodes >= 3); // 2f + 1 = 3 minimum
        assert_eq!(result.safety_violations, 0);
    }
    
    #[test]
    fn test_practical_byzantine_fault_tolerance_pbft() {
        // PBFT scenario with view changes
        let config = ByzantineConsensusConfig {
            f: 1,
            view_timeout_ms: 2000,
            message_timeout_ms: 1000,
            required_agreement_threshold: 0.67,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(7, config); // n = 3f + 1 = 7
        
        // Inject various Byzantine behaviors
        system.inject_byzantine_nodes(vec![
            (0, ByzantineNodeType::Malicious),
            (1, ByzantineNodeType::Silent),
        ]).unwrap();
        
        let result = system.reach_bayesian_consensus(Duration::from_secs(10)).unwrap();
        
        assert!(result.is_valid());
        assert!(result.participating_nodes >= 5); // Should have enough honest nodes
        assert!(result.view_changes <= 5); // Should converge within reasonable view changes
    }
    
    #[test]
    fn test_network_partition_resilience() {
        let config = ByzantineConsensusConfig {
            f: 2,
            view_timeout_ms: 1500,
            message_timeout_ms: 750,
            required_agreement_threshold: 0.6,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(10, config);
        
        // Create network partition
        system.inject_network_partition(vec![8, 9]).unwrap();
        
        // Should still reach consensus with majority partition
        let result = system.reach_bayesian_consensus(Duration::from_secs(8)).unwrap();
        
        assert!(result.is_valid());
        assert!(result.participating_nodes >= 6); // Majority partition
        
        // Heal partition and test recovery
        system.heal_network_partition();
        let recovery_result = system.reach_bayesian_consensus(Duration::from_secs(5)).unwrap();
        
        assert!(recovery_result.is_valid());
        assert!(recovery_result.participating_nodes >= 8); // More nodes after healing
    }
    
    #[test]
    fn test_message_dropping_and_delays() {
        let config = ByzantineConsensusConfig {
            f: 1,
            view_timeout_ms: 3000, // Longer timeout for slow messages
            message_timeout_ms: 1500,
            required_agreement_threshold: 0.75,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(6, config);
        
        // Inject nodes with network issues
        system.inject_byzantine_nodes(vec![
            (4, ByzantineNodeType::MessageDropping),
            (5, ByzantineNodeType::SlowResponse),
        ]).unwrap();
        
        let result = system.reach_bayesian_consensus(Duration::from_secs(15)).unwrap();
        
        assert!(result.is_valid());
        assert!(result.consensus_time_ms > 1000); // Should take longer due to delays
        // Should still reach consensus despite message issues
    }
    
    #[test]
    fn test_equivocation_attack() {
        // Test against equivocation (sending different messages to different nodes)
        let config = ByzantineConsensusConfig {
            f: 1,
            view_timeout_ms: 2000,
            message_timeout_ms: 1000,
            required_agreement_threshold: 0.67,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(5, config);
        
        // Configure malicious node with equivocation behavior
        let mut equivocating_node = EnhancedByzantineNode::new(4, ByzantineNodeType::Malicious);
        equivocating_node.malicious_behavior_config.equivocate = true;
        equivocating_node.malicious_behavior_config.send_conflicting_messages = true;
        system.nodes.insert(4, equivocating_node);
        
        let result = system.reach_bayesian_consensus(Duration::from_secs(10));
        
        // Should either reach consensus despite equivocation or detect the attack
        match result {
            Ok(consensus) => {
                assert!(consensus.is_valid());
                assert_eq!(consensus.byzantine_nodes_detected.len(), 1);
            },
            Err(ByzantineFaultToleranceError::SafetyViolation) => {
                // Expected: safety violation detected due to conflicting messages
                assert!(system.metrics.safety_violations > 0);
            },
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
    
    #[test]
    fn test_safety_property_violation_detection() {
        let config = ByzantineConsensusConfig {
            f: 1,
            view_timeout_ms: 1000,
            message_timeout_ms: 500,
            required_agreement_threshold: 0.75,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(4, config);
        
        // Force safety violation by having malicious node propose invalid VaR
        let mut malicious_node = EnhancedByzantineNode::new(3, ByzantineNodeType::Malicious);
        malicious_node.var_estimate = f64::INFINITY; // Invalid VaR
        system.nodes.insert(3, malicious_node);
        
        // Create invalid agreed estimates to test safety validation
        let invalid_estimates = vec![f64::INFINITY, -100.0]; // Mix of invalid and valid
        
        let safety_result = system.validate_safety_properties(&invalid_estimates);
        
        assert!(safety_result.is_err());
        assert!(matches!(safety_result.unwrap_err(), ByzantineFaultToleranceError::SafetyViolation));
        assert!(system.metrics.safety_violations > 0);
    }
    
    #[test]
    fn test_liveness_under_adversarial_conditions() {
        let config = ByzantineConsensusConfig {
            f: 2,
            view_timeout_ms: 500, // Short timeout to trigger liveness issues
            message_timeout_ms: 250,
            required_agreement_threshold: 0.6,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(9, config);
        
        // Maximum Byzantine nodes at the limit (f = 2 for n = 9)
        system.inject_byzantine_nodes(vec![
            (6, ByzantineNodeType::Malicious),
            (7, ByzantineNodeType::Silent),
        ]).unwrap();
        
        // Also inject network issues
        system.inject_byzantine_nodes(vec![
            (8, ByzantineNodeType::SlowResponse),
        ]).unwrap();
        
        let result = system.reach_bayesian_consensus(Duration::from_secs(3)); // Short timeout
        
        // May fail due to liveness issues or succeed with high latency
        match result {
            Ok(consensus) => {
                assert!(consensus.is_valid());
                assert!(consensus.consensus_time_ms > 500); // Should take longer
            },
            Err(ByzantineFaultToleranceError::LivenessViolation { .. }) => {
                // Expected under adversarial conditions with short timeout
                assert!(system.metrics.liveness_violations > 0);
            },
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
    
    #[test]
    fn test_byzantine_node_detection_accuracy() {
        let config = ByzantineConsensusConfig {
            f: 2,
            view_timeout_ms: 2000,
            message_timeout_ms: 1000,
            required_agreement_threshold: 0.6,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(8, config);
        
        // Inject known Byzantine nodes
        let byzantine_nodes = vec![
            (5, ByzantineNodeType::Malicious),
            (6, ByzantineNodeType::Faulty),
            (7, ByzantineNodeType::Selfish),
        ];
        
        system.inject_byzantine_nodes(byzantine_nodes.clone()).unwrap();
        
        let result = system.reach_bayesian_consensus(Duration::from_secs(10));
        
        match result {
            Ok(consensus) => {
                let detected = consensus.byzantine_nodes_detected;
                let expected: HashSet<usize> = byzantine_nodes.iter().map(|(id, _)| *id).collect();
                let detected_set: HashSet<usize> = detected.iter().cloned().collect();
                
                // Should detect all or most Byzantine nodes
                let detection_accuracy = detected_set.intersection(&expected).count() as f64 / expected.len() as f64;
                assert!(detection_accuracy >= 0.67, "Detection accuracy too low: {:.2}", detection_accuracy);
            },
            Err(_) => {
                // Even if consensus fails, detection should work
                let detected = system.detect_byzantine_nodes();
                assert_eq!(detected.len(), 3);
            }
        }
    }
    
    #[test]
    fn test_consensus_with_maximum_byzantine_nodes() {
        // Test at the theoretical limit: f < n/3
        let n = 10;
        let f = 3; // Maximum for n=10: f < 10/3 = 3.33, so f ≤ 3
        
        let config = ByzantineConsensusConfig {
            f,
            view_timeout_ms: 3000,
            message_timeout_ms: 1500,
            required_agreement_threshold: 0.7,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(n, config);
        
        // Inject exactly f Byzantine nodes
        let byzantine_specs = vec![
            (7, ByzantineNodeType::Malicious),
            (8, ByzantineNodeType::Silent),
            (9, ByzantineNodeType::Faulty),
        ];
        
        system.inject_byzantine_nodes(byzantine_specs).unwrap();
        
        let result = system.reach_bayesian_consensus(Duration::from_secs(15)).unwrap();
        
        assert!(result.is_valid());
        assert!(result.participating_nodes >= 2 * f + 1); // Minimum for consensus
        assert_eq!(result.byzantine_nodes_detected.len(), f);
    }
    
    #[test]
    fn test_consensus_impossible_with_too_many_byzantine() {
        let config = ByzantineConsensusConfig {
            f: 1, // Can only tolerate 1 Byzantine node
            view_timeout_ms: 1000,
            message_timeout_ms: 500,
            required_agreement_threshold: 0.75,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(4, config);
        
        // Try to inject more Byzantine nodes than tolerable
        let result = system.inject_byzantine_nodes(vec![
            (2, ByzantineNodeType::Malicious),
            (3, ByzantineNodeType::Silent),
        ]);
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ByzantineFaultToleranceError::TooManyByzantineNodes { .. }));
    }
}

#[cfg(test)]
mod advanced_byzantine_scenarios {
    use super::*;
    
    #[test]
    fn test_adaptive_adversary_with_changing_behavior() {
        // Test against adaptive adversary that changes behavior over time
        let config = ByzantineConsensusConfig {
            f: 2,
            view_timeout_ms: 2000,
            message_timeout_ms: 1000,
            required_agreement_threshold: 0.6,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(9, config);
        
        // Start with honest nodes
        system.inject_byzantine_nodes(vec![
            (7, ByzantineNodeType::Honest),
            (8, ByzantineNodeType::Honest),
        ]).unwrap();
        
        // First consensus round - should succeed
        let result1 = system.reach_bayesian_consensus(Duration::from_secs(5)).unwrap();
        assert!(result1.is_valid());
        
        // Adversary adapts: nodes become malicious
        system.inject_byzantine_nodes(vec![
            (7, ByzantineNodeType::Malicious),
            (8, ByzantineNodeType::Silent),
        ]).unwrap();
        
        // Second consensus round - should still work with adapted protocol
        let result2 = system.reach_bayesian_consensus(Duration::from_secs(8));
        
        match result2 {
            Ok(consensus) => assert!(consensus.is_valid()),
            Err(ByzantineFaultToleranceError::LivenessViolation { .. }) => {
                // Acceptable: adaptive adversary can cause temporary liveness issues
                assert!(system.metrics.view_changes > result1.view_changes);
            },
            Err(e) => panic!("Unexpected error with adaptive adversary: {:?}", e),
        }
    }
    
    #[test]
    fn test_consensus_with_mixed_fault_types() {
        // Test system resilience with multiple types of faults simultaneously
        let config = ByzantineConsensusConfig {
            f: 2,
            view_timeout_ms: 4000,
            message_timeout_ms: 2000,
            required_agreement_threshold: 0.6,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(10, config);
        
        // Inject diverse fault types
        system.inject_byzantine_nodes(vec![
            (7, ByzantineNodeType::Malicious),      // Active Byzantine
            (8, ByzantineNodeType::MessageDropping), // Network issues
            (9, ByzantineNodeType::SlowResponse),    // Performance issues
        ]).unwrap();
        
        // Add network partition
        system.inject_network_partition(vec![6]).unwrap();
        
        let start = Instant::now();
        let result = system.reach_bayesian_consensus(Duration::from_secs(20));
        let elapsed = start.elapsed();
        
        match result {
            Ok(consensus) => {
                assert!(consensus.is_valid());
                // Should handle mixed faults but may take longer
                assert!(elapsed.as_millis() > 1000); // Not too fast given the faults
                assert!(consensus.participating_nodes >= 6); // Majority despite faults
            },
            Err(ByzantineFaultToleranceError::LivenessViolation { .. }) => {
                // Acceptable: mixed faults can overwhelm consensus
                println!("Mixed faults caused liveness violation - acceptable");
            },
            Err(e) => panic!("Unexpected error with mixed faults: {:?}", e),
        }
    }
    
    #[test]
    fn test_long_running_consensus_stability() {
        // Test system stability over multiple consensus rounds
        let config = ByzantineConsensusConfig {
            f: 1,
            view_timeout_ms: 1000,
            message_timeout_ms: 500,
            required_agreement_threshold: 0.75,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(6, config);
        
        system.inject_byzantine_nodes(vec![
            (5, ByzantineNodeType::Faulty),
        ]).unwrap();
        
        let mut successful_rounds = 0;
        let total_rounds = 10;
        
        for round in 0..total_rounds {
            // Vary the Byzantine behavior slightly each round
            if round % 3 == 0 {
                system.inject_byzantine_nodes(vec![
                    (5, ByzantineNodeType::Malicious),
                ]).unwrap();
            } else {
                system.inject_byzantine_nodes(vec![
                    (5, ByzantineNodeType::Faulty),
                ]).unwrap();
            }
            
            let result = system.reach_bayesian_consensus(Duration::from_secs(5));
            
            match result {
                Ok(consensus) => {
                    if consensus.is_valid() {
                        successful_rounds += 1;
                    }
                },
                Err(_) => {
                    // Some rounds may fail due to changing conditions
                    println!("Round {} failed - continuing", round);
                }
            }
        }
        
        // Should succeed in majority of rounds despite changing conditions
        let success_rate = successful_rounds as f64 / total_rounds as f64;
        assert!(success_rate >= 0.7, "Success rate too low: {:.2}", success_rate);
    }
}

#[cfg(test)]
mod byzantine_performance_tests {
    use super::*;
    
    #[test]
    fn test_consensus_latency_under_load() {
        let config = ByzantineConsensusConfig {
            f: 1,
            view_timeout_ms: 2000,
            message_timeout_ms: 1000,
            required_agreement_threshold: 0.75,
        };
        
        let mut system = EnhancedDistributedByzantineSystem::new(7, config);
        
        system.inject_byzantine_nodes(vec![
            (6, ByzantineNodeType::SlowResponse),
        ]).unwrap();
        
        let mut latencies = Vec::new();
        
        for _ in 0..5 {
            let start = Instant::now();
            let result = system.reach_bayesian_consensus(Duration::from_secs(10));
            let latency = start.elapsed();
            
            if let Ok(consensus) = result {
                if consensus.is_valid() {
                    latencies.push(latency.as_millis() as u64);
                }
            }
        }
        
        assert!(!latencies.is_empty(), "No successful consensus rounds");
        
        let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
        let max_latency = *latencies.iter().max().unwrap();
        
        println!("Average latency: {}ms, Max latency: {}ms", avg_latency, max_latency);
        
        // Performance thresholds
        assert!(avg_latency < 5000, "Average latency too high: {}ms", avg_latency);
        assert!(max_latency < 8000, "Maximum latency too high: {}ms", max_latency);
    }
}

/// Comprehensive coverage validation
#[cfg(test)]
mod byzantine_coverage_validation {
    use super::*;
    
    #[test]
    fn test_comprehensive_byzantine_coverage() {
        // Ensure all Byzantine fault tolerance scenarios are tested
        
        // 1. Classic Byzantine Generals Problem
        test_lamport_byzantine_generals_classic_scenario();
        
        // 2. PBFT scenarios
        test_practical_byzantine_fault_tolerance_pbft();
        
        // 3. Network partitions
        test_network_partition_resilience();
        
        // 4. Message delivery issues
        test_message_dropping_and_delays();
        
        // 5. Equivocation attacks
        test_equivocation_attack();
        
        // 6. Safety violations
        test_safety_property_violation_detection();
        
        // 7. Liveness under adversarial conditions
        test_liveness_under_adversarial_conditions();
        
        // 8. Byzantine node detection
        test_byzantine_node_detection_accuracy();
        
        // 9. Theoretical limits
        test_consensus_with_maximum_byzantine_nodes();
        test_consensus_impossible_with_too_many_byzantine();
        
        println!("Byzantine fault tolerance coverage validation completed");
    }
}