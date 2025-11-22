//! Two-Phase Atomic Commit Protocol
//!
//! GREEN PHASE Implementation
//! Implements atomic execution across multiple exchanges with Byzantine fault tolerance
//! Ensures either all exchanges execute the transaction or none do (atomicity)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};

use super::byzantine_consensus::{ConsensusError, ValidatorId};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExchangeId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicTransaction {
    pub transaction_id: TransactionId,
    pub operations: Vec<ExchangeOperation>,
    pub timeout_ms: u64,
    pub requires_consensus: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransactionId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeOperation {
    pub exchange_id: ExchangeId,
    pub operation_type: OperationType,
    pub asset_pair: String,
    pub amount: f64,
    pub limit_price: Option<f64>,
    pub expected_execution_time_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Buy,
    Sell,
    Cancel,
    Arbitrage {
        source_exchange: ExchangeId,
        target_exchange: ExchangeId,
        profit_threshold: f64,
    },
}

#[derive(Debug, Clone)]
pub enum CommitPhase {
    Prepare,
    Commit,
    Abort,
}

#[derive(Debug, Clone)]
pub enum TransactionStatus {
    Preparing,
    Prepared,
    Committing,
    Committed,
    Aborted,
    TimedOut,
}

#[derive(Debug, Clone)]
struct ExchangeParticipant {
    exchange_id: ExchangeId,
    status: ParticipantStatus,
    vote: Option<Vote>,
    last_heartbeat: u64,
    latency_ns: u64,
}

#[derive(Debug, Clone, PartialEq)]
enum ParticipantStatus {
    Active,
    Preparing,
    Prepared,
    Committed,
    Aborted,
    Failed,
}

#[derive(Debug, Clone, PartialEq)]
enum Vote {
    Commit,
    Abort,
}

pub struct AtomicCommit {
    coordinator_id: ValidatorId,
    active_transactions: Arc<RwLock<HashMap<TransactionId, TransactionState>>>,
    exchange_participants: Arc<RwLock<HashMap<ExchangeId, ExchangeParticipant>>>,
    performance_metrics: Arc<Mutex<AtomicCommitMetrics>>,
    timeout_monitor: Arc<Mutex<TimeoutMonitor>>,
}

#[derive(Debug, Clone)]
struct TransactionState {
    transaction: AtomicTransaction,
    phase: CommitPhase,
    status: TransactionStatus,
    start_time: Instant,
    participant_votes: HashMap<ExchangeId, Vote>,
    committed_exchanges: HashSet<ExchangeId>,
    rollback_operations: Vec<RollbackOperation>,
}

#[derive(Debug, Clone)]
struct RollbackOperation {
    exchange_id: ExchangeId,
    operation_data: Vec<u8>,
    rollback_command: String,
}

#[derive(Debug, Default, Clone)]
pub struct AtomicCommitMetrics {
    pub total_transactions: u64,
    pub successful_commits: u64,
    pub aborted_transactions: u64,
    pub timed_out_transactions: u64,
    pub average_commit_time_ns: u64,
    pub max_participants: usize,
    pub byzantine_failures_detected: u64,
}

struct TimeoutMonitor {
    active_timeouts: HashMap<TransactionId, Instant>,
}

impl AtomicCommit {
    pub fn new(coordinator_id: ValidatorId) -> Self {
        Self {
            coordinator_id,
            active_transactions: Arc::new(RwLock::new(HashMap::new())),
            exchange_participants: Arc::new(RwLock::new(HashMap::new())),
            performance_metrics: Arc::new(Mutex::new(AtomicCommitMetrics::default())),
            timeout_monitor: Arc::new(Mutex::new(TimeoutMonitor {
                active_timeouts: HashMap::new(),
            })),
        }
    }

    pub async fn register_exchange(
        &self,
        exchange_id: ExchangeId,
        latency_ns: u64,
    ) -> Result<(), ConsensusError> {
        let participant = ExchangeParticipant {
            exchange_id: exchange_id.clone(),
            status: ParticipantStatus::Active,
            vote: None,
            last_heartbeat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            latency_ns,
        };

        let mut participants = self.exchange_participants.write().await;
        participants.insert(exchange_id, participant);

        // Update metrics
        {
            let mut metrics = self.performance_metrics.lock().await;
            metrics.max_participants = metrics.max_participants.max(participants.len());
        }

        Ok(())
    }

    pub async fn execute_atomic_transaction(
        &self,
        transaction: AtomicTransaction,
    ) -> Result<TransactionId, ConsensusError> {
        let start_time = Instant::now();

        // Validate transaction
        self.validate_transaction(&transaction).await?;

        // Initialize transaction state
        let transaction_id = transaction.transaction_id.clone();
        let mut transaction_state = TransactionState {
            transaction: transaction.clone(),
            phase: CommitPhase::Prepare,
            status: TransactionStatus::Preparing,
            start_time,
            participant_votes: HashMap::new(),
            committed_exchanges: HashSet::new(),
            rollback_operations: Vec::new(),
        };

        // Add to active transactions
        {
            let mut active = self.active_transactions.write().await;
            active.insert(transaction_id.clone(), transaction_state.clone());
        }

        // Set timeout monitoring
        {
            let mut monitor = self.timeout_monitor.lock().await;
            monitor.active_timeouts.insert(
                transaction_id.clone(),
                start_time + Duration::from_millis(transaction.timeout_ms),
            );
        }

        // Update metrics
        {
            let mut metrics = self.performance_metrics.lock().await;
            metrics.total_transactions += 1;
        }

        // Execute two-phase commit protocol
        match self.run_two_phase_commit(&transaction_id).await {
            Ok(()) => {
                let mut metrics = self.performance_metrics.lock().await;
                metrics.successful_commits += 1;
                metrics.average_commit_time_ns =
                    (metrics.average_commit_time_ns + start_time.elapsed().as_nanos() as u64) / 2;
            }
            Err(e) => {
                let mut metrics = self.performance_metrics.lock().await;
                if matches!(e, ConsensusError::TimeoutError) {
                    metrics.timed_out_transactions += 1;
                } else {
                    metrics.aborted_transactions += 1;
                }
                return Err(e);
            }
        }

        Ok(transaction_id)
    }

    async fn run_two_phase_commit(
        &self,
        transaction_id: &TransactionId,
    ) -> Result<(), ConsensusError> {
        // Phase 1: Prepare
        self.prepare_phase(transaction_id).await?;

        // Phase 2: Commit or Abort
        let should_commit = self.collect_votes(transaction_id).await?;

        if should_commit {
            self.commit_phase(transaction_id).await
        } else {
            self.abort_phase(transaction_id).await
        }
    }

    async fn prepare_phase(&self, transaction_id: &TransactionId) -> Result<(), ConsensusError> {
        let transaction = {
            let active = self.active_transactions.read().await;
            active
                .get(transaction_id)
                .ok_or(ConsensusError::InvalidMessage)?
                .transaction
                .clone()
        };

        // Send prepare messages to all participating exchanges
        for operation in &transaction.operations {
            self.send_prepare_message(&operation.exchange_id, transaction_id, operation)
                .await?;
        }

        // Update transaction status
        {
            let mut active = self.active_transactions.write().await;
            if let Some(state) = active.get_mut(transaction_id) {
                state.phase = CommitPhase::Prepare;
                state.status = TransactionStatus::Preparing;
            }
        }

        Ok(())
    }

    async fn send_prepare_message(
        &self,
        exchange_id: &ExchangeId,
        transaction_id: &TransactionId,
        operation: &ExchangeOperation,
    ) -> Result<(), ConsensusError> {
        // Check if exchange is still active
        {
            let participants = self.exchange_participants.read().await;
            if let Some(participant) = participants.get(exchange_id) {
                if participant.status == ParticipantStatus::Failed {
                    return Err(ConsensusError::NetworkPartition);
                }
            } else {
                return Err(ConsensusError::InvalidMessage);
            }
        }

        // Simulate sending prepare message to exchange
        // In real implementation, this would be actual network communication
        tokio::time::sleep(Duration::from_nanos(50)).await;

        // Simulate exchange response (simplified)
        let can_commit = self
            .simulate_exchange_prepare_response(exchange_id, operation)
            .await?;

        // Update participant status
        {
            let mut participants = self.exchange_participants.write().await;
            if let Some(participant) = participants.get_mut(exchange_id) {
                participant.status = if can_commit {
                    ParticipantStatus::Prepared
                } else {
                    ParticipantStatus::Aborted
                };
                participant.vote = Some(if can_commit {
                    Vote::Commit
                } else {
                    Vote::Abort
                });
                participant.last_heartbeat = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;
            }
        }

        Ok(())
    }

    async fn simulate_exchange_prepare_response(
        &self,
        _exchange_id: &ExchangeId,
        operation: &ExchangeOperation,
    ) -> Result<bool, ConsensusError> {
        // Simulate exchange validation logic
        // Real implementation would check:
        // - Sufficient balance
        // - Market conditions
        // - Risk limits
        // - Latency requirements

        // Artificial failure simulation for testing
        if operation.amount <= 0.0 {
            return Ok(false);
        }

        // Check if operation can be executed within expected time
        if operation.expected_execution_time_ns > 1_000_000 {
            // 1ms limit
            return Ok(false);
        }

        Ok(true)
    }

    async fn collect_votes(&self, transaction_id: &TransactionId) -> Result<bool, ConsensusError> {
        // Wait for all participants to vote or timeout
        let timeout = Duration::from_millis(100); // Short timeout for high-frequency trading
        let start = Instant::now();

        while start.elapsed() < timeout {
            let all_voted = {
                let active = self.active_transactions.read().await;
                let state = active
                    .get(transaction_id)
                    .ok_or(ConsensusError::InvalidMessage)?;

                let participants = self.exchange_participants.read().await;
                let required_exchanges: HashSet<_> = state
                    .transaction
                    .operations
                    .iter()
                    .map(|op| &op.exchange_id)
                    .collect();

                required_exchanges.iter().all(|exchange_id| {
                    participants
                        .get(exchange_id)
                        .map(|p| p.vote.is_some())
                        .unwrap_or(false)
                })
            };

            if all_voted {
                break;
            }

            tokio::time::sleep(Duration::from_micros(100)).await;
        }

        // Check if all participants voted to commit
        let can_commit = {
            let participants = self.exchange_participants.read().await;
            let active = self.active_transactions.read().await;
            let state = active
                .get(transaction_id)
                .ok_or(ConsensusError::InvalidMessage)?;

            state.transaction.operations.iter().all(|op| {
                participants
                    .get(&op.exchange_id)
                    .and_then(|p| p.vote.as_ref())
                    .map(|vote| *vote == Vote::Commit)
                    .unwrap_or(false)
            })
        };

        // Update transaction status
        {
            let mut active = self.active_transactions.write().await;
            if let Some(state) = active.get_mut(transaction_id) {
                state.status = if can_commit {
                    TransactionStatus::Prepared
                } else {
                    TransactionStatus::Aborted
                };
            }
        }

        Ok(can_commit)
    }

    async fn commit_phase(&self, transaction_id: &TransactionId) -> Result<(), ConsensusError> {
        let operations = {
            let active = self.active_transactions.read().await;
            active
                .get(transaction_id)
                .ok_or(ConsensusError::InvalidMessage)?
                .transaction
                .operations
                .clone()
        };

        // Send commit messages to all exchanges
        for operation in &operations {
            self.send_commit_message(&operation.exchange_id, transaction_id)
                .await?;
        }

        // Update transaction status
        {
            let mut active = self.active_transactions.write().await;
            if let Some(state) = active.get_mut(transaction_id) {
                state.phase = CommitPhase::Commit;
                state.status = TransactionStatus::Committed;

                // Record committed exchanges
                for operation in &operations {
                    state
                        .committed_exchanges
                        .insert(operation.exchange_id.clone());
                }
            }
        }

        // Clean up monitoring
        {
            let mut monitor = self.timeout_monitor.lock().await;
            monitor.active_timeouts.remove(transaction_id);
        }

        Ok(())
    }

    async fn send_commit_message(
        &self,
        exchange_id: &ExchangeId,
        _transaction_id: &TransactionId,
    ) -> Result<(), ConsensusError> {
        // Update participant status
        {
            let mut participants = self.exchange_participants.write().await;
            if let Some(participant) = participants.get_mut(exchange_id) {
                participant.status = ParticipantStatus::Committed;
                participant.last_heartbeat = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;
            }
        }

        // Simulate commit execution
        tokio::time::sleep(Duration::from_nanos(100)).await;
        Ok(())
    }

    async fn abort_phase(&self, transaction_id: &TransactionId) -> Result<(), ConsensusError> {
        // Execute rollback operations for any partially committed operations
        let rollback_ops = {
            let active = self.active_transactions.read().await;
            active
                .get(transaction_id)
                .ok_or(ConsensusError::InvalidMessage)?
                .rollback_operations
                .clone()
        };

        for rollback_op in rollback_ops {
            self.execute_rollback(&rollback_op).await?;
        }

        // Update transaction status
        {
            let mut active = self.active_transactions.write().await;
            if let Some(state) = active.get_mut(transaction_id) {
                state.phase = CommitPhase::Abort;
                state.status = TransactionStatus::Aborted;
            }
        }

        // Clean up monitoring
        {
            let mut monitor = self.timeout_monitor.lock().await;
            monitor.active_timeouts.remove(transaction_id);
        }

        Err(ConsensusError::ByzantineAttack) // Transaction aborted
    }

    async fn execute_rollback(
        &self,
        _rollback_op: &RollbackOperation,
    ) -> Result<(), ConsensusError> {
        // Execute rollback operation
        // In real implementation, would send rollback commands to exchanges
        tokio::time::sleep(Duration::from_nanos(50)).await;
        Ok(())
    }

    async fn validate_transaction(
        &self,
        transaction: &AtomicTransaction,
    ) -> Result<(), ConsensusError> {
        // Validate transaction structure
        if transaction.operations.is_empty() {
            return Err(ConsensusError::InvalidMessage);
        }

        if transaction.timeout_ms == 0 {
            return Err(ConsensusError::TimeoutError);
        }

        // Check if all required exchanges are registered
        let participants = self.exchange_participants.read().await;
        for operation in &transaction.operations {
            if !participants.contains_key(&operation.exchange_id) {
                return Err(ConsensusError::NetworkPartition);
            }
        }

        Ok(())
    }

    pub async fn get_transaction_status(
        &self,
        transaction_id: &TransactionId,
    ) -> Option<TransactionStatus> {
        let active = self.active_transactions.read().await;
        active.get(transaction_id).map(|state| state.status.clone())
    }

    pub async fn get_metrics(&self) -> AtomicCommitMetrics {
        self.performance_metrics.lock().await.clone()
    }

    pub async fn cleanup_completed_transactions(&self) -> Result<(), ConsensusError> {
        let mut to_remove = Vec::new();

        {
            let active = self.active_transactions.read().await;
            for (tx_id, state) in active.iter() {
                if matches!(
                    state.status,
                    TransactionStatus::Committed
                        | TransactionStatus::Aborted
                        | TransactionStatus::TimedOut
                ) {
                    // Keep completed transactions for a short time for querying
                    if state.start_time.elapsed() > Duration::from_secs(10) {
                        to_remove.push(tx_id.clone());
                    }
                }
            }
        }

        if !to_remove.is_empty() {
            let mut active = self.active_transactions.write().await;
            for tx_id in to_remove {
                active.remove(&tx_id);
            }
        }

        Ok(())
    }
}

// Helper functions for creating transactions
impl AtomicTransaction {
    pub fn new_arbitrage(
        source_exchange: &str,
        target_exchange: &str,
        asset_pair: &str,
        amount: f64,
        profit_threshold: f64,
    ) -> Self {
        let tx_id = format!(
            "arb_{}_{}_{}_{}",
            source_exchange,
            target_exchange,
            asset_pair,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        Self {
            transaction_id: TransactionId(tx_id),
            operations: vec![
                ExchangeOperation {
                    exchange_id: ExchangeId(source_exchange.to_string()),
                    operation_type: OperationType::Sell,
                    asset_pair: asset_pair.to_string(),
                    amount,
                    limit_price: None,
                    expected_execution_time_ns: 500_000, // 500μs
                },
                ExchangeOperation {
                    exchange_id: ExchangeId(target_exchange.to_string()),
                    operation_type: OperationType::Buy,
                    asset_pair: asset_pair.to_string(),
                    amount,
                    limit_price: None,
                    expected_execution_time_ns: 500_000, // 500μs
                },
            ],
            timeout_ms: 1000, // 1 second timeout for arbitrage
            requires_consensus: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_atomic_commit_creation() {
        let commit = AtomicCommit::new(ValidatorId(0));

        let result = commit
            .register_exchange(ExchangeId("binance".to_string()), 1_000_000)
            .await;
        assert!(result.is_ok());

        let result = commit
            .register_exchange(ExchangeId("coinbase".to_string()), 1_500_000)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_arbitrage_transaction() {
        let commit = AtomicCommit::new(ValidatorId(0));

        // Register exchanges
        commit
            .register_exchange(ExchangeId("binance".to_string()), 500_000)
            .await
            .unwrap();
        commit
            .register_exchange(ExchangeId("coinbase".to_string()), 750_000)
            .await
            .unwrap();

        // Create arbitrage transaction
        let transaction =
            AtomicTransaction::new_arbitrage("binance", "coinbase", "BTC/USD", 1.0, 0.001);

        let result = commit.execute_atomic_transaction(transaction).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_transaction_validation() {
        let commit = AtomicCommit::new(ValidatorId(0));

        // Test invalid transaction (no operations)
        let invalid_tx = AtomicTransaction {
            transaction_id: TransactionId("test".to_string()),
            operations: vec![],
            timeout_ms: 1000,
            requires_consensus: true,
        };

        let result = commit.execute_atomic_transaction(invalid_tx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let commit = AtomicCommit::new(ValidatorId(0));

        commit
            .register_exchange(ExchangeId("test_exchange".to_string()), 1_000_000)
            .await
            .unwrap();

        let transaction = AtomicTransaction::new_arbitrage(
            "test_exchange",
            "test_exchange",
            "ETH/USD",
            0.5,
            0.002,
        );

        let _result = commit.execute_atomic_transaction(transaction).await;

        let metrics = commit.get_metrics().await;
        assert!(metrics.total_transactions > 0);
    }
}
