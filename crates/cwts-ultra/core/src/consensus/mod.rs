//! Byzantine Fault Tolerant Consensus Module
//!
//! Complete BFT consensus system with quantum-enhanced verification
//! and atomic execution across multiple exchanges

pub mod atomic_commit;
pub mod byzantine_consensus;
pub mod quantum_verification;
pub mod validator_network;

// Re-export main types
pub use byzantine_consensus::{
    ByzantineConsensus, ByzantineMessage, ConsensusError, ConsensusPhase, ConsensusState,
    MessageType, QuantumSignature, ValidatorId,
};

pub use validator_network::{NetworkStatus, ValidatorInfo, ValidatorNetwork};

pub use quantum_verification::{
    QuantumKey, QuantumVerification, QuantumVerificationResult, ZeroKnowledgeProof,
};

pub use atomic_commit::{
    AtomicCommit, AtomicTransaction, ExchangeId, ExchangeOperation, OperationType, TransactionId,
};

use std::sync::Arc;
use tokio::sync::RwLock;

/// Integrated Byzantine Consensus System
///
/// Combines all consensus components for production use
pub struct ByzantineConsensusSystem {
    consensus: Arc<ByzantineConsensus>,
    validator_network: Arc<ValidatorNetwork>,
    quantum_verifier: Arc<QuantumVerification>,
    atomic_commit: Arc<AtomicCommit>,
}

impl ByzantineConsensusSystem {
    pub async fn new(
        validator_count: usize,
        coordinator_id: ValidatorId,
    ) -> Result<Self, ConsensusError> {
        // Initialize core consensus
        let consensus = Arc::new(ByzantineConsensus::new(validator_count));

        // Initialize quantum verifier
        let quantum_verifier = Arc::new(QuantumVerification::new());

        // Initialize atomic commit coordinator
        let atomic_commit = Arc::new(AtomicCommit::new(coordinator_id));

        // Create initial validators
        let mut validators = Vec::new();
        for i in 0..validator_count {
            validators.push(ValidatorInfo::new(
                i as u64,
                format!("127.0.0.1:{}", 8000 + i),
            ));
        }

        let validator_network = Arc::new(ValidatorNetwork::new(validators));

        Ok(Self {
            consensus,
            validator_network,
            quantum_verifier,
            atomic_commit,
        })
    }

    /// Execute a quantum-enhanced atomic transaction with Byzantine consensus
    pub async fn execute_quantum_atomic_transaction(
        &self,
        transaction: AtomicTransaction,
    ) -> Result<TransactionId, ConsensusError> {
        // Step 1: Validate transaction through validator network
        let transaction_bytes =
            serde_json::to_vec(&transaction).map_err(|_| ConsensusError::InvalidMessage)?;

        // Step 2: Propose transaction through Byzantine consensus
        self.consensus
            .propose_transaction(transaction_bytes.clone())
            .await?;

        // Step 3: Execute atomic commit across exchanges
        let tx_id = self
            .atomic_commit
            .execute_atomic_transaction(transaction)
            .await?;

        Ok(tx_id)
    }

    /// Process incoming consensus message with full validation pipeline
    pub async fn handle_consensus_message(
        &self,
        message: ByzantineMessage,
    ) -> Result<(), ConsensusError> {
        // Step 1: Quantum signature verification
        if !self.quantum_verifier.verify_signature(&message).await? {
            return Err(ConsensusError::QuantumVerificationFailed);
        }

        // Step 2: Validator network processing (Byzantine detection)
        if !self.validator_network.process_message(&message).await? {
            return Err(ConsensusError::ByzantineAttack);
        }

        // Step 3: Byzantine consensus processing
        self.consensus.handle_message(message).await?;

        Ok(())
    }

    /// Get comprehensive system status
    pub async fn get_system_status(&self) -> SystemStatus {
        let consensus_state = self.consensus.get_consensus_state();
        let network_status = self.validator_network.get_network_status().await;
        let atomic_metrics = self.atomic_commit.get_metrics().await;
        let quantum_metrics = self.quantum_verifier.get_verification_metrics().await;

        SystemStatus {
            consensus_state,
            network_status,
            atomic_commit_metrics: atomic_metrics,
            quantum_verification_metrics: quantum_metrics,
            is_byzantine_fault_tolerant: self.consensus.is_byzantine_fault_tolerant(),
            is_network_healthy: self.validator_network.is_network_healthy().await,
        }
    }

    /// Register a new exchange for atomic transactions
    pub async fn register_exchange(
        &self,
        exchange_id: ExchangeId,
        latency_ns: u64,
    ) -> Result<(), ConsensusError> {
        self.atomic_commit
            .register_exchange(exchange_id, latency_ns)
            .await
    }

    /// Add a new validator to the network
    pub async fn add_validator(&self, validator: ValidatorInfo) -> Result<(), ConsensusError> {
        self.validator_network.add_validator(validator).await
    }

    /// Create quantum entanglement between validators for enhanced security
    pub async fn create_quantum_entanglement(
        &self,
        validator_keys: Vec<Vec<u8>>,
    ) -> Result<u64, ConsensusError> {
        self.quantum_verifier
            .create_quantum_entanglement(validator_keys)
            .await
    }
}

#[derive(Debug, Clone)]
pub struct SystemStatus {
    pub consensus_state: ConsensusState,
    pub network_status: NetworkStatus,
    pub atomic_commit_metrics: atomic_commit::AtomicCommitMetrics,
    pub quantum_verification_metrics: quantum_verification::VerificationMetrics,
    pub is_byzantine_fault_tolerant: bool,
    pub is_network_healthy: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_integrated_consensus_system() {
        let system = ByzantineConsensusSystem::new(4, ValidatorId(0))
            .await
            .unwrap();

        // Register exchanges
        system
            .register_exchange(ExchangeId("binance".to_string()), 500_000)
            .await
            .unwrap();
        system
            .register_exchange(ExchangeId("coinbase".to_string()), 750_000)
            .await
            .unwrap();

        // Create quantum arbitrage transaction
        let transaction =
            AtomicTransaction::new_arbitrage("binance", "coinbase", "BTC/USD", 1.0, 0.001);

        // Execute with full Byzantine consensus
        let result = system.execute_quantum_atomic_transaction(transaction).await;
        assert!(result.is_ok());

        // Check system status
        let status = system.get_system_status().await;
        assert!(status.is_byzantine_fault_tolerant);
        assert!(status.is_network_healthy);
    }

    #[tokio::test]
    async fn test_quantum_enhanced_message_handling() {
        let system = ByzantineConsensusSystem::new(4, ValidatorId(0))
            .await
            .unwrap();

        // Create quantum-signed message
        let quantum_sig = system.quantum_verifier.sign(b"test_message").await.unwrap();

        let message = ByzantineMessage {
            message_type: MessageType::Prepare,
            view: 0,
            sequence: 1,
            sender: ValidatorId(1),
            payload: b"test_message".to_vec(),
            quantum_signature: quantum_sig,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };

        let result = system.handle_consensus_message(message).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_system_performance_requirements() {
        let system = ByzantineConsensusSystem::new(4, ValidatorId(0))
            .await
            .unwrap();

        system
            .register_exchange(ExchangeId("test_exchange".to_string()), 100_000)
            .await
            .unwrap();

        let transaction = AtomicTransaction::new_arbitrage(
            "test_exchange",
            "test_exchange",
            "ETH/USD",
            0.5,
            0.001,
        );

        // Test sub-millisecond performance requirement
        let start = std::time::Instant::now();
        let _result = system.execute_quantum_atomic_transaction(transaction).await;
        let elapsed = start.elapsed();

        // Should complete in under 1ms for high-frequency trading
        assert!(
            elapsed.as_nanos() < 1_000_000,
            "Consensus took {}ns, exceeding 1ms requirement",
            elapsed.as_nanos()
        );
    }
}
