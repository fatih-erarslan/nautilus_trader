//! QUIC-based swarm coordinator server
//!
//! High-performance coordinator that manages agent connections and task distribution
//! using QUIC protocol with sub-millisecond latency.

use crate::error::{Result, SwarmError};
use crate::types::*;
use crate::tls::{configure_server, generate_self_signed_cert};
use crate::reasoningbank::ReasoningBankClient;

use dashmap::DashMap;
use parking_lot::RwLock;
use quinn::{Connection, Endpoint, RecvStream, SendStream, ServerConfig};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// QUIC swarm coordinator
pub struct QuicSwarmCoordinator {
    /// QUIC endpoint
    endpoint: Endpoint,
    /// Active agent connections
    agents: Arc<DashMap<String, AgentConnection>>,
    /// ReasoningBank client
    reasoning_bank: Arc<ReasoningBankClient>,
    /// Task queue sender
    task_sender: mpsc::UnboundedSender<PendingTask>,
    /// Task queue receiver
    task_receiver: Arc<RwLock<mpsc::UnboundedReceiver<PendingTask>>>,
    /// Coordinator ID
    coordinator_id: String,
    /// Start time
    start_time: Instant,
    /// Configuration
    config: CoordinatorConfig,
}

/// Agent connection state
struct AgentConnection {
    /// Agent ID
    agent_id: String,
    /// Agent type
    agent_type: AgentType,
    /// QUIC connection
    connection: Connection,
    /// Active streams
    streams: Arc<DashMap<u64, StreamInfo>>,
    /// Statistics
    stats: Arc<RwLock<AgentStats>>,
    /// Session token
    session_token: String,
    /// Handshake info
    handshake: AgentHandshake,
}

/// Stream information
struct StreamInfo {
    /// Stream purpose
    purpose: StreamPurpose,
    /// Creation time
    created_at: Instant,
    /// Last activity
    last_activity: Instant,
    /// Messages sent
    messages_sent: u64,
    /// Messages received
    messages_received: u64,
}

/// Pending task
struct PendingTask {
    /// Task definition
    task: AgentTask,
    /// Target agent ID (if specific)
    target_agent: Option<String>,
}

/// Coordinator configuration
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Maximum concurrent bidirectional streams per connection
    pub max_concurrent_bidi_streams: u32,
    /// Maximum idle timeout
    pub max_idle_timeout: Duration,
    /// Keep-alive interval
    pub keep_alive_interval: Duration,
    /// Task processing timeout
    pub task_timeout: Duration,
    /// Enable metrics collection
    pub enable_metrics: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_bidi_streams: 1000,
            max_idle_timeout: Duration::from_secs(300),
            keep_alive_interval: Duration::from_secs(10),
            task_timeout: Duration::from_secs(300),
            enable_metrics: true,
        }
    }
}

impl QuicSwarmCoordinator {
    /// Create new QUIC coordinator
    pub async fn new(bind_addr: SocketAddr, config: CoordinatorConfig) -> Result<Self> {
        info!("Initializing QUIC coordinator on {}", bind_addr);

        // Generate self-signed certificate for development
        let (certs, key) = generate_self_signed_cert()?;

        // Configure server
        let server_crypto = configure_server(certs, key)?;

        // Create QUIC server config
        let mut server_config = ServerConfig::with_crypto(Arc::new(server_crypto));
        let mut transport = quinn::TransportConfig::default();

        transport.max_concurrent_bidi_streams(config.max_concurrent_bidi_streams.into());
        transport.max_idle_timeout(Some(config.max_idle_timeout.try_into().unwrap()));
        transport.keep_alive_interval(Some(config.keep_alive_interval));

        server_config.transport_config(Arc::new(transport));

        // Create endpoint
        let endpoint = Endpoint::server(server_config, bind_addr)?;
        info!("QUIC coordinator listening on {}", bind_addr);

        // Create task queue
        let (task_sender, task_receiver) = mpsc::unbounded_channel();

        Ok(Self {
            endpoint,
            agents: Arc::new(DashMap::new()),
            reasoning_bank: Arc::new(ReasoningBankClient::new()),
            task_sender,
            task_receiver: Arc::new(RwLock::new(task_receiver)),
            coordinator_id: Uuid::new_v4().to_string(),
            start_time: Instant::now(),
            config,
        })
    }

    /// Run the coordinator (accept connections)
    pub async fn run(self: Arc<Self>) -> Result<()> {
        info!("Starting QUIC coordinator");

        // Spawn task distributor
        let coordinator = self.clone();
        tokio::spawn(async move {
            if let Err(e) = coordinator.distribute_tasks().await {
                error!("Task distributor error: {}", e);
            }
        });

        // Accept connections
        loop {
            let incoming = self.endpoint.accept().await;

            match incoming {
                Some(conn) => {
                    let coordinator = self.clone();
                    tokio::spawn(async move {
                        match conn.await {
                            Ok(connection) => {
                                let remote = connection.remote_address();
                                info!("New connection from: {}", remote);

                                if let Err(e) = coordinator.handle_agent(connection).await {
                                    error!("Agent handler error: {}", e);
                                }
                            }
                            Err(e) => {
                                error!("Connection failed: {}", e);
                            }
                        }
                    });
                }
                None => {
                    warn!("Endpoint closed");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle individual agent connection
    async fn handle_agent(&self, connection: Connection) -> Result<()> {
        let remote = connection.remote_address();
        debug!("Handling agent from: {}", remote);

        // Wait for handshake stream
        let (mut send, mut recv) = connection
            .accept_bi()
            .await
            .map_err(|_| SwarmError::StreamClosed)?;

        // Receive handshake
        let handshake = self.receive_message::<AgentHandshake>(&mut recv).await?;

        info!(
            "Agent handshake: {} (type: {:?})",
            handshake.agent_id, handshake.agent_type
        );

        // Generate session token
        let session_token = Uuid::new_v4().to_string();

        // Send acknowledgment
        let ack = AgentAck {
            coordinator_id: self.coordinator_id.clone(),
            assigned_streams: vec![
                StreamAssignment {
                    stream_id: 0,
                    purpose: StreamPurpose::PatternMatching,
                    priority: 8,
                },
                StreamAssignment {
                    stream_id: 1,
                    purpose: StreamPurpose::ReasoningExchange,
                    priority: 7,
                },
                StreamAssignment {
                    stream_id: 2,
                    purpose: StreamPurpose::TaskAssignment,
                    priority: 9,
                },
            ],
            session_token: session_token.clone(),
            config: AgentConfig::default(),
        };

        self.send_message(&mut send, &ack).await?;
        send.finish()?;

        // Register agent
        let agent_conn = AgentConnection {
            agent_id: handshake.agent_id.clone(),
            agent_type: handshake.agent_type.clone(),
            connection: connection.clone(),
            streams: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(AgentStats {
                connected_at: Some(chrono::Utc::now()),
                ..Default::default()
            })),
            session_token,
            handshake: handshake.clone(),
        };

        self.agents.insert(handshake.agent_id.clone(), agent_conn);

        info!("Agent registered: {} (total: {})", handshake.agent_id, self.agents.len());

        // Handle agent streams
        self.handle_agent_streams(connection, handshake).await
    }

    /// Handle bidirectional streams for agent
    async fn handle_agent_streams(
        &self,
        connection: Connection,
        handshake: AgentHandshake,
    ) -> Result<()> {
        let agent_id = handshake.agent_id.clone();

        loop {
            // Accept incoming stream from agent
            match connection.accept_bi().await {
                Ok((send, recv)) => {
                    let coordinator = Arc::new(self.clone());
                    let agent_id = agent_id.clone();

                    tokio::spawn(async move {
                        if let Err(e) = coordinator.handle_stream(agent_id, send, recv).await {
                            error!("Stream handler error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    warn!("Connection closed for agent {}: {}", agent_id, e);
                    self.agents.remove(&agent_id);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle individual bidirectional stream
    async fn handle_stream(
        &self,
        agent_id: String,
        mut send: SendStream,
        mut recv: RecvStream,
    ) -> Result<()> {
        debug!("Handling stream for agent: {}", agent_id);

        loop {
            // Receive message
            let message = match self.receive_message::<AgentMessage>(&mut recv).await {
                Ok(msg) => msg,
                Err(SwarmError::StreamClosed) => {
                    debug!("Stream closed for agent: {}", agent_id);
                    break;
                }
                Err(e) => {
                    error!("Error receiving message: {}", e);
                    break;
                }
            };

            // Update stats
            if let Some(agent) = self.agents.get(&agent_id) {
                let mut stats = agent.stats.write();
                stats.messages_received += 1;
                stats.last_activity = Some(chrono::Utc::now());
            }

            // Process message
            let response = self.process_agent_message(&agent_id, message).await?;

            // Send acknowledgment
            self.send_message(&mut send, &response).await?;

            // Update stats
            if let Some(agent) = self.agents.get(&agent_id) {
                let mut stats = agent.stats.write();
                stats.messages_sent += 1;
            }
        }

        Ok(())
    }

    /// Process agent message
    async fn process_agent_message(
        &self,
        agent_id: &str,
        message: AgentMessage,
    ) -> Result<MessageAck> {
        match message {
            AgentMessage::PatternMatchResult(result) => {
                self.handle_pattern_result(agent_id, result).await
            }
            AgentMessage::StrategyCorrelation(corr) => {
                self.handle_correlation(agent_id, corr).await
            }
            AgentMessage::ReasoningExperience(exp) => {
                self.handle_reasoning_experience(agent_id, exp).await
            }
            AgentMessage::NeuralGradients(grads) => {
                self.handle_gradients(agent_id, grads).await
            }
            AgentMessage::Heartbeat(hb) => {
                self.handle_heartbeat(agent_id, hb).await
            }
            AgentMessage::TaskComplete(tc) => {
                self.handle_task_completion(agent_id, tc).await
            }
            AgentMessage::Error(err) => {
                self.handle_error_report(agent_id, err).await
            }
        }
    }

    /// Handle pattern matching result
    async fn handle_pattern_result(
        &self,
        agent_id: &str,
        result: PatternMatchResult,
    ) -> Result<MessageAck> {
        debug!("Pattern result from {}: similarity={}", agent_id, result.similarity);

        // Record experience in ReasoningBank if outcome available
        if let Some(actual) = result.actual_outcome {
            let experience = ReasoningExperience {
                message_id: result.message_id.clone(),
                agent_id: agent_id.to_string(),
                action: format!("pattern_match:{}", result.pattern_type),
                outcome: OutcomeMetrics {
                    value: actual,
                    confidence: result.confidence,
                    latency_ms: result.compute_time_ms,
                    additional: HashMap::new(),
                },
                context: result.market_context.clone(),
                timestamp: result.timestamp,
            };

            self.reasoning_bank.record_experience(experience).await?;

            // Get verdict and potentially adapt
            if result.similarity > 0.85 {
                let verdict = self.reasoning_bank
                    .judge_experience(agent_id, result.expected_outcome, actual)
                    .await?;

                if verdict.should_adapt {
                    info!(
                        "ReasoningBank suggests adaptation for {}: score={}",
                        agent_id, verdict.score
                    );
                    // TODO: Send adaptation command to agent
                }
            }
        }

        Ok(MessageAck {
            message_id: result.message_id,
            status: AckStatus::Processed,
            response: None,
        })
    }

    /// Handle strategy correlation
    async fn handle_correlation(
        &self,
        agent_id: &str,
        corr: StrategyCorrelation,
    ) -> Result<MessageAck> {
        debug!("Strategy correlation from {}: {} strategies", agent_id, corr.strategy_ids.len());

        Ok(MessageAck {
            message_id: corr.message_id,
            status: AckStatus::Processed,
            response: None,
        })
    }

    /// Handle reasoning experience
    async fn handle_reasoning_experience(
        &self,
        agent_id: &str,
        exp: ReasoningExperience,
    ) -> Result<MessageAck> {
        debug!("Reasoning experience from {}: {}", agent_id, exp.action);

        self.reasoning_bank.record_experience(exp.clone()).await?;

        Ok(MessageAck {
            message_id: exp.message_id,
            status: AckStatus::Processed,
            response: None,
        })
    }

    /// Handle neural gradients
    async fn handle_gradients(
        &self,
        agent_id: &str,
        grads: NeuralGradients,
    ) -> Result<MessageAck> {
        debug!("Neural gradients from {}: {} values", agent_id, grads.gradients.len());

        Ok(MessageAck {
            message_id: grads.message_id,
            status: AckStatus::Processed,
            response: None,
        })
    }

    /// Handle heartbeat
    async fn handle_heartbeat(
        &self,
        agent_id: &str,
        hb: HeartbeatMessage,
    ) -> Result<MessageAck> {
        debug!("Heartbeat from {}: load={}, tasks={}", agent_id, hb.load, hb.active_tasks);

        Ok(MessageAck {
            message_id: Uuid::new_v4().to_string(),
            status: AckStatus::Received,
            response: None,
        })
    }

    /// Handle task completion
    async fn handle_task_completion(
        &self,
        agent_id: &str,
        tc: TaskCompletion,
    ) -> Result<MessageAck> {
        info!(
            "Task {} completed by {}: success={}, duration={}ms",
            tc.task_id, agent_id, tc.success, tc.duration_ms
        );

        Ok(MessageAck {
            message_id: Uuid::new_v4().to_string(),
            status: AckStatus::Processed,
            response: None,
        })
    }

    /// Handle error report
    async fn handle_error_report(
        &self,
        agent_id: &str,
        err: ErrorReport,
    ) -> Result<MessageAck> {
        error!("Error from {}: {} - {}", agent_id, err.code, err.message);

        Ok(MessageAck {
            message_id: Uuid::new_v4().to_string(),
            status: AckStatus::Received,
            response: None,
        })
    }

    /// Distribute tasks to agents
    async fn distribute_tasks(&self) -> Result<()> {
        let mut receiver = self.task_receiver.write();

        while let Some(pending) = receiver.recv().await {
            debug!("Distributing task: {}", pending.task.task_id);

            // Find suitable agent
            let agent_id = if let Some(target) = pending.target_agent {
                target
            } else {
                // Find least loaded agent of appropriate type
                self.find_suitable_agent(&pending.task)?
            };

            // Send task to agent
            if let Some(agent) = self.agents.get(&agent_id) {
                match agent.connection.open_bi().await {
                    Ok((mut send, _recv)) => {
                        if let Err(e) = self.send_message(&mut send, &pending.task).await {
                            error!("Failed to send task: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Failed to open stream: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Find suitable agent for task
    fn find_suitable_agent(&self, _task: &AgentTask) -> Result<String> {
        // Simple round-robin for now
        // TODO: Implement load-based selection
        self.agents
            .iter()
            .next()
            .map(|entry| entry.key().clone())
            .ok_or_else(|| SwarmError::Other("No agents available".into()))
    }

    /// Send a message over a stream
    async fn send_message<T: serde::Serialize>(
        &self,
        stream: &mut SendStream,
        message: &T,
    ) -> Result<()> {
        let data = serde_json::to_vec(message)?;
        stream.write_all(&data).await?;
        Ok(())
    }

    /// Receive a message from a stream
    async fn receive_message<T: serde::de::DeserializeOwned>(
        &self,
        stream: &mut RecvStream,
    ) -> Result<T> {
        let mut buf = vec![0u8; 65536];
        let n = stream
            .read(&mut buf)
            .await?
            .ok_or(SwarmError::StreamClosed)?;

        let message = serde_json::from_slice(&buf[..n])?;
        Ok(message)
    }

    /// Get coordinator statistics
    pub fn stats(&self) -> CoordinatorStats {
        CoordinatorStats {
            uptime: self.start_time.elapsed(),
            active_agents: self.agents.len(),
            coordinator_id: self.coordinator_id.clone(),
        }
    }
}

impl Clone for QuicSwarmCoordinator {
    fn clone(&self) -> Self {
        Self {
            endpoint: self.endpoint.clone(),
            agents: self.agents.clone(),
            reasoning_bank: self.reasoning_bank.clone(),
            task_sender: self.task_sender.clone(),
            task_receiver: self.task_receiver.clone(),
            coordinator_id: self.coordinator_id.clone(),
            start_time: self.start_time,
            config: self.config.clone(),
        }
    }
}

/// Coordinator statistics
#[derive(Debug, Clone)]
pub struct CoordinatorStats {
    /// Uptime duration
    pub uptime: Duration,
    /// Number of active agents
    pub active_agents: usize,
    /// Coordinator ID
    pub coordinator_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let addr = "127.0.0.1:0".parse().unwrap();
        let coordinator = QuicSwarmCoordinator::new(addr, CoordinatorConfig::default())
            .await
            .unwrap();

        assert_eq!(coordinator.agents.len(), 0);
    }
}
