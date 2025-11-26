//! QUIC-based swarm agent client
//!
//! Agents connect to the coordinator and process tasks using QUIC protocol.

use crate::error::{Result, SwarmError};
use crate::types::*;
use crate::tls::configure_client_insecure;

use quinn::{Connection, Endpoint, RecvStream, SendStream};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// QUIC swarm agent
pub struct QuicSwarmAgent {
    /// Agent ID
    agent_id: String,
    /// Agent type
    agent_type: AgentType,
    /// QUIC connection
    connection: Option<Connection>,
    /// Session token
    session_token: Option<String>,
    /// Message sender
    message_sender: mpsc::UnboundedSender<AgentMessage>,
    /// Message receiver
    message_receiver: Arc<parking_lot::Mutex<mpsc::UnboundedReceiver<AgentMessage>>>,
    /// Configuration
    config: AgentConfig,
}

impl QuicSwarmAgent {
    /// Connect to coordinator
    pub async fn connect(
        agent_id: String,
        agent_type: AgentType,
        coordinator_addr: SocketAddr,
    ) -> Result<Self> {
        info!("Connecting agent {} to coordinator at {}", agent_id, coordinator_addr);

        // Configure client (insecure for self-signed certs in development)
        let client_config = configure_client_insecure()?;

        // Create endpoint
        let mut endpoint = Endpoint::client("[::]:0".parse()?)?;
        endpoint.set_default_client_config(quinn::ClientConfig::new(Arc::new(client_config)));

        // Connect to coordinator
        let connection = endpoint
            .connect(coordinator_addr, "localhost")?
            .await?;

        info!("Connected to coordinator");

        // Send handshake
        let (mut send, mut recv) = connection.open_bi().await?;

        let handshake = AgentHandshake {
            agent_id: agent_id.clone(),
            agent_type: agent_type.clone(),
            capabilities: vec![
                "pattern_matching".to_string(),
                "dtw".to_string(),
                "lcs".to_string(),
            ],
            version: "1.0.0".to_string(),
            metadata: std::collections::HashMap::new(),
        };

        let handshake_bytes = serde_json::to_vec(&handshake)?;
        send.write_all(&handshake_bytes).await?;
        send.finish()?;

        // Receive acknowledgment
        let mut buf = vec![0u8; 8192];
        let n = recv
            .read(&mut buf)
            .await?
            .ok_or(SwarmError::StreamClosed)?;

        let ack: AgentAck = serde_json::from_slice(&buf[..n])?;

        info!(
            "Agent registered with coordinator: {} (streams: {})",
            ack.coordinator_id,
            ack.assigned_streams.len()
        );

        // Create message channel
        let (message_sender, message_receiver) = mpsc::unbounded_channel();

        Ok(Self {
            agent_id,
            agent_type,
            connection: Some(connection),
            session_token: Some(ack.session_token),
            message_sender,
            message_receiver: Arc::new(parking_lot::Mutex::new(message_receiver)),
            config: ack.config,
        })
    }

    /// Run agent main loop
    pub async fn run(&mut self) -> Result<()> {
        let connection = self
            .connection
            .as_ref()
            .ok_or_else(|| SwarmError::Other("Not connected".into()))?;

        info!("Agent {} starting main loop", self.agent_id);

        loop {
            tokio::select! {
                // Accept incoming task streams
                stream = connection.accept_bi() => {
                    match stream {
                        Ok((send, recv)) => {
                            let agent_id = self.agent_id.clone();
                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_task_stream(agent_id, send, recv).await {
                                    error!("Task stream error: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            error!("Failed to accept stream: {}", e);
                            break;
                        }
                    }
                }

                // Send outgoing messages
                message = async {
                    self.message_receiver.lock().recv().await
                } => {
                    if let Some(msg) = message {
                        if let Err(e) = self.send_message(msg).await {
                            error!("Failed to send message: {}", e);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle task stream
    async fn handle_task_stream(
        agent_id: String,
        mut send: SendStream,
        mut recv: RecvStream,
    ) -> Result<()> {
        debug!("Handling task stream for agent: {}", agent_id);

        let mut buf = vec![0u8; 65536];
        let n = recv
            .read(&mut buf)
            .await?
            .ok_or(SwarmError::StreamClosed)?;

        let task: AgentTask = serde_json::from_slice(&buf[..n])?;

        info!("Received task: {} (type: {:?})", task.task_id, task.task_type);

        // Process task
        let start = Instant::now();
        let result = Self::process_task(task.clone()).await;
        let duration = start.elapsed().as_secs_f64() * 1000.0;

        // Send completion message
        let completion = TaskCompletion {
            task_id: task.task_id,
            agent_id,
            success: result.is_ok(),
            result: result.ok(),
            error: None,
            duration_ms: duration,
            timestamp: chrono::Utc::now(),
        };

        let message = AgentMessage::TaskComplete(completion);
        let msg_bytes = serde_json::to_vec(&message)?;
        send.write_all(&msg_bytes).await?;

        Ok(())
    }

    /// Process task
    async fn process_task(task: AgentTask) -> Result<serde_json::Value> {
        // Placeholder implementation
        match task.task_type {
            TaskType::PatternMatch { current, historical } => {
                // Simulate pattern matching
                let similarity = current
                    .iter()
                    .zip(historical.first().unwrap_or(&vec![]).iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / current.len() as f32;

                Ok(serde_json::json!({
                    "similarity": 1.0 - similarity.min(1.0),
                    "pattern_type": "dtw",
                }))
            }
            TaskType::Compute { description, input } => {
                Ok(serde_json::json!({
                    "result": format!("Processed: {}", description),
                    "input": input,
                }))
            }
            _ => Ok(serde_json::json!({ "status": "completed" })),
        }
    }

    /// Send message to coordinator
    async fn send_message(&self, message: AgentMessage) -> Result<()> {
        let connection = self
            .connection
            .as_ref()
            .ok_or_else(|| SwarmError::Other("Not connected".into()))?;

        let (mut send, mut recv) = connection.open_bi().await?;

        // Send message
        let msg_bytes = serde_json::to_vec(&message)?;
        send.write_all(&msg_bytes).await?;

        // Wait for acknowledgment
        let mut buf = vec![0u8; 1024];
        let n = recv
            .read(&mut buf)
            .await?
            .ok_or(SwarmError::StreamClosed)?;

        let _ack: MessageAck = serde_json::from_slice(&buf[..n])?;

        Ok(())
    }

    /// Send pattern match result
    pub async fn send_pattern_result(&self, result: PatternMatchResult) -> Result<()> {
        self.message_sender
            .send(AgentMessage::PatternMatchResult(result))
            .map_err(|_| SwarmError::Other("Failed to queue message".into()))
    }

    /// Send heartbeat
    pub async fn send_heartbeat(&self, load: f64, active_tasks: usize) -> Result<()> {
        let heartbeat = HeartbeatMessage {
            agent_id: self.agent_id.clone(),
            load,
            active_tasks,
            memory_usage: 0, // TODO: Get actual memory usage
            timestamp: chrono::Utc::now(),
        };

        self.message_sender
            .send(AgentMessage::Heartbeat(heartbeat))
            .map_err(|_| SwarmError::Other("Failed to queue message".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_creation() {
        // This test requires a running coordinator
        // Skip in CI
    }

    #[test]
    fn test_agent_id() {
        let agent_id = Uuid::new_v4().to_string();
        assert!(!agent_id.is_empty());
    }
}
