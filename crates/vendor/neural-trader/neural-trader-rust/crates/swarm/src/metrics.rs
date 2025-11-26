//! Metrics collection for QUIC swarm coordination

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Swarm metrics collector
#[derive(Clone)]
pub struct SwarmMetrics {
    /// Total connections
    pub total_connections: Arc<AtomicU64>,
    /// Active connections
    pub active_connections: Arc<AtomicU64>,
    /// Total messages sent
    pub messages_sent: Arc<AtomicU64>,
    /// Total messages received
    pub messages_received: Arc<AtomicU64>,
    /// Total bytes sent
    pub bytes_sent: Arc<AtomicU64>,
    /// Total bytes received
    pub bytes_received: Arc<AtomicU64>,
    /// Total errors
    pub errors: Arc<AtomicU64>,
}

impl SwarmMetrics {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            total_connections: Arc::new(AtomicU64::new(0)),
            active_connections: Arc::new(AtomicU64::new(0)),
            messages_sent: Arc::new(AtomicU64::new(0)),
            messages_received: Arc::new(AtomicU64::new(0)),
            bytes_sent: Arc::new(AtomicU64::new(0)),
            bytes_received: Arc::new(AtomicU64::new(0)),
            errors: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Increment total connections
    pub fn inc_connections(&self) {
        self.total_connections.fetch_add(1, Ordering::Relaxed);
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active connections
    pub fn dec_connections(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record message sent
    pub fn record_sent(&self, bytes: u64) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record message received
    pub fn record_received(&self, bytes: u64) {
        self.messages_received.fetch_add(1, Ordering::Relaxed);
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record error
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            total_connections: self.total_connections.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

impl Default for SwarmMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics snapshot
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Total connections
    pub total_connections: u64,
    /// Active connections
    pub active_connections: u64,
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Total errors
    pub errors: u64,
}

impl MetricsSnapshot {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "Connections: {}/{} | Messages: {}↑ {}↓ | Bytes: {}↑ {}↓ | Errors: {}",
            self.active_connections,
            self.total_connections,
            self.messages_sent,
            self.messages_received,
            self.format_bytes(self.bytes_sent),
            self.format_bytes(self.bytes_received),
            self.errors
        )
    }

    fn format_bytes(&self, bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = 1024 * KB;
        const GB: u64 = 1024 * MB;

        if bytes >= GB {
            format!("{:.2}GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2}MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2}KB", bytes as f64 / KB as f64)
        } else {
            format!("{}B", bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collection() {
        let metrics = SwarmMetrics::new();

        metrics.inc_connections();
        metrics.record_sent(1024);
        metrics.record_received(2048);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_connections, 1);
        assert_eq!(snapshot.active_connections, 1);
        assert_eq!(snapshot.messages_sent, 1);
        assert_eq!(snapshot.messages_received, 1);
        assert_eq!(snapshot.bytes_sent, 1024);
        assert_eq!(snapshot.bytes_received, 2048);
    }

    #[test]
    fn test_format_bytes() {
        let snapshot = MetricsSnapshot {
            total_connections: 0,
            active_connections: 0,
            messages_sent: 0,
            messages_received: 0,
            bytes_sent: 1024,
            bytes_received: 1024 * 1024,
            errors: 0,
        };

        let formatted = snapshot.format();
        assert!(formatted.contains("1.00KB"));
        assert!(formatted.contains("1.00MB"));
    }
}
