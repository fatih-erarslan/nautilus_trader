use beclever_common::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionRequest {
    pub input: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionResult {
    pub is_threat: bool,
    pub confidence: f64,
    pub threat_types: Vec<String>,
}

#[cfg_attr(test, mockall::automock)]
pub trait ThreatDetector: Send + Sync {
    fn detect(&self, request: ThreatDetectionRequest) -> Result<ThreatDetectionResult>;
}
