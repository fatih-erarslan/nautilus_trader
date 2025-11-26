//! Odds API integration module (stub for future implementation)

use crate::{Error, Result};
use serde::{Serialize, Deserialize};

/// Odds from betting market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Odds {
    pub sport: String,
    pub event: String,
    pub home: f64,
    pub away: f64,
    pub draw: Option<f64>,
}

/// Odds API client (stub)
pub struct OddsApiClient {
    api_key: String,
}

impl OddsApiClient {
    /// Create new odds API client
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }

    /// Fetch odds for sport (stub - to be implemented)
    pub async fn fetch_odds(&self, _sport: &str) -> Result<Vec<Odds>> {
        // TODO: Implement actual API integration
        Err(Error::Internal("Odds API not yet implemented".to_string()))
    }
}
