//! Reference Point Management for Prospect Theory

use crate::{MarketData, Position, ProspectTheoryError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct ReferencePointManager {
    current_reference: f64,
    adaptation_rate: f64,
}

impl ReferencePointManager {
    pub fn new() -> Self {
        Self {
            current_reference: 0.0,
            adaptation_rate: 0.1,
        }
    }
    
    pub fn update_reference_point(&mut self, market_data: &MarketData, position: Option<&Position>) -> Result<f64> {
        // Simple adaptive reference point
        let new_reference = if let Some(pos) = position {
            pos.entry_price * 0.7 + market_data.current_price * 0.3
        } else {
            market_data.current_price
        };
        
        self.current_reference = self.current_reference * (1.0 - self.adaptation_rate) + 
                                new_reference * self.adaptation_rate;
        
        Ok(self.current_reference)
    }
}