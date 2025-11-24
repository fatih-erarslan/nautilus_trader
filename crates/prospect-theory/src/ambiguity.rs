//! Ambiguity aversion module for Prospect Theory

use num_traits::Float;
use serde::{Deserialize, Serialize};

use crate::{ProspectError, ProspectResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguityHandler<F: Float> {
    pub aversion_coefficient: F,
    pub seeking_threshold: F,
}

impl<F: Float> AmbiguityHandler<F> 
where
    F: From<f64> + Copy + PartialOrd,
{
    pub fn new(aversion_coefficient: F) -> ProspectResult<Self> {
        Ok(Self {
            aversion_coefficient,
            seeking_threshold: <F as From<f64>>::from(0.2),
        })
    }
    
    pub fn apply_ambiguity_aversion(&self, value: F, ambiguity_level: F) -> ProspectResult<F> {
        if ambiguity_level < self.seeking_threshold {
            // Low ambiguity - might seek it
            Ok(value * (F::one() + ambiguity_level * <F as From<f64>>::from(0.1)))
        } else {
            // High ambiguity - averse to it
            let aversion_factor = F::one() - self.aversion_coefficient * ambiguity_level;
            Ok(value * aversion_factor)
        }
    }
    
    pub fn reset(&mut self) {
        // Reset any state if needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ambiguity_handler() {
        let handler = AmbiguityHandler::new(1.0).unwrap();
        
        // Low ambiguity should increase value
        let low_ambiguity = handler.apply_ambiguity_aversion(100.0, 0.1).unwrap();
        assert!(low_ambiguity > 100.0);
        
        // High ambiguity should decrease value
        let high_ambiguity = handler.apply_ambiguity_aversion(100.0, 0.8).unwrap();
        assert!(high_ambiguity < 100.0);
    }
}