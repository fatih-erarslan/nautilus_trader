//! Momentum observer implementation

use crate::prelude::*;

/// Observer for momentum analysis
#[derive(Debug, Clone)]
pub struct MomentumObserver {
    pub window: usize,
    pub threshold: f64,
}

impl MomentumObserver {
    pub fn new(window: usize, threshold: f64) -> Self {
        Self { window, threshold }
    }
    
    pub fn observe(&self, data: &[f64]) -> f64 {
        if data.len() < self.window {
            return 0.0;
        }
        
        let recent = &data[data.len() - self.window..];
        let first = recent[0];
        let last = recent[recent.len() - 1];
        
        (last - first) / first
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_momentum_observer() {
        let observer = MomentumObserver::new(3, 0.1);
        let data = vec![100.0, 105.0, 110.0, 115.0];
        let momentum = observer.observe(&data);
        assert!(momentum > 0.0);
    }
}