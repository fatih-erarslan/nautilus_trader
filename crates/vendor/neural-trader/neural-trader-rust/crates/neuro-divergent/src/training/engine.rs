//! Training engine for neural models

use crate::{Result, NeuroDivergentError};

/// Early stopping helper
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
    best_loss: Option<f64>,
    counter: usize,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: None,
            counter: 0,
        }
    }

    pub fn should_stop(&mut self, val_loss: f64) -> bool {
        if let Some(best) = self.best_loss {
            if val_loss < best - self.min_delta {
                self.best_loss = Some(val_loss);
                self.counter = 0;
                false
            } else {
                self.counter += 1;
                self.counter >= self.patience
            }
        } else {
            self.best_loss = Some(val_loss);
            false
        }
    }

    pub fn reset(&mut self) {
        self.best_loss = None;
        self.counter = 0;
    }
}

/// Training engine
pub struct TrainingEngine {
    pub early_stopping: Option<EarlyStopping>,
}

impl TrainingEngine {
    pub fn new(patience: Option<usize>) -> Self {
        Self {
            early_stopping: patience.map(|p| EarlyStopping::new(p, 1e-4)),
        }
    }

    pub fn check_early_stopping(&mut self, val_loss: f64) -> Result<bool> {
        if let Some(ref mut es) = self.early_stopping {
            Ok(es.should_stop(val_loss))
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping() {
        let mut es = EarlyStopping::new(3, 0.01);

        assert!(!es.should_stop(1.0));
        assert!(!es.should_stop(0.9));
        assert!(!es.should_stop(0.89));
        assert!(!es.should_stop(0.891));
        assert!(!es.should_stop(0.892));
        assert!(es.should_stop(0.893));
    }
}
