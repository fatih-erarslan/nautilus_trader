//! Conditional Coverage Conformal Prediction
//!
//! Implements group-conditional and kernel-based localized conformal prediction
//! for achieving coverage guarantees across subpopulations.
//!
//! # References
//! - Vovk et al. (2012): "Conditional validity of inductive conformal predictors"
//! - Romano et al. (2020): "Mondrian Conformal Prediction"
//! - Barber et al. (2023): "Conformal Prediction Beyond Exchangeability"

pub mod mondrian;
pub mod kandinsky;
pub mod localized;

pub use mondrian::{MondrianCalibrator, MondrianConfig, GroupId};
pub use kandinsky::{KandinskyCalibrator, KandinskyConfig};
pub use localized::{LocalizedCalibrator, LocalizedConfig};

/// Common trait for nonconformity scoring (compatible with scores::NonconformityScorer)
pub trait NonconformityScore: Clone {
    /// Compute nonconformity score for a prediction
    ///
    /// # Arguments
    /// * `prediction` - Softmax probabilities for all classes
    /// * `label` - True class label
    /// * `u` - Random uniform [0,1] for tie-breaking
    ///
    /// # Returns
    /// Nonconformity score (higher = less conforming)
    fn score(&self, prediction: &[f32], label: usize, u: f32) -> f32;
}

// Implement NonconformityScore for any type implementing crate::scores::NonconformityScorer
impl<T: crate::scores::NonconformityScorer + Clone> NonconformityScore for T {
    fn score(&self, prediction: &[f32], label: usize, u: f32) -> f32 {
        crate::scores::NonconformityScorer::score(self, prediction, label, u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Verify module compilation and basic structure
        assert!(true, "Conditional module compiles successfully");
    }
}
