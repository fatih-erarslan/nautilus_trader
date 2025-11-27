//! Localized Conformal Prediction
//!
//! Combines elements of Mondrian and Kandinsky approaches for
//! flexible conditional coverage guarantees.
//!
//! Can use either discrete groups or continuous kernel weighting.

use super::{NonconformityScore, GroupId};
use super::mondrian::MondrianCalibrator;
use super::kandinsky::{KandinskyCalibrator, KernelType};

/// Configuration for localized conformal predictor
#[derive(Clone, Debug)]
pub struct LocalizedConfig {
    pub alpha: f32,
    pub localization_type: LocalizationType,
}

/// Type of localization to use
#[derive(Clone, Debug)]
pub enum LocalizationType {
    /// Use Mondrian (group-based)
    Mondrian {
        min_group_size: usize,
        fallback_to_marginal: bool,
    },
    /// Use Kandinsky (kernel-based)
    Kandinsky {
        bandwidth: f32,
        kernel_type: KernelType,
        min_effective_samples: f32,
    },
    /// Hybrid: try Mondrian first, fallback to Kandinsky
    Hybrid {
        mondrian_min_size: usize,
        kandinsky_bandwidth: f32,
    },
}

impl Default for LocalizedConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            localization_type: LocalizationType::Mondrian {
                min_group_size: 30,
                fallback_to_marginal: true,
            },
        }
    }
}

/// Unified localized conformal predictor
///
/// Adapts between Mondrian and Kandinsky based on configuration
pub struct LocalizedCalibrator<S: NonconformityScore> {
    config: LocalizedConfig,
    mondrian: Option<MondrianCalibrator<S>>,
    kandinsky: Option<KandinskyCalibrator<S>>,
}

impl<S: NonconformityScore> LocalizedCalibrator<S> {
    pub fn new(config: LocalizedConfig, scorer: S) -> Self {
        match &config.localization_type {
            LocalizationType::Mondrian { min_group_size, fallback_to_marginal } => {
                let mondrian_config = super::mondrian::MondrianConfig {
                    alpha: config.alpha,
                    min_group_size: *min_group_size,
                    fallback_to_marginal: *fallback_to_marginal,
                    conservative: true,
                };
                Self {
                    config,
                    mondrian: Some(MondrianCalibrator::new(mondrian_config, scorer)),
                    kandinsky: None,
                }
            }
            LocalizationType::Kandinsky { bandwidth, kernel_type, min_effective_samples } => {
                let kandinsky_config = super::kandinsky::KandinskyConfig {
                    alpha: config.alpha,
                    bandwidth: *bandwidth,
                    kernel_type: kernel_type.clone(),
                    min_effective_samples: *min_effective_samples,
                    fallback_to_marginal: true,
                };
                Self {
                    config,
                    mondrian: None,
                    kandinsky: Some(KandinskyCalibrator::new(kandinsky_config, scorer.clone())),
                }
            }
            LocalizationType::Hybrid { mondrian_min_size, kandinsky_bandwidth } => {
                let mondrian_config = super::mondrian::MondrianConfig {
                    alpha: config.alpha,
                    min_group_size: *mondrian_min_size,
                    fallback_to_marginal: false,
                    conservative: true,
                };
                let kandinsky_config = super::kandinsky::KandinskyConfig {
                    alpha: config.alpha,
                    bandwidth: *kandinsky_bandwidth,
                    kernel_type: KernelType::Gaussian,
                    min_effective_samples: 10.0,
                    fallback_to_marginal: true,
                };
                Self {
                    config,
                    mondrian: Some(MondrianCalibrator::new(mondrian_config, scorer.clone())),
                    kandinsky: Some(KandinskyCalibrator::new(kandinsky_config, scorer)),
                }
            }
        }
    }

    /// Calibrate with both groups and features (for hybrid mode)
    pub fn calibrate_with_groups(
        &mut self,
        predictions: &[Vec<f32>],
        labels: &[usize],
        groups: &[GroupId],
        u_values: &[f32],
    ) {
        if let Some(ref mut mondrian) = self.mondrian {
            mondrian.calibrate(predictions, labels, groups, u_values);
        }
    }

    /// Calibrate with features (for Kandinsky mode)
    pub fn calibrate_with_features(
        &mut self,
        predictions: &[Vec<f32>],
        labels: &[usize],
        features: &[Vec<f32>],
        u_values: &[f32],
    ) {
        if let Some(ref mut kandinsky) = self.kandinsky {
            kandinsky.calibrate(predictions, labels, features, u_values);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct MockScorer;

    impl NonconformityScore for MockScorer {
        fn score(&self, prediction: &[f32], label: usize, _u: f32) -> f32 {
            1.0 - prediction[label]
        }
    }

    #[test]
    fn test_localized_mondrian_mode() {
        let config = LocalizedConfig {
            alpha: 0.1,
            localization_type: LocalizationType::Mondrian {
                min_group_size: 2,
                fallback_to_marginal: true,
            },
        };

        let calibrator = LocalizedCalibrator::new(config, MockScorer);
        assert!(calibrator.mondrian.is_some());
        assert!(calibrator.kandinsky.is_none());
    }

    #[test]
    fn test_localized_kandinsky_mode() {
        let config = LocalizedConfig {
            alpha: 0.1,
            localization_type: LocalizationType::Kandinsky {
                bandwidth: 1.0,
                kernel_type: KernelType::Gaussian,
                min_effective_samples: 10.0,
            },
        };

        let calibrator = LocalizedCalibrator::new(config, MockScorer);
        assert!(calibrator.mondrian.is_none());
        assert!(calibrator.kandinsky.is_some());
    }

    #[test]
    fn test_localized_hybrid_mode() {
        let config = LocalizedConfig {
            alpha: 0.1,
            localization_type: LocalizationType::Hybrid {
                mondrian_min_size: 30,
                kandinsky_bandwidth: 1.0,
            },
        };

        let calibrator = LocalizedCalibrator::new(config, MockScorer);
        assert!(calibrator.mondrian.is_some());
        assert!(calibrator.kandinsky.is_some());
    }
}
