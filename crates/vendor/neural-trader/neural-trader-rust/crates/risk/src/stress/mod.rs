//! Stress testing scenarios and sensitivity analysis
//!
//! Portfolio stress testing and risk factor analysis:
//! - Historical scenarios (2008 Financial Crisis, 2020 COVID Crash)
//! - Custom stress scenarios
//! - Multi-factor sensitivity analysis
//! - Cross-sensitivity (2D grid analysis)

pub mod scenarios;
pub mod sensitivity;

pub use crate::types::StressTestResult;
pub use scenarios::{
    StressScenario, StressTester, CustomScenario, AssetClass
};
pub use sensitivity::{
    SensitivityAnalyzer, SensitivityResult, SensitivityScenario,
    MarketFactor, CrossSensitivityResult, CrossSensitivityPoint
};
