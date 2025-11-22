/// Risk management and quantitative finance modules
pub mod greeks;
pub mod var;
pub mod metrics;
pub mod engine;

pub use greeks::{OptionParams, Greeks, calculate_black_scholes, calculate_put_greeks};
pub use var::{VarModel, GarchParams, calculate_var, historical_var, garch_var, ewma_var};
pub use metrics::RiskMetrics;
pub use engine::{RiskEngine, RiskConfig};
