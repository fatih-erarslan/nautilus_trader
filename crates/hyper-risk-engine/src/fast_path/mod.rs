//! Fast path optimizations for sub-100μs risk checks.
//!
//! This module contains highly optimized routines for the
//! critical path of pre-trade risk evaluation.
//!
//! ## Optimization Techniques
//!
//! 1. **Cache-line alignment**: Data structures aligned to 64 bytes
//! 2. **Branch prediction hints**: Likely/unlikely annotations
//! 3. **SIMD vectorization**: Batch processing where applicable
//! 4. **Lock-free algorithms**: Atomic operations only
//! 5. **Zero allocation**: No heap allocation in hot path
//!
//! ## Latency Budget
//!
//! | Component | Target | Implementation |
//! |-----------|--------|----------------|
//! | Data access | 5μs | Cache-aligned structures |
//! | Limit checks | 10μs | Branchless comparisons |
//! | Quantile lookup | 15μs | Pre-computed tables |
//! | Decision | 5μs | Simple logic |
//! | **Total** | **35μs** | |

pub mod pre_trade;
pub mod limit_checker;
pub mod anomaly_detector;

pub use pre_trade::{PreTradeChecker, PreTradeResult};
pub use limit_checker::{LimitChecker, LimitViolation};
pub use anomaly_detector::{FastAnomalyDetector, AnomalyScore};
