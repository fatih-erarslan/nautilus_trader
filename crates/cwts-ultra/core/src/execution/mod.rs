pub mod atomic_orders;
pub mod branchless;
pub mod iceberg_orders;
pub mod simple_orders;
pub mod smart_order_routing;
pub mod twap_vwap;

// Re-export main types from atomic_orders as the canonical versions
pub use atomic_orders::{AtomicOrder, FillResult, OrderSide, OrderStatus, OrderType, Trade};
pub use branchless::*;
pub use iceberg_orders::*;
pub use smart_order_routing::{ExecutionResult, SmartOrderRouter};
pub use twap_vwap::{ExecutionEngine, ExecutionSlice, StrategyType, TWAPStrategy, VWAPStrategy};
