//! Topological mapping of market data structures
//!
//! This module provides functionality to map financial market data
//! into topological spaces for analysis using hyperphysics-geometry.
//!
//! Phase 2 will implement:
//! - Price-time manifold mappings
//! - Order book topology (bid-ask spread as metric)
//! - Volume-weighted topology
//! - Correlation manifolds for multi-asset analysis

pub mod mapper;

pub use mapper::MarketTopologyMapper;
