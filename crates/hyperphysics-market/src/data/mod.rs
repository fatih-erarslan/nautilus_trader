//! Market data structures and types

pub mod bar;
pub mod orderbook;
pub mod tick;
pub mod lockfree_orderbook;

pub use bar::{Bar, Timeframe};
pub use orderbook::{OrderBook, OrderBookLevel};
pub use tick::Tick;

// Lock-free order book for HFT (ultra-low latency)
pub use lockfree_orderbook::{
    LockFreeOrderBook, LockFreeSkipList,
    PriceLevel, Order, OrderSide, OrderStatus, AtomicOrderStatus,
    OrderBookStats, MarketDepth,
};
