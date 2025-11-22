// Simplified Orders System for MCP Server
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Simplified order with atomic operations
#[derive(Debug)]
pub struct AtomicOrder {
    pub order_id: AtomicU64,
    pub price: AtomicU64,
    pub quantity: AtomicU64,
    pub filled_quantity: AtomicU64,
    pub side: AtomicU32,
    pub order_type: AtomicU32,
    pub status: AtomicU32,
    pub created_ns: AtomicU64,
    pub updated_ns: AtomicU64,
    pub is_active: AtomicBool,
    pub is_immediate_or_cancel: AtomicBool,
    pub is_fill_or_kill: AtomicBool,
    pub is_post_only: AtomicBool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy = 0,
    Sell = 1,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Market = 0,
    Limit = 1,
    Stop = 2,
    StopLimit = 3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderStatus {
    New = 0,
    PartiallyFilled = 1,
    Filled = 2,
    Cancelled = 3,
    Rejected = 4,
}

#[derive(Debug, Clone)]
pub enum FillResult {
    Success {
        filled: u64,
        remaining: u64,
        status: OrderStatus,
    },
    OrderInactive,
    AlreadyFilled,
    InsufficientQuantity,
}

impl AtomicOrder {
    pub fn new(
        order_id: u64,
        price: u64,
        quantity: u64,
        side: OrderSide,
        order_type: OrderType,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self {
            order_id: AtomicU64::new(order_id),
            price: AtomicU64::new(price),
            quantity: AtomicU64::new(quantity),
            filled_quantity: AtomicU64::new(0),
            side: AtomicU32::new(side as u32),
            order_type: AtomicU32::new(order_type as u32),
            status: AtomicU32::new(OrderStatus::New as u32),
            created_ns: AtomicU64::new(now),
            updated_ns: AtomicU64::new(now),
            is_active: AtomicBool::new(true),
            is_immediate_or_cancel: AtomicBool::new(false),
            is_fill_or_kill: AtomicBool::new(false),
            is_post_only: AtomicBool::new(false),
        }
    }

    pub fn atomic_fill(&self, fill_quantity: u64) -> FillResult {
        if !self.is_active.load(Ordering::Acquire) {
            return FillResult::OrderInactive;
        }

        loop {
            let current_filled = self.filled_quantity.load(Ordering::Acquire);
            let total_quantity = self.quantity.load(Ordering::Acquire);
            let remaining = total_quantity.saturating_sub(current_filled);

            if remaining == 0 {
                return FillResult::AlreadyFilled;
            }

            let to_fill = fill_quantity.min(remaining);
            let new_filled = current_filled + to_fill;

            match self.filled_quantity.compare_exchange_weak(
                current_filled,
                new_filled,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let new_status = if new_filled == total_quantity {
                        OrderStatus::Filled
                    } else {
                        OrderStatus::PartiallyFilled
                    };

                    self.status.store(new_status as u32, Ordering::Release);
                    self.updated_ns
                        .store(Self::timestamp_ns(), Ordering::Release);

                    return FillResult::Success {
                        filled: to_fill,
                        remaining: total_quantity - new_filled,
                        status: new_status,
                    };
                }
                Err(_) => continue,
            }
        }
    }

    pub fn atomic_cancel(&self) -> bool {
        match self
            .is_active
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                self.status
                    .store(OrderStatus::Cancelled as u32, Ordering::Release);
                self.updated_ns
                    .store(Self::timestamp_ns(), Ordering::Release);
                true
            }
            Err(_) => false,
        }
    }

    pub fn atomic_modify_price(&self, new_price: u64) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return false;
        }

        self.price.store(new_price, Ordering::Release);
        self.updated_ns
            .store(Self::timestamp_ns(), Ordering::Release);
        true
    }

    pub fn atomic_modify_quantity(&self, new_quantity: u64) -> bool {
        loop {
            let filled = self.filled_quantity.load(Ordering::Acquire);

            if new_quantity <= filled {
                return false;
            }

            if !self.is_active.load(Ordering::Acquire) {
                return false;
            }

            let old_quantity = self.quantity.load(Ordering::Acquire);

            match self.quantity.compare_exchange_weak(
                old_quantity,
                new_quantity,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.updated_ns
                        .store(Self::timestamp_ns(), Ordering::Release);
                    return true;
                }
                Err(_) => continue,
            }
        }
    }

    fn timestamp_ns() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

/// Simplified matching engine
pub struct AtomicMatchingEngine {
    orders: Arc<RwLock<HashMap<u64, Arc<AtomicOrder>>>>,
    last_trade_price: AtomicU64,
    total_volume: AtomicU64,
    trade_count: AtomicU64,
}

impl Default for AtomicMatchingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicMatchingEngine {
    pub fn new() -> Self {
        Self {
            orders: Arc::new(RwLock::new(HashMap::new())),
            last_trade_price: AtomicU64::new(0),
            total_volume: AtomicU64::new(0),
            trade_count: AtomicU64::new(0),
        }
    }

    pub fn submit_order(&self, order: AtomicOrder) -> bool {
        let order_id = order.order_id.load(Ordering::Acquire);
        let arc_order = Arc::new(order);

        tokio::spawn({
            let orders = self.orders.clone();
            let arc_order = arc_order.clone();
            async move {
                let mut orders = orders.write().await;
                orders.insert(order_id, arc_order);
            }
        });

        true
    }

    pub fn match_orders(&self) -> Vec<Trade> {
        // Simplified matching - returns empty vec for now
        // In a real implementation, this would match orders
        Vec::new()
    }

    pub fn get_stats(&self) -> EngineStats {
        EngineStats {
            buy_orders: 0,  // Simplified
            sell_orders: 0, // Simplified
            last_price: self.last_trade_price.load(Ordering::Acquire),
            total_volume: self.total_volume.load(Ordering::Acquire),
            trade_count: self.trade_count.load(Ordering::Acquire),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub buy_order_id: u64,
    pub sell_order_id: u64,
    pub price: u64,
    pub quantity: u64,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone)]
pub struct EngineStats {
    pub buy_orders: u64,
    pub sell_orders: u64,
    pub last_price: u64,
    pub total_volume: u64,
    pub trade_count: u64,
}
