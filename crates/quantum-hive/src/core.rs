//! Core types and structures for the quantum trading hive

use serde::{Serialize, Deserialize};
use std::time::Instant;

/// Quantum state representation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QuantumState {
    pub amplitude: [f64; 2],
    pub phase: f64,
    pub entanglement_strength: f64,
}

impl Default for QuantumState {
    fn default() -> Self {
        Self {
            amplitude: [1.0, 0.0],
            phase: 0.0,
            entanglement_strength: 0.0,
        }
    }
}

/// Trade action to be executed
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TradeAction {
    pub action_type: ActionType,
    pub quantity: f64,
    pub confidence: f64,
    pub risk_factor: f64,
}

impl Default for TradeAction {
    fn default() -> Self {
        Self {
            action_type: ActionType::Hold,
            quantity: 0.0,
            confidence: 0.0,
            risk_factor: 0.0,
        }
    }
}

/// Type of trading action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActionType {
    Buy = 0,
    Sell = 1,
    Hold = 2,
    Hedge = 3,
}

/// Market tick data
#[derive(Debug, Clone, Copy)]
pub struct MarketTick {
    pub symbol: [u8; 8], // Fixed-size for zero-allocation
    pub price: f64,
    pub volume: f64,
    pub timestamp: u64,
    pub bid: f64,
    pub ask: f64,
}

impl Default for MarketTick {
    fn default() -> Self {
        Self {
            symbol: [0; 8],
            price: 0.0,
            volume: 0.0,
            timestamp: 0,
            bid: 0.0,
            ask: 0.0,
        }
    }
}

/// Pre-computed quantum strategy lookup table for nanosecond execution
#[derive(Debug)]
pub struct QuantumStrategyLUT {
    // 65536 price buckets for ultra-fast lookup
    pub price_actions: Box<[TradeAction; 65536]>,
    pub volatility_actions: Box<[TradeAction; 1024]>,
    pub correlation_matrix: [[f64; 16]; 16], // 16x16 asset correlation
    pub last_update: Instant,
    pub generation: u64,
}

impl QuantumStrategyLUT {
    /// Get action for a given price index with zero-cost abstraction
    #[inline(always)]
    pub unsafe fn get_action(&self, price_index: u16) -> TradeAction {
        *self.price_actions.get_unchecked(price_index as usize)
    }
    
    /// Safe version of get_action
    #[inline]
    pub fn get_action_safe(&self, price: f64) -> TradeAction {
        let price_index = ((price * 65535.0) as u16).min(65535);
        self.price_actions[price_index as usize]
    }
}

impl Default for QuantumStrategyLUT {
    fn default() -> Self {
        Self {
            price_actions: Box::new([TradeAction::default(); 65536]),
            volatility_actions: Box::new([TradeAction::default(); 1024]),
            correlation_matrix: [[0.0; 16]; 16],
            last_update: Instant::now(),
            generation: 0,
        }
    }
}

/// Bell state types for quantum entanglement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BellStateType {
    PhiPlus,
    PhiMinus,
    PsiPlus,
    PsiMinus,
}

/// Market regime classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MarketRegime {
    Trending,
    MeanReverting,
    HighVolatility,
    LowVolatility,
}

/// Execution statistics for performance tracking
#[derive(Debug, Default, Clone)]
pub struct ExecutionStats {
    pub trades_executed: u64,
    pub total_pnl: f64,
    pub avg_latency_ns: u64,
    pub error_count: u64,
    pub success_rate: f64,
}

/// Circular buffer for efficient data storage
#[derive(Debug)]
pub struct CircularBuffer<T> {
    pub data: Vec<T>,
    head: usize,
    pub size: usize,
    capacity: usize,
}

impl<T: Copy + Default> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            head: 0,
            size: 0,
            capacity,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, item: T) {
        unsafe {
            *self.data.get_unchecked_mut(self.head) = item;
        }
        self.head = (self.head + 1) % self.capacity;
        self.size = self.size.saturating_add(1).min(self.capacity);
    }

    #[inline(always)]
    pub fn latest(&self) -> Option<T> {
        if self.size > 0 {
            let idx = if self.head == 0 { self.capacity - 1 } else { self.head - 1 };
            unsafe { Some(*self.data.get_unchecked(idx)) }
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let start = if self.size < self.capacity { 0 } else { self.head };
        (0..self.size).map(move |i| &self.data[(start + i) % self.capacity])
    }
}

/// Quantum job types for PennyLane bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumJobType {
    StrategyOptimization,
    RiskAssessment,
    CorrelationAnalysis,
    RegimeDetection,
    AnomalyDetection,
}

/// Quantum computation job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumJob {
    pub job_id: u64,
    pub job_type: QuantumJobType,
    pub priority: u8,
    pub created_at: u64,
    pub parameters: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_strategy_lut_performance() {
        let lut = QuantumStrategyLUT::default();
        let start = Instant::now();
        
        // Test nanosecond lookup performance
        for i in 0..1_000_000 {
            let price_index = (i % 65536) as u16;
            unsafe {
                let _action = lut.get_action(price_index);
            }
        }
        
        let duration = start.elapsed();
        println!("1M lookups took: {:?}", duration);
        assert!(duration.as_millis() < 10); // Should be < 10ms for 1M lookups
    }

    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(10);
        
        for i in 0..20 {
            buffer.push(i);
        }
        
        assert_eq!(buffer.latest(), Some(19));
        assert_eq!(buffer.size, 10);
    }
}