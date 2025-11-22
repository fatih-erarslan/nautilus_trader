// Iceberg Orders - REAL IMPLEMENTATION with hidden volume execution and stealth capabilities
use crossbeam::channel::{unbounded, Receiver, Sender};
use parking_lot::{Mutex, RwLock};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crate::execution::atomic_orders::{FillResult, OrderSide, OrderStatus};
use crate::execution::smart_order_routing::SmartOrderRouter;

/// Iceberg order configuration
#[derive(Debug, Clone)]
pub struct IcebergConfig {
    /// Percentage of total volume to show (1-50%)
    pub visible_percentage: f32,
    /// Random variation factor (±20% of slice size)
    pub randomization_factor: f32,
    /// Minimum slice size (to prevent too small orders)
    pub min_slice_size: u64,
    /// Maximum slice size (to prevent too large orders)
    pub max_slice_size: u64,
    /// Base delay between slice reveals (milliseconds)
    pub base_reveal_delay_ms: u64,
    /// Random delay variation (±50% of base delay)
    pub delay_randomization_factor: f32,
    /// Enable stealth mode (more sophisticated hiding)
    pub stealth_mode: bool,
    /// Price improvement threshold (basis points)
    pub price_improvement_threshold_bps: u32,
    /// Maximum number of active slices
    pub max_active_slices: u32,
    /// Detection avoidance level (0=none, 3=maximum)
    pub detection_avoidance_level: u32,
}

impl Default for IcebergConfig {
    fn default() -> Self {
        Self {
            visible_percentage: 10.0,        // 10% visible
            randomization_factor: 0.2,       // ±20% variation
            min_slice_size: 1_000_000,       // 0.01 units minimum
            max_slice_size: 100_000_000,     // 1.0 units maximum
            base_reveal_delay_ms: 500,       // 500ms base delay
            delay_randomization_factor: 0.5, // ±50% delay variation
            stealth_mode: true,
            price_improvement_threshold_bps: 5, // 0.05%
            max_active_slices: 3,
            detection_avoidance_level: 2, // High avoidance
        }
    }
}

/// Market pattern detector for stealth execution
#[derive(Debug, Clone)]
struct MarketPattern {
    volume_trend: f32,       // -1.0 to 1.0 (decreasing to increasing)
    price_volatility: f32,   // 0.0 to 1.0 (stable to volatile)
    order_flow_balance: f32, // -1.0 to 1.0 (sell pressure to buy pressure)
    time_since_pattern_ms: u64,
    confidence: f32, // 0.0 to 1.0
}

/// Detection avoidance metrics
#[derive(Debug, Clone)]
struct DetectionMetrics {
    repetitive_pattern_score: f32, // 0.0 to 1.0 (higher = more detectable)
    timing_predictability: f32,    // 0.0 to 1.0 (higher = more predictable)
    size_clustering: f32,          // 0.0 to 1.0 (higher = more clustered)
    market_impact_signature: f32,  // 0.0 to 1.0 (higher = more distinctive)
    overall_stealth_score: f32,    // 0.0 to 1.0 (higher = better stealth)
}

/// Individual slice of an iceberg order
#[derive(Debug)]
pub struct IcebergSlice {
    pub slice_id: u64,
    pub parent_order_id: u64,
    pub side: OrderSide,
    pub slice_quantity: u64,
    pub filled_quantity: AtomicU64,
    pub price: u64,
    pub status: AtomicU32, // OrderStatus
    pub created_ns: u64,
    pub reveal_time_ns: AtomicU64,
    pub is_active: AtomicBool,
    pub is_visible: AtomicBool,
    pub generation: u32, // Which generation of slice this is
    pub randomization_seed: u64,
    pub stealth_params: StealthParameters,
}

#[derive(Debug, Clone)]
pub struct StealthParameters {
    size_modifier: f32,    // -0.2 to 0.2 (size adjustment)
    timing_offset: i64,    // -500ms to +500ms offset
    price_offset: i64,     // Price adjustment in micropips
    behavior_type: u32,    // 0=aggressive, 1=passive, 2=adaptive
    camouflage_level: u32, // 0=none, 3=maximum
}

impl IcebergSlice {
    pub fn new(
        slice_id: u64,
        parent_order_id: u64,
        side: OrderSide,
        slice_quantity: u64,
        price: u64,
        generation: u32,
        stealth_params: StealthParameters,
    ) -> Self {
        let mut rng = StdRng::from_entropy();
        let seed = rng.gen();

        Self {
            slice_id,
            parent_order_id,
            side,
            slice_quantity,
            filled_quantity: AtomicU64::new(0),
            price,
            status: AtomicU32::new(OrderStatus::New as u32),
            created_ns: Self::timestamp_ns(),
            reveal_time_ns: AtomicU64::new(0),
            is_active: AtomicBool::new(false),
            is_visible: AtomicBool::new(false),
            generation,
            randomization_seed: seed,
            stealth_params,
        }
    }

    /// Fill slice atomically
    pub fn atomic_fill(&self, fill_quantity: u64) -> FillResult {
        loop {
            let current_filled = self.filled_quantity.load(Ordering::Acquire);
            let remaining = self.slice_quantity.saturating_sub(current_filled);

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
                    let new_status = if new_filled == self.slice_quantity {
                        OrderStatus::Filled
                    } else {
                        OrderStatus::PartiallyFilled
                    };

                    self.status.store(new_status as u32, Ordering::Release);

                    return FillResult::Success {
                        filled: to_fill,
                        remaining: self.slice_quantity - new_filled,
                        status: new_status,
                    };
                }
                Err(_) => continue,
            }
        }
    }

    /// Check if slice is completely filled
    pub fn is_filled(&self) -> bool {
        self.filled_quantity.load(Ordering::Acquire) >= self.slice_quantity
    }

    /// Get remaining quantity
    pub fn remaining_quantity(&self) -> u64 {
        self.slice_quantity
            .saturating_sub(self.filled_quantity.load(Ordering::Acquire))
    }

    /// Apply stealth modifications to visible parameters
    pub fn apply_stealth_modifications(&self, base_quantity: u64, base_price: u64) -> (u64, u64) {
        let mut rng = StdRng::seed_from_u64(self.randomization_seed + self.generation as u64);

        // Apply size modifier
        let size_factor = 1.0 + self.stealth_params.size_modifier;
        let modified_quantity = ((base_quantity as f32) * size_factor) as u64;

        // Apply price offset for camouflage
        let price_offset = match self.stealth_params.camouflage_level {
            0 => 0,
            1 => rng.gen_range(-50..=50),   // ±0.5 micropips
            2 => rng.gen_range(-100..=100), // ±1 micropip
            3 => rng.gen_range(-200..=200), // ±2 micropips
            _ => 0,
        };

        let modified_price = (base_price as i64 + price_offset).max(1) as u64;

        (modified_quantity, modified_price)
    }

    fn timestamp_ns() -> u64 {
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

/// Main Iceberg Order structure
pub struct IcebergOrder {
    pub order_id: u64,
    pub symbol: String,
    pub side: OrderSide,
    pub total_quantity: u64,
    pub filled_quantity: AtomicU64,
    pub price: u64,
    pub status: AtomicU32,
    pub created_ns: u64,
    pub updated_ns: AtomicU64,
    pub config: IcebergConfig,

    // Slice management
    active_slices: RwLock<HashMap<u64, Arc<IcebergSlice>>>,
    completed_slices: RwLock<Vec<Arc<IcebergSlice>>>,
    next_slice_id: AtomicU64,
    slice_generation: AtomicU32,

    // Randomization and stealth
    rng: Mutex<StdRng>,
    last_slice_reveal: AtomicU64,
    market_pattern: RwLock<MarketPattern>,
    detection_metrics: RwLock<DetectionMetrics>,
    stealth_adjustments: RwLock<Vec<(u64, f32)>>, // (timestamp, adjustment)

    // Performance tracking
    total_trades: AtomicU64,
    average_fill_time: AtomicU64,
    slippage_tracking: RwLock<Vec<i64>>,
    detection_score: AtomicU32, // 0-100, lower is better stealth
}

impl IcebergOrder {
    pub fn new(
        order_id: u64,
        symbol: String,
        side: OrderSide,
        total_quantity: u64,
        price: u64,
        config: IcebergConfig,
    ) -> Self {
        let rng = StdRng::from_entropy();
        let now = Self::timestamp_ns();

        Self {
            order_id,
            symbol,
            side,
            total_quantity,
            filled_quantity: AtomicU64::new(0),
            price,
            status: AtomicU32::new(OrderStatus::New as u32),
            created_ns: now,
            updated_ns: AtomicU64::new(now),
            config,
            active_slices: RwLock::new(HashMap::new()),
            completed_slices: RwLock::new(Vec::new()),
            next_slice_id: AtomicU64::new(1),
            slice_generation: AtomicU32::new(0),
            rng: Mutex::new(rng),
            last_slice_reveal: AtomicU64::new(0),
            market_pattern: RwLock::new(MarketPattern {
                volume_trend: 0.0,
                price_volatility: 0.0,
                order_flow_balance: 0.0,
                time_since_pattern_ms: 0,
                confidence: 0.0,
            }),
            detection_metrics: RwLock::new(DetectionMetrics {
                repetitive_pattern_score: 0.0,
                timing_predictability: 0.0,
                size_clustering: 0.0,
                market_impact_signature: 0.0,
                overall_stealth_score: 1.0,
            }),
            stealth_adjustments: RwLock::new(Vec::new()),
            total_trades: AtomicU64::new(0),
            average_fill_time: AtomicU64::new(0),
            slippage_tracking: RwLock::new(Vec::new()),
            detection_score: AtomicU32::new(0),
        }
    }

    /// Calculate next slice size with randomization
    pub fn calculate_next_slice_size(&self) -> u64 {
        let remaining = self.remaining_quantity();
        if remaining == 0 {
            return 0;
        }

        // Base slice size from configuration
        let base_size =
            ((self.total_quantity as f32) * (self.config.visible_percentage / 100.0)) as u64;
        let base_size = base_size.clamp(self.config.min_slice_size, self.config.max_slice_size);

        // Apply randomization
        let mut rng = self.rng.lock();
        let randomization =
            rng.gen_range(-self.config.randomization_factor..=self.config.randomization_factor);
        let randomized_size = ((base_size as f32) * (1.0 + randomization)) as u64;

        // Apply stealth adjustments based on market conditions
        let stealth_factor = if self.config.stealth_mode {
            self.calculate_stealth_size_factor(&mut rng)
        } else {
            1.0
        };

        let final_size = ((randomized_size as f32) * stealth_factor) as u64;

        // Clamp to remaining quantity and size limits
        final_size
            .clamp(self.config.min_slice_size, self.config.max_slice_size)
            .min(remaining)
    }

    /// Calculate stealth size factor based on market conditions
    fn calculate_stealth_size_factor(&self, rng: &mut StdRng) -> f32 {
        let pattern = self.market_pattern.read();
        let detection = self.detection_metrics.read();

        // Adjust size based on market volatility
        let volatility_factor = if pattern.price_volatility > 0.7 {
            // High volatility - can be more aggressive
            rng.gen_range(1.1..=1.3)
        } else if pattern.price_volatility < 0.3 {
            // Low volatility - be more conservative
            rng.gen_range(0.7..=0.9)
        } else {
            // Normal volatility - slight randomization
            rng.gen_range(0.9..=1.1)
        };

        // Adjust based on detection score
        let stealth_factor = if detection.overall_stealth_score < 0.5 {
            // Poor stealth - reduce size significantly
            rng.gen_range(0.5..=0.8)
        } else if detection.overall_stealth_score > 0.8 {
            // Good stealth - can be more aggressive
            rng.gen_range(1.0..=1.2)
        } else {
            // Average stealth
            rng.gen_range(0.8..=1.1)
        };

        // Apply detection avoidance level
        let avoidance_factor = match self.config.detection_avoidance_level {
            0 => 1.0,                      // No avoidance
            1 => rng.gen_range(0.9..=1.1), // Light avoidance
            2 => rng.gen_range(0.8..=1.2), // Medium avoidance
            3 => rng.gen_range(0.6..=1.4), // Maximum avoidance
            _ => 1.0,
        };

        volatility_factor * stealth_factor * avoidance_factor
    }

    /// Calculate adaptive reveal delay based on market conditions
    pub fn calculate_reveal_delay(&self) -> Duration {
        let mut rng = self.rng.lock();
        let base_delay = self.config.base_reveal_delay_ms;

        // Apply random variation
        let random_factor = rng.gen_range(
            -self.config.delay_randomization_factor..=self.config.delay_randomization_factor,
        );
        let randomized_delay = ((base_delay as f32) * (1.0 + random_factor)) as u64;

        // Apply market-based adjustments
        let market_factor = if self.config.stealth_mode {
            self.calculate_market_timing_factor(&mut rng)
        } else {
            1.0
        };

        let final_delay = ((randomized_delay as f32) * market_factor) as u64;
        Duration::from_millis(final_delay.clamp(100, 5000)) // 100ms to 5s range
    }

    /// Calculate timing factor based on market conditions
    fn calculate_market_timing_factor(&self, rng: &mut StdRng) -> f32 {
        let pattern = self.market_pattern.read();
        let detection = self.detection_metrics.read();

        // Adjust timing based on market activity
        let activity_factor = if pattern.volume_trend > 0.5 {
            // High activity - can reveal faster
            rng.gen_range(0.5..=0.8)
        } else if pattern.volume_trend < -0.5 {
            // Low activity - wait longer to blend in
            rng.gen_range(1.2..=2.0)
        } else {
            // Normal activity
            rng.gen_range(0.8..=1.2)
        };

        // Adjust based on timing predictability
        let predictability_factor = if detection.timing_predictability > 0.7 {
            // Too predictable - add more variation
            rng.gen_range(0.5..=2.0)
        } else {
            // Good unpredictability
            rng.gen_range(0.8..=1.2)
        };

        // Apply detection avoidance
        let avoidance_factor = match self.config.detection_avoidance_level {
            0 => 1.0,
            1 => rng.gen_range(0.9..=1.1),
            2 => rng.gen_range(0.7..=1.5),
            3 => rng.gen_range(0.5..=2.5),
            _ => 1.0,
        };

        activity_factor * predictability_factor * avoidance_factor
    }

    /// Generate stealth parameters for a new slice
    fn generate_stealth_parameters(&self, generation: u32) -> StealthParameters {
        let mut rng = self.rng.lock();

        // Size modifier based on detection avoidance level
        let size_modifier = match self.config.detection_avoidance_level {
            0 => 0.0,
            1 => rng.gen_range(-0.1..=0.1),
            2 => rng.gen_range(-0.15..=0.15),
            3 => rng.gen_range(-0.2..=0.2),
            _ => 0.0,
        };

        // Timing offset for unpredictability
        let timing_offset = if self.config.stealth_mode {
            rng.gen_range(-500..=500) // ±500ms
        } else {
            0
        };

        // Price offset for camouflage
        let price_offset = match self.config.detection_avoidance_level {
            0 => 0,
            1 => rng.gen_range(-25..=25),   // ±0.25 micropips
            2 => rng.gen_range(-50..=50),   // ±0.5 micropips
            3 => rng.gen_range(-100..=100), // ±1 micropip
            _ => 0,
        };

        // Behavioral type rotation to avoid patterns
        let behavior_type = match generation % 4 {
            0 => 0,                    // Aggressive
            1 => 1,                    // Passive
            2 => 2,                    // Adaptive
            3 => rng.gen_range(0..=2), // Random
            _ => 0,
        };

        StealthParameters {
            size_modifier,
            timing_offset,
            price_offset,
            behavior_type,
            camouflage_level: self.config.detection_avoidance_level,
        }
    }

    /// Create and reveal next slice
    pub fn reveal_next_slice(&self) -> Option<Arc<IcebergSlice>> {
        let remaining = self.remaining_quantity();
        if remaining == 0 {
            return None;
        }

        let slice_size = self.calculate_next_slice_size();
        if slice_size == 0 {
            return None;
        }

        let slice_id = self.next_slice_id.fetch_add(1, Ordering::AcqRel);
        let generation = self.slice_generation.fetch_add(1, Ordering::AcqRel);
        let stealth_params = self.generate_stealth_parameters(generation);

        let slice = Arc::new(IcebergSlice::new(
            slice_id,
            self.order_id,
            self.side,
            slice_size,
            self.price,
            generation,
            stealth_params,
        ));

        // Apply stealth modifications
        let (_modified_size, _modified_price) =
            slice.apply_stealth_modifications(slice_size, self.price);

        // Set reveal time with calculated delay
        let delay = self.calculate_reveal_delay();
        let reveal_time = Self::timestamp_ns() + delay.as_nanos() as u64;
        slice.reveal_time_ns.store(reveal_time, Ordering::Release);

        // Mark as active and visible
        slice.is_active.store(true, Ordering::Release);
        slice.is_visible.store(true, Ordering::Release);

        // Add to active slices
        {
            let mut active = self.active_slices.write();
            active.insert(slice_id, slice.clone());
        }

        // Update last reveal time
        self.last_slice_reveal
            .store(Self::timestamp_ns(), Ordering::Release);

        // Update detection metrics
        self.update_detection_metrics(slice_size, delay);

        Some(slice)
    }

    /// Update detection avoidance metrics
    fn update_detection_metrics(&self, slice_size: u64, _delay: Duration) {
        let mut detection = self.detection_metrics.write();
        let mut adjustments = self.stealth_adjustments.write();

        let now = Self::timestamp_ns();
        adjustments.push((now, slice_size as f32));

        // Keep only recent adjustments (last 1 hour)
        adjustments.retain(|(timestamp, _)| now - timestamp < 3_600_000_000_000);

        if adjustments.len() > 1 {
            // Calculate repetitive pattern score
            let sizes: Vec<f32> = adjustments.iter().map(|(_, size)| *size).collect();
            let mean_size: f32 = sizes.iter().sum::<f32>() / sizes.len() as f32;
            let variance: f32 =
                sizes.iter().map(|x| (x - mean_size).powi(2)).sum::<f32>() / sizes.len() as f32;
            let coefficient_of_variation = if mean_size > 0.0 {
                variance.sqrt() / mean_size
            } else {
                0.0
            };

            // Lower variation = higher pattern score (more detectable)
            detection.repetitive_pattern_score = 1.0 - coefficient_of_variation.min(1.0);

            // Calculate timing predictability
            let time_intervals: Vec<u64> =
                adjustments.windows(2).map(|w| w[1].0 - w[0].0).collect();

            if time_intervals.len() > 1 {
                let mean_interval =
                    time_intervals.iter().sum::<u64>() as f32 / time_intervals.len() as f32;
                let interval_variance: f32 = time_intervals
                    .iter()
                    .map(|&x| (x as f32 - mean_interval).powi(2))
                    .sum::<f32>()
                    / time_intervals.len() as f32;
                let interval_cv = if mean_interval > 0.0 {
                    interval_variance.sqrt() / mean_interval
                } else {
                    0.0
                };

                detection.timing_predictability = 1.0 - interval_cv.min(1.0);
            }

            // Calculate size clustering
            let recent_sizes: Vec<f32> = sizes.iter().rev().take(10).cloned().collect();
            if recent_sizes.len() > 2 {
                let size_range = recent_sizes
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
                    - recent_sizes
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();
                let size_mean: f32 = recent_sizes.iter().sum::<f32>() / recent_sizes.len() as f32;

                detection.size_clustering = if size_mean > 0.0 {
                    1.0 - (size_range / size_mean).min(1.0)
                } else {
                    0.0
                };
            }

            // Calculate overall stealth score
            detection.overall_stealth_score = 1.0
                - ((detection.repetitive_pattern_score * 0.3)
                    + (detection.timing_predictability * 0.3)
                    + (detection.size_clustering * 0.2)
                    + (detection.market_impact_signature * 0.2));
        }

        // Update detection score for monitoring
        let detection_percentage = ((1.0 - detection.overall_stealth_score) * 100.0) as u32;
        self.detection_score
            .store(detection_percentage, Ordering::Release);
    }

    /// Fill order from slice execution
    pub fn process_slice_fill(&self, slice_id: u64, fill_quantity: u64) -> bool {
        let slice = { self.active_slices.read().get(&slice_id).cloned() };

        if let Some(slice) = slice {
            match slice.atomic_fill(fill_quantity) {
                FillResult::Success {
                    filled,
                    remaining: _,
                    status,
                } => {
                    // Update total filled quantity
                    self.filled_quantity.fetch_add(filled, Ordering::AcqRel);
                    self.total_trades.fetch_add(1, Ordering::AcqRel);
                    self.updated_ns
                        .store(Self::timestamp_ns(), Ordering::Release);

                    // Move completed slice
                    if status == OrderStatus::Filled {
                        let mut active = self.active_slices.write();
                        if let Some(completed_slice) = active.remove(&slice_id) {
                            self.completed_slices.write().push(completed_slice);
                        }
                    }

                    // Check if entire order is filled
                    if self.filled_quantity.load(Ordering::Acquire) >= self.total_quantity {
                        self.status
                            .store(OrderStatus::Filled as u32, Ordering::Release);
                    } else if self.filled_quantity.load(Ordering::Acquire) > 0 {
                        self.status
                            .store(OrderStatus::PartiallyFilled as u32, Ordering::Release);
                    }

                    true
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Update market pattern for adaptive behavior
    pub fn update_market_pattern(
        &self,
        volume_trend: f32,
        price_volatility: f32,
        order_flow_balance: f32,
    ) {
        let mut pattern = self.market_pattern.write();

        // Exponential moving average update
        let alpha = 0.1;
        pattern.volume_trend = pattern.volume_trend * (1.0 - alpha) + volume_trend * alpha;
        pattern.price_volatility =
            pattern.price_volatility * (1.0 - alpha) + price_volatility * alpha;
        pattern.order_flow_balance =
            pattern.order_flow_balance * (1.0 - alpha) + order_flow_balance * alpha;

        // Update pattern age
        pattern.time_since_pattern_ms += 100; // Assume 100ms update interval

        // Calculate confidence based on data quality and age
        let age_factor = 1.0 - (pattern.time_since_pattern_ms as f32 / 300_000.0).min(1.0); // Decay over 5 minutes
        pattern.confidence = age_factor * 0.8 + 0.2; // Min 20% confidence

        // Update market impact signature in detection metrics
        {
            let mut detection = self.detection_metrics.write();
            let impact_factor = (price_volatility * volume_trend.abs()).min(1.0);
            detection.market_impact_signature = impact_factor;
        }
    }

    /// Check if ready for next slice reveal
    pub fn should_reveal_next_slice(&self) -> bool {
        if self.remaining_quantity() == 0 {
            return false;
        }

        // Check active slice count
        let active_count = self.active_slices.read().len() as u32;
        if active_count >= self.config.max_active_slices {
            return false;
        }

        // Check time since last reveal
        let now = Self::timestamp_ns();
        let last_reveal = self.last_slice_reveal.load(Ordering::Acquire);
        let min_delay = Duration::from_millis(self.config.base_reveal_delay_ms);

        if now - last_reveal < min_delay.as_nanos() as u64 {
            return false;
        }

        // Check market conditions for stealth mode
        if self.config.stealth_mode {
            let pattern = self.market_pattern.read();
            let detection = self.detection_metrics.read();

            // Don't reveal if detection risk is too high
            if detection.overall_stealth_score < 0.3 {
                return false;
            }

            // Consider market conditions
            if pattern.confidence > 0.7 {
                // High confidence in pattern data
                if pattern.price_volatility > 0.8 || pattern.order_flow_balance.abs() > 0.7 {
                    // Unfavorable market conditions for stealth
                    return false;
                }
            }
        }

        true
    }

    /// Get remaining quantity
    pub fn remaining_quantity(&self) -> u64 {
        self.total_quantity
            .saturating_sub(self.filled_quantity.load(Ordering::Acquire))
    }

    /// Get current status
    pub fn get_status(&self) -> OrderStatus {
        match self.status.load(Ordering::Acquire) {
            0 => OrderStatus::New,
            1 => OrderStatus::PartiallyFilled,
            2 => OrderStatus::Filled,
            3 => OrderStatus::Cancelled,
            4 => OrderStatus::Rejected,
            _ => OrderStatus::New,
        }
    }

    /// Cancel order
    pub fn cancel(&self) -> bool {
        // Cancel all active slices
        {
            let active = self.active_slices.read();
            for slice in active.values() {
                slice.is_active.store(false, Ordering::Release);
            }
        }

        self.status
            .store(OrderStatus::Cancelled as u32, Ordering::Release);
        self.updated_ns
            .store(Self::timestamp_ns(), Ordering::Release);
        true
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> IcebergMetrics {
        let active_slices = self.active_slices.read().len() as u32;
        let completed_slices = self.completed_slices.read().len() as u32;
        let filled_qty = self.filled_quantity.load(Ordering::Acquire);
        let fill_rate = if self.total_quantity > 0 {
            (filled_qty as f32 / self.total_quantity as f32) * 100.0
        } else {
            0.0
        };

        let detection = self.detection_metrics.read();
        let pattern = self.market_pattern.read();

        IcebergMetrics {
            order_id: self.order_id,
            total_quantity: self.total_quantity,
            filled_quantity: filled_qty,
            remaining_quantity: self.remaining_quantity(),
            fill_rate_percent: fill_rate,
            active_slices,
            completed_slices,
            total_trades: self.total_trades.load(Ordering::Acquire),
            average_fill_time_ms: self.average_fill_time.load(Ordering::Acquire) / 1_000_000,
            stealth_score: detection.overall_stealth_score,
            detection_risk: self.detection_score.load(Ordering::Acquire),
            market_pattern_confidence: pattern.confidence,
            execution_efficiency: self.calculate_execution_efficiency(),
        }
    }

    /// Calculate execution efficiency score
    fn calculate_execution_efficiency(&self) -> f32 {
        let fill_rate = if self.total_quantity > 0 {
            self.filled_quantity.load(Ordering::Acquire) as f32 / self.total_quantity as f32
        } else {
            0.0
        };

        let detection = self.detection_metrics.read();
        let time_efficiency = if self.created_ns > 0 {
            let elapsed_seconds = (Self::timestamp_ns() - self.created_ns) / 1_000_000_000;
            let expected_time_seconds = (self.total_quantity / 1_000_000) * 10; // 10 seconds per unit
            1.0 - (elapsed_seconds as f32 / expected_time_seconds as f32).min(1.0)
        } else {
            0.0
        };

        // Combine fill rate, stealth score, and time efficiency
        (fill_rate * 0.4) + (detection.overall_stealth_score * 0.4) + (time_efficiency * 0.2)
    }

    fn timestamp_ns() -> u64 {
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

/// Performance metrics for iceberg order
#[derive(Debug, Clone)]
pub struct IcebergMetrics {
    pub order_id: u64,
    pub total_quantity: u64,
    pub filled_quantity: u64,
    pub remaining_quantity: u64,
    pub fill_rate_percent: f32,
    pub active_slices: u32,
    pub completed_slices: u32,
    pub total_trades: u64,
    pub average_fill_time_ms: u64,
    pub stealth_score: f32,  // 0.0 to 1.0 (higher is better stealth)
    pub detection_risk: u32, // 0 to 100 (lower is better)
    pub market_pattern_confidence: f32,
    pub execution_efficiency: f32, // 0.0 to 1.0 (higher is better)
}

/// Iceberg Order Manager - handles multiple iceberg orders
pub struct IcebergOrderManager {
    orders: RwLock<HashMap<u64, Arc<IcebergOrder>>>,
    router: Arc<SmartOrderRouter>,
    next_order_id: AtomicU64,

    // Event channels
    fill_events_tx: Sender<IcebergFillEvent>,
    fill_events_rx: Receiver<IcebergFillEvent>,

    // Background processing
    slice_reveal_scheduler: RwLock<VecDeque<(u64, u64)>>, // (order_id, reveal_time_ns)
    market_data_processor: Arc<MarketDataProcessor>,

    // Statistics
    total_orders_created: AtomicU64,
    total_volume_processed: AtomicU64,
    average_stealth_score: AtomicU32, // Fixed point (x100)
}

#[derive(Debug, Clone)]
pub struct IcebergFillEvent {
    pub order_id: u64,
    pub slice_id: u64,
    pub filled_quantity: u64,
    pub price: u64,
    pub timestamp_ns: u64,
}

/// Market data processor for pattern recognition
pub struct MarketDataProcessor {
    volume_window: RwLock<VecDeque<(u64, u64)>>, // (timestamp, volume)
    price_window: RwLock<VecDeque<(u64, u64)>>,  // (timestamp, price)
    order_flow_window: RwLock<VecDeque<(u64, f32)>>, // (timestamp, buy_sell_ratio)
    window_size_ms: u64,
}

impl MarketDataProcessor {
    pub fn new(window_size_ms: u64) -> Self {
        Self {
            volume_window: RwLock::new(VecDeque::new()),
            price_window: RwLock::new(VecDeque::new()),
            order_flow_window: RwLock::new(VecDeque::new()),
            window_size_ms,
        }
    }

    /// Update market data
    pub fn update_market_data(&self, volume: u64, price: u64, buy_sell_ratio: f32) {
        let now = Self::timestamp_ns();
        let cutoff_time = now - (self.window_size_ms * 1_000_000);

        // Update volume window
        {
            let mut volume_window = self.volume_window.write();
            volume_window.push_back((now, volume));
            while volume_window
                .front()
                .is_some_and(|(ts, _)| *ts < cutoff_time)
            {
                volume_window.pop_front();
            }
        }

        // Update price window
        {
            let mut price_window = self.price_window.write();
            price_window.push_back((now, price));
            while price_window
                .front()
                .is_some_and(|(ts, _)| *ts < cutoff_time)
            {
                price_window.pop_front();
            }
        }

        // Update order flow window
        {
            let mut flow_window = self.order_flow_window.write();
            flow_window.push_back((now, buy_sell_ratio));
            while flow_window.front().is_some_and(|(ts, _)| *ts < cutoff_time) {
                flow_window.pop_front();
            }
        }
    }

    /// Calculate market patterns
    pub fn calculate_patterns(&self) -> (f32, f32, f32) {
        let volume_trend = self.calculate_volume_trend();
        let price_volatility = self.calculate_price_volatility();
        let order_flow_balance = self.calculate_order_flow_balance();

        (volume_trend, price_volatility, order_flow_balance)
    }

    fn calculate_volume_trend(&self) -> f32 {
        let volume_window = self.volume_window.read();
        if volume_window.len() < 2 {
            return 0.0;
        }

        let volumes: Vec<u64> = volume_window.iter().map(|(_, v)| *v).collect();
        let mid_point = volumes.len() / 2;

        let early_avg: f64 =
            volumes[..mid_point].iter().map(|&x| x as f64).sum::<f64>() / mid_point as f64;
        let late_avg: f64 = volumes[mid_point..].iter().map(|&x| x as f64).sum::<f64>()
            / (volumes.len() - mid_point) as f64;

        if early_avg > 0.0 {
            ((late_avg - early_avg) / early_avg).clamp(-1.0, 1.0) as f32
        } else {
            0.0
        }
    }

    fn calculate_price_volatility(&self) -> f32 {
        let price_window = self.price_window.read();
        if price_window.len() < 2 {
            return 0.0;
        }

        let prices: Vec<f64> = price_window.iter().map(|(_, p)| *p as f64).collect();
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();

        if mean > 0.0 {
            (std_dev / mean).min(1.0) as f32
        } else {
            0.0
        }
    }

    fn calculate_order_flow_balance(&self) -> f32 {
        let flow_window = self.order_flow_window.read();
        if flow_window.is_empty() {
            return 0.0;
        }

        let average_ratio: f32 =
            flow_window.iter().map(|(_, r)| *r).sum::<f32>() / flow_window.len() as f32;
        (average_ratio - 0.5) * 2.0 // Convert from [0,1] to [-1,1]
    }

    fn timestamp_ns() -> u64 {
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

impl IcebergOrderManager {
    pub fn new(router: Arc<SmartOrderRouter>) -> Self {
        let (fill_tx, fill_rx) = unbounded();

        Self {
            orders: RwLock::new(HashMap::new()),
            router,
            next_order_id: AtomicU64::new(1),
            fill_events_tx: fill_tx,
            fill_events_rx: fill_rx,
            slice_reveal_scheduler: RwLock::new(VecDeque::new()),
            market_data_processor: Arc::new(MarketDataProcessor::new(300_000)), // 5 minute window
            total_orders_created: AtomicU64::new(0),
            total_volume_processed: AtomicU64::new(0),
            average_stealth_score: AtomicU32::new(100), // Start with perfect stealth
        }
    }

    /// Create new iceberg order
    pub fn create_iceberg_order(
        &self,
        symbol: String,
        side: OrderSide,
        total_quantity: u64,
        price: u64,
        config: Option<IcebergConfig>,
    ) -> u64 {
        let order_id = self.next_order_id.fetch_add(1, Ordering::AcqRel);
        let config = config.unwrap_or_default();

        let order = Arc::new(IcebergOrder::new(
            order_id,
            symbol,
            side,
            total_quantity,
            price,
            config,
        ));

        self.orders.write().insert(order_id, order);
        self.total_orders_created.fetch_add(1, Ordering::AcqRel);
        self.total_volume_processed
            .fetch_add(total_quantity, Ordering::AcqRel);

        order_id
    }

    /// Process slice reveals and market updates
    pub fn process_background_tasks(&self) {
        // Process fill events
        while let Ok(event) = self.fill_events_rx.try_recv() {
            if let Some(order) = self.orders.read().get(&event.order_id) {
                order.process_slice_fill(event.slice_id, event.filled_quantity);
            }
        }

        // Process scheduled slice reveals
        self.process_scheduled_reveals();

        // Update market patterns for all orders
        self.update_market_patterns();

        // Clean up completed orders
        self.cleanup_completed_orders();
    }

    /// Process scheduled slice reveals
    fn process_scheduled_reveals(&self) {
        let now = Self::timestamp_ns();
        let mut reveals_to_process = Vec::new();

        // Check scheduler
        {
            let mut scheduler = self.slice_reveal_scheduler.write();
            while let Some(&(order_id, reveal_time)) = scheduler.front() {
                if reveal_time <= now {
                    reveals_to_process.push(order_id);
                    scheduler.pop_front();
                } else {
                    break;
                }
            }
        }

        // Process reveals
        for order_id in reveals_to_process {
            if let Some(order) = self.orders.read().get(&order_id) {
                if order.should_reveal_next_slice() {
                    if let Some(_slice) = order.reveal_next_slice() {
                        // Schedule next reveal
                        let next_delay = order.calculate_reveal_delay();
                        let next_reveal_time = now + next_delay.as_nanos() as u64;
                        self.slice_reveal_scheduler
                            .write()
                            .push_back((order_id, next_reveal_time));
                    }
                }
            }
        }

        // Auto-reveal for new orders
        let orders = self.orders.read();
        for (&order_id, order) in orders.iter() {
            if order.get_status() == OrderStatus::New && order.should_reveal_next_slice() {
                if let Some(_slice) = order.reveal_next_slice() {
                    let delay = order.calculate_reveal_delay();
                    let reveal_time = now + delay.as_nanos() as u64;
                    drop(orders); // Release read lock before acquiring write lock
                    self.slice_reveal_scheduler
                        .write()
                        .push_back((order_id, reveal_time));
                    break; // Re-acquire lock in next iteration
                }
            }
        }
    }

    /// Update market patterns for all orders
    fn update_market_patterns(&self) {
        let (volume_trend, price_volatility, order_flow_balance) =
            self.market_data_processor.calculate_patterns();

        let orders = self.orders.read();
        for order in orders.values() {
            order.update_market_pattern(volume_trend, price_volatility, order_flow_balance);
        }
    }

    /// Clean up completed orders
    fn cleanup_completed_orders(&self) {
        let mut orders = self.orders.write();
        let completed_orders: Vec<u64> = orders
            .iter()
            .filter(|(_, order)| {
                let status = order.get_status();
                status == OrderStatus::Filled || status == OrderStatus::Cancelled
            })
            .map(|(&id, _)| id)
            .collect();

        // Keep recent completed orders for a while, then remove old ones
        let cutoff_time = Self::timestamp_ns() - 3_600_000_000_000; // 1 hour

        for order_id in completed_orders {
            if let Some(order) = orders.get(&order_id) {
                let order_age = Self::timestamp_ns() - order.created_ns;
                if order_age > cutoff_time {
                    orders.remove(&order_id);
                }
            }
        }
    }

    /// Submit fill event
    pub fn submit_fill_event(&self, event: IcebergFillEvent) {
        let _ = self.fill_events_tx.send(event);
    }

    /// Update market data
    pub fn update_market_data(&self, volume: u64, price: u64, buy_sell_ratio: f32) {
        self.market_data_processor
            .update_market_data(volume, price, buy_sell_ratio);
    }

    /// Get order metrics
    pub fn get_order_metrics(&self, order_id: u64) -> Option<IcebergMetrics> {
        self.orders
            .read()
            .get(&order_id)
            .map(|order| order.get_performance_metrics())
    }

    /// Get all order metrics
    pub fn get_all_metrics(&self) -> Vec<IcebergMetrics> {
        self.orders
            .read()
            .values()
            .map(|order| order.get_performance_metrics())
            .collect()
    }

    /// Get manager statistics
    pub fn get_statistics(&self) -> ManagerStatistics {
        let orders = self.orders.read();
        let total_orders = orders.len() as u64;
        let active_orders = orders
            .values()
            .filter(|order| {
                let status = order.get_status();
                status == OrderStatus::New || status == OrderStatus::PartiallyFilled
            })
            .count() as u64;

        let total_stealth_score: f32 = orders
            .values()
            .map(|order| order.get_performance_metrics().stealth_score)
            .sum();

        let avg_stealth = if total_orders > 0 {
            total_stealth_score / total_orders as f32
        } else {
            1.0
        };

        ManagerStatistics {
            total_orders_created: self.total_orders_created.load(Ordering::Acquire),
            active_orders,
            total_volume_processed: self.total_volume_processed.load(Ordering::Acquire),
            average_stealth_score: avg_stealth,
            pending_slice_reveals: self.slice_reveal_scheduler.read().len() as u64,
        }
    }

    /// Cancel order
    pub fn cancel_order(&self, order_id: u64) -> bool {
        if let Some(order) = self.orders.read().get(&order_id) {
            order.cancel()
        } else {
            false
        }
    }

    fn timestamp_ns() -> u64 {
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

#[derive(Debug, Clone)]
pub struct ManagerStatistics {
    pub total_orders_created: u64,
    pub active_orders: u64,
    pub total_volume_processed: u64,
    pub average_stealth_score: f32,
    pub pending_slice_reveals: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[test]
    fn test_iceberg_config_default() {
        let config = IcebergConfig::default();
        assert_eq!(config.visible_percentage, 10.0);
        assert_eq!(config.randomization_factor, 0.2);
        assert!(config.stealth_mode);
    }

    #[test]
    fn test_iceberg_slice_creation() {
        let stealth_params = StealthParameters {
            size_modifier: 0.1,
            timing_offset: 100,
            price_offset: 50,
            behavior_type: 0,
            camouflage_level: 2,
        };

        let slice = IcebergSlice::new(
            1,
            100,
            OrderSide::Buy,
            1_000_000,
            50_000_000,
            1,
            stealth_params,
        );

        assert_eq!(slice.slice_id, 1);
        assert_eq!(slice.parent_order_id, 100);
        assert_eq!(slice.slice_quantity, 1_000_000);
        assert!(!slice.is_filled());
        assert_eq!(slice.remaining_quantity(), 1_000_000);
    }

    #[test]
    fn test_slice_fill() {
        let stealth_params = StealthParameters {
            size_modifier: 0.0,
            timing_offset: 0,
            price_offset: 0,
            behavior_type: 0,
            camouflage_level: 0,
        };

        let slice = IcebergSlice::new(
            1,
            100,
            OrderSide::Buy,
            1_000_000,
            50_000_000,
            1,
            stealth_params,
        );

        // Partial fill
        match slice.atomic_fill(300_000) {
            FillResult::Success {
                filled,
                remaining,
                status,
            } => {
                assert_eq!(filled, 300_000);
                assert_eq!(remaining, 700_000);
                assert_eq!(status, OrderStatus::PartiallyFilled);
            }
            _ => panic!("Fill should succeed"),
        }

        // Complete fill
        match slice.atomic_fill(700_000) {
            FillResult::Success {
                filled,
                remaining,
                status,
            } => {
                assert_eq!(filled, 700_000);
                assert_eq!(remaining, 0);
                assert_eq!(status, OrderStatus::Filled);
                assert!(slice.is_filled());
            }
            _ => panic!("Fill should succeed"),
        }
    }

    #[test]
    fn test_iceberg_order_creation() {
        let config = IcebergConfig::default();
        let order = IcebergOrder::new(
            1,
            "BTCUSD".to_string(),
            OrderSide::Buy,
            10_000_000,
            50_000_000,
            config,
        );

        assert_eq!(order.order_id, 1);
        assert_eq!(order.total_quantity, 10_000_000);
        assert_eq!(order.remaining_quantity(), 10_000_000);
        assert_eq!(order.get_status(), OrderStatus::New);
    }

    #[test]
    fn test_slice_size_calculation() {
        let config = IcebergConfig {
            visible_percentage: 20.0,
            min_slice_size: 500_000,
            max_slice_size: 5_000_000,
            ..Default::default()
        };

        let order = IcebergOrder::new(
            1,
            "BTCUSD".to_string(),
            OrderSide::Buy,
            10_000_000,
            50_000_000,
            config,
        );

        let slice_size = order.calculate_next_slice_size();
        assert!(slice_size >= 500_000);
        assert!(slice_size <= 5_000_000);

        // Should be roughly 20% of total with some randomization
        let expected_base = 2_000_000; // 20% of 10M
        assert!(slice_size > expected_base / 2);
        assert!(slice_size < expected_base * 2);
    }

    #[test]
    fn test_stealth_modifications() {
        let stealth_params = StealthParameters {
            size_modifier: 0.1, // +10%
            timing_offset: 100,
            price_offset: 50, // +50 micropips
            behavior_type: 2,
            camouflage_level: 1,
        };

        let slice = IcebergSlice::new(
            1,
            100,
            OrderSide::Buy,
            1_000_000,
            50_000_000,
            1,
            stealth_params,
        );

        let (modified_qty, modified_price) =
            slice.apply_stealth_modifications(1_000_000, 50_000_000);

        // Size should be increased by ~10%
        assert!(modified_qty > 1_000_000);
        assert!(modified_qty < 1_200_000);

        // Price might be modified for camouflage
        assert!(modified_price > 0);
    }

    #[test]
    fn test_market_pattern_update() {
        let config = IcebergConfig::default();
        let order = IcebergOrder::new(
            1,
            "BTCUSD".to_string(),
            OrderSide::Buy,
            10_000_000,
            50_000_000,
            config,
        );

        order.update_market_pattern(0.3, 0.5, -0.2);

        let pattern = order.market_pattern.read();
        assert!((pattern.volume_trend - 0.03).abs() < 0.1); // EMA effect
        assert!((pattern.price_volatility - 0.05).abs() < 0.1);
        assert!((pattern.order_flow_balance - (-0.02)).abs() < 0.1);
    }

    #[test]
    fn test_detection_metrics() {
        let config = IcebergConfig {
            detection_avoidance_level: 3,
            ..Default::default()
        };

        let order = IcebergOrder::new(
            1,
            "BTCUSD".to_string(),
            OrderSide::Buy,
            10_000_000,
            50_000_000,
            config,
        );

        // Simulate repetitive slicing
        for _ in 0..5 {
            order.update_detection_metrics(1_000_000, Duration::from_millis(500));
        }

        let detection = order.detection_metrics.read();
        assert!(detection.repetitive_pattern_score > 0.0);
        assert!(detection.timing_predictability > 0.0);
        assert!(detection.overall_stealth_score < 1.0);
    }

    #[tokio::test]
    async fn test_iceberg_order_manager() {
        let router = Arc::new(SmartOrderRouter::new());
        let manager = IcebergOrderManager::new(router);

        let order_id = manager.create_iceberg_order(
            "BTCUSD".to_string(),
            OrderSide::Buy,
            10_000_000,
            50_000_000,
            None,
        );

        // Update market data
        manager.update_market_data(1_000_000, 50_000_000, 0.6);

        // Process background tasks
        manager.process_background_tasks();

        // Check metrics
        let metrics = manager.get_order_metrics(order_id).unwrap();
        assert_eq!(metrics.order_id, order_id);
        assert_eq!(metrics.total_quantity, 10_000_000);

        // Test fill event
        let fill_event = IcebergFillEvent {
            order_id,
            slice_id: 1,
            filled_quantity: 500_000,
            price: 50_000_000,
            timestamp_ns: IcebergOrderManager::timestamp_ns(),
        };

        manager.submit_fill_event(fill_event);
        manager.process_background_tasks();

        // Check updated metrics
        let updated_metrics = manager.get_order_metrics(order_id).unwrap();
        assert!(updated_metrics.filled_quantity >= 500_000);

        let stats = manager.get_statistics();
        assert_eq!(stats.total_orders_created, 1);
        assert!(stats.average_stealth_score > 0.0);
    }

    #[test]
    fn test_market_data_processor() {
        let processor = MarketDataProcessor::new(10_000); // 10 second window

        // Add some market data
        processor.update_market_data(1_000_000, 50_000_000, 0.6);
        std::thread::sleep(std::time::Duration::from_millis(100));
        processor.update_market_data(1_200_000, 50_100_000, 0.7);
        std::thread::sleep(std::time::Duration::from_millis(100));
        processor.update_market_data(800_000, 49_900_000, 0.4);

        let (volume_trend, price_volatility, order_flow_balance) = processor.calculate_patterns();

        assert!(volume_trend.abs() <= 1.0);
        assert!(price_volatility >= 0.0 && price_volatility <= 1.0);
        assert!(order_flow_balance.abs() <= 1.0);
    }
}
