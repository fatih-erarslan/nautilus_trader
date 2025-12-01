//! # Neuromorphic Technical Indicators
//!
//! This example demonstrates how to use the pBit architecture for financial
//! technical analysis. Unlike traditional indicators (moving averages, RSI),
//! neuromorphic indicators learn patterns through collective dynamics.
//!
//! ## Concepts
//!
//! 1. **Market Events → Spikes**: Price/volume changes become input spikes
//! 2. **pBit Fabric → Pattern Memory**: Couplings encode learned correlations
//! 3. **Metropolis Dynamics → Signal Generation**: Collective state = indicator
//!
//! Run with: cargo run --example financial_indicators --release --no-default-features

use hyperphysics_cortical_bus::prelude::*;
use hyperphysics_cortical_bus::scalable_pbit::{
    ScalablePBitFabric, ScalablePBitConfig, SparseCouplings,
};
use std::collections::VecDeque;

// =============================================================================
// PART 1: Market Data Structures
// =============================================================================

/// A single OHLCV bar
#[derive(Debug, Clone, Copy)]
pub struct Bar {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Bar {
    /// Calculate returns
    pub fn returns(&self) -> f64 {
        (self.close - self.open) / self.open
    }

    /// Calculate bar range
    pub fn range(&self) -> f64 {
        (self.high - self.low) / self.open
    }

    /// Is this a bullish bar?
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

// =============================================================================
// PART 2: Spike Encoder - Convert Market Data to Spikes
// =============================================================================

/// Encodes market data into spike trains
///
/// Maps different market features to different pBit regions:
/// - pBits 0-63: Price momentum (different timeframes)
/// - pBits 64-127: Volume spikes
/// - pBits 128-191: Volatility regime
/// - pBits 192-255: Trend direction
pub struct MarketSpikeEncoder {
    /// Historical prices for momentum calculation
    price_history: VecDeque<f64>,
    /// Historical volumes for relative comparison
    volume_history: VecDeque<f64>,
    /// Lookback periods for different indicators
    lookback_periods: Vec<usize>,
    /// Threshold for spike generation
    spike_threshold: f64,
}

impl MarketSpikeEncoder {
    pub fn new() -> Self {
        Self {
            price_history: VecDeque::with_capacity(256),
            volume_history: VecDeque::with_capacity(256),
            lookback_periods: vec![1, 5, 10, 20, 50, 100, 200], // Multiple timeframes
            spike_threshold: 0.001, // 0.1% move triggers spike
        }
    }

    /// Process a new bar and generate spikes
    pub fn encode(&mut self, bar: &Bar) -> Vec<(usize, f32)> {
        let mut spikes = Vec::new();

        // Update history
        self.price_history.push_back(bar.close);
        self.volume_history.push_back(bar.volume);
        if self.price_history.len() > 256 {
            self.price_history.pop_front();
            self.volume_history.pop_front();
        }

        // === MOMENTUM SPIKES (pBits 0-63) ===
        // Each lookback period gets 8 pBits
        for (period_idx, &period) in self.lookback_periods.iter().enumerate() {
            if self.price_history.len() > period {
                let current = bar.close;
                let past = self.price_history[self.price_history.len() - 1 - period];
                let momentum = (current - past) / past;

                // Quantize momentum into 8 levels
                let base_pbit = period_idx * 8;
                let level = ((momentum / self.spike_threshold).clamp(-4.0, 4.0) + 4.0) as usize;
                
                // Activate the appropriate pBit
                if level < 8 {
                    let strength = momentum.abs().min(1.0) as f32;
                    spikes.push((base_pbit + level, strength));
                }
            }
        }

        // === VOLUME SPIKES (pBits 64-127) ===
        if self.volume_history.len() > 20 {
            let avg_volume: f64 = self.volume_history.iter().rev().take(20).sum::<f64>() / 20.0;
            let volume_ratio = bar.volume / avg_volume;

            // Volume surge detection (8 levels)
            let volume_level = ((volume_ratio - 0.5).clamp(0.0, 3.0) * 2.0) as usize;
            let base_pbit = 64 + volume_level * 8;
            
            for i in 0..8 {
                if volume_ratio > 1.0 + (i as f64 * 0.25) {
                    spikes.push((base_pbit + i, (volume_ratio - 1.0).min(2.0) as f32));
                }
            }
        }

        // === VOLATILITY REGIME (pBits 128-191) ===
        let range = bar.range();
        let volatility_level = (range / 0.01).clamp(0.0, 7.0) as usize;
        spikes.push((128 + volatility_level * 8, range.min(1.0) as f32));

        // === TREND DIRECTION (pBits 192-255) ===
        // Use multiple timeframe trend agreement
        let mut bullish_count = 0;
        let mut bearish_count = 0;
        
        for &period in &[5, 10, 20, 50] {
            if self.price_history.len() > period {
                let sma: f64 = self.price_history.iter().rev().take(period).sum::<f64>() 
                    / period as f64;
                if bar.close > sma {
                    bullish_count += 1;
                } else {
                    bearish_count += 1;
                }
            }
        }

        // Trend strength indicator
        let trend_pbit = 192 + (bullish_count * 8);
        spikes.push((trend_pbit, (bullish_count as f32 / 4.0)));

        spikes
    }
}

// =============================================================================
// PART 3: Neuromorphic Indicator - The pBit Pattern Detector
// =============================================================================

/// Neuromorphic technical indicator using pBit dynamics
///
/// This is NOT a traditional indicator. Instead of calculating a formula,
/// it learns patterns through the collective dynamics of coupled pBits.
pub struct NeuromorphicIndicator {
    /// pBit fabric for pattern detection
    fabric: ScalablePBitFabric,
    /// Market spike encoder
    encoder: MarketSpikeEncoder,
    /// Output signal history
    signal_history: VecDeque<f64>,
    /// Number of sweeps per bar
    sweeps_per_bar: usize,
    /// Signal smoothing factor
    smoothing: f64,
}

impl NeuromorphicIndicator {
    /// Create a new neuromorphic indicator
    pub fn new() -> Self {
        // Configure pBit fabric: 256 pBits with structured couplings
        let config = ScalablePBitConfig {
            num_pbits: 256,
            avg_degree: 12,
            temperature: 1.0,
            use_simd: true,
            coupling_capacity: Some(256 * 12),
        };

        let mut fabric = ScalablePBitFabric::new(config);
        
        // Build structured couplings that encode market knowledge
        Self::build_market_couplings(&mut fabric);

        Self {
            fabric,
            encoder: MarketSpikeEncoder::new(),
            signal_history: VecDeque::with_capacity(100),
            sweeps_per_bar: 10,
            smoothing: 0.2,
        }
    }

    /// Build couplings that encode market structure
    fn build_market_couplings(fabric: &mut ScalablePBitFabric) {
        // === INTRA-REGION COUPLINGS ===
        // Within each region, nearby pBits should correlate
        
        for region_start in [0, 64, 128, 192] {
            for i in region_start..(region_start + 64) {
                for j in (i + 1)..(region_start + 64).min(i + 4) {
                    // Nearby pBits in same region: positive coupling
                    fabric.add_coupling(i, j, 0.5);
                }
            }
        }

        // === CROSS-REGION COUPLINGS (learned patterns) ===
        
        // Momentum → Trend alignment
        // Strong momentum should align with trend
        for momentum_pbit in 0..56 {
            for trend_pbit in 192..256 {
                if momentum_pbit % 8 > 4 {  // Bullish momentum
                    fabric.add_coupling(momentum_pbit, trend_pbit, 0.3);
                }
            }
        }

        // Volume → Volatility correlation
        // High volume often accompanies high volatility
        for volume_pbit in 64..128 {
            for volatility_pbit in 128..192 {
                fabric.add_coupling(volume_pbit, volatility_pbit, 0.2);
            }
        }

        // Momentum divergence detection
        // Short-term vs long-term momentum disagreement
        for short_term in 0..16 {
            for long_term in 40..56 {
                // Negative coupling: divergence creates tension
                fabric.add_coupling(short_term, long_term, -0.4);
            }
        }

        // Finalize CSR structure
        // (In the actual implementation, this happens automatically)
    }

    /// Process a new bar and update the indicator
    pub fn update(&mut self, bar: &Bar) -> IndicatorSignal {
        // 1. Encode market data into spikes
        let spikes = self.encoder.encode(bar);

        // 2. Apply spikes as bias modulations
        for (pbit_idx, strength) in &spikes {
            self.fabric.set_bias(*pbit_idx, *strength);
        }

        // 3. Run Metropolis dynamics to let patterns emerge
        for _ in 0..self.sweeps_per_bar {
            self.fabric.metropolis_sweep();
        }

        // 4. Read output signal from fabric state
        let signal = self.compute_signal();

        // 5. Smooth the signal
        let smoothed = if let Some(&last) = self.signal_history.back() {
            self.smoothing * signal + (1.0 - self.smoothing) * last
        } else {
            signal
        };

        self.signal_history.push_back(smoothed);
        if self.signal_history.len() > 100 {
            self.signal_history.pop_front();
        }

        // 6. Generate indicator signal
        IndicatorSignal {
            value: smoothed,
            direction: self.interpret_direction(smoothed),
            strength: self.fabric.magnetization().abs(),
            regime: self.detect_regime(),
        }
    }

    /// Compute raw signal from pBit states
    fn compute_signal(&self) -> f64 {
        // Weight different regions differently
        let mut signal = 0.0;

        // Momentum contribution (weight: 0.4)
        let momentum_active: f64 = (0..64)
            .filter(|&i| self.fabric.get_state(i))
            .map(|i| if i % 8 > 4 { 1.0 } else { -1.0 })
            .sum();
        signal += 0.4 * momentum_active / 64.0;

        // Volume contribution (weight: 0.2)
        let volume_active = (64..128)
            .filter(|&i| self.fabric.get_state(i))
            .count() as f64;
        signal += 0.2 * (volume_active / 32.0 - 1.0);

        // Trend contribution (weight: 0.4)
        let trend_active: f64 = (192..256)
            .filter(|&i| self.fabric.get_state(i))
            .count() as f64;
        signal += 0.4 * (trend_active / 32.0 - 1.0);

        signal.clamp(-1.0, 1.0)
    }

    /// Interpret signal direction
    fn interpret_direction(&self, signal: f64) -> Direction {
        if signal > 0.3 {
            Direction::Bullish
        } else if signal < -0.3 {
            Direction::Bearish
        } else {
            Direction::Neutral
        }
    }

    /// Detect market regime from volatility pBits
    fn detect_regime(&self) -> Regime {
        let volatility_active = (128..192)
            .filter(|&i| self.fabric.get_state(i))
            .count();

        if volatility_active > 48 {
            Regime::HighVolatility
        } else if volatility_active < 16 {
            Regime::LowVolatility
        } else {
            Regime::Normal
        }
    }

    /// Get recent signal history for plotting
    pub fn history(&self) -> Vec<f64> {
        self.signal_history.iter().copied().collect()
    }

    /// Adjust temperature (exploration vs exploitation)
    pub fn set_temperature(&mut self, temp: f64) {
        self.fabric.set_temperature(temp);
    }
}

/// Output signal from the indicator
#[derive(Debug, Clone)]
pub struct IndicatorSignal {
    /// Raw signal value [-1, +1]
    pub value: f64,
    /// Interpreted direction
    pub direction: Direction,
    /// Signal strength (0 to 1)
    pub strength: f64,
    /// Detected market regime
    pub regime: Regime,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Bullish,
    Bearish,
    Neutral,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Regime {
    HighVolatility,
    Normal,
    LowVolatility,
}

// =============================================================================
// PART 4: Demonstration
// =============================================================================

fn main() -> Result<()> {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     📊 Neuromorphic Technical Indicator Demo 📊                   ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Using pBit dynamics for pattern-based signal generation          ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Create indicator
    let mut indicator = NeuromorphicIndicator::new();

    // Generate synthetic market data (trending then ranging)
    println!("📈 Generating synthetic market data...\n");
    let bars = generate_synthetic_data(200);

    // Process bars
    println!("🔮 Processing bars through neuromorphic indicator...\n");
    println!("   Bar │ Close    │ Signal │ Direction │ Regime");
    println!("  ─────┼──────────┼────────┼───────────┼────────────────");

    let mut signals = Vec::new();
    let mut directions = Vec::new();

    for (i, bar) in bars.iter().enumerate() {
        let signal = indicator.update(bar);
        signals.push(signal.value);
        directions.push(signal.direction);

        // Print every 10th bar
        if i % 10 == 0 {
            println!(
                "  {:>4} │ {:>8.4} │ {:>+6.3} │ {:^9} │ {:?}",
                i,
                bar.close,
                signal.value,
                format!("{:?}", signal.direction),
                signal.regime
            );
        }
    }

    // Analyze results
    println!("\n📊 Signal Analysis:");
    println!("  ─────────────────────────────────────────");

    // Count direction changes
    let mut direction_changes = 0;
    for i in 1..directions.len() {
        if directions[i] != directions[i - 1] {
            direction_changes += 1;
        }
    }

    let bullish_count = directions.iter().filter(|&&d| d == Direction::Bullish).count();
    let bearish_count = directions.iter().filter(|&&d| d == Direction::Bearish).count();
    let neutral_count = directions.iter().filter(|&&d| d == Direction::Neutral).count();

    println!("   Bullish signals:  {:>3} ({:.1}%)", bullish_count, bullish_count as f64 / bars.len() as f64 * 100.0);
    println!("   Bearish signals:  {:>3} ({:.1}%)", bearish_count, bearish_count as f64 / bars.len() as f64 * 100.0);
    println!("   Neutral signals:  {:>3} ({:.1}%)", neutral_count, neutral_count as f64 / bars.len() as f64 * 100.0);
    println!("   Direction changes: {:>3}", direction_changes);

    // Signal statistics
    let avg_signal: f64 = signals.iter().sum::<f64>() / signals.len() as f64;
    let signal_std: f64 = (signals.iter().map(|s| (s - avg_signal).powi(2)).sum::<f64>() 
        / signals.len() as f64).sqrt();

    println!("\n   Average signal: {:>+.4}", avg_signal);
    println!("   Signal std dev: {:>.4}", signal_std);

    // Visualize signal
    println!("\n📉 Signal Visualization (last 60 bars):");
    println!("   ─────────────────────────────────────────────────────────────");
    visualize_signal(&signals[signals.len().saturating_sub(60)..]);

    println!("\n✅ Demo complete!");
    println!("\n   Key insight: The pBit indicator doesn't calculate a formula.");
    println!("   Instead, it lets patterns EMERGE from collective dynamics.\n");

    Ok(())
}

/// Generate synthetic market data
fn generate_synthetic_data(n: usize) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(n);
    let mut price = 100.0;
    let mut rng = fastrand::Rng::with_seed(42);

    for i in 0..n {
        // Create trend/range phases
        let trend = if i < 50 {
            0.001  // Uptrend
        } else if i < 100 {
            -0.0005  // Downtrend
        } else if i < 150 {
            0.0  // Ranging
        } else {
            0.0008  // Uptrend again
        };

        // Add noise
        let noise = (rng.f64() - 0.5) * 0.02;
        let change = trend + noise;

        let open = price;
        let close = price * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.f64() * 0.005);
        let low = open.min(close) * (1.0 - rng.f64() * 0.005);
        let volume = 1000.0 * (1.0 + rng.f64() * 0.5 + change.abs() * 10.0);

        bars.push(Bar {
            timestamp: i as u64,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    bars
}

/// ASCII visualization of signal
fn visualize_signal(signals: &[f64]) {
    let height = 10;
    let width = signals.len().min(60);

    // Create grid
    for row in 0..height {
        let threshold = 1.0 - (row as f64 / (height - 1) as f64) * 2.0;
        print!("   {:>+.1} │", threshold);

        for col in 0..width {
            let signal = signals[col];
            let next_threshold = 1.0 - ((row + 1) as f64 / (height - 1) as f64) * 2.0;

            if signal >= next_threshold && signal < threshold {
                if signal > 0.0 {
                    print!("█");
                } else {
                    print!("▓");
                }
            } else if row == height / 2 {
                print!("─");
            } else {
                print!(" ");
            }
        }
        println!();
    }
    println!("        └────────────────────────────────────────────────────────────");
}

use fastrand;
