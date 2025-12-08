//! Panarchy Cycle Detection
//! 
//! Implementation of panarchy theory for financial markets, detecting the four phases
//! of adaptive cycles: growth, conservation, release, and reorganization.

use crate::market_data::MarketData;
use anyhow::Result;
use std::collections::VecDeque;

pub struct PanarchyDetector {
    cycle_window: usize,
    momentum_threshold: f64,
    volatility_threshold: f64,
    volume_threshold: f64,
    phase_history: VecDeque<PhaseData>,
}

#[derive(Debug, Clone)]
pub struct PanarchyPhase {
    pub current_phase: String,
    pub phase_progress: f64,      // 0.0 to 1.0
    pub cycle_maturity: f64,      // 0.0 to 1.0
    pub transition_probability: f64,
    pub next_phase: String,
    pub phase_strength: f64,
    pub adaptive_capacity: f64,
}

#[derive(Debug, Clone)]
struct PhaseData {
    phase: CyclePhase,
    strength: f64,
    duration: usize,
    timestamp: usize,
    characteristics: PhaseCharacteristics,
}

#[derive(Debug, Clone, PartialEq)]
enum CyclePhase {
    Growth,        // r - rapid growth, high potential
    Conservation,  // K - efficiency, rigidity, stable
    Release,       // Ω (Omega) - collapse, rapid change
    Reorganization,// α (Alpha) - innovation, flexibility
}

#[derive(Debug, Clone)]
struct PhaseCharacteristics {
    potential: f64,    // Available energy/resources
    connectedness: f64, // Rigidity/flexibility
    resilience: f64,   // Ability to absorb shocks
    momentum: f64,     // Rate of change
    volatility: f64,   // Uncertainty level
    volume_pattern: f64, // Trading activity pattern
}

impl PanarchyDetector {
    pub fn new() -> Self {
        Self {
            cycle_window: 100,
            momentum_threshold: 0.02,
            volatility_threshold: 0.03,
            volume_threshold: 1.5,
            phase_history: VecDeque::with_capacity(1000),
        }
    }

    pub fn detect_cycle_phase(&mut self, data: &MarketData) -> Result<String> {
        if data.len() < self.cycle_window {
            return Ok("unknown".to_string());
        }

        let phase_analysis = self.analyze_current_phase(data)?;
        self.update_phase_history(&phase_analysis);

        // Return the phase name as string for integration
        Ok(match phase_analysis.phase {
            CyclePhase::Growth => "growth",
            CyclePhase::Conservation => "conservation", 
            CyclePhase::Release => "release",
            CyclePhase::Reorganization => "reorganization",
        }.to_string())
    }

    pub fn get_detailed_phase_info(&mut self, data: &MarketData) -> Result<PanarchyPhase> {
        if data.len() < self.cycle_window {
            return Ok(PanarchyPhase::default());
        }

        let phase_analysis = self.analyze_current_phase(data)?;
        let characteristics = &phase_analysis.characteristics;

        let current_phase = match phase_analysis.phase {
            CyclePhase::Growth => "growth",
            CyclePhase::Conservation => "conservation",
            CyclePhase::Release => "release", 
            CyclePhase::Reorganization => "reorganization",
        }.to_string();

        let next_phase = self.predict_next_phase(&phase_analysis.phase);
        let transition_prob = self.calculate_transition_probability(&phase_analysis);
        let cycle_maturity = self.calculate_cycle_maturity();

        Ok(PanarchyPhase {
            current_phase,
            phase_progress: phase_analysis.strength,
            cycle_maturity,
            transition_probability: transition_prob,
            next_phase,
            phase_strength: phase_analysis.strength,
            adaptive_capacity: characteristics.resilience,
        })
    }

    fn analyze_current_phase(&self, data: &MarketData) -> Result<PhaseData> {
        let characteristics = self.calculate_phase_characteristics(data)?;
        let (phase, strength) = self.classify_phase(&characteristics);

        Ok(PhaseData {
            phase,
            strength,
            duration: 1, // Will be updated when tracking history
            timestamp: data.len() - 1,
            characteristics,
        })
    }

    fn calculate_phase_characteristics(&self, data: &MarketData) -> Result<PhaseCharacteristics> {
        let window_start = data.len().saturating_sub(self.cycle_window);
        let window_prices = &data.prices[window_start..];
        let window_volumes = &data.volumes[window_start..];

        // Calculate potential (trend strength and momentum)
        let potential = self.calculate_potential(window_prices)?;
        
        // Calculate connectedness (correlation and interdependence)
        let connectedness = self.calculate_connectedness(data)?;
        
        // Calculate resilience (ability to absorb shocks)
        let resilience = self.calculate_resilience(data)?;
        
        // Calculate momentum (rate of change)
        let momentum = self.calculate_momentum(window_prices)?;
        
        // Calculate volatility (uncertainty)
        let volatility = self.calculate_phase_volatility(window_prices)?;
        
        // Calculate volume pattern
        let volume_pattern = self.calculate_volume_pattern(window_volumes)?;

        Ok(PhaseCharacteristics {
            potential,
            connectedness,
            resilience,
            momentum,
            volatility,
            volume_pattern,
        })
    }

    fn calculate_potential(&self, prices: &[f64]) -> Result<f64> {
        if prices.len() < 20 {
            return Ok(0.5);
        }

        // Potential = trend strength + price level relative to range
        let trend = self.calculate_linear_trend(prices);
        let price_range = self.calculate_price_position(prices);
        
        // Normalize trend component
        let trend_component = (trend + 1.0) / 2.0; // -1 to 1 -> 0 to 1
        
        // Combine components
        let potential = (trend_component * 0.7 + price_range * 0.3).clamp(0.0, 1.0);
        
        Ok(potential)
    }

    fn calculate_connectedness(&self, data: &MarketData) -> Result<f64> {
        if data.len() < 50 {
            return Ok(0.5);
        }

        // Connectedness = correlation stability + volume consistency
        let price_autocorr = self.calculate_price_autocorrelation(&data.prices);
        let volume_consistency = self.calculate_volume_consistency(&data.volumes);
        
        let connectedness = (price_autocorr * 0.6 + volume_consistency * 0.4).clamp(0.0, 1.0);
        
        Ok(connectedness)
    }

    fn calculate_resilience(&self, data: &MarketData) -> Result<f64> {
        if data.len() < 30 {
            return Ok(0.5);
        }

        // Resilience = recovery speed + volatility absorption
        let recovery_speed = self.calculate_recovery_speed(data)?;
        let volatility_absorption = self.calculate_volatility_absorption(data)?;
        
        let resilience = (recovery_speed * 0.6 + volatility_absorption * 0.4).clamp(0.0, 1.0);
        
        Ok(resilience)
    }

    fn calculate_momentum(&self, prices: &[f64]) -> Result<f64> {
        if prices.len() < 10 {
            return Ok(0.0);
        }

        // Calculate rate of change over multiple timeframes
        let short_momentum = self.calculate_roc(prices, 5);
        let medium_momentum = self.calculate_roc(prices, 10);
        let long_momentum = self.calculate_roc(prices, 20);

        // Weighted combination
        let momentum = (short_momentum * 0.5 + medium_momentum * 0.3 + long_momentum * 0.2).abs();
        
        Ok(momentum.min(1.0))
    }

    fn calculate_phase_volatility(&self, prices: &[f64]) -> Result<f64> {
        if prices.len() < 20 {
            return Ok(0.0);
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;

        let volatility = variance.sqrt();
        
        Ok((volatility * 100.0).min(1.0)) // Scale and cap
    }

    fn calculate_volume_pattern(&self, volumes: &[f64]) -> Result<f64> {
        if volumes.len() < 20 {
            return Ok(0.5);
        }

        // Analyze volume distribution and consistency
        let recent_avg = volumes.iter().rev().take(10).sum::<f64>() / 10.0;
        let historical_avg = volumes.iter().sum::<f64>() / volumes.len() as f64;
        
        let volume_ratio = if historical_avg > 0.0 {
            recent_avg / historical_avg
        } else {
            1.0
        };

        // Normalize volume pattern
        let pattern = ((volume_ratio - 1.0).abs()).min(2.0) / 2.0;
        
        Ok(pattern)
    }

    fn classify_phase(&self, chars: &PhaseCharacteristics) -> (CyclePhase, f64) {
        // Growth phase: High potential, low connectedness, building momentum
        let growth_score = chars.potential * 0.4 + 
                          (1.0 - chars.connectedness) * 0.3 + 
                          chars.momentum * 0.3;

        // Conservation phase: High potential, high connectedness, low volatility
        let conservation_score = chars.potential * 0.3 + 
                                chars.connectedness * 0.4 + 
                                (1.0 - chars.volatility) * 0.3;

        // Release phase: High connectedness, high volatility, negative momentum
        let release_score = chars.connectedness * 0.3 + 
                           chars.volatility * 0.4 + 
                           (1.0 - chars.resilience) * 0.3;

        // Reorganization phase: Low connectedness, high resilience, moderate volatility
        let reorg_score = (1.0 - chars.connectedness) * 0.4 + 
                         chars.resilience * 0.3 + 
                         chars.volatility * 0.3;

        // Find dominant phase
        let scores = vec![
            (CyclePhase::Growth, growth_score),
            (CyclePhase::Conservation, conservation_score),
            (CyclePhase::Release, release_score),
            (CyclePhase::Reorganization, reorg_score),
        ];

        let (phase, strength) = scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((CyclePhase::Growth, 0.5));

        (phase, strength)
    }

    fn predict_next_phase(&self, current: &CyclePhase) -> String {
        match current {
            CyclePhase::Growth => "conservation",
            CyclePhase::Conservation => "release",
            CyclePhase::Release => "reorganization",
            CyclePhase::Reorganization => "growth",
        }.to_string()
    }

    fn calculate_transition_probability(&self, phase_data: &PhaseData) -> f64 {
        // Higher probability if phase is mature (high strength) and showing signs of stress
        let maturity_factor = phase_data.strength;
        let stress_factor = match phase_data.phase {
            CyclePhase::Growth => phase_data.characteristics.connectedness, // Growing rigidity
            CyclePhase::Conservation => phase_data.characteristics.volatility, // Increasing instability  
            CyclePhase::Release => 1.0 - phase_data.characteristics.momentum, // Losing momentum
            CyclePhase::Reorganization => phase_data.characteristics.potential, // Building potential
        };

        (maturity_factor * 0.6 + stress_factor * 0.4).clamp(0.0, 1.0)
    }

    fn calculate_cycle_maturity(&self) -> f64 {
        if self.phase_history.len() < 4 {
            return 0.0;
        }

        // Estimate how far through the complete cycle we are
        let recent_phases: Vec<_> = self.phase_history.iter()
            .rev()
            .take(20)
            .map(|p| &p.phase)
            .collect();

        // Look for phase transitions
        let mut phase_changes = 0;
        let mut current_phase_duration = 1;

        for i in 1..recent_phases.len() {
            if recent_phases[i] != recent_phases[i-1] {
                phase_changes += 1;
                if i == 1 {
                    current_phase_duration = 1;
                }
            } else if i == 1 {
                current_phase_duration += 1;
            }
        }

        // Maturity based on phase changes and duration
        let transition_maturity = (phase_changes as f64 / 4.0).min(1.0); // 4 phases in full cycle
        let duration_maturity = (current_phase_duration as f64 / 20.0).min(1.0);

        (transition_maturity * 0.7 + duration_maturity * 0.3)
    }

    fn update_phase_history(&mut self, phase_data: &PhaseData) {
        // Update duration if same phase continues
        if let Some(last_phase) = self.phase_history.back_mut() {
            if last_phase.phase == phase_data.phase {
                last_phase.duration += 1;
                last_phase.strength = (last_phase.strength + phase_data.strength) / 2.0;
                return;
            }
        }

        // Add new phase
        self.phase_history.push_back(phase_data.clone());

        // Keep history manageable
        while self.phase_history.len() > 1000 {
            self.phase_history.pop_front();
        }
    }

    // Helper methods
    fn calculate_linear_trend(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let n = prices.len() as f64;
        let x_sum: f64 = (0..prices.len()).map(|i| i as f64).sum();
        let y_sum: f64 = prices.iter().sum();
        let xy_sum: f64 = prices.iter().enumerate()
            .map(|(i, &price)| i as f64 * price)
            .sum();
        let x2_sum: f64 = (0..prices.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * x2_sum - x_sum * x_sum;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (n * xy_sum - x_sum * y_sum) / denominator
    }

    fn calculate_price_position(&self, prices: &[f64]) -> f64 {
        if prices.is_empty() {
            return 0.5;
        }

        let min_price = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_price = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let current_price = prices[prices.len() - 1];

        if max_price > min_price {
            (current_price - min_price) / (max_price - min_price)
        } else {
            0.5
        }
    }

    fn calculate_price_autocorrelation(&self, prices: &[f64]) -> f64 {
        if prices.len() < 20 {
            return 0.5;
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        self.autocorrelation(&returns, 1).abs()
    }

    fn calculate_volume_consistency(&self, volumes: &[f64]) -> f64 {
        if volumes.len() < 20 {
            return 0.5;
        }

        let mean = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let cv = if mean > 0.0 {
            let std = self.standard_deviation(volumes, mean);
            std / mean
        } else {
            1.0
        };

        (1.0 - cv.min(1.0)).max(0.0)
    }

    fn calculate_recovery_speed(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        if returns.len() < 30 {
            return Ok(0.5);
        }

        // Find drawdowns and recovery times
        let mut recovery_times = Vec::new();
        let mut drawdown_start = None;
        let mut peak = 0.0;

        for (i, &ret) in returns.iter().enumerate() {
            if ret < -0.02 && drawdown_start.is_none() {
                drawdown_start = Some(i);
                peak = 0.0;
            }

            if let Some(start) = drawdown_start {
                peak += ret;
                if peak > 0.0 { // Recovery
                    recovery_times.push(i - start);
                    drawdown_start = None;
                }
            }
        }

        if recovery_times.is_empty() {
            return Ok(0.5);
        }

        let avg_recovery = recovery_times.iter().sum::<usize>() as f64 / recovery_times.len() as f64;
        Ok((1.0 - (avg_recovery / 20.0).min(1.0)).max(0.0))
    }

    fn calculate_volatility_absorption(&self, data: &MarketData) -> Result<f64> {
        let returns = data.returns();
        if returns.len() < 20 {
            return Ok(0.5);
        }

        // Measure how well the system absorbs volatility shocks
        let volatility = self.standard_deviation(&returns, 0.0);
        let extreme_returns = returns.iter()
            .filter(|&&r| r.abs() > 2.0 * volatility)
            .count();

        let absorption = 1.0 - (extreme_returns as f64 / returns.len() as f64);
        Ok(absorption.clamp(0.0, 1.0))
    }

    fn calculate_roc(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() <= period {
            return 0.0;
        }

        let current = prices[prices.len() - 1];
        let past = prices[prices.len() - 1 - period];
        
        if past != 0.0 {
            (current - past) / past
        } else {
            0.0
        }
    }

    fn autocorrelation(&self, data: &[f64], lag: usize) -> f64 {
        if data.len() <= lag {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in lag..data.len() {
            numerator += (data[i] - mean) * (data[i - lag] - mean);
        }

        for &value in data {
            denominator += (value - mean).powi(2);
        }

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn standard_deviation(&self, data: &[f64], mean: f64) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }

        let actual_mean = if mean == 0.0 {
            data.iter().sum::<f64>() / data.len() as f64
        } else {
            mean
        };

        let variance = data.iter()
            .map(|&x| (x - actual_mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;

        variance.sqrt()
    }
}

impl Default for PanarchyPhase {
    fn default() -> Self {
        Self {
            current_phase: "unknown".to_string(),
            phase_progress: 0.0,
            cycle_maturity: 0.0,
            transition_probability: 0.0,
            next_phase: "unknown".to_string(),
            phase_strength: 0.0,
            adaptive_capacity: 0.0,
        }
    }
}

impl Default for PanarchyDetector {
    fn default() -> Self {
        Self::new()
    }
}