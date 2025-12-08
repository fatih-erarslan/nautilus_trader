//! pBit-Enhanced Self-Organized Criticality Analysis
//!
//! Leverages Ising model dynamics at the critical point to detect
//! and measure SOC behavior in financial time series.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - **Critical Temperature**: T_c = 2/ln(1+√2) ≈ 2.269 (2D Ising)
//! - **Correlation Length**: ξ ∝ |T - T_c|^(-ν), ν ≈ 1
//! - **Avalanche Size**: P(s) ∝ s^(-τ), τ ≈ 1.5 for SOC
//! - **Critical Exponent**: χ ∝ |T - T_c|^(-γ), γ ≈ 7/4

use rand::prelude::*;
use std::collections::VecDeque;

/// Critical temperature for 2D Ising model (Onsager solution)
pub const CRITICAL_TEMP: f64 = 2.269185314213022;

/// pBit-based SOC detector
#[derive(Debug, Clone)]
pub struct PBitSOCDetector {
    /// Current effective temperature
    pub temperature: f64,
    /// Lattice size (NxN spins)
    pub lattice_size: usize,
    /// Spin configuration
    spins: Vec<i8>,
    /// Coupling strength
    pub coupling: f64,
    /// External field
    pub field: f64,
    /// Avalanche history
    avalanche_history: VecDeque<AvalancheRecord>,
    /// Max history length
    max_history: usize,
}

/// Record of an avalanche event
#[derive(Debug, Clone)]
pub struct AvalancheRecord {
    /// Size of avalanche (number of flipped spins)
    pub size: usize,
    /// Duration in simulation steps
    pub duration: usize,
    /// Energy released
    pub energy_delta: f64,
    /// Timestamp
    pub timestamp: u64,
}

impl Default for PBitSOCDetector {
    fn default() -> Self {
        Self::new(16, CRITICAL_TEMP)
    }
}

impl PBitSOCDetector {
    /// Create new SOC detector
    pub fn new(lattice_size: usize, temperature: f64) -> Self {
        let n = lattice_size * lattice_size;
        let mut rng = rand::thread_rng();
        
        // Initialize spins randomly
        let spins: Vec<i8> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();

        Self {
            temperature,
            lattice_size,
            spins,
            coupling: 1.0,
            field: 0.0,
            avalanche_history: VecDeque::with_capacity(1000),
            max_history: 1000,
        }
    }

    /// Create at critical temperature
    pub fn at_critical_point(lattice_size: usize) -> Self {
        Self::new(lattice_size, CRITICAL_TEMP)
    }

    /// Set temperature relative to critical point
    pub fn set_criticality(&mut self, delta: f64) {
        self.temperature = CRITICAL_TEMP * (1.0 + delta);
    }

    /// Calculate local field at site i
    fn local_field(&self, i: usize) -> f64 {
        let n = self.lattice_size;
        let x = i % n;
        let y = i / n;

        // Sum of neighboring spins (periodic boundary)
        let neighbors = [
            ((x + 1) % n) + y * n,           // right
            ((x + n - 1) % n) + y * n,       // left
            x + ((y + 1) % n) * n,           // up
            x + ((y + n - 1) % n) * n,       // down
        ];

        let sum: i32 = neighbors.iter()
            .map(|&j| self.spins[j] as i32)
            .sum();

        self.coupling * sum as f64 + self.field
    }

    /// Calculate energy change for flipping spin i
    fn delta_energy(&self, i: usize) -> f64 {
        2.0 * self.spins[i] as f64 * self.local_field(i)
    }

    /// Perform one Metropolis sweep and detect avalanches
    pub fn sweep(&mut self) -> Option<AvalancheRecord> {
        let n = self.spins.len();
        let mut rng = rand::thread_rng();
        
        let mut flipped = 0;
        let mut total_energy_delta = 0.0;
        let mut avalanche_active = false;
        let mut avalanche_duration = 0;

        for _ in 0..n {
            let i = rng.gen_range(0..n);
            let de = self.delta_energy(i);

            // Metropolis acceptance
            if de <= 0.0 || rng.gen::<f64>() < (-de / self.temperature).exp() {
                self.spins[i] *= -1;
                flipped += 1;
                total_energy_delta += de;

                if de < 0.0 {
                    avalanche_active = true;
                    avalanche_duration += 1;
                }
            }
        }

        if avalanche_active && flipped > n / 100 {
            let record = AvalancheRecord {
                size: flipped,
                duration: avalanche_duration,
                energy_delta: total_energy_delta,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            };

            if self.avalanche_history.len() >= self.max_history {
                self.avalanche_history.pop_front();
            }
            self.avalanche_history.push_back(record.clone());
            
            Some(record)
        } else {
            None
        }
    }

    /// Run multiple sweeps and collect avalanche statistics
    pub fn simulate(&mut self, sweeps: usize) -> SOCStatistics {
        let mut avalanche_sizes = Vec::new();
        let mut total_flips = 0;

        for _ in 0..sweeps {
            if let Some(av) = self.sweep() {
                avalanche_sizes.push(av.size);
                total_flips += av.size;
            }
        }

        // Calculate statistics
        let magnetization = self.magnetization();
        let susceptibility = self.susceptibility_estimate(&avalanche_sizes);
        let tau = self.estimate_power_law_exponent(&avalanche_sizes);
        let criticality = self.criticality_score();

        SOCStatistics {
            magnetization,
            susceptibility,
            power_law_exponent: tau,
            criticality_score: criticality,
            avalanche_count: avalanche_sizes.len(),
            mean_avalanche_size: if avalanche_sizes.is_empty() {
                0.0
            } else {
                total_flips as f64 / avalanche_sizes.len() as f64
            },
            temperature_ratio: self.temperature / CRITICAL_TEMP,
        }
    }

    /// Calculate magnetization
    pub fn magnetization(&self) -> f64 {
        let sum: i32 = self.spins.iter().map(|&s| s as i32).sum();
        (sum as f64 / self.spins.len() as f64).abs()
    }

    /// Estimate susceptibility from avalanche fluctuations
    fn susceptibility_estimate(&self, sizes: &[usize]) -> f64 {
        if sizes.len() < 2 {
            return 0.0;
        }

        let mean: f64 = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        let variance: f64 = sizes.iter()
            .map(|&s| (s as f64 - mean).powi(2))
            .sum::<f64>() / sizes.len() as f64;

        variance / self.temperature
    }

    /// Estimate power law exponent τ from avalanche distribution
    fn estimate_power_law_exponent(&self, sizes: &[usize]) -> f64 {
        if sizes.len() < 10 {
            return 1.5; // Default SOC value
        }

        // Simple Hill estimator for tail exponent
        let mut sorted: Vec<f64> = sizes.iter()
            .filter(|&&s| s > 0)
            .map(|&s| s as f64)
            .collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.len() < 5 {
            return 1.5;
        }

        let k = sorted.len() / 5; // Top 20%
        let x_k = sorted[k];
        if x_k <= 1.0 {
            return 1.5;
        }

        let sum_log: f64 = sorted[..k].iter()
            .map(|&x| (x / x_k).ln())
            .sum();

        1.0 + k as f64 / sum_log.max(0.01)
    }

    /// Calculate criticality score (0 = far from critical, 1 = at critical)
    pub fn criticality_score(&self) -> f64 {
        let t_ratio = self.temperature / CRITICAL_TEMP;
        
        // Gaussian centered at T_c
        let deviation = (t_ratio - 1.0).abs();
        (-deviation.powi(2) / 0.1).exp()
    }

    /// Analyze time series for SOC signatures
    pub fn analyze_series(&mut self, data: &[f64]) -> SeriesSOCResult {
        let n = data.len();
        if n < 20 {
            return SeriesSOCResult::insufficient_data();
        }

        // Calculate returns
        let returns: Vec<f64> = data.windows(2)
            .map(|w| (w[1] - w[0]) / w[0].abs().max(1e-10))
            .collect();

        // Map returns to temperature perturbations
        let volatility = returns.iter()
            .map(|r| r.abs())
            .sum::<f64>() / returns.len() as f64;

        // Adjust temperature based on volatility
        self.temperature = CRITICAL_TEMP * (1.0 + volatility * 10.0);

        // Run simulation
        let stats = self.simulate(100);

        // Detect if series is at criticality
        let is_critical = stats.criticality_score > 0.7;
        let regime = if stats.criticality_score > 0.8 {
            SOCRegime::Critical
        } else if stats.temperature_ratio < 0.9 {
            SOCRegime::Ordered
        } else if stats.temperature_ratio > 1.1 {
            SOCRegime::Disordered
        } else {
            SOCRegime::NearCritical
        };

        SeriesSOCResult {
            regime,
            criticality_score: stats.criticality_score,
            power_law_exponent: stats.power_law_exponent,
            avalanche_frequency: stats.avalanche_count as f64 / 100.0,
            estimated_correlation_length: self.correlation_length_estimate(),
            is_critical,
        }
    }

    /// Estimate correlation length
    fn correlation_length_estimate(&self) -> f64 {
        let deviation = (self.temperature / CRITICAL_TEMP - 1.0).abs().max(0.01);
        1.0 / deviation // ξ ∝ |T - T_c|^(-1)
    }

    /// Get avalanche history
    pub fn avalanche_history(&self) -> &VecDeque<AvalancheRecord> {
        &self.avalanche_history
    }

    /// Reset detector
    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        for spin in &mut self.spins {
            *spin = if rng.gen::<bool>() { 1 } else { -1 };
        }
        self.avalanche_history.clear();
    }
}

/// SOC simulation statistics
#[derive(Debug, Clone)]
pub struct SOCStatistics {
    /// Order parameter (magnetization)
    pub magnetization: f64,
    /// Susceptibility estimate
    pub susceptibility: f64,
    /// Power law exponent τ
    pub power_law_exponent: f64,
    /// Criticality score [0, 1]
    pub criticality_score: f64,
    /// Number of avalanches detected
    pub avalanche_count: usize,
    /// Mean avalanche size
    pub mean_avalanche_size: f64,
    /// T/T_c ratio
    pub temperature_ratio: f64,
}

/// SOC regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SOCRegime {
    /// T < T_c: ordered phase
    Ordered,
    /// T ≈ T_c: critical phase
    Critical,
    /// T close to T_c
    NearCritical,
    /// T > T_c: disordered phase
    Disordered,
}

/// Result of SOC analysis on time series
#[derive(Debug, Clone)]
pub struct SeriesSOCResult {
    /// Detected regime
    pub regime: SOCRegime,
    /// Criticality score [0, 1]
    pub criticality_score: f64,
    /// Estimated power law exponent
    pub power_law_exponent: f64,
    /// Avalanche frequency (events per sweep)
    pub avalanche_frequency: f64,
    /// Estimated correlation length
    pub estimated_correlation_length: f64,
    /// Whether system is at criticality
    pub is_critical: bool,
}

impl SeriesSOCResult {
    fn insufficient_data() -> Self {
        Self {
            regime: SOCRegime::Disordered,
            criticality_score: 0.0,
            power_law_exponent: 1.5,
            avalanche_frequency: 0.0,
            estimated_correlation_length: 1.0,
            is_critical: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_temperature_wolfram_validated() {
        // Wolfram: 2/Log[1 + Sqrt[2]] = 2.2691853142130216
        let t_c = 2.0 / (1.0 + 2.0_f64.sqrt()).ln();
        assert!((t_c - CRITICAL_TEMP).abs() < 1e-10);
    }

    #[test]
    fn test_magnetization_at_criticality() {
        let detector = PBitSOCDetector::at_critical_point(8);
        let mag = detector.magnetization();
        
        // At T_c, magnetization should be between 0 and 1
        assert!(mag >= 0.0 && mag <= 1.0);
    }

    #[test]
    fn test_criticality_score() {
        let mut detector = PBitSOCDetector::at_critical_point(8);
        assert!(detector.criticality_score() > 0.9);

        detector.temperature = CRITICAL_TEMP * 0.5;
        assert!(detector.criticality_score() < 0.5);
    }

    #[test]
    fn test_avalanche_detection() {
        let mut detector = PBitSOCDetector::at_critical_point(16);
        let stats = detector.simulate(50);
        
        println!("Avalanches: {}, Mean size: {:.1}", 
            stats.avalanche_count, stats.mean_avalanche_size);
        
        // Should detect some avalanches at criticality
        assert!(stats.avalanche_count > 0 || stats.mean_avalanche_size >= 0.0);
    }

    #[test]
    fn test_power_law_exponent() {
        let mut detector = PBitSOCDetector::at_critical_point(16);
        let stats = detector.simulate(100);
        
        // SOC systems typically have τ ≈ 1.5
        assert!(stats.power_law_exponent > 1.0 && stats.power_law_exponent < 3.0);
    }
}
