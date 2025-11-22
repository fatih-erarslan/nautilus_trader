use std::collections::HashMap;
use nalgebra::{DVector, DMatrix};
use rand::{Rng, thread_rng};

/// Self-Organized Criticality system for modeling market avalanches and emergent behavior
/// Based on the Bak-Tang-Wiesenfeld sandpile model and extended for economic systems
pub struct SelfOrganizedCriticality {
    /// Grid representing system state (e.g., market pressure, agent density)
    grid: DMatrix<f64>,
    /// Critical threshold for avalanche initiation
    critical_threshold: f64,
    /// Avalanche history for pattern analysis
    avalanche_history: Vec<AvalancheEvent>,
    /// System parameters
    params: SocParameters,
}

#[derive(Clone, Debug)]
pub struct AvalancheEvent {
    /// Size of the avalanche (number of affected cells)
    pub size: usize,
    /// Duration of the avalanche
    pub duration: f64,
    /// Starting position
    pub origin: (usize, usize),
    /// Energy dissipated
    pub energy: f64,
    /// Timestamp
    pub timestamp: f64,
}

#[derive(Clone, Debug)]
pub struct SocParameters {
    /// Grid dimensions
    pub width: usize,
    pub height: usize,
    /// Critical threshold
    pub threshold: f64,
    /// Energy redistribution factor
    pub redistribution_factor: f64,
    /// Dissipation rate
    pub dissipation_rate: f64,
    /// External drive rate
    pub drive_rate: f64,
}

impl Default for SocParameters {
    fn default() -> Self {
        Self {
            width: 100,
            height: 100,
            threshold: 4.0,
            redistribution_factor: 0.25,
            dissipation_rate: 0.01,
            drive_rate: 0.1,
        }
    }
}

impl SelfOrganizedCriticality {
    /// Create new SOC system
    pub fn new(params: SocParameters) -> Self {
        let grid = DMatrix::zeros(params.height, params.width);
        
        Self {
            grid,
            critical_threshold: params.threshold,
            avalanche_history: Vec::new(),
            params,
        }
    }

    /// Initialize with random state
    pub fn initialize_random(&mut self) {
        let mut rng = thread_rng();
        for i in 0..self.params.height {
            for j in 0..self.params.width {
                self.grid[(i, j)] = rng.gen::<f64>() * self.critical_threshold * 0.8;
            }
        }
    }

    /// Add energy to random location (external drive)
    pub fn add_energy(&mut self, amount: f64) {
        let mut rng = thread_rng();
        let i = rng.gen_range(0..self.params.height);
        let j = rng.gen_range(0..self.params.width);
        self.grid[(i, j)] += amount;
    }

    /// Perform one simulation step
    pub fn step(&mut self, timestamp: f64) -> Option<AvalancheEvent> {
        // Add external drive
        self.add_energy(self.params.drive_rate);

        // Check for criticality and trigger avalanche
        if let Some(critical_site) = self.find_critical_site() {
            return Some(self.trigger_avalanche(critical_site, timestamp));
        }

        // Apply dissipation
        self.apply_dissipation();

        None
    }

    /// Find site that exceeds critical threshold
    fn find_critical_site(&self) -> Option<(usize, usize)> {
        for i in 0..self.params.height {
            for j in 0..self.params.width {
                if self.grid[(i, j)] >= self.critical_threshold {
                    return Some((i, j));
                }
            }
        }
        None
    }

    /// Trigger avalanche from critical site
    fn trigger_avalanche(&mut self, origin: (usize, usize), timestamp: f64) -> AvalancheEvent {
        let mut avalanche_size = 0;
        let mut total_energy = 0.0;
        let mut active_sites = vec![origin];
        let mut visited = vec![vec![false; self.params.width]; self.params.height];
        
        let start_time = std::time::Instant::now();

        while !active_sites.is_empty() {
            let mut new_active_sites = Vec::new();

            for &(i, j) in &active_sites {
                if visited[i][j] || self.grid[(i, j)] < self.critical_threshold {
                    continue;
                }

                visited[i][j] = true;
                avalanche_size += 1;
                
                // Redistribute energy to neighbors
                let excess_energy = self.grid[(i, j)] - self.critical_threshold;
                total_energy += excess_energy;
                self.grid[(i, j)] = self.critical_threshold * 0.1; // Reset to low value

                // Distribute to neighbors
                let neighbors = self.get_neighbors(i, j);
                let energy_per_neighbor = excess_energy * self.params.redistribution_factor / neighbors.len() as f64;

                for (ni, nj) in neighbors {
                    self.grid[(ni, nj)] += energy_per_neighbor;
                    if self.grid[(ni, nj)] >= self.critical_threshold {
                        new_active_sites.push((ni, nj));
                    }
                }
            }

            active_sites = new_active_sites;
        }

        let duration = start_time.elapsed().as_secs_f64();

        let event = AvalancheEvent {
            size: avalanche_size,
            duration,
            origin,
            energy: total_energy,
            timestamp,
        };

        self.avalanche_history.push(event.clone());
        event
    }

    /// Get valid neighbors for a cell
    fn get_neighbors(&self, i: usize, j: usize) -> Vec<(usize, usize)> {
        let mut neighbors = Vec::new();
        
        for di in -1i32..=1 {
            for dj in -1i32..=1 {
                if di == 0 && dj == 0 { continue; }
                
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                
                if ni >= 0 && ni < self.params.height as i32 && 
                   nj >= 0 && nj < self.params.width as i32 {
                    neighbors.push((ni as usize, nj as usize));
                }
            }
        }
        
        neighbors
    }

    /// Apply energy dissipation across the grid
    fn apply_dissipation(&mut self) {
        for i in 0..self.params.height {
            for j in 0..self.params.width {
                self.grid[(i, j)] *= 1.0 - self.params.dissipation_rate;
            }
        }
    }

    /// Calculate power law distribution of avalanche sizes
    pub fn analyze_power_law(&self) -> PowerLawAnalysis {
        if self.avalanche_history.is_empty() {
            return PowerLawAnalysis::default();
        }

        let mut size_counts: HashMap<usize, usize> = HashMap::new();
        for event in &self.avalanche_history {
            *size_counts.entry(event.size).or_insert(0) += 1;
        }

        let mut sizes: Vec<usize> = size_counts.keys().cloned().collect();
        sizes.sort();

        let mut log_sizes = Vec::new();
        let mut log_counts = Vec::new();

        for size in sizes {
            if let Some(&count) = size_counts.get(&size) {
                log_sizes.push((size as f64).ln());
                log_counts.push((count as f64).ln());
            }
        }

        // Simple linear regression to estimate power law exponent
        let exponent = if log_sizes.len() > 1 {
            self.linear_regression(&log_sizes, &log_counts).0
        } else {
            0.0
        };

        PowerLawAnalysis {
            exponent,
            r_squared: self.calculate_r_squared(&log_sizes, &log_counts, exponent),
            sample_size: self.avalanche_history.len(),
            size_range: (
                self.avalanche_history.iter().map(|e| e.size).min().unwrap_or(0),
                self.avalanche_history.iter().map(|e| e.size).max().unwrap_or(0)
            ),
        }
    }

    /// Simple linear regression
    fn linear_regression(&self, x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }

    /// Calculate R-squared for goodness of fit
    fn calculate_r_squared(&self, x: &[f64], y: &[f64], slope: f64) -> f64 {
        if x.is_empty() { return 0.0; }

        let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        let (_, intercept) = self.linear_regression(x, y);

        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(y.iter())
            .map(|(xi, yi)| {
                let y_pred = slope * xi + intercept;
                (yi - y_pred).powi(2)
            })
            .sum();

        if ss_tot == 0.0 { 0.0 } else { 1.0 - ss_res / ss_tot }
    }

    /// Get current system state
    pub fn get_state(&self) -> SocState {
        let total_energy: f64 = self.grid.iter().sum();
        let max_energy = self.grid.iter().fold(0.0, |max, &val| val.max(max));
        let critical_sites = self.count_critical_sites();

        SocState {
            total_energy,
            max_energy,
            critical_sites,
            avalanche_count: self.avalanche_history.len(),
            grid_snapshot: self.grid.clone(),
        }
    }

    /// Count sites near criticality
    fn count_critical_sites(&self) -> usize {
        let mut count = 0;
        let threshold = self.critical_threshold * 0.9; // Near-critical threshold
        
        for i in 0..self.params.height {
            for j in 0..self.params.width {
                if self.grid[(i, j)] >= threshold {
                    count += 1;
                }
            }
        }
        
        count
    }

    /// Reset system state
    pub fn reset(&mut self) {
        self.grid = DMatrix::zeros(self.params.height, self.params.width);
        self.avalanche_history.clear();
    }

    /// Get avalanche history
    pub fn get_avalanche_history(&self) -> &[AvalancheEvent] {
        &self.avalanche_history
    }
}

#[derive(Clone, Debug, Default)]
pub struct PowerLawAnalysis {
    /// Power law exponent (negative for proper power law)
    pub exponent: f64,
    /// R-squared goodness of fit
    pub r_squared: f64,
    /// Sample size
    pub sample_size: usize,
    /// Range of avalanche sizes
    pub size_range: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct SocState {
    /// Total energy in the system
    pub total_energy: f64,
    /// Maximum energy at any site
    pub max_energy: f64,
    /// Number of near-critical sites
    pub critical_sites: usize,
    /// Total number of avalanches recorded
    pub avalanche_count: usize,
    /// Current grid state
    pub grid_snapshot: DMatrix<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soc_initialization() {
        let params = SocParameters::default();
        let mut soc = SelfOrganizedCriticality::new(params);
        soc.initialize_random();
        
        let state = soc.get_state();
        assert!(state.total_energy > 0.0);
        assert!(state.max_energy < soc.critical_threshold);
    }

    #[test]
    fn test_avalanche_triggering() {
        let mut params = SocParameters::default();
        params.width = 10;
        params.height = 10;
        params.threshold = 2.0;
        
        let mut soc = SelfOrganizedCriticality::new(params);
        
        // Manually set a critical site
        soc.grid[(5, 5)] = 3.0; // Above threshold
        
        let avalanche = soc.step(0.0);
        assert!(avalanche.is_some());
        
        let event = avalanche.unwrap();
        assert!(event.size > 0);
        assert_eq!(event.origin, (5, 5));
    }

    #[test]
    fn test_power_law_analysis() {
        let params = SocParameters::default();
        let mut soc = SelfOrganizedCriticality::new(params);
        
        // Simulate multiple avalanches
        for _ in 0..100 {
            soc.add_energy(1.0);
            soc.step(0.0);
        }
        
        let analysis = soc.analyze_power_law();
        assert!(analysis.sample_size <= 100);
        assert!(analysis.exponent != 0.0 || analysis.sample_size == 0);
    }
}