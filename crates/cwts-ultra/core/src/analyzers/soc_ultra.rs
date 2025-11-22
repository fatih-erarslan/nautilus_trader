// Self-Organized Criticality Ultra Analyzer - REAL IMPLEMENTATION
use std::collections::{VecDeque, HashMap};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::{E, PI};

/// Self-Organized Criticality detector for market avalanches
pub struct SOCAnalyzer {
    // Sandpile model parameters
    grid_size: usize,
    sandpile: DMatrix<f64>,
    critical_threshold: f64,
    
    // Avalanche tracking
    avalanche_history: VecDeque<Avalanche>,
    current_avalanche: Option<Avalanche>,
    
    // Power law detection
    avalanche_sizes: Vec<f64>,
    power_law_exponent: f64,
    
    // Market microstructure
    order_flow_imbalance: VecDeque<f64>,
    price_returns: VecDeque<f64>,
    volume_bursts: VecDeque<f64>,
    
    // Network topology
    correlation_matrix: DMatrix<f64>,
    adjacency_matrix: DMatrix<bool>,
    
    // Critical state indicators
    susceptibility: f64,
    branching_ratio: f64,
    correlation_length: f64,
    
    // Parameters
    history_window: usize,
    detection_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct Avalanche {
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub size: f64,
    pub duration: u64,
    pub affected_nodes: Vec<(usize, usize)>,
    pub trigger_location: (usize, usize),
    pub propagation_path: Vec<(usize, usize)>,
    pub energy_dissipated: f64,
}

#[derive(Debug, Clone)]
pub struct CriticalityState {
    pub is_critical: bool,
    pub criticality_score: f64,
    pub avalanche_probability: f64,
    pub expected_avalanche_size: f64,
    pub warning_level: WarningLevel,
    pub phase: MarketPhase,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WarningLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketPhase {
    Subcritical,    // Stable, small fluctuations
    Critical,       // Power-law avalanches
    Supercritical,  // System-wide cascades
}

impl SOCAnalyzer {
    pub fn new(grid_size: usize, history_window: usize) -> Self {
        Self {
            grid_size,
            sandpile: DMatrix::zeros(grid_size, grid_size),
            critical_threshold: 4.0, // Classic BTW model threshold
            avalanche_history: VecDeque::with_capacity(history_window),
            current_avalanche: None,
            avalanche_sizes: Vec::new(),
            power_law_exponent: 1.5, // Typical for financial markets
            order_flow_imbalance: VecDeque::with_capacity(history_window),
            price_returns: VecDeque::with_capacity(history_window),
            volume_bursts: VecDeque::with_capacity(history_window),
            correlation_matrix: DMatrix::identity(grid_size, grid_size),
            adjacency_matrix: DMatrix::from_element(grid_size, grid_size, true),
            susceptibility: 0.0,
            branching_ratio: 1.0,
            correlation_length: 0.0,
            history_window,
            detection_threshold: 0.7,
        }
    }
    
    /// Add market event (grain of sand)
    pub fn add_event(&mut self, x: usize, y: usize, magnitude: f64, timestamp: u64) -> Option<CriticalityState> {
        // Add sand grain to the pile
        self.sandpile[(x, y)] += magnitude;
        
        // Check for avalanche
        if self.sandpile[(x, y)] >= self.critical_threshold {
            let avalanche = self.trigger_avalanche(x, y, timestamp);
            self.avalanche_history.push_back(avalanche.clone());
            
            if self.avalanche_history.len() > self.history_window {
                self.avalanche_history.pop_front();
            }
            
            self.avalanche_sizes.push(avalanche.size);
            
            // Update criticality metrics
            self.update_criticality_metrics();
            
            return Some(self.get_criticality_state());
        }
        
        None
    }
    
    /// Trigger and propagate avalanche
    fn trigger_avalanche(&mut self, x: usize, y: usize, timestamp: u64) -> Avalanche {
        let mut avalanche = Avalanche {
            start_time: timestamp,
            end_time: None,
            size: 0.0,
            duration: 0,
            affected_nodes: Vec::new(),
            trigger_location: (x, y),
            propagation_path: Vec::new(),
            energy_dissipated: 0.0,
        };
        
        let mut to_topple = vec![(x, y)];
        let mut toppled = HashMap::new();
        
        while !to_topple.is_empty() {
            let (cx, cy) = to_topple.pop().unwrap();
            
            if self.sandpile[(cx, cy)] >= self.critical_threshold {
                // Topple this site
                let excess = self.sandpile[(cx, cy)] - self.critical_threshold;
                self.sandpile[(cx, cy)] = 0.0;
                
                avalanche.size += excess;
                avalanche.energy_dissipated += excess * excess;
                avalanche.affected_nodes.push((cx, cy));
                avalanche.propagation_path.push((cx, cy));
                
                *toppled.entry((cx, cy)).or_insert(0) += 1;
                
                // Distribute to neighbors
                let neighbors = self.get_neighbors(cx, cy);
                let distribution = excess / neighbors.len() as f64;
                
                for (nx, ny) in neighbors {
                    self.sandpile[(nx, ny)] += distribution;
                    
                    // Check if neighbor needs to topple
                    if self.sandpile[(nx, ny)] >= self.critical_threshold {
                        if !to_topple.contains(&(nx, ny)) {
                            to_topple.push((nx, ny));
                        }
                    }
                }
            }
        }
        
        avalanche.duration = toppled.len() as u64;
        avalanche.end_time = Some(timestamp + avalanche.duration);
        
        avalanche
    }
    
    /// Get valid neighbors considering boundaries
    fn get_neighbors(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut neighbors = Vec::new();
        
        // Check adjacency matrix for connected nodes
        for i in 0..self.grid_size {
            for j in 0..self.grid_size {
                if self.adjacency_matrix[(x, y)] && self.adjacency_matrix[(i, j)] {
                    let dist = ((i as i32 - x as i32).abs() + (j as i32 - y as i32).abs()) as usize;
                    if dist == 1 {
                        neighbors.push((i, j));
                    }
                }
            }
        }
        
        // If no neighbors found, use von Neumann neighborhood
        if neighbors.is_empty() {
            if x > 0 { neighbors.push((x - 1, y)); }
            if x < self.grid_size - 1 { neighbors.push((x + 1, y)); }
            if y > 0 { neighbors.push((x, y - 1)); }
            if y < self.grid_size - 1 { neighbors.push((x, y + 1)); }
        }
        
        neighbors
    }
    
    /// Update criticality metrics
    fn update_criticality_metrics(&mut self) {
        // Calculate susceptibility (response to perturbation)
        self.susceptibility = self.calculate_susceptibility();
        
        // Calculate branching ratio (avalanche propagation tendency)
        self.branching_ratio = self.calculate_branching_ratio();
        
        // Calculate correlation length
        self.correlation_length = self.calculate_correlation_length();
        
        // Fit power law to avalanche sizes
        if self.avalanche_sizes.len() > 10 {
            self.power_law_exponent = self.fit_power_law(&self.avalanche_sizes);
        }
    }
    
    /// Calculate susceptibility (chi)
    fn calculate_susceptibility(&self) -> f64 {
        if self.avalanche_sizes.is_empty() {
            return 0.0;
        }
        
        // Chi = <s²> - <s>²
        let mean_size = self.avalanche_sizes.iter().sum::<f64>() / self.avalanche_sizes.len() as f64;
        let mean_size_squared = self.avalanche_sizes.iter()
            .map(|s| s * s)
            .sum::<f64>() / self.avalanche_sizes.len() as f64;
        
        mean_size_squared - mean_size * mean_size
    }
    
    /// Calculate branching ratio
    fn calculate_branching_ratio(&self) -> f64 {
        if self.avalanche_history.is_empty() {
            return 1.0;
        }
        
        let mut total_ratio = 0.0;
        let mut count = 0;
        
        for avalanche in &self.avalanche_history {
            if avalanche.affected_nodes.len() > 1 {
                // Ratio of triggered sites to initial site
                let ratio = (avalanche.affected_nodes.len() - 1) as f64;
                total_ratio += ratio;
                count += 1;
            }
        }
        
        if count > 0 {
            total_ratio / count as f64
        } else {
            1.0
        }
    }
    
    /// Calculate correlation length
    fn calculate_correlation_length(&self) -> f64 {
        // Use sandpile configuration to estimate correlation length
        let mut correlation_sum = 0.0;
        let mut pair_count = 0;
        
        for i in 0..self.grid_size {
            for j in 0..self.grid_size {
                for k in 0..self.grid_size {
                    for l in 0..self.grid_size {
                        if (i, j) != (k, l) {
                            let distance = ((i as f64 - k as f64).powi(2) + 
                                          (j as f64 - l as f64).powi(2)).sqrt();
                            
                            if distance > 0.0 {
                                let correlation = (self.sandpile[(i, j)] - 2.0) * 
                                                 (self.sandpile[(k, l)] - 2.0);
                                correlation_sum += correlation / distance;
                                pair_count += 1;
                            }
                        }
                    }
                }
            }
        }
        
        if pair_count > 0 {
            (correlation_sum / pair_count as f64).abs().sqrt()
        } else {
            0.0
        }
    }
    
    /// Fit power law distribution
    fn fit_power_law(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 1.5;
        }
        
        // Maximum likelihood estimation for power law exponent
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let xmin = sorted_data[0];
        let n = sorted_data.len() as f64;
        
        let sum_log = sorted_data.iter()
            .filter(|&&x| x > xmin)
            .map(|x| (x / xmin).ln())
            .sum::<f64>();
        
        if sum_log > 0.0 {
            1.0 + n / sum_log
        } else {
            1.5
        }
    }
    
    /// Process market data
    pub fn process_market_data(
        &mut self,
        price: f64,
        volume: f64,
        order_imbalance: f64,
        timestamp: u64,
    ) -> CriticalityState {
        // Update market microstructure
        if let Some(last_price) = self.price_returns.back() {
            let return_val = (price / last_price).ln();
            self.price_returns.push_back(return_val);
        } else {
            self.price_returns.push_back(0.0);
        }
        
        self.volume_bursts.push_back(volume);
        self.order_flow_imbalance.push_back(order_imbalance);
        
        // Maintain window size
        if self.price_returns.len() > self.history_window {
            self.price_returns.pop_front();
        }
        if self.volume_bursts.len() > self.history_window {
            self.volume_bursts.pop_front();
        }
        if self.order_flow_imbalance.len() > self.history_window {
            self.order_flow_imbalance.pop_front();
        }
        
        // Map market state to sandpile
        let x = ((price % self.grid_size as f64) as usize).min(self.grid_size - 1);
        let y = ((volume % self.grid_size as f64) as usize).min(self.grid_size - 1);
        let magnitude = order_imbalance.abs();
        
        // Add event and check for criticality
        if let Some(state) = self.add_event(x, y, magnitude, timestamp) {
            state
        } else {
            self.get_criticality_state()
        }
    }
    
    /// Get current criticality state
    pub fn get_criticality_state(&self) -> CriticalityState {
        let criticality_score = self.calculate_criticality_score();
        
        let phase = if self.branching_ratio < 0.9 {
            MarketPhase::Subcritical
        } else if self.branching_ratio > 1.1 {
            MarketPhase::Supercritical
        } else {
            MarketPhase::Critical
        };
        
        let warning_level = if criticality_score < 0.3 {
            WarningLevel::Low
        } else if criticality_score < 0.6 {
            WarningLevel::Medium
        } else if criticality_score < 0.8 {
            WarningLevel::High
        } else {
            WarningLevel::Critical
        };
        
        CriticalityState {
            is_critical: criticality_score > self.detection_threshold,
            criticality_score,
            avalanche_probability: self.calculate_avalanche_probability(),
            expected_avalanche_size: self.calculate_expected_avalanche_size(),
            warning_level,
            phase,
        }
    }
    
    /// Calculate criticality score
    fn calculate_criticality_score(&self) -> f64 {
        // Combine multiple indicators
        let susceptibility_score = (self.susceptibility / 100.0).min(1.0);
        let branching_score = (self.branching_ratio - 1.0).abs();
        let correlation_score = (self.correlation_length / self.grid_size as f64).min(1.0);
        let power_law_score = ((self.power_law_exponent - 1.5).abs() / 0.5).min(1.0);
        
        // Weighted average
        (susceptibility_score * 0.3 + 
         (1.0 - branching_score) * 0.3 + 
         correlation_score * 0.2 + 
         (1.0 - power_law_score) * 0.2).min(1.0).max(0.0)
    }
    
    /// Calculate avalanche probability
    fn calculate_avalanche_probability(&self) -> f64 {
        // Based on current sandpile configuration
        let mut unstable_sites = 0;
        
        for i in 0..self.grid_size {
            for j in 0..self.grid_size {
                if self.sandpile[(i, j)] >= self.critical_threshold * 0.9 {
                    unstable_sites += 1;
                }
            }
        }
        
        unstable_sites as f64 / (self.grid_size * self.grid_size) as f64
    }
    
    /// Calculate expected avalanche size
    fn calculate_expected_avalanche_size(&self) -> f64 {
        if self.avalanche_sizes.is_empty() {
            return 0.0;
        }
        
        // Use power law distribution
        let mean_size = self.avalanche_sizes.iter().sum::<f64>() / self.avalanche_sizes.len() as f64;
        
        // Scale by current criticality
        mean_size * (1.0 + self.susceptibility / 100.0)
    }
    
    /// Detect dragon kings (outliers beyond power law)
    pub fn detect_dragon_king(&self) -> Option<DragonKing> {
        if self.avalanche_sizes.len() < 20 {
            return None;
        }
        
        let mut sorted_sizes = self.avalanche_sizes.clone();
        sorted_sizes.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        // Check if largest avalanche deviates from power law
        let largest = sorted_sizes[0];
        let second_largest = sorted_sizes[1];
        
        // Dragon king if largest is significantly larger than expected
        if largest > second_largest * 3.0 {
            Some(DragonKing {
                size: largest,
                deviation_factor: largest / second_largest,
                probability: 1.0 / self.avalanche_sizes.len() as f64,
                mechanism: DragonKingMechanism::PositiveFeedback,
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct DragonKing {
    pub size: f64,
    pub deviation_factor: f64,
    pub probability: f64,
    pub mechanism: DragonKingMechanism,
}

#[derive(Debug, Clone, Copy)]
pub enum DragonKingMechanism {
    PositiveFeedback,
    NetworkEffect,
    Synchronization,
    PhaseTransition,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_soc_analyzer_creation() {
        let analyzer = SOCAnalyzer::new(10, 100);
        assert_eq!(analyzer.grid_size, 10);
        assert_eq!(analyzer.critical_threshold, 4.0);
    }
    
    #[test]
    fn test_avalanche_triggering() {
        let mut analyzer = SOCAnalyzer::new(5, 100);
        
        // Add sand until avalanche
        for i in 0..10 {
            let state = analyzer.add_event(2, 2, 0.5, i as u64);
            if let Some(s) = state {
                if s.is_critical {
                    assert!(analyzer.avalanche_history.len() > 0);
                    break;
                }
            }
        }
    }
    
    #[test]
    fn test_market_data_processing() {
        let mut analyzer = SOCAnalyzer::new(10, 100);
        
        // Simulate market data
        for i in 0..50 {
            let price = 100.0 + (i as f64).sin() * 5.0;
            let volume = 1000.0 + (i as f64 * 0.5).cos() * 200.0;
            let imbalance = (i as f64 * 0.3).sin();
            
            let state = analyzer.process_market_data(price, volume, imbalance, i as u64);
            
            assert!(state.criticality_score >= 0.0 && state.criticality_score <= 1.0);
        }
    }
    
    #[test]
    fn test_power_law_fitting() {
        let analyzer = SOCAnalyzer::new(10, 100);
        
        // Generate power law distributed data
        let data: Vec<f64> = (1..100).map(|i| 1.0 / (i as f64).powf(1.5)).collect();
        
        let exponent = analyzer.fit_power_law(&data);
        assert!(exponent > 1.0 && exponent < 3.0);
    }
    
    #[test]
    fn test_dragon_king_detection() {
        let mut analyzer = SOCAnalyzer::new(10, 100);
        
        // Add normal avalanches
        for i in 1..20 {
            analyzer.avalanche_sizes.push(1.0 / i as f64);
        }
        
        // Add dragon king
        analyzer.avalanche_sizes.push(100.0);
        
        let dragon_king = analyzer.detect_dragon_king();
        assert!(dragon_king.is_some());
        
        if let Some(dk) = dragon_king {
            assert!(dk.deviation_factor > 3.0);
        }
    }
}