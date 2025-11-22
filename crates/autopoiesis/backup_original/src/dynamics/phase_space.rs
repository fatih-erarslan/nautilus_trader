use nalgebra::{DVector, DMatrix};
use std::collections::VecDeque;

/// Phase space reconstruction and attractor analysis for temporal dynamics
/// Implements Takens' embedding theorem and chaotic attractor detection
pub struct PhaseSpaceReconstructor {
    /// Time series data buffer
    time_series: VecDeque<f64>,
    /// Embedding dimension
    embedding_dimension: usize,
    /// Time delay for embedding
    time_delay: usize,
    /// Reconstructed phase space points
    phase_space: Vec<DVector<f64>>,
    /// Parameters for reconstruction
    params: PhaseSpaceParameters,
}

#[derive(Clone, Debug)]
pub struct PhaseSpaceParameters {
    /// Maximum embedding dimension to consider
    pub max_embedding_dim: usize,
    /// Maximum time delay to consider
    pub max_time_delay: usize,
    /// Window size for analysis
    pub window_size: usize,
    /// Minimum data points required
    pub min_data_points: usize,
    /// Tolerance for neighbor searching
    pub tolerance: f64,
}

impl Default for PhaseSpaceParameters {
    fn default() -> Self {
        Self {
            max_embedding_dim: 10,
            max_time_delay: 20,
            window_size: 1000,
            min_data_points: 100,
            tolerance: 1e-6,
        }
    }
}

/// Attractor types that can be detected
#[derive(Clone, Debug, PartialEq)]
pub enum AttractorType {
    FixedPoint,
    LimitCycle,
    Torus,
    ChaoticAttractor,
    Strange,
    Unknown,
}

/// Results of attractor analysis
#[derive(Clone, Debug)]
pub struct AttractorAnalysis {
    /// Detected attractor type
    pub attractor_type: AttractorType,
    /// Correlation dimension
    pub correlation_dimension: f64,
    /// Largest Lyapunov exponent
    pub largest_lyapunov_exponent: f64,
    /// Embedding dimension used
    pub embedding_dimension: usize,
    /// Time delay used
    pub time_delay: usize,
    /// False nearest neighbors percentage
    pub false_neighbors_percentage: f64,
    /// Entropy estimate
    pub entropy: f64,
    /// Predictability measure
    pub predictability: f64,
}

impl PhaseSpaceReconstructor {
    /// Create new phase space reconstructor
    pub fn new(params: PhaseSpaceParameters) -> Self {
        Self {
            time_series: VecDeque::new(),
            embedding_dimension: 3, // Default embedding dimension
            time_delay: 1,         // Default time delay
            phase_space: Vec::new(),
            params,
        }
    }

    /// Add new data point to time series
    pub fn add_data_point(&mut self, value: f64) {
        self.time_series.push_back(value);
        
        // Maintain window size
        if self.time_series.len() > self.params.window_size {
            self.time_series.pop_front();
        }
        
        // Update phase space if we have enough data
        if self.time_series.len() >= self.minimum_required_points() {
            self.update_phase_space();
        }
    }

    /// Add multiple data points
    pub fn add_data_points(&mut self, values: &[f64]) {
        for &value in values {
            self.add_data_point(value);
        }
    }

    /// Determine optimal embedding parameters using mutual information and false nearest neighbors
    pub fn determine_optimal_parameters(&mut self) -> (usize, usize) {
        if self.time_series.len() < self.params.min_data_points {
            return (self.embedding_dimension, self.time_delay);
        }

        // Find optimal time delay using mutual information
        let optimal_delay = self.find_optimal_time_delay();
        
        // Find optimal embedding dimension using false nearest neighbors
        let optimal_dimension = self.find_optimal_embedding_dimension(optimal_delay);
        
        self.time_delay = optimal_delay;
        self.embedding_dimension = optimal_dimension;
        
        (optimal_dimension, optimal_delay)
    }

    /// Find optimal time delay using mutual information
    fn find_optimal_time_delay(&self) -> usize {
        let mut min_mutual_info = f64::INFINITY;
        let mut optimal_delay = 1;
        
        for delay in 1..=self.params.max_time_delay {
            let mutual_info = self.calculate_mutual_information(delay);
            
            if mutual_info < min_mutual_info {
                min_mutual_info = mutual_info;
                optimal_delay = delay;
            }
        }
        
        optimal_delay
    }

    /// Calculate mutual information for given time delay
    fn calculate_mutual_information(&self, delay: usize) -> f64 {
        if self.time_series.len() <= delay {
            return f64::INFINITY;
        }
        
        let n = self.time_series.len() - delay;
        let mut x: Vec<f64> = Vec::with_capacity(n);
        let mut y: Vec<f64> = Vec::with_capacity(n);
        
        for i in 0..n {
            x.push(self.time_series[i]);
            y.push(self.time_series[i + delay]);
        }
        
        // Estimate mutual information using histogram-based approach
        self.estimate_mutual_information(&x, &y)
    }

    /// Estimate mutual information between two vectors
    fn estimate_mutual_information(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        // Simple histogram-based mutual information estimation
        let bins = 20;
        let x_min = x.iter().fold(f64::INFINITY, |min, &val| val.min(min));
        let x_max = x.iter().fold(f64::NEG_INFINITY, |max, &val| val.max(max));
        let y_min = y.iter().fold(f64::INFINITY, |min, &val| val.min(min));
        let y_max = y.iter().fold(f64::NEG_INFINITY, |max, &val| val.max(max));
        
        if x_max <= x_min || y_max <= y_min {
            return 0.0;
        }
        
        let x_step = (x_max - x_min) / bins as f64;
        let y_step = (y_max - y_min) / bins as f64;
        
        // Build histograms
        let mut joint_hist = vec![vec![0; bins]; bins];
        let mut x_hist = vec![0; bins];
        let mut y_hist = vec![0; bins];
        
        for i in 0..x.len() {
            let x_bin = ((x[i] - x_min) / x_step).floor() as usize;
            let y_bin = ((y[i] - y_min) / y_step).floor() as usize;
            
            let x_bin = x_bin.min(bins - 1);
            let y_bin = y_bin.min(bins - 1);
            
            joint_hist[x_bin][y_bin] += 1;
            x_hist[x_bin] += 1;
            y_hist[y_bin] += 1;
        }
        
        // Calculate mutual information
        let n = x.len() as f64;
        let mut mutual_info = 0.0;
        
        for i in 0..bins {
            for j in 0..bins {
                if joint_hist[i][j] > 0 && x_hist[i] > 0 && y_hist[j] > 0 {
                    let p_xy = joint_hist[i][j] as f64 / n;
                    let p_x = x_hist[i] as f64 / n;
                    let p_y = y_hist[j] as f64 / n;
                    
                    mutual_info += p_xy * (p_xy / (p_x * p_y)).ln();
                }
            }
        }
        
        mutual_info
    }

    /// Find optimal embedding dimension using false nearest neighbors
    fn find_optimal_embedding_dimension(&self, delay: usize) -> usize {
        let mut optimal_dimension = 3;
        
        for dim in 2..=self.params.max_embedding_dim {
            let false_neighbors_pct = self.calculate_false_nearest_neighbors(dim, delay);
            
            if false_neighbors_pct < 0.05 { // Less than 5% false neighbors
                optimal_dimension = dim;
                break;
            }
        }
        
        optimal_dimension
    }

    /// Calculate percentage of false nearest neighbors
    fn calculate_false_nearest_neighbors(&self, dimension: usize, delay: usize) -> f64 {
        let phase_points = self.reconstruct_phase_space_with_params(dimension, delay);
        
        if phase_points.len() < 20 {
            return 1.0; // Not enough data
        }
        
        let mut false_neighbors = 0;
        let total_points = phase_points.len().min(100); // Sample for efficiency
        
        for i in 0..total_points {
            // Find nearest neighbor in d-dimensional space
            let mut min_distance = f64::INFINITY;
            let mut nearest_neighbor = 0;
            
            for j in 0..phase_points.len() {
                if i == j { continue; }
                
                let distance = (&phase_points[i] - &phase_points[j]).norm();
                if distance < min_distance {
                    min_distance = distance;
                    nearest_neighbor = j;
                }
            }
            
            // Check if this neighbor is false in (d+1)-dimensional space
            if nearest_neighbor < phase_points.len() && 
               i + delay < self.time_series.len() && 
               nearest_neighbor + delay < self.time_series.len() {
                
                let current_next = self.time_series[i + delay];
                let neighbor_next = self.time_series[nearest_neighbor + delay];
                
                let next_distance = (current_next - neighbor_next).abs();
                
                // Criterion for false nearest neighbor
                if min_distance > 0.0 && next_distance / min_distance > 10.0 {
                    false_neighbors += 1;
                }
            }
        }
        
        false_neighbors as f64 / total_points as f64
    }

    /// Reconstruct phase space with given parameters
    fn reconstruct_phase_space_with_params(&self, dimension: usize, delay: usize) -> Vec<DVector<f64>> {
        let required_length = (dimension - 1) * delay + 1;
        
        if self.time_series.len() < required_length {
            return Vec::new();
        }
        
        let mut phase_points = Vec::new();
        
        for i in 0..=(self.time_series.len() - required_length) {
            let mut point = DVector::zeros(dimension);
            
            for j in 0..dimension {
                point[j] = self.time_series[i + j * delay];
            }
            
            phase_points.push(point);
        }
        
        phase_points
    }

    /// Update phase space reconstruction
    fn update_phase_space(&mut self) {
        self.phase_space = self.reconstruct_phase_space_with_params(
            self.embedding_dimension, 
            self.time_delay
        );
    }

    /// Get minimum required points for current parameters
    fn minimum_required_points(&self) -> usize {
        (self.embedding_dimension - 1) * self.time_delay + 1
    }

    /// Perform comprehensive attractor analysis
    pub fn analyze_attractor(&self) -> AttractorAnalysis {
        if self.phase_space.is_empty() {
            return AttractorAnalysis {
                attractor_type: AttractorType::Unknown,
                correlation_dimension: 0.0,
                largest_lyapunov_exponent: 0.0,
                embedding_dimension: self.embedding_dimension,
                time_delay: self.time_delay,
                false_neighbors_percentage: 1.0,
                entropy: 0.0,
                predictability: 0.0,
            };
        }

        let correlation_dimension = self.calculate_correlation_dimension();
        let lyapunov_exponent = self.calculate_largest_lyapunov_exponent();
        let entropy = self.calculate_entropy();
        let predictability = self.calculate_predictability();
        
        let attractor_type = self.classify_attractor(
            correlation_dimension, 
            lyapunov_exponent, 
            entropy
        );

        AttractorAnalysis {
            attractor_type,
            correlation_dimension,
            largest_lyapunov_exponent: lyapunov_exponent,
            embedding_dimension: self.embedding_dimension,
            time_delay: self.time_delay,
            false_neighbors_percentage: self.calculate_false_nearest_neighbors(
                self.embedding_dimension, 
                self.time_delay
            ),
            entropy,
            predictability,
        }
    }

    /// Calculate correlation dimension using Grassberger-Procaccia algorithm
    fn calculate_correlation_dimension(&self) -> f64 {
        if self.phase_space.len() < 50 {
            return 0.0;
        }
        
        let sample_size = self.phase_space.len().min(500); // Sample for efficiency
        let mut distances = Vec::new();
        
        // Calculate all pairwise distances
        for i in 0..sample_size {
            for j in (i + 1)..sample_size {
                let distance = (&self.phase_space[i] - &self.phase_space[j]).norm();
                distances.push(distance);
            }
        }
        
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        if distances.is_empty() {
            return 0.0;
        }
        
        // Calculate correlation integral for different radius values
        let mut log_r = Vec::new();
        let mut log_c = Vec::new();
        
        let min_r = distances[distances.len() / 100]; // 1st percentile
        let max_r = distances[distances.len() * 99 / 100]; // 99th percentile
        
        for i in 1..20 {
            let r = min_r * (max_r / min_r).powf(i as f64 / 19.0);
            let count = distances.iter().filter(|&&d| d < r).count();
            let correlation_integral = count as f64 / distances.len() as f64;
            
            if correlation_integral > 0.0 {
                log_r.push(r.ln());
                log_c.push(correlation_integral.ln());
            }
        }
        
        // Linear regression to find slope (correlation dimension)
        if log_r.len() < 3 {
            return 0.0;
        }
        
        self.linear_regression_slope(&log_r, &log_c)
    }

    /// Calculate largest Lyapunov exponent
    fn calculate_largest_lyapunov_exponent(&self) -> f64 {
        if self.phase_space.len() < 100 {
            return 0.0;
        }
        
        let mut lyapunov_sum = 0.0;
        let mut count = 0;
        let evolution_time = 10; // Steps to evolve
        
        for i in 0..(self.phase_space.len() - evolution_time - 1) {
            // Find nearest neighbor
            let mut min_distance = f64::INFINITY;
            let mut nearest_idx = 0;
            
            for j in 0..self.phase_space.len() {
                if (i as i32 - j as i32).abs() < 50 { continue; } // Avoid temporal correlation
                
                let distance = (&self.phase_space[i] - &self.phase_space[j]).norm();
                if distance < min_distance && distance > self.params.tolerance {
                    min_distance = distance;
                    nearest_idx = j;
                }
            }
            
            // Check if we can evolve both points
            if nearest_idx + evolution_time < self.phase_space.len() &&
               i + evolution_time < self.phase_space.len() {
                
                let evolved_distance = (&self.phase_space[i + evolution_time] - 
                                      &self.phase_space[nearest_idx + evolution_time]).norm();
                
                if evolved_distance > self.params.tolerance && min_distance > self.params.tolerance {
                    lyapunov_sum += (evolved_distance / min_distance).ln();
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            lyapunov_sum / (count as f64 * evolution_time as f64)
        } else {
            0.0
        }
    }

    /// Calculate entropy of the attractor
    fn calculate_entropy(&self) -> f64 {
        if self.phase_space.is_empty() {
            return 0.0;
        }
        
        // Box-counting approach for entropy estimation
        let boxes = 20;
        let mut box_counts = vec![vec![vec![0; boxes]; boxes]; boxes];
        
        // Find bounds
        let mut min_vals = self.phase_space[0].clone();
        let mut max_vals = self.phase_space[0].clone();
        
        for point in &self.phase_space {
            for i in 0..point.len().min(3) { // Use first 3 dimensions
                min_vals[i] = min_vals[i].min(point[i]);
                max_vals[i] = max_vals[i].max(point[i]);
            }
        }
        
        // Count points in each box
        for point in &self.phase_space {
            let mut box_idx = [0; 3];
            
            for i in 0..3.min(point.len()) {
                if max_vals[i] > min_vals[i] {
                    let normalized = (point[i] - min_vals[i]) / (max_vals[i] - min_vals[i]);
                    box_idx[i] = (normalized * boxes as f64).floor() as usize;
                    box_idx[i] = box_idx[i].min(boxes - 1);
                }
            }
            
            box_counts[box_idx[0]][box_idx[1]][box_idx[2]] += 1;
        }
        
        // Calculate entropy
        let total_points = self.phase_space.len() as f64;
        let mut entropy = 0.0;
        
        for i in 0..boxes {
            for j in 0..boxes {
                for k in 0..boxes {
                    if box_counts[i][j][k] > 0 {
                        let probability = box_counts[i][j][k] as f64 / total_points;
                        entropy -= probability * probability.ln();
                    }
                }
            }
        }
        
        entropy
    }

    /// Calculate predictability measure
    fn calculate_predictability(&self) -> f64 {
        if self.phase_space.len() < 50 {
            return 0.0;
        }
        
        let prediction_horizon = 5;
        let mut prediction_errors = Vec::new();
        
        for i in 0..(self.phase_space.len() - prediction_horizon) {
            // Find k nearest neighbors
            let k = 5;
            let mut neighbors = Vec::new();
            
            for j in 0..self.phase_space.len() {
                if (i as i32 - j as i32).abs() < 10 { continue; } // Avoid temporal correlation
                
                let distance = (&self.phase_space[i] - &self.phase_space[j]).norm();
                neighbors.push((distance, j));
            }
            
            neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            neighbors.truncate(k);
            
            if neighbors.len() == k {
                // Make prediction based on neighbors
                let mut predicted_point = DVector::zeros(self.phase_space[0].len());
                
                for (_, neighbor_idx) in &neighbors {
                    if *neighbor_idx + prediction_horizon < self.phase_space.len() {
                        predicted_point += &self.phase_space[*neighbor_idx + prediction_horizon];
                    }
                }
                predicted_point /= neighbors.len() as f64;
                
                // Calculate prediction error
                if i + prediction_horizon < self.phase_space.len() {
                    let actual_point = &self.phase_space[i + prediction_horizon];
                    let error = (actual_point - &predicted_point).norm();
                    prediction_errors.push(error);
                }
            }
        }
        
        if prediction_errors.is_empty() {
            return 0.0;
        }
        
        let mean_error: f64 = prediction_errors.iter().sum::<f64>() / prediction_errors.len() as f64;
        
        // Convert error to predictability (lower error = higher predictability)
        if mean_error > 0.0 {
            1.0 / (1.0 + mean_error)
        } else {
            1.0
        }
    }

    /// Classify attractor type based on computed measures
    fn classify_attractor(&self, correlation_dim: f64, lyapunov: f64, entropy: f64) -> AttractorType {
        if correlation_dim < 0.5 {
            AttractorType::FixedPoint
        } else if correlation_dim < 1.5 && lyapunov <= 0.0 {
            AttractorType::LimitCycle
        } else if correlation_dim < 2.5 && lyapunov <= 0.0 {
            AttractorType::Torus
        } else if lyapunov > 0.001 && correlation_dim > 1.5 {
            if correlation_dim.fract() > 0.1 {
                AttractorType::Strange
            } else {
                AttractorType::ChaoticAttractor
            }
        } else {
            AttractorType::Unknown
        }
    }

    /// Linear regression slope calculation
    fn linear_regression_slope(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();
        
        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }
        
        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Get current phase space points
    pub fn get_phase_space(&self) -> &[DVector<f64>] {
        &self.phase_space
    }

    /// Get time series data
    pub fn get_time_series(&self) -> Vec<f64> {
        self.time_series.iter().cloned().collect()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.time_series.clear();
        self.phase_space.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_space_reconstruction() {
        let params = PhaseSpaceParameters::default();
        let mut reconstructor = PhaseSpaceReconstructor::new(params);
        
        // Add some test data (simple sine wave)
        for i in 0..200 {
            let value = (i as f64 * 0.1).sin();
            reconstructor.add_data_point(value);
        }
        
        assert!(!reconstructor.phase_space.is_empty());
        assert!(reconstructor.time_series.len() > 0);
    }

    #[test]
    fn test_mutual_information() {
        let params = PhaseSpaceParameters::default();
        let reconstructor = PhaseSpaceReconstructor::new(params);
        
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfectly correlated
        
        let mi = reconstructor.estimate_mutual_information(&x, &y);
        assert!(mi >= 0.0);
    }

    #[test]
    fn test_attractor_analysis() {
        let params = PhaseSpaceParameters::default();
        let mut reconstructor = PhaseSpaceReconstructor::new(params);
        
        // Add chaotic Lorenz-like data
        for i in 0..500 {
            let t = i as f64 * 0.01;
            let value = t.sin() + 0.5 * (3.0 * t).sin() + 0.25 * (7.0 * t).sin();
            reconstructor.add_data_point(value);
        }
        
        let analysis = reconstructor.analyze_attractor();
        assert!(analysis.correlation_dimension >= 0.0);
        println!("Attractor type: {:?}", analysis.attractor_type);
        println!("Correlation dimension: {}", analysis.correlation_dimension);
        println!("Lyapunov exponent: {}", analysis.largest_lyapunov_exponent);
    }
}