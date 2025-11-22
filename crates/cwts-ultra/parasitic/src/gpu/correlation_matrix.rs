//! Correlation Matrix Data Structure
//!
//! Efficient storage and manipulation of correlation matrices for organism pairs.
//! Optimized for cache performance and mathematical operations.

use super::*;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

/// Correlation matrix with optimized storage and operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    data: Vec<f32>,
    size: usize,
    is_symmetric: bool,
    creation_time: std::time::SystemTime,
}

impl CorrelationMatrix {
    /// Create new correlation matrix
    pub fn new(size: usize, data: Vec<f32>) -> Result<Self, CorrelationError> {
        if data.len() != size * size {
            return Err(CorrelationError::ComputationError(format!(
                "Data size {} doesn't match matrix size {}x{}",
                data.len(),
                size,
                size
            )));
        }

        // Validate correlation values
        for (i, &value) in data.iter().enumerate() {
            if value.is_nan() || value.is_infinite() {
                return Err(CorrelationError::ComputationError(format!(
                    "Invalid correlation value at index {}: {}",
                    i, value
                )));
            }

            if value < -1.0 || value > 1.0 {
                return Err(CorrelationError::ComputationError(format!(
                    "Correlation value {} out of range [-1, 1] at index {}",
                    value, i
                )));
            }
        }

        let matrix = Self {
            data,
            size,
            is_symmetric: false,
            creation_time: std::time::SystemTime::now(),
        };

        // Verify matrix properties
        if !matrix.diagonal_ones() {
            return Err(CorrelationError::ComputationError(
                "Diagonal elements must be 1.0 for correlation matrix".to_string(),
            ));
        }

        Ok(matrix)
    }

    /// Create identity correlation matrix
    pub fn identity(size: usize) -> Self {
        let mut data = vec![0.0; size * size];

        for i in 0..size {
            data[i * size + i] = 1.0;
        }

        Self {
            data,
            size,
            is_symmetric: true,
            creation_time: std::time::SystemTime::now(),
        }
    }

    /// Get matrix size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get correlation value at (i, j)
    pub fn get(&self, i: usize, j: usize) -> f32 {
        if i >= self.size || j >= self.size {
            0.0 // Return 0 for out-of-bounds access
        } else {
            self.data[i * self.size + j]
        }
    }

    /// Set correlation value at (i, j)
    pub fn set(&mut self, i: usize, j: usize, value: f32) {
        if i < self.size && j < self.size {
            self.data[i * self.size + j] = value;

            // Invalidate symmetry flag if needed
            if self.is_symmetric && i != j && value != self.get(j, i) {
                self.is_symmetric = false;
            }
        }
    }

    /// Get row as slice
    pub fn row(&self, i: usize) -> &[f32] {
        if i >= self.size {
            &[]
        } else {
            &self.data[i * self.size..(i + 1) * self.size]
        }
    }

    /// Get mutable row as slice
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        if i >= self.size {
            &mut []
        } else {
            self.is_symmetric = false; // Assume modification breaks symmetry
            &mut self.data[i * self.size..(i + 1) * self.size]
        }
    }

    /// Check if matrix is symmetric
    pub fn is_symmetric(&self) -> bool {
        if self.is_symmetric {
            return true;
        }

        // Check symmetry
        for i in 0..self.size {
            for j in 0..i {
                if (self.get(i, j) - self.get(j, i)).abs() > 1e-6 {
                    return false;
                }
            }
        }

        true
    }

    /// Check if diagonal elements are 1.0
    pub fn diagonal_ones(&self) -> bool {
        for i in 0..self.size {
            if (self.get(i, i) - 1.0).abs() > 1e-6 {
                return false;
            }
        }
        true
    }

    /// Get diagonal elements
    pub fn diagonal(&self) -> Vec<f32> {
        (0..self.size).map(|i| self.get(i, i)).collect()
    }

    /// Get upper triangular part (excluding diagonal)
    pub fn upper_triangular(&self) -> Vec<(usize, usize, f32)> {
        let mut result = Vec::new();

        for i in 0..self.size {
            for j in (i + 1)..self.size {
                result.push((i, j, self.get(i, j)));
            }
        }

        result
    }

    /// Get lower triangular part (excluding diagonal)
    pub fn lower_triangular(&self) -> Vec<(usize, usize, f32)> {
        let mut result = Vec::new();

        for i in 1..self.size {
            for j in 0..i {
                result.push((i, j, self.get(i, j)));
            }
        }

        result
    }

    /// Find highest correlations
    pub fn highest_correlations(&self, count: usize) -> Vec<(usize, usize, f32)> {
        let mut correlations = Vec::new();

        // Collect all off-diagonal correlations
        for i in 0..self.size {
            for j in (i + 1)..self.size {
                correlations.push((i, j, self.get(i, j)));
            }
        }

        // Sort by correlation value (descending)
        correlations.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N
        correlations.truncate(count);
        correlations
    }

    /// Find lowest correlations (most negative)
    pub fn lowest_correlations(&self, count: usize) -> Vec<(usize, usize, f32)> {
        let mut correlations = Vec::new();

        // Collect all off-diagonal correlations
        for i in 0..self.size {
            for j in (i + 1)..self.size {
                correlations.push((i, j, self.get(i, j)));
            }
        }

        // Sort by correlation value (ascending)
        correlations.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N
        correlations.truncate(count);
        correlations
    }

    /// Calculate matrix statistics
    pub fn statistics(&self) -> CorrelationStatistics {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut count = 0;

        // Calculate statistics for off-diagonal elements only
        for i in 0..self.size {
            for j in 0..self.size {
                if i != j {
                    let val = self.get(i, j);
                    sum += val;
                    sum_sq += val * val;
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                    count += 1;
                }
            }
        }

        let mean = if count > 0 { sum / count as f32 } else { 0.0 };
        let variance = if count > 0 {
            (sum_sq / count as f32) - (mean * mean)
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        CorrelationStatistics {
            mean,
            std_dev,
            min: if min_val.is_finite() { min_val } else { 0.0 },
            max: if max_val.is_finite() { max_val } else { 0.0 },
            variance,
            count,
        }
    }

    /// Convert to dense matrix representation
    pub fn to_dense(&self) -> Vec<Vec<f32>> {
        (0..self.size).map(|i| self.row(i).to_vec()).collect()
    }

    /// Convert to sparse representation (upper triangular only)
    pub fn to_sparse(&self) -> SparseCorrelationMatrix {
        let mut entries = Vec::new();

        for i in 0..self.size {
            for j in i..self.size {
                let value = self.get(i, j);
                if value.abs() > 1e-9 {
                    // Only store non-zero values
                    entries.push(SparseEntry {
                        row: i,
                        col: j,
                        value,
                    });
                }
            }
        }

        SparseCorrelationMatrix {
            size: self.size,
            entries,
        }
    }

    /// Apply threshold to correlation values
    pub fn apply_threshold(&mut self, threshold: f32) {
        for i in 0..self.size {
            for j in 0..self.size {
                if i != j && self.get(i, j).abs() < threshold {
                    self.set(i, j, 0.0);
                }
            }
        }
    }

    /// Ensure symmetry by averaging (i,j) and (j,i) values
    pub fn enforce_symmetry(&mut self) {
        for i in 0..self.size {
            for j in 0..i {
                let avg = (self.get(i, j) + self.get(j, i)) / 2.0;
                self.set(i, j, avg);
                self.set(j, i, avg);
            }
        }
        self.is_symmetric = true;
    }

    /// Get creation timestamp
    pub fn creation_time(&self) -> std::time::SystemTime {
        self.creation_time
    }

    /// Get age in milliseconds
    pub fn age_ms(&self) -> u64 {
        self.creation_time.elapsed().unwrap_or_default().as_millis() as u64
    }

    /// Clone with threshold applied
    pub fn with_threshold(&self, threshold: f32) -> Self {
        let mut cloned = self.clone();
        cloned.apply_threshold(threshold);
        cloned
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of_val(self) + self.data.len() * std::mem::size_of::<f32>()
    }
}

impl Index<(usize, usize)> for CorrelationMatrix {
    type Output = f32;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[i * self.size + j]
    }
}

impl IndexMut<(usize, usize)> for CorrelationMatrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        self.is_symmetric = false; // Modification may break symmetry
        &mut self.data[i * self.size + j]
    }
}

/// Correlation matrix statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationStatistics {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub variance: f32,
    pub count: usize,
}

impl CorrelationStatistics {
    /// Check if correlations are well-distributed
    pub fn is_well_distributed(&self) -> bool {
        // Good distribution should have reasonable standard deviation
        // and values spanning a good range
        self.std_dev > 0.1 && (self.max - self.min) > 0.5
    }

    /// Get distribution quality score [0, 1]
    pub fn distribution_quality(&self) -> f32 {
        let range_score = ((self.max - self.min) / 2.0).min(1.0).max(0.0);
        let spread_score = (self.std_dev / 0.5).min(1.0).max(0.0);

        (range_score + spread_score) / 2.0
    }
}

/// Sparse correlation matrix representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseCorrelationMatrix {
    pub size: usize,
    pub entries: Vec<SparseEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseEntry {
    pub row: usize,
    pub col: usize,
    pub value: f32,
}

impl SparseCorrelationMatrix {
    /// Convert back to dense matrix
    pub fn to_dense(&self) -> CorrelationMatrix {
        let mut data = vec![0.0; self.size * self.size];

        // Set diagonal to 1.0
        for i in 0..self.size {
            data[i * self.size + i] = 1.0;
        }

        // Fill from sparse entries
        for entry in &self.entries {
            data[entry.row * self.size + entry.col] = entry.value;
            if entry.row != entry.col {
                data[entry.col * self.size + entry.row] = entry.value;
            }
        }

        CorrelationMatrix::new(self.size, data)
            .unwrap_or_else(|_| CorrelationMatrix::identity(self.size))
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of_val(self) + self.entries.len() * std::mem::size_of::<SparseEntry>()
    }
}

/// Correlation matrix builder for incremental construction
pub struct CorrelationMatrixBuilder {
    size: usize,
    data: Vec<f32>,
    pairs_computed: usize,
    total_pairs: usize,
}

impl CorrelationMatrixBuilder {
    pub fn new(size: usize) -> Self {
        let mut data = vec![0.0; size * size];

        // Initialize diagonal to 1.0
        for i in 0..size {
            data[i * size + i] = 1.0;
        }

        Self {
            size,
            data,
            pairs_computed: 0,
            total_pairs: size * (size - 1) / 2, // Upper triangular without diagonal
        }
    }

    /// Set correlation value and its symmetric counterpart
    pub fn set_correlation(&mut self, i: usize, j: usize, value: f32) {
        if i < self.size && j < self.size && i != j {
            self.data[i * self.size + j] = value;
            self.data[j * self.size + i] = value;

            if i < j {
                self.pairs_computed += 1;
            }
        }
    }

    /// Check if construction is complete
    pub fn is_complete(&self) -> bool {
        self.pairs_computed >= self.total_pairs
    }

    /// Get completion percentage
    pub fn completion_percentage(&self) -> f32 {
        if self.total_pairs == 0 {
            100.0
        } else {
            (self.pairs_computed as f32 / self.total_pairs as f32) * 100.0
        }
    }

    /// Build final correlation matrix
    pub fn build(self) -> Result<CorrelationMatrix, CorrelationError> {
        CorrelationMatrix::new(self.size, self.data)
    }

    /// Build partial matrix (even if not complete)
    pub fn build_partial(self) -> CorrelationMatrix {
        CorrelationMatrix::new(self.size, self.data)
            .unwrap_or_else(|_| CorrelationMatrix::identity(self.size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_matrix_creation() {
        let size = 3;
        let data = vec![1.0, 0.5, -0.2, 0.5, 1.0, 0.8, -0.2, 0.8, 1.0];

        let matrix = CorrelationMatrix::new(size, data).unwrap();

        assert_eq!(matrix.size(), size);
        assert!(matrix.diagonal_ones());
        assert!(matrix.is_symmetric());

        assert_eq!(matrix.get(0, 1), 0.5);
        assert_eq!(matrix.get(1, 0), 0.5);
        assert_eq!(matrix.get(0, 2), -0.2);

        println!("✅ Correlation matrix creation verified");
    }

    #[test]
    fn test_matrix_statistics() {
        let size = 3;
        let data = vec![1.0, 0.8, -0.6, 0.8, 1.0, 0.2, -0.6, 0.2, 1.0];

        let matrix = CorrelationMatrix::new(size, data).unwrap();
        let stats = matrix.statistics();

        assert!(stats.min >= -1.0);
        assert!(stats.max <= 1.0);
        assert!(stats.count == 6); // 3x3 matrix has 6 off-diagonal elements

        println!(
            "✅ Matrix statistics: mean={:.3}, std={:.3}, range=[{:.3}, {:.3}]",
            stats.mean, stats.std_dev, stats.min, stats.max
        );
    }

    #[test]
    fn test_highest_correlations() {
        let size = 4;
        let data = vec![
            1.0, 0.9, 0.1, -0.3, 0.9, 1.0, 0.7, -0.5, 0.1, 0.7, 1.0, 0.2, -0.3, -0.5, 0.2, 1.0,
        ];

        let matrix = CorrelationMatrix::new(size, data).unwrap();
        let highest = matrix.highest_correlations(3);

        assert!(highest.len() <= 3);
        assert_eq!(highest[0].2, 0.9); // Highest correlation

        println!("✅ Highest correlations: {:?}", highest);
    }

    #[test]
    fn test_sparse_conversion() {
        let size = 3;
        let data = vec![1.0, 0.8, 0.0, 0.8, 1.0, 0.5, 0.0, 0.5, 1.0];

        let matrix = CorrelationMatrix::new(size, data).unwrap();
        let sparse = matrix.to_sparse();
        let reconstructed = sparse.to_dense();

        // Verify reconstruction
        for i in 0..size {
            for j in 0..size {
                let original = matrix.get(i, j);
                let reconstructed_val = reconstructed.get(i, j);
                assert!((original - reconstructed_val).abs() < 1e-6);
            }
        }

        println!(
            "✅ Sparse conversion verified ({} entries)",
            sparse.entries.len()
        );
    }

    #[test]
    fn test_matrix_builder() {
        let size = 3;
        let mut builder = CorrelationMatrixBuilder::new(size);

        assert!(!builder.is_complete());
        assert_eq!(builder.completion_percentage(), 0.0);

        builder.set_correlation(0, 1, 0.7);
        builder.set_correlation(0, 2, -0.3);
        builder.set_correlation(1, 2, 0.5);

        assert!(builder.is_complete());
        assert_eq!(builder.completion_percentage(), 100.0);

        let matrix = builder.build().unwrap();
        assert_eq!(matrix.get(0, 1), 0.7);
        assert_eq!(matrix.get(1, 0), 0.7); // Symmetry

        println!("✅ Matrix builder verified");
    }

    #[test]
    fn test_invalid_matrix_creation() {
        // Test with invalid correlation values
        let size = 2;
        let data = vec![1.0, 2.0, 2.0, 1.0]; // 2.0 is out of range

        let result = CorrelationMatrix::new(size, data);
        assert!(result.is_err());

        // Test with NaN values
        let data = vec![1.0, f32::NAN, f32::NAN, 1.0];
        let result = CorrelationMatrix::new(size, data);
        assert!(result.is_err());

        println!("✅ Invalid matrix creation properly rejected");
    }
}
