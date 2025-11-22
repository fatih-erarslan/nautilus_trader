//! Pattern observer implementation

use crate::prelude::*;

/// Observer for pattern recognition in market data
#[derive(Debug, Clone)]
pub struct PatternObserver {
    pub window_size: usize,
    pub min_pattern_length: usize,
    pub similarity_threshold: f64,
    pub patterns: Vec<PatternTemplate>,
}

#[derive(Debug, Clone)]
pub struct PatternTemplate {
    pub name: String,
    pub data: Vec<f64>,
    pub significance: f64,
    pub frequency: usize,
}

#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub template: PatternTemplate,
    pub position: usize,
    pub similarity: f64,
    pub confidence: f64,
    pub projected_outcome: Option<f64>,
}

impl PatternObserver {
    pub fn new(window_size: usize, min_pattern_length: usize, similarity_threshold: f64) -> Self {
        Self {
            window_size,
            min_pattern_length,
            similarity_threshold,
            patterns: Vec::new(),
        }
    }
    
    pub fn add_pattern(&mut self, name: String, data: Vec<f64>, significance: f64) {
        self.patterns.push(PatternTemplate {
            name,
            data,
            significance,
            frequency: 0,
        });
    }
    
    pub fn observe(&self, price_data: &[f64]) -> Vec<PatternMatch> {
        if price_data.len() < self.window_size {
            return Vec::new();
        }
        
        let recent_data = &price_data[price_data.len() - self.window_size..];
        let normalized_data = self.normalize_data(recent_data);
        
        let mut matches = Vec::new();
        
        for pattern in &self.patterns {
            if let Some(pattern_match) = self.match_pattern(&normalized_data, pattern) {
                matches.push(pattern_match);
            }
        }
        
        // Sort by confidence/similarity
        matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        matches
    }
    
    fn normalize_data(&self, data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return Vec::new();
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return vec![0.0; data.len()];
        }
        
        data.iter()
            .map(|x| (x - mean) / std_dev)
            .collect()
    }
    
    fn match_pattern(&self, data: &[f64], pattern: &PatternTemplate) -> Option<PatternMatch> {
        if pattern.data.len() > data.len() {
            return None;
        }
        
        let normalized_pattern = self.normalize_data(&pattern.data);
        let mut best_similarity = 0.0;
        let mut best_position = 0;
        
        // Sliding window pattern matching
        for i in 0..=(data.len() - normalized_pattern.len()) {
            let window = &data[i..i + normalized_pattern.len()];
            let similarity = self.calculate_similarity(window, &normalized_pattern);
            
            if similarity > best_similarity {
                best_similarity = similarity;
                best_position = i;
            }
        }
        
        if best_similarity >= self.similarity_threshold {
            let confidence = self.calculate_confidence(best_similarity, pattern);
            let projected_outcome = self.project_outcome(pattern, best_similarity);
            
            Some(PatternMatch {
                template: pattern.clone(),
                position: best_position,
                similarity: best_similarity,
                confidence,
                projected_outcome,
            })
        } else {
            None
        }
    }
    
    fn calculate_similarity(&self, data1: &[f64], data2: &[f64]) -> f64 {
        if data1.len() != data2.len() {
            return 0.0;
        }
        
        // Pearson correlation coefficient
        let n = data1.len() as f64;
        let sum1: f64 = data1.iter().sum();
        let sum2: f64 = data2.iter().sum();
        let sum1_sq: f64 = data1.iter().map(|x| x * x).sum();
        let sum2_sq: f64 = data2.iter().map(|x| x * x).sum();
        let sum12: f64 = data1.iter().zip(data2.iter()).map(|(x, y)| x * y).sum();
        
        let numerator = n * sum12 - sum1 * sum2;
        let denominator = ((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2)).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            (numerator / denominator).abs() // Absolute correlation
        }
    }
    
    fn calculate_confidence(&self, similarity: f64, pattern: &PatternTemplate) -> f64 {
        // Weight by pattern significance and historical frequency
        let base_confidence = similarity;
        let significance_weight = pattern.significance;
        let frequency_weight = (pattern.frequency as f64).ln().max(1.0) / 10.0;
        
        (base_confidence * significance_weight * frequency_weight).min(1.0)
    }
    
    fn project_outcome(&self, pattern: &PatternTemplate, similarity: f64) -> Option<f64> {
        // Simple projection based on pattern's historical outcomes
        // In a real implementation, this would use more sophisticated methods
        if similarity > 0.8 {
            Some(pattern.significance * similarity * 0.1) // 10% of significance as projection
        } else {
            None
        }
    }
    
    pub fn learn_pattern(&mut self, name: String, data: Vec<f64>, outcome: f64) {
        if data.len() < self.min_pattern_length {
            return;
        }
        
        // Check if similar pattern exists
        let normalized_data = self.normalize_data(&data);
        let mut found_similar = false;
        
        for i in 0..self.patterns.len() {
            let normalized_pattern = self.normalize_data(&self.patterns[i].data);
            let similarity = self.calculate_similarity(&normalized_data, &normalized_pattern);
            
            if similarity > 0.9 { // Very similar pattern
                // Update existing pattern
                self.patterns[i].frequency += 1;
                self.patterns[i].significance = (self.patterns[i].significance + outcome.abs()) / 2.0;
                found_similar = true;
                break;
            }
        }
        
        if !found_similar {
            // Add new pattern
            self.patterns.push(PatternTemplate {
                name,
                data,
                significance: outcome.abs(),
                frequency: 1,
            });
        }
    }
    
    pub fn get_pattern_statistics(&self) -> PatternStatistics {
        let total_patterns = self.patterns.len();
        let avg_significance = if total_patterns > 0 {
            self.patterns.iter().map(|p| p.significance).sum::<f64>() / total_patterns as f64
        } else {
            0.0
        };
        
        let most_frequent = self.patterns.iter()
            .max_by_key(|p| p.frequency)
            .cloned();
            
        let most_significant = self.patterns.iter()
            .max_by(|a, b| a.significance.partial_cmp(&b.significance).unwrap())
            .cloned();
        
        PatternStatistics {
            total_patterns,
            average_significance: avg_significance,
            most_frequent_pattern: most_frequent,
            most_significant_pattern: most_significant,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatternStatistics {
    pub total_patterns: usize,
    pub average_significance: f64,
    pub most_frequent_pattern: Option<PatternTemplate>,
    pub most_significant_pattern: Option<PatternTemplate>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pattern_observer() {
        let mut observer = PatternObserver::new(20, 5, 0.7);
        
        // Add a simple trend pattern
        let trend_pattern = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
        observer.add_pattern("uptrend".to_string(), trend_pattern, 0.8);
        
        // Test data with similar trend
        let test_data: Vec<f64> = (0..25).map(|i| 100.0 + i as f64 * 0.5).collect();
        
        let matches = observer.observe(&test_data);
        assert!(!matches.is_empty());
    }
    
    #[test]
    fn test_pattern_learning() {
        let mut observer = PatternObserver::new(15, 3, 0.6);
        
        let pattern_data = vec![1.0, 0.9, 1.1, 1.2, 0.8];
        observer.learn_pattern("test_pattern".to_string(), pattern_data, 0.5);
        
        assert_eq!(observer.patterns.len(), 1);
        assert_eq!(observer.patterns[0].name, "test_pattern");
    }
}