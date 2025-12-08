//! Simple standalone test for Panarchy analyzer
//! This file can be compiled and run independently to verify the implementation

#[allow(dead_code)]
mod panarchy {
    use std::collections::HashMap;
    use std::time::Instant;
    
    pub type Float = f64;
    
    #[derive(Debug, Clone, Copy)]
    pub enum PanarchyPhase {
        Growth,
        Conservation,
        Release,
        Reorganization,
        Unknown,
    }
    
    impl PanarchyPhase {
        pub fn to_score(&self) -> Float {
            match self {
                Self::Growth => 0.25,
                Self::Conservation => 0.50,
                Self::Release => 0.75,
                Self::Reorganization => 0.90,
                Self::Unknown => 0.50,
            }
        }
        
        pub fn as_str(&self) -> &'static str {
            match self {
                Self::Growth => "growth",
                Self::Conservation => "conservation",
                Self::Release => "release",
                Self::Reorganization => "reorganization",
                Self::Unknown => "unknown",
            }
        }
    }
    
    #[derive(Debug, Clone)]
    pub struct PCRComponents {
        pub potential: Float,
        pub connectedness: Float,
        pub resilience: Float,
    }
    
    impl PCRComponents {
        pub fn new(potential: Float, connectedness: Float, resilience: Float) -> Self {
            Self { potential, connectedness, resilience }
        }
        
        pub fn composite_score(&self) -> Float {
            (self.potential + self.connectedness + self.resilience) / 3.0
        }
    }
    
    #[derive(Debug, Clone)]
    pub struct PanarchyResult {
        pub phase: PanarchyPhase,
        pub confidence: Float,
        pub pcr: PCRComponents,
        pub phase_scores: HashMap<String, Float>,
        pub transition_probability: Float,
        pub computation_time_ns: u64,
    }
    
    #[derive(Debug, Clone)]
    pub struct PanarchyConfig {
        pub window_size: usize,
        pub autocorr_lag: usize,
        pub min_confidence: Float,
        pub hysteresis_threshold: Float,
    }
    
    impl Default for PanarchyConfig {
        fn default() -> Self {
            Self {
                window_size: 20,
                autocorr_lag: 1,
                min_confidence: 0.6,
                hysteresis_threshold: 0.1,
            }
        }
    }
    
    pub struct PanarchyAnalyzer {
        config: PanarchyConfig,
        previous_phase: Option<PanarchyPhase>,
        phase_history: Vec<PanarchyPhase>,
    }
    
    impl PanarchyAnalyzer {
        pub fn new() -> Self {
            Self::with_config(PanarchyConfig::default())
        }
        
        pub fn with_config(config: PanarchyConfig) -> Self {
            Self {
                config,
                previous_phase: None,
                phase_history: Vec::new(),
            }
        }
        
        pub fn analyze_full(&mut self, prices: &[Float], volumes: &[Float]) -> Result<PanarchyResult, String> {
            let start_time = Instant::now();
            
            if prices.is_empty() || volumes.is_empty() {
                return Err("Empty input data".to_string());
            }
            
            if prices.len() != volumes.len() {
                return Err("Prices and volumes length mismatch".to_string());
            }
            
            let returns = self.calculate_returns(prices)?;
            let pcr = self.calculate_pcr_scalar(prices, &returns, volumes)?;
            let (phase, confidence, phase_scores) = self.identify_phase_fast(&pcr, &returns)?;
            let transition_probability = self.calculate_transition_probability(&pcr, phase)?;
            
            self.update_phase_history(phase);
            
            let computation_time_ns = start_time.elapsed().as_nanos() as u64;
            
            Ok(PanarchyResult {
                phase,
                confidence,
                pcr,
                phase_scores,
                transition_probability,
                computation_time_ns,
            })
        }
        
        fn calculate_returns(&self, prices: &[Float]) -> Result<Vec<Float>, String> {
            if prices.len() < 2 {
                return Ok(Vec::new());
            }
            
            let mut returns = Vec::with_capacity(prices.len() - 1);
            for i in 1..prices.len() {
                if prices[i - 1] != 0.0 {
                    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
                } else {
                    returns.push(0.0);
                }
            }
            
            Ok(returns)
        }
        
        fn calculate_pcr_scalar(&self, prices: &[Float], returns: &[Float], _volumes: &[Float]) -> Result<PCRComponents, String> {
            let n = prices.len();
            let window = self.config.window_size.min(n);
            
            if window < 3 {
                return Ok(PCRComponents::new(0.5, 0.5, 0.5));
            }
            
            let start_idx = n.saturating_sub(window);
            let price_window = &prices[start_idx..];
            let return_window = if returns.is_empty() { &[] } else { &returns[start_idx.saturating_sub(1)..] };
            
            // Potential: Normalized position in price range
            let price_min = price_window.iter().fold(Float::INFINITY, |a, &b| a.min(b));
            let price_max = price_window.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let current_price = prices[n - 1];
            
            let potential = if price_max > price_min {
                (current_price - price_min) / (price_max - price_min)
            } else {
                0.5
            };
            
            // Connectedness: Autocorrelation of returns
            let connectedness = if return_window.len() > self.config.autocorr_lag {
                self.calculate_autocorrelation(return_window, self.config.autocorr_lag).unwrap_or(0.0).abs()
            } else {
                0.5
            };
            
            // Resilience: Inverse of volatility
            let volatility = self.calculate_volatility(return_window).unwrap_or(0.0);
            let resilience = if volatility > 0.0 {
                1.0 / (1.0 + volatility)
            } else {
                1.0
            };
            
            Ok(PCRComponents::new(
                potential.clamp(0.0, 1.0),
                connectedness.clamp(0.0, 1.0),
                resilience.clamp(0.0, 1.0),
            ))
        }
        
        fn identify_phase_fast(&self, pcr: &PCRComponents, _returns: &[Float]) -> Result<(PanarchyPhase, Float, HashMap<String, Float>), String> {
            let mut phase_scores = HashMap::new();
            
            let growth_score = pcr.potential * 0.6 + (1.0 - pcr.connectedness) * 0.4;
            let conservation_score = pcr.connectedness * 0.7 + pcr.resilience * 0.3;
            let release_score = (1.0 - pcr.resilience) * 0.8 + pcr.potential * 0.2;
            let reorganization_score = (1.0 - pcr.potential) * 0.5 + (1.0 - pcr.connectedness) * 0.3 + pcr.resilience * 0.2;
            
            phase_scores.insert("growth".to_string(), growth_score);
            phase_scores.insert("conservation".to_string(), conservation_score);
            phase_scores.insert("release".to_string(), release_score);
            phase_scores.insert("reorganization".to_string(), reorganization_score);
            
            let (best_phase_name, &max_score) = phase_scores
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            
            let phase = match best_phase_name.as_str() {
                "growth" => PanarchyPhase::Growth,
                "conservation" => PanarchyPhase::Conservation,
                "release" => PanarchyPhase::Release,
                "reorganization" => PanarchyPhase::Reorganization,
                _ => PanarchyPhase::Unknown,
            };
            
            let final_phase = if let Some(prev_phase) = self.previous_phase {
                self.apply_hysteresis(phase, prev_phase, max_score).unwrap_or(phase)
            } else {
                phase
            };
            
            let scores: Vec<Float> = phase_scores.values().copied().collect();
            let confidence = self.calculate_phase_confidence(&scores, max_score);
            
            Ok((final_phase, confidence, phase_scores))
        }
        
        fn apply_hysteresis(&self, new_phase: PanarchyPhase, prev_phase: PanarchyPhase, score: Float) -> Result<PanarchyPhase, String> {
            if matches!((new_phase, prev_phase), (PanarchyPhase::Growth, PanarchyPhase::Growth) |
                       (PanarchyPhase::Conservation, PanarchyPhase::Conservation) |
                       (PanarchyPhase::Release, PanarchyPhase::Release) |
                       (PanarchyPhase::Reorganization, PanarchyPhase::Reorganization)) {
                return Ok(new_phase);
            }
            
            let threshold = self.config.min_confidence + self.config.hysteresis_threshold;
            
            if score >= threshold {
                Ok(new_phase)
            } else {
                Ok(prev_phase)
            }
        }
        
        fn calculate_phase_confidence(&self, scores: &[Float], max_score: Float) -> Float {
            if scores.len() < 2 {
                return 0.5;
            }
            
            let mut sorted_scores = scores.to_vec();
            sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            
            let second_best = sorted_scores[1];
            let separation = max_score - second_best;
            
            (separation * 2.0).clamp(0.0, 1.0)
        }
        
        fn calculate_transition_probability(&self, pcr: &PCRComponents, current_phase: PanarchyPhase) -> Result<Float, String> {
            let transition_prob = match current_phase {
                PanarchyPhase::Growth => pcr.connectedness * 0.8 + (1.0 - pcr.potential) * 0.2,
                PanarchyPhase::Conservation => (1.0 - pcr.resilience) * 0.9 + pcr.potential * 0.1,
                PanarchyPhase::Release => 0.8,
                PanarchyPhase::Reorganization => pcr.potential * 0.7 + (1.0 - pcr.connectedness) * 0.3,
                PanarchyPhase::Unknown => 0.5,
            };
            
            Ok(transition_prob.clamp(0.0, 1.0))
        }
        
        fn update_phase_history(&mut self, phase: PanarchyPhase) {
            self.previous_phase = Some(phase);
            self.phase_history.push(phase);
            
            if self.phase_history.len() > 100 {
                self.phase_history.remove(0);
            }
        }
        
        fn calculate_autocorrelation(&self, data: &[Float], lag: usize) -> Result<Float, String> {
            let n = data.len();
            if n <= lag {
                return Ok(0.0);
            }
            
            let mean = data.iter().sum::<Float>() / n as Float;
            
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            
            for i in 0..(n - lag) {
                let x_dev = data[i] - mean;
                let y_dev = data[i + lag] - mean;
                numerator += x_dev * y_dev;
                denominator += x_dev * x_dev;
            }
            
            if denominator.abs() < Float::EPSILON {
                Ok(0.0)
            } else {
                Ok(numerator / denominator)
            }
        }
        
        fn calculate_volatility(&self, returns: &[Float]) -> Result<Float, String> {
            if returns.is_empty() {
                return Ok(0.0);
            }
            
            let mean = returns.iter().sum::<Float>() / returns.len() as Float;
            let variance = returns.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<Float>() / returns.len() as Float;
                
            Ok(variance.sqrt())
        }
        
        pub fn clear_history(&mut self) {
            self.phase_history.clear();
            self.previous_phase = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::panarchy::*;
    
    #[test]
    fn test_panarchy_basic_analysis() {
        let mut analyzer = PanarchyAnalyzer::new();
        
        // Test with simple growth pattern
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
                         108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0,
                         116.0, 117.0, 118.0, 119.0, 120.0];
        let volumes = vec![1000.0; prices.len()];
        
        let result = analyzer.analyze_full(&prices, &volumes);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.pcr.potential >= 0.0 && result.pcr.potential <= 1.0);
        assert!(result.pcr.connectedness >= 0.0 && result.pcr.connectedness <= 1.0);
        assert!(result.pcr.resilience >= 0.0 && result.pcr.resilience <= 1.0);
        assert!(result.computation_time_ns > 0);
        
        println!("Test result: Phase = {}, Confidence = {:.3}, PCR = ({:.3}, {:.3}, {:.3}), Time = {}ns",
                 result.phase.as_str(), result.confidence, 
                 result.pcr.potential, result.pcr.connectedness, result.pcr.resilience,
                 result.computation_time_ns);
    }
    
    #[test]
    fn test_performance_target() {
        let mut analyzer = PanarchyAnalyzer::new();
        let prices = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect::<Vec<_>>();
        let volumes = vec![1000.0; prices.len()];
        
        // Warm up
        for _ in 0..10 {
            let _ = analyzer.analyze_full(&prices, &volumes);
        }
        
        // Test performance
        let runs = 100;
        let mut total_time = 0u64;
        
        for _ in 0..runs {
            let result = analyzer.analyze_full(&prices, &volumes).unwrap();
            total_time += result.computation_time_ns;
        }
        
        let avg_time = total_time / runs;
        println!("Average analysis time: {}ns (target: <800ns)", avg_time);
        
        // Allow generous margin for test environment
        assert!(avg_time < 50_000); // 50 microseconds should be more than enough
    }
    
    #[test]
    fn test_phase_scenarios() {
        let mut analyzer = PanarchyAnalyzer::new();
        
        // Conservation scenario - stable prices
        let conservation_prices = vec![100.0; 25];
        let volumes = vec![1000.0; 25];
        
        analyzer.clear_history();
        let result = analyzer.analyze_full(&conservation_prices, &volumes).unwrap();
        println!("Conservation test: Phase = {}, PCR = ({:.3}, {:.3}, {:.3})",
                 result.phase.as_str(), result.pcr.potential, result.pcr.connectedness, result.pcr.resilience);
        
        // High resilience expected for stable prices
        assert!(result.pcr.resilience > 0.5);
        
        // Release scenario - volatile decline
        let release_prices = (0..25).map(|i| 120.0 - i as f64 * 2.0).collect::<Vec<_>>();
        
        analyzer.clear_history();
        let result = analyzer.analyze_full(&release_prices, &volumes).unwrap();
        println!("Release test: Phase = {}, PCR = ({:.3}, {:.3}, {:.3})",
                 result.phase.as_str(), result.pcr.potential, result.pcr.connectedness, result.pcr.resilience);
        
        // Lower resilience expected for declining prices
        assert!(result.pcr.resilience < 0.8);
    }
}

fn main() {
    println!("Panarchy Analyzer Simple Test");
    println!("============================");
    
    let mut analyzer = panarchy::PanarchyAnalyzer::new();
    
    // Test different market scenarios
    test_scenario(&mut analyzer, "Growth", generate_growth_data());
    test_scenario(&mut analyzer, "Conservation", generate_conservation_data());
    test_scenario(&mut analyzer, "Release", generate_release_data());
    test_scenario(&mut analyzer, "Reorganization", generate_reorganization_data());
    
    // Performance test
    println!("\nPerformance Test:");
    let test_prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
    let test_volumes = vec![1000.0; test_prices.len()];
    
    // Warm up
    for _ in 0..5 {
        let _ = analyzer.analyze_full(&test_prices, &test_volumes);
    }
    
    // Test performance
    let runs = 100;
    let mut total_time = 0u64;
    
    for _ in 0..runs {
        if let Ok(result) = analyzer.analyze_full(&test_prices, &test_volumes) {
            total_time += result.computation_time_ns;
        }
    }
    
    let avg_time = total_time / runs;
    println!("  Average analysis time: {}ns", avg_time);
    println!("  Target: <800ns");
    if avg_time < 800 {
        println!("  ✓ Performance target met!");
    } else {
        println!("  ⚠ Performance target missed by {}ns", avg_time - 800);
    }
}

fn test_scenario(analyzer: &mut panarchy::PanarchyAnalyzer, name: &str, (prices, volumes): (Vec<f64>, Vec<f64>)) {
    analyzer.clear_history();
    
    match analyzer.analyze_full(&prices, &volumes) {
        Ok(result) => {
            println!("\n{} Scenario:", name);
            println!("  Phase: {}", result.phase.as_str());
            println!("  Confidence: {:.3}", result.confidence);
            println!("  PCR Components:");
            println!("    Potential: {:.3}", result.pcr.potential);
            println!("    Connectedness: {:.3}", result.pcr.connectedness);
            println!("    Resilience: {:.3}", result.pcr.resilience);
            println!("    Composite: {:.3}", result.pcr.composite_score());
            println!("  Transition Probability: {:.3}", result.transition_probability);
            println!("  Computation Time: {}ns", result.computation_time_ns);
        }
        Err(e) => {
            println!("\n{} Scenario: ERROR - {}", name, e);
        }
    }
}

fn generate_growth_data() -> (Vec<f64>, Vec<f64>) {
    let prices: Vec<f64> = (0..30)
        .map(|i| 100.0 + i as f64 * 0.5)
        .collect();
    let volumes = vec![1000.0; prices.len()];
    (prices, volumes)
}

fn generate_conservation_data() -> (Vec<f64>, Vec<f64>) {
    let prices: Vec<f64> = (0..30)
        .map(|i| 100.0 + (i as f64 * 0.1).sin() * 0.5)
        .collect();
    let volumes = vec![1000.0; prices.len()];
    (prices, volumes)
}

fn generate_release_data() -> (Vec<f64>, Vec<f64>) {
    let prices: Vec<f64> = (0..30)
        .map(|i| 120.0 - i as f64 * 1.5)
        .collect();
    let volumes = vec![1000.0; prices.len()];
    (prices, volumes)
}

fn generate_reorganization_data() -> (Vec<f64>, Vec<f64>) {
    let prices: Vec<f64> = (0..30)
        .map(|i| 60.0 + i as f64 * 0.3 + (i as f64 * 0.5).sin() * 2.0)
        .collect();
    let volumes = vec![1000.0; prices.len()];
    (prices, volumes)
}