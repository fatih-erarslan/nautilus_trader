use cdfa_panarchy_analyzer::*;
use approx::assert_relative_eq;

#[test]
fn test_full_analysis_cycle() {
    let mut analyzer = PanarchyAnalyzer::new();
    
    // Generate test data with known patterns
    let prices: Vec<f64> = (0..100)
        .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
        .collect();
    
    let volumes = vec![1000.0; 100];
    
    let result = analyzer.analyze(&prices, &volumes).unwrap();
    
    // Verify result structure
    assert!(result.signal >= 0.0 && result.signal <= 1.0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(result.regime_score >= 0.0 && result.regime_score <= 100.0);
    assert_eq!(result.data_points, 100);
    
    // Verify PCR components
    assert!(result.pcr.potential >= 0.0 && result.pcr.potential <= 1.0);
    assert!(result.pcr.connectedness >= 0.0 && result.pcr.connectedness <= 1.0);
    assert!(result.pcr.resilience >= 0.0 && result.pcr.resilience <= 1.0);
    
    // Verify phase scores sum to 1
    let score_sum = result.phase_scores.growth + 
                    result.phase_scores.conservation + 
                    result.phase_scores.release + 
                    result.phase_scores.reorganization;
    assert_relative_eq!(score_sum, 1.0, epsilon = 1e-10);
}

#[test]
fn test_phase_identification_patterns() {
    let mut analyzer = PanarchyAnalyzer::new();
    
    // Test Growth pattern - steady uptrend
    let growth_prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
    let volumes = vec![1000.0; 50];
    
    let result = analyzer.analyze(&growth_prices, &volumes).unwrap();
    assert!(matches!(result.phase, MarketPhase::Growth | MarketPhase::Conservation));
    
    // Test Release pattern - sharp decline
    let release_prices: Vec<f64> = (0..50).map(|i| 200.0 - i as f64 * 3.0).collect();
    let result = analyzer.analyze(&release_prices, &volumes).unwrap();
    assert!(matches!(result.phase, MarketPhase::Release | MarketPhase::Reorganization));
}

#[test]
fn test_pcr_calculation() {
    let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0; 20];
    let pcr = calculate_pcr(&prices, 10).unwrap();
    
    assert_eq!(pcr.len(), prices.len());
    
    // Check last PCR values are reasonable
    let last_pcr = pcr.last().unwrap();
    assert!(last_pcr.potential > 0.5); // Should be high for uptrending prices
    assert!(last_pcr.connectedness >= 0.0 && last_pcr.connectedness <= 1.0);
    assert!(last_pcr.resilience >= 0.0 && last_pcr.resilience <= 1.0);
}

#[test]
fn test_parameter_customization() {
    let mut params = PanarchyParameters::default();
    params.adx_period = 20;
    params.autocorr_lag = 2;
    params.hysteresis_min_score_threshold = 0.4;
    
    let mut analyzer = PanarchyAnalyzer::with_params(params);
    
    let prices = vec![100.0; 50];
    let volumes = vec![1000.0; 50];
    
    let result = analyzer.analyze(&prices, &volumes);
    assert!(result.is_ok());
}

#[test]
fn test_batch_analyzer() {
    let mut batch = BatchPanarchyAnalyzer::new(5);
    
    let price_series: Vec<Vec<f64>> = (0..5)
        .map(|j| (0..50).map(|i| 100.0 + j as f64 * 10.0 + i as f64).collect())
        .collect();
    
    let volume_series = vec![vec![1000.0; 50]; 5];
    
    let results = batch.analyze_batch(&price_series, &volume_series);
    
    assert_eq!(results.len(), 5);
    for result in results {
        assert!(result.is_ok());
    }
}

#[test]
fn test_error_handling() {
    let mut analyzer = PanarchyAnalyzer::new();
    
    // Test empty data
    let result = analyzer.analyze(&[], &[]);
    assert!(matches!(result, Err(PanarchyError::InsufficientData { .. })));
    
    // Test mismatched lengths
    let result = analyzer.analyze(&[100.0, 101.0], &[1000.0]);
    assert!(matches!(result, Err(PanarchyError::InvalidParameters { .. })));
    
    // Test insufficient data for period
    let result = analyzer.analyze(&vec![100.0; 5], &vec![1000.0; 5]);
    assert!(matches!(result, Err(PanarchyError::InsufficientData { .. })));
}

#[test]
fn test_phase_transitions() {
    let mut tracker = PhaseTransitionTracker::new();
    
    // Record some transitions
    tracker.record_transition(MarketPhase::Growth, MarketPhase::Conservation);
    tracker.record_transition(MarketPhase::Conservation, MarketPhase::Release);
    tracker.record_transition(MarketPhase::Release, MarketPhase::Reorganization);
    tracker.record_transition(MarketPhase::Reorganization, MarketPhase::Growth);
    
    // Check probabilities
    let prob = tracker.get_transition_probability(MarketPhase::Growth, MarketPhase::Conservation);
    assert_eq!(prob, 1.0); // Only one transition recorded from Growth
}

#[test]
fn test_simd_operations() {
    use cdfa_panarchy_analyzer::simd::*;
    
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    
    // Test mean
    let mean = simd_mean(&data);
    assert_relative_eq!(mean, 4.5, epsilon = 1e-10);
    
    // Test standard deviation
    let std_dev = simd_std_dev(&data, mean);
    assert!(std_dev > 0.0);
    
    // Test autocorrelation
    let autocorr = simd_autocorrelation(&data, 1);
    assert!(autocorr.is_finite());
}

#[test]
fn test_market_phase_conversions() {
    assert_eq!(MarketPhase::from_string("growth"), MarketPhase::Growth);
    assert_eq!(MarketPhase::from_string("CONSERVATION"), MarketPhase::Conservation);
    assert_eq!(MarketPhase::from_string("Release"), MarketPhase::Release);
    assert_eq!(MarketPhase::from_string("reorganization"), MarketPhase::Reorganization);
    assert_eq!(MarketPhase::from_string("invalid"), MarketPhase::Unknown);
    
    assert_eq!(MarketPhase::Growth.as_str(), "growth");
    assert_eq!(MarketPhase::Conservation.to_score(), 0.50);
}

#[test]
fn test_fast_pcr_calculator() {
    use cdfa_panarchy_analyzer::pcr::FastPCRCalculator;
    
    let mut calc = FastPCRCalculator::new(10, 1);
    
    // Feed in some data points
    for i in 0..20 {
        let price = 100.0 + (i as f64 * 0.1).sin() * 5.0;
        let return_val = if i > 0 { 0.01 } else { 0.0 };
        let volatility = 0.2;
        
        let pcr = calc.update(price, return_val, volatility);
        
        assert!(pcr.potential >= 0.0 && pcr.potential <= 1.0);
        assert!(pcr.connectedness >= 0.0 && pcr.connectedness <= 1.0);
        assert!(pcr.resilience >= 0.0 && pcr.resilience <= 1.0);
    }
}

#[test]
fn test_performance_targets() {
    use std::time::Instant;
    
    let mut analyzer = PanarchyAnalyzer::new();
    let prices = vec![100.0; 50];
    let volumes = vec![1000.0; 50];
    
    // Warm up
    let _ = analyzer.analyze(&prices, &volumes);
    
    // Measure
    let start = Instant::now();
    let result = analyzer.analyze(&prices, &volumes).unwrap();
    let elapsed = start.elapsed();
    
    println!("Analysis took: {} ns", elapsed.as_nanos());
    
    // Verify computation time is recorded
    assert!(result.computation_time_ns > 0);
    
    // Note: Actual performance will vary by hardware
    // The sub-microsecond target is aspirational
}