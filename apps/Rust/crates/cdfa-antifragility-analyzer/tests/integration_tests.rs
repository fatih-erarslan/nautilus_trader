use cdfa_antifragility_analyzer::{
    AntifragilityAnalyzer, AntifragilityParameters, AntifragilityError
};

#[test]
fn test_basic_analysis() {
    let analyzer = AntifragilityAnalyzer::new();
    let (prices, volumes) = generate_test_data(200);
    
    let result = analyzer.analyze_prices(&prices, &volumes);
    assert!(result.is_ok());
    
    let analysis = result.unwrap();
    assert!(analysis.antifragility_index >= 0.0);
    assert!(analysis.antifragility_index <= 1.0);
    assert!(analysis.fragility_score >= 0.0);
    assert!(analysis.fragility_score <= 1.0);
    assert_eq!(analysis.data_points, 200);
}

#[test]
fn test_insufficient_data() {
    let analyzer = AntifragilityAnalyzer::new();
    let (prices, volumes) = generate_test_data(50);
    
    let result = analyzer.analyze_prices(&prices, &volumes);
    assert!(result.is_err());
    
    match result.unwrap_err() {
        AntifragilityError::InsufficientData { required, actual } => {
            assert!(actual < required);
        }
        _ => panic!("Expected InsufficientData error"),
    }
}

#[test]
fn test_mismatched_arrays() {
    let analyzer = AntifragilityAnalyzer::new();
    let prices = vec![100.0, 101.0, 102.0];
    let volumes = vec![1000.0, 1100.0]; // Different length
    
    let result = analyzer.analyze_prices(&prices, &volumes);
    assert!(result.is_err());
}

#[test]
fn test_parameter_validation() {
    let mut params = AntifragilityParameters::default();
    params.convexity_weight = 0.5;
    params.asymmetry_weight = 0.5;
    params.recovery_weight = 0.0;
    params.benefit_ratio_weight = 0.0;
    
    assert!(params.validate().is_ok());
    
    params.convexity_weight = 0.6; // Sum > 1.0
    assert!(params.validate().is_err());
}

#[test]
fn test_cache_functionality() {
    let analyzer = AntifragilityAnalyzer::new();
    let (prices, volumes) = generate_test_data(200);
    
    // First analysis
    let result1 = analyzer.analyze_prices(&prices, &volumes);
    assert!(result1.is_ok());
    
    // Second analysis (should be faster due to caching)
    let result2 = analyzer.analyze_prices(&prices, &volumes);
    assert!(result2.is_ok());
    
    // Results should be identical
    let analysis1 = result1.unwrap();
    let analysis2 = result2.unwrap();
    
    assert_eq!(analysis1.antifragility_index, analysis2.antifragility_index);
    assert_eq!(analysis1.fragility_score, analysis2.fragility_score);
}

#[test]
fn test_performance_metrics() {
    let analyzer = AntifragilityAnalyzer::new();
    let (prices, volumes) = generate_test_data(200);
    
    let _result = analyzer.analyze_prices(&prices, &volumes);
    
    let metrics = analyzer.get_performance_metrics();
    assert!(metrics.total_analyses > 0);
    assert!(metrics.total_analysis_time.as_nanos() > 0);
    assert!(metrics.success_count > 0);
    assert_eq!(metrics.error_count, 0);
}

#[test]
fn test_different_parameter_configurations() {
    let test_configs = vec![
        // Balanced
        (0.25, 0.25, 0.25, 0.25),
        // Convexity focused
        (0.70, 0.10, 0.10, 0.10),
        // Recovery focused
        (0.10, 0.10, 0.70, 0.10),
    ];
    
    for (conv_w, asym_w, rec_w, ben_w) in test_configs {
        let mut params = AntifragilityParameters::default();
        params.convexity_weight = conv_w;
        params.asymmetry_weight = asym_w;
        params.recovery_weight = rec_w;
        params.benefit_ratio_weight = ben_w;
        
        let analyzer = AntifragilityAnalyzer::with_params(params);
        let (prices, volumes) = generate_test_data(200);
        
        let result = analyzer.analyze_prices(&prices, &volumes);
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert!(analysis.antifragility_index >= 0.0);
        assert!(analysis.antifragility_index <= 1.0);
    }
}

#[test]
fn test_edge_cases() {
    let analyzer = AntifragilityAnalyzer::new();
    
    // Constant prices
    let constant_prices = vec![100.0; 200];
    let constant_volumes = vec![1000.0; 200];
    
    let result = analyzer.analyze_prices(&constant_prices, &constant_volumes);
    assert!(result.is_ok());
    
    let analysis = result.unwrap();
    // Should return neutral values for constant data
    assert!((analysis.antifragility_index - 0.5).abs() < 0.3);
    
    // Extreme volatility
    let mut extreme_prices = vec![100.0];
    for i in 1..200 {
        let price = if i % 2 == 0 { 50.0 } else { 150.0 };
        extreme_prices.push(price);
    }
    let extreme_volumes = vec![1000.0; 200];
    
    let result = analyzer.analyze_prices(&extreme_prices, &extreme_volumes);
    assert!(result.is_ok());
}

#[test]
fn test_classification_accuracy() {
    let analyzer = AntifragilityAnalyzer::new();
    
    // Generate antifragile-like data (performs better during volatility)
    let (antifragile_prices, antifragile_volumes) = generate_antifragile_data();
    let result = analyzer.analyze_prices(&antifragile_prices, &antifragile_volumes);
    assert!(result.is_ok());
    
    let analysis = result.unwrap();
    // Should detect antifragile characteristics
    assert!(analysis.antifragility_index > 0.5);
    
    // Generate fragile-like data (performs worse during volatility)
    let (fragile_prices, fragile_volumes) = generate_fragile_data();
    let result = analyzer.analyze_prices(&fragile_prices, &fragile_volumes);
    assert!(result.is_ok());
    
    let analysis = result.unwrap();
    // Should detect fragile characteristics
    assert!(analysis.fragility_score > 0.5);
}

fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut prices = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);
    
    let mut price = 100.0;
    for i in 0..n {
        let return_rate = 0.001 * ((i as f64) * 0.1).sin();
        price *= 1.0 + return_rate;
        prices.push(price);
        volumes.push(1000.0 + 100.0 * ((i as f64) * 0.05).cos());
    }
    
    (prices, volumes)
}

fn generate_antifragile_data() -> (Vec<f64>, Vec<f64>) {
    let n = 300;
    let mut prices = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);
    
    let mut price = 100.0;
    for i in 0..n {
        let t = i as f64 * 0.01;
        
        // Create system that benefits from volatility
        let base_return = 0.0001;
        let volatility = 0.02 * (t * 5.0).sin().abs();
        let antifragile_effect = volatility * 0.5; // Benefit from volatility
        
        let return_rate = base_return + antifragile_effect;
        price *= 1.0 + return_rate;
        prices.push(price);
        
        volumes.push(1000.0 + 200.0 * volatility);
    }
    
    (prices, volumes)
}

fn generate_fragile_data() -> (Vec<f64>, Vec<f64>) {
    let n = 300;
    let mut prices = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);
    
    let mut price = 100.0;
    for i in 0..n {
        let t = i as f64 * 0.01;
        
        // Create system that suffers from volatility
        let base_return = 0.0001;
        let volatility = 0.02 * (t * 5.0).sin().abs();
        let fragile_effect = -volatility * 0.3; // Harmed by volatility
        
        let return_rate = base_return + fragile_effect;
        price *= 1.0 + return_rate;
        prices.push(price);
        
        volumes.push(1000.0 + 200.0 * volatility);
    }
    
    (prices, volumes)
}