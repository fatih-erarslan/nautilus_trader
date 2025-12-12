use cdfa_fibonacci_pattern_detector::{FibonacciPatternDetector, PatternParameters};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fibonacci Pattern Detection Example");
    println!("==================================");
    
    // Initialize the pattern detector
    cdfa_fibonacci_pattern_detector::init()?;
    
    // Create sample market data
    let high = Array1::from_vec(vec![
        1.0, 1.2, 1.1, 1.3, 1.05, 1.25, 1.15, 1.35, 1.08, 1.28,
        1.18, 1.38, 1.12, 1.32, 1.22, 1.42, 1.16, 1.36, 1.26, 1.46,
    ]);
    
    let low = Array1::from_vec(vec![
        0.95, 1.15, 1.05, 1.25, 1.0, 1.2, 1.1, 1.3, 1.03, 1.23,
        1.13, 1.33, 1.07, 1.27, 1.17, 1.37, 1.11, 1.31, 1.21, 1.41,
    ]);
    
    let close = Array1::from_vec(vec![
        1.0, 1.18, 1.08, 1.28, 1.02, 1.22, 1.12, 1.32, 1.05, 1.25,
        1.15, 1.35, 1.09, 1.29, 1.19, 1.39, 1.13, 1.33, 1.23, 1.43,
    ]);
    
    // Create detector with custom parameters
    let mut params = PatternParameters::default();
    params.min_pattern_bars = 10;
    params.max_pattern_bars = 50;
    params.ratio_tolerance = 0.05;
    params.min_confidence = 0.7;
    
    let detector = FibonacciPatternDetector::with_params(params);
    
    // Detect patterns
    println!("Detecting patterns in {} bars of data...", high.len());
    let result = detector.detect_patterns(&high, &low, &close)?;
    
    println!("Analysis Results:");
    println!("================");
    println!("Computation time: {} ns", result.computation_time_ns);
    println!("Scan time: {} ns", result.scan_time_ns);
    println!("Validation time: {} ns", result.validation_time_ns);
    println!("Swing points detected: {}", result.swing_points_detected);
    println!("Patterns found: {}", result.patterns_found);
    
    if result.has_patterns() {
        println!("\nDetected Patterns:");
        println!("-----------------");
        
        for (i, pattern) in result.detected_patterns.iter().enumerate() {
            println!("Pattern {}: {} (confidence: {:.2}%)", 
                     i + 1, pattern.pattern_type.as_str(), pattern.confidence * 100.0);
            println!("  Type: {}", if pattern.is_bullish { "Bullish" } else { "Bearish" });
            println!("  Validation score: {:.3}", pattern.validation_score);
            println!("  Duration: {} bars", pattern.pattern_duration);
            println!("  Height: {:.4}", pattern.pattern_height);
            
            println!("  Ratios:");
            println!("    AB/XA: {:.3}", pattern.ab_xa_ratio);
            println!("    BC/AB: {:.3}", pattern.bc_ab_ratio);
            println!("    CD/BC: {:.3}", pattern.cd_bc_ratio);
            println!("    AD/XA: {:.3}", pattern.ad_xa_ratio);
            
            println!("  Points:");
            for point in &pattern.points {
                println!("    {}: index {}, price {:.4}", 
                         point.role, point.index, point.price);
            }
            println!();
        }
        
        // Show pattern type distribution
        let gartley_count = result.get_patterns_by_type(cdfa_fibonacci_pattern_detector::PatternType::Gartley).len();
        let butterfly_count = result.get_patterns_by_type(cdfa_fibonacci_pattern_detector::PatternType::Butterfly).len();
        let bat_count = result.get_patterns_by_type(cdfa_fibonacci_pattern_detector::PatternType::Bat).len();
        let crab_count = result.get_patterns_by_type(cdfa_fibonacci_pattern_detector::PatternType::Crab).len();
        let shark_count = result.get_patterns_by_type(cdfa_fibonacci_pattern_detector::PatternType::Shark).len();
        
        println!("Pattern Distribution:");
        if gartley_count > 0 { println!("  Gartley: {}", gartley_count); }
        if butterfly_count > 0 { println!("  Butterfly: {}", butterfly_count); }
        if bat_count > 0 { println!("  Bat: {}", bat_count); }
        if crab_count > 0 { println!("  Crab: {}", crab_count); }
        if shark_count > 0 { println!("  Shark: {}", shark_count); }
        
        // Show bullish vs bearish
        let bullish_count = result.get_bullish_patterns().len();
        let bearish_count = result.get_bearish_patterns().len();
        println!("  Bullish: {}", bullish_count);
        println!("  Bearish: {}", bearish_count);
        
        // Show highest confidence pattern
        if let Some(best_pattern) = result.highest_confidence_pattern() {
            println!("\nHighest Confidence Pattern:");
            println!("  {} with {:.2}% confidence", 
                     best_pattern.pattern_type.as_str(), 
                     best_pattern.confidence * 100.0);
        }
    } else {
        println!("No patterns detected in the provided data.");
        println!("Try with:");
        println!("- More data points (current: {})", high.len());
        println!("- Lower confidence threshold (current: {:.2})", detector.params.min_confidence);
        println!("- Higher ratio tolerance (current: {:.3})", detector.params.ratio_tolerance);
    }
    
    // Performance information
    println!("\nPerformance Information:");
    println!("=======================");
    println!("Hardware acceleration: {}", 
             if cdfa_fibonacci_pattern_detector::acceleration_available() { "Available" } else { "Not available" });
    println!("Version: {}", cdfa_fibonacci_pattern_detector::version());
    
    // Check performance targets
    if result.computation_time_ns <= cdfa_fibonacci_pattern_detector::perf::FULL_DETECTION_TARGET_NS {
        println!("✅ Performance target met!");
    } else {
        println!("⚠️  Performance target missed ({} ns > {} ns)", 
                 result.computation_time_ns, 
                 cdfa_fibonacci_pattern_detector::perf::FULL_DETECTION_TARGET_NS);
    }
    
    Ok(())
}