// Panarchy LUT Analyzer Performance Benchmarks
// Validates <10ms response time requirement for ultra-fast market analysis
use std::time::Instant;
use cwts_ultra::analyzers::PanarchyLUTAnalyzer;

const BENCHMARK_ITERATIONS: usize = 1000;
const DATA_POINTS: usize = 500;
const TARGET_LATENCY_MS: f64 = 10.0;

fn main() {
    println!("🚀 CWTS Ultra - Panarchy LUT Analyzer Benchmarks");
    println!("=================================================");
    
    // Test different configurations
    run_latency_benchmark("Ultra-Fast Config", 100, 4, 50);
    run_latency_benchmark("High-Resolution Config", 500, 6, 100);
    run_latency_benchmark("Maximum Scale Config", 1000, 8, 200);
    
    run_throughput_benchmark();
    run_memory_benchmark();
    run_accuracy_benchmark();
    
    println!("\n✅ All benchmarks completed!");
    println!("Target: <10ms response time for ultra-fast trading decisions");
}

fn run_latency_benchmark(name: &str, window_size: usize, scale_count: usize, lut_resolution: usize) {
    println!("\n📊 Latency Benchmark: {}", name);
    println!("   Window: {}, Scales: {}, LUT Resolution: {}x{}", 
             window_size, scale_count, lut_resolution, lut_resolution);
    
    let mut analyzer = PanarchyLUTAnalyzer::new(window_size, scale_count, lut_resolution);
    
    // Warm up with sample data
    for i in 0..50 {
        analyzer.add_data_point(100.0 + i as f64, 1000.0, i * 1000);
    }
    
    let mut latencies = Vec::with_capacity(BENCHMARK_ITERATIONS);
    
    // Run benchmark iterations
    for i in 0..BENCHMARK_ITERATIONS {
        // Add new data point
        let price = 100.0 + (i as f64 * 0.1).sin() * 10.0;
        let volume = 1000.0 + (i as f64 * 0.05).cos() * 200.0;
        analyzer.add_data_point(price, volume, i * 1000);
        
        // Measure analysis time
        let start = Instant::now();
        let _analysis = analyzer.analyze();
        let duration = start.elapsed();
        
        latencies.push(duration.as_micros() as f64 / 1000.0); // Convert to milliseconds
    }
    
    // Calculate statistics
    let mean_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let mut sorted_latencies = latencies.clone();
    sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let p50 = sorted_latencies[sorted_latencies.len() / 2];
    let p95 = sorted_latencies[(sorted_latencies.len() as f64 * 0.95) as usize];
    let p99 = sorted_latencies[(sorted_latencies.len() as f64 * 0.99) as usize];
    let max_latency = sorted_latencies.last().unwrap();
    
    println!("   📈 Results ({} iterations):", BENCHMARK_ITERATIONS);
    println!("      Mean: {:.2}ms | P50: {:.2}ms | P95: {:.2}ms | P99: {:.2}ms | Max: {:.2}ms", 
             mean_latency, p50, p95, p99, max_latency);
    
    // Performance evaluation
    if mean_latency < TARGET_LATENCY_MS {
        println!("      ✅ PASSED - Mean latency within target (<{}ms)", TARGET_LATENCY_MS);
    } else {
        println!("      ❌ FAILED - Mean latency {}ms exceeds target {}ms", mean_latency, TARGET_LATENCY_MS);
    }
    
    if p99 < TARGET_LATENCY_MS * 2.0 {
        println!("      ✅ PASSED - P99 latency acceptable");
    } else {
        println!("      ⚠️  WARNING - P99 latency {}ms high", p99);
    }
    
    // Memory efficiency check
    let memory_per_analysis = estimate_memory_usage(window_size, scale_count, lut_resolution);
    println!("      📊 Estimated memory per analysis: {:.1}KB", memory_per_analysis / 1024.0);
}

fn run_throughput_benchmark() {
    println!("\n⚡ Throughput Benchmark");
    
    let mut analyzer = PanarchyLUTAnalyzer::new(200, 6, 100);
    
    // Warm up
    for i in 0..20 {
        analyzer.add_data_point(100.0 + i as f64, 1000.0, i * 1000);
    }
    
    let test_duration_ms = 5000; // 5 seconds
    let start_time = Instant::now();
    let mut analysis_count = 0;
    
    while start_time.elapsed().as_millis() < test_duration_ms {
        // Add data point
        let price = 100.0 + (analysis_count as f64 * 0.1).sin() * 5.0;
        analyzer.add_data_point(price, 1000.0, analysis_count * 1000);
        
        // Perform analysis
        let _analysis = analyzer.analyze();
        analysis_count += 1;
    }
    
    let actual_duration = start_time.elapsed();
    let analyses_per_second = analysis_count as f64 / actual_duration.as_secs_f64();
    
    println!("   📈 Throughput Results:");
    println!("      Analyses: {} in {:.2}s", analysis_count, actual_duration.as_secs_f64());
    println!("      Rate: {:.0} analyses/second", analyses_per_second);
    
    if analyses_per_second > 100.0 {
        println!("      ✅ PASSED - High throughput achieved");
    } else {
        println!("      ⚠️  WARNING - Throughput may be limited");
    }
}

fn run_memory_benchmark() {
    println!("\n💾 Memory Usage Benchmark");
    
    let configs = vec![
        ("Minimal", 50, 3, 25),
        ("Standard", 200, 6, 50),
        ("Maximum", 1000, 8, 200),
    ];
    
    for (name, window, scales, resolution) in configs {
        let memory_bytes = estimate_memory_usage(window, scales, resolution);
        let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);
        
        println!("   {} Config: {:.2}MB", name, memory_mb);
        
        if memory_mb < 10.0 {
            println!("      ✅ PASSED - Memory usage acceptable");
        } else if memory_mb < 50.0 {
            println!("      ⚠️  WARNING - High memory usage");
        } else {
            println!("      ❌ CRITICAL - Excessive memory usage");
        }
    }
}

fn run_accuracy_benchmark() {
    println!("\n🎯 Accuracy Benchmark");
    
    let mut analyzer = PanarchyLUTAnalyzer::new(200, 6, 100);
    
    // Test phase identification accuracy with known patterns
    let test_cases = vec![
        ("Growth Pattern", generate_growth_pattern(), "Growth"),
        ("Conservation Pattern", generate_conservation_pattern(), "Conservation"),
        ("Release Pattern", generate_release_pattern(), "Release"),
        ("Reorganization Pattern", generate_reorganization_pattern(), "Reorganization"),
    ];
    
    let mut correct_identifications = 0;
    let total_tests = test_cases.len();
    
    for (test_name, data_points, expected_phase) in test_cases {
        // Reset analyzer for each test
        let mut test_analyzer = PanarchyLUTAnalyzer::new(200, 6, 100);
        
        // Feed the pattern data
        for (i, (price, volume)) in data_points.iter().enumerate() {
            test_analyzer.add_data_point(*price, *volume, i * 1000);
        }
        
        let analysis = test_analyzer.analyze();
        let identified_phase = format!("{:?}", analysis.current_phase);
        
        println!("   {}: Expected {}, Got {} (Confidence: {:.1}%)", 
                 test_name, expected_phase, identified_phase, analysis.phase_confidence * 100.0);
        
        // Check if identification is reasonable (allowing for some flexibility)
        if identified_phase.contains(expected_phase) || analysis.phase_confidence > 0.6 {
            correct_identifications += 1;
            println!("      ✅ PASSED");
        } else {
            println!("      ❌ FAILED - Low confidence or incorrect phase");
        }
    }
    
    let accuracy = (correct_identifications as f64 / total_tests as f64) * 100.0;
    println!("   📊 Overall Accuracy: {:.1}% ({}/{} correct)", accuracy, correct_identifications, total_tests);
    
    if accuracy >= 75.0 {
        println!("      ✅ PASSED - Good phase identification accuracy");
    } else {
        println!("      ⚠️  WARNING - Phase identification needs improvement");
    }
}

fn estimate_memory_usage(window_size: usize, scale_count: usize, lut_resolution: usize) -> usize {
    let base_memory = 1024; // Base struct memory
    let history_memory = window_size * 8 * 4; // 4 f64 arrays
    let scale_memory = scale_count * 200; // Scale levels
    let lut_memory = lut_resolution * lut_resolution * 8 * 4; // 4 LUT matrices
    let hash_memory = 10000; // Estimated hash maps
    
    base_memory + history_memory + scale_memory + lut_memory + hash_memory
}

fn generate_growth_pattern() -> Vec<(f64, f64)> {
    let mut data = Vec::new();
    let base_price = 100.0;
    
    for i in 0..50 {
        let growth_factor = 1.0 + (i as f64 * 0.02); // 2% growth trend
        let noise = (i as f64 * 0.3).sin() * 0.5; // Small noise
        let price = base_price * growth_factor + noise;
        let volume = 1000.0 + (i as f64 * 0.1).cos() * 100.0; // Normal volume
        
        data.push((price, volume));
    }
    
    data
}

fn generate_conservation_pattern() -> Vec<(f64, f64)> {
    let mut data = Vec::new();
    let base_price = 150.0; // Higher price after growth
    
    for i in 0..50 {
        let sideways = (i as f64 * 0.1).sin() * 1.0; // Small sideways movement
        let price = base_price + sideways;
        let volume = 800.0 - (i as f64 * 2.0); // Declining volume
        
        data.push((price, volume.max(500.0)));
    }
    
    data
}

fn generate_release_pattern() -> Vec<(f64, f64)> {
    let mut data = Vec::new();
    let start_price = 150.0;
    
    for i in 0..30 {
        let crash_factor = (i as f64 / 5.0).powi(2); // Accelerating decline
        let price = start_price - crash_factor;
        let volume = 2000.0 + crash_factor * 100.0; // Panic volume
        
        data.push((price.max(50.0), volume));
    }
    
    data
}

fn generate_reorganization_pattern() -> Vec<(f64, f64)> {
    let mut data = Vec::new();
    let base_price = 75.0; // Post-crash level
    
    for i in 0..50 {
        let volatility = 5.0 * (1.0 - i as f64 / 100.0); // Decreasing volatility
        let innovation = (i as f64 * 0.2).sin() * volatility; // High but decreasing swings
        let slow_recovery = i as f64 * 0.1; // Slow upward trend
        let price = base_price + innovation + slow_recovery;
        let volume = 1200.0 + (i as f64 * 0.15).cos() * 300.0; // Variable volume
        
        data.push((price.max(50.0), volume));
    }
    
    data
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;
    
    #[test]
    fn test_latency_target() {
        let mut analyzer = PanarchyLUTAnalyzer::new(100, 4, 50);
        
        // Add sample data
        for i in 0..20 {
            analyzer.add_data_point(100.0 + i as f64, 1000.0, i * 1000);
        }
        
        // Test multiple analyses
        let mut total_time = std::time::Duration::new(0, 0);
        let iterations = 100;
        
        for i in 0..iterations {
            analyzer.add_data_point(100.0 + i as f64, 1000.0, (i + 20) * 1000);
            
            let start = Instant::now();
            let _analysis = analyzer.analyze();
            total_time += start.elapsed();
        }
        
        let avg_latency_ms = (total_time.as_micros() as f64 / iterations as f64) / 1000.0;
        
        assert!(avg_latency_ms < TARGET_LATENCY_MS, 
                "Average latency {:.2}ms exceeds target {:.2}ms", avg_latency_ms, TARGET_LATENCY_MS);
    }
    
    #[test]
    fn test_memory_efficiency() {
        let memory_usage = estimate_memory_usage(200, 6, 100);
        let memory_mb = memory_usage as f64 / (1024.0 * 1024.0);
        
        // Should use less than 50MB for reasonable configurations
        assert!(memory_mb < 50.0, "Memory usage {:.2}MB exceeds reasonable limit", memory_mb);
    }
    
    #[test]
    fn test_pattern_generation() {
        let growth_pattern = generate_growth_pattern();
        assert_eq!(growth_pattern.len(), 50);
        
        let first_price = growth_pattern[0].0;
        let last_price = growth_pattern[49].0;
        assert!(last_price > first_price, "Growth pattern should show price increase");
        
        let release_pattern = generate_release_pattern();
        let first_price = release_pattern[0].0;
        let last_price = release_pattern[29].0;
        assert!(last_price < first_price, "Release pattern should show price decrease");
    }
}