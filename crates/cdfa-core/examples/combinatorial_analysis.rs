//! Combinatorial Diversity Fusion Analysis Example
//! 
//! Demonstrates the enhanced CDFA system with combinatorial capabilities,
//! algorithm pool management, and synergy detection.

use cdfa_core::prelude::*;
use ndarray::array;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Combinatorial Diversity Fusion Analysis Demo");
    println!("================================================\n");
    
    // Create sample multi-algorithm data (simulating different prediction algorithms)
    let algorithm_data = array![
        [0.85, 0.72, 0.91, 0.43, 0.67, 0.89, 0.55, 0.78],  // Algorithm A: Conservative predictor
        [0.78, 0.85, 0.69, 0.52, 0.81, 0.74, 0.63, 0.87],  // Algorithm B: Aggressive predictor  
        [0.92, 0.61, 0.83, 0.48, 0.75, 0.66, 0.71, 0.59],  // Algorithm C: Momentum-based
        [0.69, 0.79, 0.77, 0.64, 0.58, 0.82, 0.76, 0.73],  // Algorithm D: Mean-reverting
        [0.73, 0.68, 0.85, 0.57, 0.79, 0.71, 0.64, 0.81],  // Algorithm E: Hybrid approach
    ];
    
    println!("📊 Sample Data: {} algorithms × {} predictions", 
             algorithm_data.nrows(), algorithm_data.ncols());
    
    // Ground truth for validation (optional)
    let ground_truth = array![0.80, 0.75, 0.82, 0.55, 0.72, 0.77, 0.65, 0.79];
    
    // 1. Create Combinatorial Diversity Fusion Analyzer with custom configuration
    println!("\n🔧 Initializing Combinatorial CDFA with custom configuration...");
    
    let config = CombinatorialConfig {
        max_k: 4,                    // Evaluate combinations up to 4 algorithms
        diversity_threshold: 0.3,    // Minimum diversity for inclusion
        performance_weight: 0.7,     // 70% weight on performance vs diversity
        parallel_evaluation: true,   // Enable parallel processing
        simd_level: crate::combinatorial::SIMDLevel::Auto,
    };
    
    let mut analyzer = CombinatorialDiversityFusionAnalyzer::with_config(config);
    
    // 2. Demonstrate single combination analysis
    println!("\n🔍 Single Combination Analysis");
    println!("-------------------------------");
    
    let start = Instant::now();
    
    // Select first 3 algorithms for focused analysis
    let algorithm_ids = vec!["builtin_average".to_string(), "builtin_borda_count".to_string()];
    
    let single_result = analyzer.analyze_single_combination(
        &algorithm_ids,
        &algorithm_data.view(),
        FusionMethod::Hybrid,
        Some(&ground_truth),
    )?;
    
    println!("✅ Combination: {:?}", single_result.algorithm_ids);
    println!("📈 Fusion Result: {:?}", single_result.fusion_result);
    println!("⚡ Performance: {:.0}ns (Target: <1μs = 1000ns)", 
             single_result.computational_cost.total_analysis_ns);
    println!("🎯 Meets Target: {}", 
             single_result.performance_profile.evaluation_metrics.performance.meets_target);
    
    // Display synergy analysis
    println!("\n🤝 Synergy Analysis:");
    for interaction in &single_result.synergy_analysis {
        println!("  {} ↔ {}: {:.3} synergy, {:?} relationship",
                 interaction.algorithm_a,
                 interaction.algorithm_b,
                 interaction.synergy_metrics.complementarity,
                 interaction.interaction_type);
    }
    
    println!("⏱️  Single analysis completed in: {:.2?}", start.elapsed());
    
    // 3. Complete combinatorial analysis
    println!("\n🌊 Complete Combinatorial Analysis");
    println!("===================================");
    
    let start = Instant::now();
    
    let full_analysis = analyzer.analyze_combinations(
        &algorithm_data.view(),
        Some(&ground_truth),
    )?;
    
    println!("✅ Analysis completed in: {:.2?}", start.elapsed());
    println!("📊 Total combinations evaluated: {}", full_analysis.combination_profiles.len());
    println!("⚡ Performance target achieved by: {:.1}% of combinations",
             full_analysis.analysis_summary.target_meeting_combinations as f64 / 
             full_analysis.analysis_summary.total_combinations as f64 * 100.0);
    
    // 4. Display optimal combinations
    println!("\n🏆 Optimal Combinations Found:");
    println!("------------------------------");
    
    if let Some(ref best_acc) = full_analysis.optimal_combinations.best_accuracy {
        println!("🎯 Best Accuracy: {}", best_acc);
    }
    
    if let Some(ref best_perf) = full_analysis.optimal_combinations.best_performance {
        println!("⚡ Best Performance: {}", best_perf);
    }
    
    if let Some(ref best_overall) = full_analysis.optimal_combinations.best_overall {
        println!("⭐ Best Overall: {}", best_overall);
    }
    
    if let Some(ref most_diverse) = full_analysis.optimal_combinations.most_diverse {
        println!("🌈 Most Diverse: {}", most_diverse);
    }
    
    if let Some(ref most_synergistic) = full_analysis.optimal_combinations.most_synergistic {
        println!("🤝 Most Synergistic: {}", most_synergistic);
    }
    
    // 5. Synergy Matrix Analysis
    println!("\n📈 Algorithm Interaction Matrix:");
    println!("--------------------------------");
    
    let matrix = &full_analysis.interaction_matrix;
    println!("Average Synergy Score: {:.3}", matrix.summary_stats.avg_synergy);
    println!("Synergistic Pairs: {}", matrix.summary_stats.synergistic_count);
    println!("Complementary Pairs: {}", matrix.summary_stats.complementary_count);
    println!("Redundant Pairs: {}", matrix.summary_stats.redundancy_count);
    
    // 6. Performance Benchmarks
    println!("\n⚡ Performance Benchmarks:");
    println!("-------------------------");
    
    let benchmarks = &full_analysis.performance_benchmarks;
    println!("Total Analysis Time: {:.0}ns ({:.2}μs)", 
             benchmarks.total_time_ns, 
             benchmarks.total_time_ns as f64 / 1000.0);
    println!("Fusion Time: {:.0}ns", benchmarks.fusion_time_ns);
    println!("Synergy Analysis Time: {:.0}ns", benchmarks.synergy_analysis_time_ns);
    println!("Memory Usage: {:.1}KB", benchmarks.memory_usage_bytes as f64 / 1024.0);
    println!("Meets <1μs Target: {}", benchmarks.meets_performance_target());
    
    // 7. Diversity Distribution Analysis
    println!("\n📊 Diversity Distribution:");
    println!("--------------------------");
    
    let diversity = &full_analysis.analysis_summary.diversity_distribution;
    println!("Mean Diversity: {:.3}", diversity.mean_diversity);
    println!("Diversity Range: {:.3} - {:.3}", diversity.min_diversity, diversity.max_diversity);
    println!("Standard Deviation: {:.3}", diversity.std_dev_diversity);
    println!("Quartiles: Q1={:.3}, Q2={:.3}, Q3={:.3}", 
             diversity.quartiles[0], diversity.quartiles[1], diversity.quartiles[2]);
    
    // 8. Recommendations
    println!("\n💡 System Recommendations:");
    println!("---------------------------");
    
    for (i, recommendation) in full_analysis.analysis_summary.recommendations.iter().enumerate() {
        println!("{}. {}", i + 1, recommendation);
    }
    
    // 9. Synergy-Guided Selection Demo
    println!("\n🎯 Synergy-Guided Combination Selection:");
    println!("----------------------------------------");
    
    let start = Instant::now();
    let synergy_combinations = analyzer.find_synergy_guided_combinations(
        &algorithm_data.view(),
        3, // k=3 algorithms per combination
        5, // top 5 combinations
    )?;
    
    println!("✅ Found {} synergy-optimized combinations in {:.2?}", 
             synergy_combinations.len(), start.elapsed());
    
    for (i, combo) in synergy_combinations.iter().enumerate() {
        println!("{}. Algorithms: {:?}", i + 1, combo.algorithm_ids);
        println!("   Performance: {:.0}ns, Quality: {:.3}",
                 combo.computational_cost.total_analysis_ns,
                 combo.performance_profile.evaluation_metrics.quality.overall_score);
    }
    
    // 10. Final Performance Statistics
    println!("\n📈 Analyzer Performance Statistics:");
    println!("-----------------------------------");
    
    let stats = analyzer.get_performance_statistics();
    println!("Total Analyses Performed: {}", stats.total_analyses);
    println!("Average Analysis Time: {:.0}ns", stats.average_analysis_time_ns);
    println!("Cache Hit Rate: {:.1}%", stats.cache_hit_rate * 100.0);
    println!("Algorithm Pool Size: {}", stats.algorithm_pool_size);
    println!("Performance Target Achievement: {:.1}%", 
             stats.performance_target_achievement_rate * 100.0);
    
    println!("\n🎉 Combinatorial CDFA Analysis Complete!");
    println!("========================================");
    println!("Enhanced CDFA system successfully demonstrates:");
    println!("✅ Algorithm pool management");
    println!("✅ Synergy detection & analysis");
    println!("✅ K-combinations evaluation");
    println!("✅ Performance optimization (<1μs target)");
    println!("✅ Comprehensive metrics & recommendations");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_combinatorial_analysis_example() {
        // This test validates that the example runs without errors
        // In a real scenario, you'd want more specific assertions
        assert!(main().is_ok());
    }
}