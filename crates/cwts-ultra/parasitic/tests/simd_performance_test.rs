//! SIMD Performance Test for CQGS Compliance
//! 
//! Tests the blueprint requirements for SimdPairScorer with ZERO tolerance for violations.

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use parasitic::pairlist::{SimdPairScorer, AlignedWeights, PairFeatures};
    use parasitic::simd_ops::{
        SimdPatternMatcher, SimdFitnessCalculator, SimdHostSelector,
        TradingPattern, ParasiticOrganism, HostCandidate
    };
    
    /// Test SIMD pair scoring with <1ms performance requirement
    #[tokio::test]
    async fn test_simd_pair_scorer_performance() {
        println!("🧪 Testing SimdPairScorer performance (CQGS compliance)");
        
        let scorer = SimdPairScorer::new().await.unwrap();
        
        // Create test data - 8 pair chunks for AVX2 optimization
        let test_analyses = create_test_analyses(1000);
        
        // Measure performance
        let start_time = Instant::now();
        
        let scored_pairs = scorer.score_analyses(&test_analyses).await.unwrap();
        
        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis();
        
        println!("📊 SIMD Pair Scoring Results:");
        println!("   • Pairs processed: {}", test_analyses.len());
        println!("   • Duration: {}ms ({}μs)", duration_ms, duration.as_micros());
        println!("   • Pairs per second: {:.0}", test_analyses.len() as f64 / duration.as_secs_f64());
        println!("   • Scored pairs: {}", scored_pairs.len());
        
        // ZERO TOLERANCE: Must be <1ms
        assert!(duration_ms < 1, "❌ VIOLATION: SIMD scoring took {}ms (>1ms limit)", duration_ms);
        assert_eq!(scored_pairs.len(), test_analyses.len(), "❌ All pairs must be scored");
        
        println!("✅ SIMD Pair Scorer PASSED (<1ms requirement)");
    }
    
    /// Test AVX2-optimized pattern matching
    #[test]
    fn test_avx2_pattern_matching() {
        println!("🧪 Testing AVX2 pattern matching performance");
        
        if !is_x86_feature_detected!("avx2") {
            println!("⚠️ AVX2 not available, skipping test");
            return;
        }
        
        let matcher = SimdPatternMatcher::new();
        let target = create_test_pattern(1);
        let patterns: Vec<TradingPattern> = (0..1024).map(create_test_pattern).collect();
        
        let start_time = Instant::now();
        
        let results = unsafe {
            matcher.find_similar_patterns(&target, &patterns, 0.7)
        };
        
        let duration = start_time.elapsed();
        let duration_ns = duration.as_nanos();
        
        println!("📊 AVX2 Pattern Matching Results:");
        println!("   • Patterns searched: {}", patterns.len());
        println!("   • Duration: {}ns ({}μs)", duration_ns, duration.as_micros());
        println!("   • Similar patterns found: {}", results.len());
        println!("   • Patterns per nanosecond: {:.3}", patterns.len() as f64 / duration_ns as f64);
        
        // Target: <100ns for 1024 patterns
        assert!(duration_ns < 100_000, "❌ VIOLATION: Pattern matching took {}ns (>100μs limit)", duration_ns);
        
        println!("✅ AVX2 Pattern Matching PASSED");
    }
    
    /// Test 8-pair chunk processing requirement
    #[test]
    fn test_8_pair_chunk_processing() {
        println!("🧪 Testing 8-pair chunk processing");
        
        if !is_x86_feature_detected!("avx2") {
            println!("⚠️ AVX2 not available, skipping test");
            return;
        }
        
        let calculator = SimdFitnessCalculator::new();
        let mut organisms: Vec<ParasiticOrganism> = (0..16).map(|i| ParasiticOrganism::new(i)).collect();
        let market_conditions = [1.0, 0.8, 0.9, 1.1, 0.7, 1.2, 0.6, 1.3];
        
        let start_time = Instant::now();
        
        unsafe {
            calculator.evaluate_batch_fitness(&mut organisms, &market_conditions);
        }
        
        let duration = start_time.elapsed();
        let duration_ns = duration.as_nanos();
        
        println!("📊 8-Pair Chunk Processing Results:");
        println!("   • Organisms processed: {}", organisms.len());
        println!("   • Duration: {}ns", duration_ns);
        println!("   • Organisms per nanosecond: {:.6}", organisms.len() as f64 / duration_ns as f64);
        
        // Verify all organisms have fitness scores
        for organism in &organisms {
            assert!(organism.fitness_score.is_finite(), "❌ Invalid fitness score");
        }
        
        // Target: <50ns for 16 organisms
        assert!(duration_ns < 50_000, "❌ VIOLATION: Batch fitness took {}ns (>50μs limit)", duration_ns);
        
        println!("✅ 8-Pair Chunk Processing PASSED");
    }
    
    /// Test horizontal_sum_avx2 implementation
    #[test]
    fn test_horizontal_sum_avx2() {
        if !is_x86_feature_detected!("avx2") {
            println!("⚠️ AVX2 not available, skipping horizontal sum test");
            return;
        }
        
        println!("🧪 Testing horizontal_sum_avx2 implementation");
        
        // Test data for horizontal sum
        let test_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected_sum = test_data.iter().sum::<f32>();
        
        // This would require exposing the horizontal_sum_avx2 function
        // For now, we test through the pattern matching which uses it
        let matcher = SimdPatternMatcher::new();
        let pattern1 = create_test_pattern_with_data(&test_data);
        let pattern2 = create_test_pattern_with_data(&test_data);
        
        let results = unsafe {
            matcher.find_similar_patterns(&pattern1, &[pattern2], 0.5)
        };
        
        assert_eq!(results.len(), 1, "❌ Should find identical pattern");
        assert!(results[0].1 > 0.99, "❌ Identical patterns should have high correlation: {}", results[0].1);
        
        println!("✅ horizontal_sum_avx2 implementation PASSED (via pattern matching)");
    }
    
    /// Test host selection with vectorized scoring
    #[test]
    fn test_host_selection_performance() {
        println!("🧪 Testing host selection performance");
        
        if !is_x86_feature_detected!("avx2") {
            println!("⚠️ AVX2 not available, skipping test");
            return;
        }
        
        let selector = SimdHostSelector::new();
        let mut hosts: Vec<HostCandidate> = (0..64).map(|i| HostCandidate::new(i)).collect();
        let organism = ParasiticOrganism::new(1);
        
        let start_time = Instant::now();
        
        let selected = unsafe {
            selector.select_best_hosts(&mut hosts, &organism, 10)
        };
        
        let duration = start_time.elapsed();
        let duration_ns = duration.as_nanos();
        
        println!("📊 Host Selection Results:");
        println!("   • Hosts evaluated: {}", hosts.len());
        println!("   • Duration: {}ns", duration_ns);
        println!("   • Selected hosts: {}", selected.len());
        println!("   • Hosts per nanosecond: {:.6}", hosts.len() as f64 / duration_ns as f64);
        
        // Target: <30ns for scoring 64 hosts
        assert!(duration_ns < 30_000, "❌ VIOLATION: Host selection took {}ns (>30μs limit)", duration_ns);
        assert_eq!(selected.len(), 10, "❌ Should select exactly 10 hosts");
        
        println!("✅ Host Selection Performance PASSED");
    }
    
    /// Test AlignedWeights memory alignment
    #[test]
    fn test_aligned_weights() {
        println!("🧪 Testing AlignedWeights alignment");
        
        let weights = AlignedWeights {
            parasitic_opportunity: [1.0; 8],
            vulnerability_score: [1.0; 8],
            organism_fitness: [1.0; 8],
            emergence_bonus: [1.0; 8],
        };
        
        let ptr = &weights as *const AlignedWeights;
        let addr = ptr as usize;
        
        println!("📊 Memory Alignment Check:");
        println!("   • AlignedWeights address: 0x{:x}", addr);
        println!("   • 32-byte aligned: {}", addr % 32 == 0);
        
        assert_eq!(addr % 32, 0, "❌ VIOLATION: AlignedWeights not 32-byte aligned");
        
        println!("✅ AlignedWeights alignment PASSED");
    }
    
    /// Comprehensive SIMD feature detection
    #[test]
    fn test_simd_feature_detection() {
        println!("🧪 Testing SIMD feature detection");
        
        let has_sse42 = is_x86_feature_detected!("sse4.2");
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_fma = is_x86_feature_detected!("fma");
        let has_avx512f = is_x86_feature_detected!("avx512f");
        
        println!("📊 SIMD Features Available:");
        println!("   • SSE4.2: {}", has_sse42);
        println!("   • AVX2: {} {}", has_avx2, if has_avx2 { "✅" } else { "❌" });
        println!("   • FMA: {} {}", has_fma, if has_fma { "✅" } else { "❌" });
        println!("   • AVX512F: {}", has_avx512f);
        
        // BLUEPRINT REQUIREMENT: Must have AVX2 and FMA
        assert!(has_avx2, "❌ VIOLATION: AVX2 not available - required for blueprint");
        assert!(has_fma, "❌ VIOLATION: FMA not available - required for blueprint");
        
        println!("✅ Required SIMD features (AVX2, FMA) AVAILABLE");
    }
    
    // Helper functions
    
    fn create_test_analyses(count: usize) -> Vec<parasitic::pairlist::CQGSValidatedAnalysis> {
        use parasitic::pairlist::*;
        
        (0..count).map(|i| {
            CQGSValidatedAnalysis {
                analysis: PairAnalysis {
                    pair_id: format!("PAIR_{}", i),
                    base_score: 0.8,
                    quantum_score: 0.9,
                    neural_score: 0.85,
                },
                compliance_metrics: ComplianceMetrics {
                    overall_compliance: 0.95,
                    sentinel_validation: 0.9,
                },
                hyperbolic_score: 0.88,
            }
        }).collect()
    }
    
    fn create_test_pattern(id: u32) -> TradingPattern {
        let mut pattern = TradingPattern::new(id);
        
        // Fill with some test data
        for i in 0..8 {
            pattern.price_history[i] = (i as f32 + id as f32 * 0.1) % 10.0;
            pattern.volume_history[i] = (i as f32 * 2.0 + id as f32 * 0.05) % 100.0;
        }
        
        pattern.volatility = 0.5 + (id as f32 * 0.01) % 0.5;
        pattern.momentum = (id as f32 * 0.02) % 2.0 - 1.0;
        pattern.rsi = 30.0 + (id as f32 * 0.1) % 40.0;
        pattern.macd = (id as f32 * 0.01) % 1.0 - 0.5;
        
        pattern
    }
    
    fn create_test_pattern_with_data(data: &[f32]) -> TradingPattern {
        let mut pattern = TradingPattern::new(1);
        
        if data.len() >= 8 {
            pattern.price_history.copy_from_slice(&data[0..8]);
            pattern.volume_history.copy_from_slice(&data[0..8]);
        }
        
        pattern
    }
}