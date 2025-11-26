// Panarchy LUT Analyzer Integration Tests
// Comprehensive testing of adaptive cycle analysis functionality
//
// NOTE: These tests are disabled because the analyzers module is currently
// disabled in lib.rs. Re-enable when analyzers module is restored.

#![cfg(feature = "analyzers")]

use cwts_ultra::analyzers::{
    AdaptiveCyclePhase, DisturbanceType, InteractionDirection, InteractionType, PanarchyAnalysis,
    PanarchyLUTAnalyzer, RecommendationType, TriggerType, WarningType,
};
use std::time::Instant;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_adaptive_cycle() {
        let mut analyzer = PanarchyLUTAnalyzer::new(200, 6, 100);

        // Simulate complete adaptive cycle: Growth -> Conservation -> Release -> Reorganization

        // Phase 1: Growth - increasing prices, innovation
        println!("Testing Growth Phase...");
        for i in 0..50 {
            let price = 100.0 + i as f64 * 2.0; // Strong growth
            let volume = 1000.0 + (i as f64 * 0.1).sin() * 200.0;
            analyzer.add_data_point(price, volume, i * 1000);
        }

        let growth_analysis = analyzer.analyze();
        println!(
            "Growth phase detected: {:?} (confidence: {:.1}%)",
            growth_analysis.current_phase,
            growth_analysis.phase_confidence * 100.0
        );

        // Phase 2: Conservation - sideways movement, efficiency focus
        println!("Testing Conservation Phase...");
        for i in 50..100 {
            let price = 200.0 + (i as f64 * 0.05).sin() * 3.0; // Sideways with low volatility
            let volume = 800.0 - (i - 50) as f64 * 5.0; // Decreasing volume
            analyzer.add_data_point(price, volume.max(400.0), i * 1000);
        }

        let conservation_analysis = analyzer.analyze();
        println!(
            "Conservation phase detected: {:?} (confidence: {:.1}%)",
            conservation_analysis.current_phase,
            conservation_analysis.phase_confidence * 100.0
        );

        // Phase 3: Release - rapid decline, breakdown
        println!("Testing Release Phase...");
        for i in 100..120 {
            let crash_factor = (i - 100) as f64 * 5.0;
            let price = 200.0 - crash_factor; // Rapid decline
            let volume = 3000.0 + crash_factor * 50.0; // Panic selling
            analyzer.add_data_point(price.max(50.0), volume, i * 1000);
        }

        let release_analysis = analyzer.analyze();
        println!(
            "Release phase detected: {:?} (confidence: {:.1}%)",
            release_analysis.current_phase,
            release_analysis.phase_confidence * 100.0
        );

        // Phase 4: Reorganization - recovery, innovation
        println!("Testing Reorganization Phase...");
        for i in 120..170 {
            let base = 100.0; // New baseline
            let volatility = 10.0 * (1.0 - (i - 120) as f64 / 100.0); // Decreasing volatility
            let recovery = (i - 120) as f64 * 0.5; // Slow recovery
            let price = base + (i as f64 * 0.3).sin() * volatility + recovery;
            let volume = 1500.0 + (i as f64 * 0.2).cos() * 400.0;
            analyzer.add_data_point(price, volume, i * 1000);
        }

        let reorganization_analysis = analyzer.analyze();
        println!(
            "Reorganization phase detected: {:?} (confidence: {:.1}%)",
            reorganization_analysis.current_phase,
            reorganization_analysis.phase_confidence * 100.0
        );

        // Verify adaptive cycle progression makes sense
        assert!(
            growth_analysis.phase_confidence > 0.2,
            "Growth phase should be detected with reasonable confidence"
        );
        assert!(
            release_analysis.vulnerability_score > 0.3,
            "Release phase should show high vulnerability"
        );
        assert!(
            reorganization_analysis.transformation_potential > 0.4,
            "Reorganization should show high transformation potential"
        );
    }

    #[test]
    fn test_cross_scale_interactions() {
        let mut analyzer = PanarchyLUTAnalyzer::new(300, 6, 100); // 6 scales for rich interactions

        // Create multi-scale patterns
        for i in 0..100 {
            let t = i as f64;

            // Multiple temporal scales
            let micro_scale = (t * 0.5).sin() * 1.0; // Fast oscillations
            let meso_scale = (t * 0.1).sin() * 5.0; // Medium cycles
            let macro_scale = (t * 0.02).sin() * 10.0; // Slow trends

            let price = 100.0 + micro_scale + meso_scale + macro_scale;
            let volume = 1000.0 + (t * 0.15).cos() * 300.0;

            analyzer.add_data_point(price, volume, i * 1000);
        }

        let analysis = analyzer.analyze();

        // Should detect cross-scale interactions
        println!(
            "Cross-scale interactions detected: {}",
            analysis.cross_scale_interactions.len()
        );

        for interaction in &analysis.cross_scale_interactions {
            println!(
                "  Scale {}→{}: {:?} (strength: {:.1}%)",
                interaction.source_scale,
                interaction.target_scale,
                interaction.interaction_type,
                interaction.strength * 100.0
            );

            // Validate interaction properties
            assert!(
                interaction.strength >= 0.0 && interaction.strength <= 1.0,
                "Interaction strength should be normalized"
            );
            assert!(
                interaction.source_scale != interaction.target_scale,
                "Self-interactions should not be reported"
            );
        }

        // Test for remember connections (slow -> fast)
        let remember_interactions: Vec<_> = analysis
            .cross_scale_interactions
            .iter()
            .filter(|i| i.interaction_type == InteractionType::Remember)
            .collect();

        // Test for revolt connections (fast -> slow)
        let revolt_interactions: Vec<_> = analysis
            .cross_scale_interactions
            .iter()
            .filter(|i| i.interaction_type == InteractionType::Revolt)
            .collect();

        println!(
            "Remember connections: {}, Revolt connections: {}",
            remember_interactions.len(),
            revolt_interactions.len()
        );
    }

    #[test]
    fn test_resilience_metrics() {
        let mut analyzer = PanarchyLUTAnalyzer::new(200, 4, 50);

        // Test different scenarios and their resilience implications

        // Scenario 1: Stable market (high resilience expected)
        for i in 0..50 {
            let price = 100.0 + (i as f64 * 0.05).sin() * 2.0; // Low volatility
            analyzer.add_data_point(price, 1000.0, i * 1000);
        }

        let stable_analysis = analyzer.analyze();
        println!(
            "Stable market resilience: {:.1}%",
            stable_analysis.resilience_metrics.overall_resilience * 100.0
        );

        // Scenario 2: Volatile market (lower resilience expected)
        for i in 50..100 {
            let price = 105.0 + (i as f64 * 0.5).sin() * 15.0; // High volatility
            analyzer.add_data_point(price, 1500.0, i * 1000);
        }

        let volatile_analysis = analyzer.analyze();
        println!(
            "Volatile market resilience: {:.1}%",
            volatile_analysis.resilience_metrics.overall_resilience * 100.0
        );

        // Validate resilience components
        assert!(stable_analysis.resilience_metrics.overall_resilience >= 0.0);
        assert!(stable_analysis.resilience_metrics.overall_resilience <= 1.0);
        assert!(stable_analysis.resilience_metrics.engineering_resilience >= 0.0);
        assert!(stable_analysis.resilience_metrics.ecological_resilience >= 0.0);
        assert!(stable_analysis.resilience_metrics.social_resilience >= 0.0);
        assert!(stable_analysis.resilience_metrics.recovery_time > 0.0);

        // Stable market should generally have better resilience than volatile market
        // (though this may not always hold depending on phase and other factors)
        println!(
            "Stability radius: {:.3} vs {:.3}",
            stable_analysis.resilience_metrics.stability_radius,
            volatile_analysis.resilience_metrics.stability_radius
        );
    }

    #[test]
    fn test_early_warning_signals() {
        let mut analyzer = PanarchyLUTAnalyzer::new(200, 4, 50);

        // Simulate critical slowing down pattern (early warning signal)
        for i in 0..80 {
            let dampening_factor = 1.0 - (i as f64 / 160.0); // Gradual loss of responsiveness
            let base_price = 100.0;
            let perturbation = (i as f64 * 0.2).sin() * 5.0 * dampening_factor;
            let price = base_price + perturbation;
            let volume = 1000.0 + (i as f64 * 0.1).cos() * 200.0;

            analyzer.add_data_point(price, volume, i * 1000);
        }

        let analysis = analyzer.analyze();

        println!(
            "Early warning signals detected: {}",
            analysis.warning_signals.len()
        );

        for signal in &analysis.warning_signals {
            println!(
                "  {:?}: strength {:.1}%, trend {:.3}",
                signal.signal_type,
                signal.strength * 100.0,
                signal.trend
            );

            // Validate signal properties
            assert!(
                signal.strength >= 0.0 && signal.strength <= 1.0,
                "Signal strength should be normalized"
            );
            assert!(
                signal.critical_threshold > 0.0,
                "Critical threshold should be positive"
            );

            match signal.signal_type {
                WarningType::CriticalSlowingDown => {
                    // Should detect slowing down in our simulated pattern
                    assert!(signal.strength > 0.1 || signal.trend < 0.0,
                            "Critical slowing down should show decreasing trend or significant strength");
                }
                WarningType::IncreasingVariance => {
                    // Variance-related warnings
                    assert!(
                        signal.strength >= 0.0,
                        "Variance signal should be non-negative"
                    );
                }
                _ => {} // Other signal types
            }
        }
    }

    #[test]
    fn test_transition_triggers() {
        let mut analyzer = PanarchyLUTAnalyzer::new(150, 4, 50);

        // Simulate conditions that should trigger phase transitions

        // Build up to conservation phase
        for i in 0..40 {
            let price = 100.0 + i as f64 * 1.5; // Growth
            analyzer.add_data_point(price, 1000.0, i * 1000);
        }

        // Add rigidity (high connectedness, complexity)
        for i in 40..60 {
            let price = 160.0 + (i as f64 * 0.02).sin() * 1.0; // Low volatility, high price
            analyzer.add_data_point(price, 800.0, i * 1000); // Decreasing volume
        }

        let analysis = analyzer.analyze();

        println!(
            "Transition triggers detected: {}",
            analysis.transition_triggers.len()
        );

        for trigger in &analysis.transition_triggers {
            println!(
                "  {:?}: probability {:.1}%, urgency {:.1}%",
                trigger.trigger_type,
                trigger.probability * 100.0,
                trigger.urgency * 100.0
            );

            // Validate trigger properties
            assert!(
                trigger.probability >= 0.0 && trigger.probability <= 1.0,
                "Trigger probability should be normalized"
            );
            assert!(
                trigger.urgency >= 0.0 && trigger.urgency <= 1.0,
                "Trigger urgency should be normalized"
            );
            assert!(
                trigger.threshold > 0.0,
                "Trigger threshold should be positive"
            );

            match trigger.trigger_type {
                TriggerType::RigidityBuildup => {
                    // Should detect building rigidity in conservation phase approach
                    assert!(
                        trigger.current_value >= 0.0,
                        "Rigidity value should be non-negative"
                    );
                }
                TriggerType::ConnectednessLoss => {
                    // Should detect loss of system connectivity
                    assert!(
                        trigger.current_value >= 0.0,
                        "Connectedness value should be non-negative"
                    );
                }
                _ => {} // Other trigger types
            }
        }

        // Should have some transition probability for next phases
        let total_transition_prob: f64 = analysis.next_phase_probability.values().sum();
        assert!(
            (total_transition_prob - 1.0).abs() < 0.1,
            "Transition probabilities should sum to approximately 1.0, got {}",
            total_transition_prob
        );
    }

    #[test]
    fn test_recommendations_system() {
        let mut analyzer = PanarchyLUTAnalyzer::new(150, 4, 50);

        // Test recommendations for different phases
        let phase_scenarios = vec![
            ("Growth", generate_growth_scenario()),
            ("Conservation", generate_conservation_scenario()),
            ("Release", generate_release_scenario()),
            ("Reorganization", generate_reorganization_scenario()),
        ];

        for (phase_name, data_points) in phase_scenarios {
            // Reset analyzer for each scenario
            let mut test_analyzer = PanarchyLUTAnalyzer::new(150, 4, 50);

            // Feed the scenario data
            for (i, (price, volume)) in data_points.iter().enumerate() {
                test_analyzer.add_data_point(*price, *volume, i * 1000);
            }

            let analysis = test_analyzer.analyze();

            println!("\n{} Phase Recommendations:", phase_name);

            for rec in &analysis.recommendations {
                println!(
                    "  {:?} - {} (Impact: {:.1}%, Confidence: {:.1}%)",
                    rec.recommendation_type,
                    rec.rationale,
                    rec.expected_impact * 100.0,
                    rec.confidence * 100.0
                );

                // Validate recommendation properties
                assert!(
                    rec.expected_impact >= 0.0 && rec.expected_impact <= 1.0,
                    "Expected impact should be normalized"
                );
                assert!(
                    rec.confidence >= 0.0 && rec.confidence <= 1.0,
                    "Confidence should be normalized"
                );
                assert!(!rec.rationale.is_empty(), "Rationale should not be empty");

                // Check phase-appropriate recommendations
                match phase_name {
                    "Growth" => {
                        // Growth phase should suggest exploration or building resilience
                        assert!(
                            matches!(
                                rec.recommendation_type,
                                RecommendationType::ExploreOpportunities
                                    | RecommendationType::BuildResilience
                                    | RecommendationType::DiversifyStrategies
                            ),
                            "Growth phase should suggest appropriate actions"
                        );
                    }
                    "Conservation" => {
                        // Conservation should focus on resilience and monitoring
                        assert!(
                            matches!(
                                rec.recommendation_type,
                                RecommendationType::BuildResilience
                                    | RecommendationType::MonitorSignals
                                    | RecommendationType::DiversifyStrategies
                            ),
                            "Conservation phase should suggest appropriate actions"
                        );
                    }
                    "Release" => {
                        // Release phase should suggest preparation for change
                        assert!(
                            matches!(
                                rec.recommendation_type,
                                RecommendationType::PrepareForChange
                                    | RecommendationType::DiversifyStrategies
                            ),
                            "Release phase should suggest preparation actions"
                        );
                    }
                    "Reorganization" => {
                        // Reorganization should suggest innovation and exploration
                        assert!(
                            matches!(
                                rec.recommendation_type,
                                RecommendationType::InnovateAdapt
                                    | RecommendationType::ExploreOpportunities
                                    | RecommendationType::DiversifyStrategies
                            ),
                            "Reorganization phase should suggest innovation actions"
                        );
                    }
                    _ => {}
                }
            }

            assert!(
                !analysis.recommendations.is_empty(),
                "Should provide recommendations for {} phase",
                phase_name
            );
        }
    }

    #[test]
    fn test_performance_requirements() {
        let mut analyzer = PanarchyLUTAnalyzer::new(200, 6, 100);

        // Warm up with data
        for i in 0..50 {
            analyzer.add_data_point(100.0 + i as f64, 1000.0, i * 1000);
        }

        // Test multiple analysis calls for performance
        let mut total_time = std::time::Duration::new(0, 0);
        let test_iterations = 100;

        for i in 0..test_iterations {
            let price = 150.0 + (i as f64 * 0.1).sin() * 5.0;
            analyzer.add_data_point(price, 1200.0, (i + 50) * 1000);

            let start = Instant::now();
            let analysis = analyzer.analyze();
            total_time += start.elapsed();

            // Validate analysis completeness
            assert!(
                !analysis.recommendations.is_empty(),
                "Analysis should include recommendations"
            );
            assert!(
                analysis.phase_confidence >= 0.0 && analysis.phase_confidence <= 1.0,
                "Phase confidence should be normalized"
            );
            assert!(
                analysis.resilience_metrics.overall_resilience >= 0.0,
                "Resilience metrics should be non-negative"
            );
        }

        let avg_time_ms = (total_time.as_micros() as f64 / test_iterations as f64) / 1000.0;

        println!(
            "Average analysis time: {:.2}ms over {} iterations",
            avg_time_ms, test_iterations
        );

        // Performance requirement: <10ms average
        assert!(
            avg_time_ms < 10.0,
            "Analysis time {:.2}ms exceeds 10ms target for ultra-fast trading",
            avg_time_ms
        );

        // No single analysis should take more than 50ms (extreme case)
        let max_single_time_ms = 50.0;

        for _i in 0..10 {
            let start = Instant::now();
            let _analysis = analyzer.analyze();
            let single_time_ms = start.elapsed().as_micros() as f64 / 1000.0;

            assert!(
                single_time_ms < max_single_time_ms,
                "Single analysis time {:.2}ms exceeds maximum allowed {:.2}ms",
                single_time_ms,
                max_single_time_ms
            );
        }
    }

    // Helper functions for generating test scenarios

    fn generate_growth_scenario() -> Vec<(f64, f64)> {
        let mut data = Vec::new();
        for i in 0..40 {
            let price = 100.0 + i as f64 * 2.0 + (i as f64 * 0.2).sin() * 1.0;
            let volume = 1000.0 + (i as f64 * 0.1).cos() * 200.0;
            data.push((price, volume));
        }
        data
    }

    fn generate_conservation_scenario() -> Vec<(f64, f64)> {
        let mut data = Vec::new();
        for i in 0..40 {
            let price = 180.0 + (i as f64 * 0.05).sin() * 2.0; // Low volatility
            let volume = 1000.0 - i as f64 * 5.0; // Declining volume
            data.push((price, volume.max(500.0)));
        }
        data
    }

    fn generate_release_scenario() -> Vec<(f64, f64)> {
        let mut data = Vec::new();
        for i in 0..25 {
            let crash_factor = (i as f64 * 0.3).powi(2);
            let price = 180.0 - crash_factor * 3.0;
            let volume = 2000.0 + crash_factor * 200.0;
            data.push((price.max(100.0), volume));
        }
        data
    }

    fn generate_reorganization_scenario() -> Vec<(f64, f64)> {
        let mut data = Vec::new();
        for i in 0..40 {
            let base = 120.0;
            let volatility = 8.0 * (1.0 - i as f64 / 80.0); // Decreasing volatility
            let innovation = (i as f64 * 0.25).sin() * volatility;
            let recovery = i as f64 * 0.2;
            let price = base + innovation + recovery;
            let volume = 1300.0 + (i as f64 * 0.18).cos() * 300.0;
            data.push((price, volume));
        }
        data
    }
}
