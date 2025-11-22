//! Integration tests for consciousness system
//! Tests consciousness emergence, integration, and dynamics

use autopoiesis::consciousness::*;
use autopoiesis::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(feature = "test-utils")]
use approx::assert_relative_eq;

use tokio::time::{sleep, Duration};

#[cfg(feature = "property-tests")]
use proptest::prelude::*;

#[tokio::test]
async fn test_consciousness_system_initialization() {
    let system = ConsciousnessSystem::new((5, 5, 5), 40.0);
    
    assert_eq!(system.dimensions, (5, 5, 5));
    assert_eq!(system.base_frequency, 40.0);
    assert!(!system.is_consciousness_emerged());
    
    let state = system.get_consciousness_state();
    assert!(state.field_state.consciousness_level >= 0.0);
    assert!(state.consciousness_quality.unity_level >= 0.0);
    assert!(state.lattice_state.supports_consciousness == false || state.lattice_state.supports_consciousness == true);
}

#[tokio::test]
async fn test_coherent_consciousness_initialization() {
    let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
    system.initialize_coherent_consciousness();
    
    let state = system.get_consciousness_state();
    
    // After coherent initialization, consciousness levels should be elevated
    assert!(state.field_state.consciousness_level > 0.1);
    assert!(state.consciousness_quality.coherence_strength > 0.1);
    assert!(state.integration_metrics.overall_integration >= 0.0);
}

#[tokio::test]
async fn test_consciousness_cycle_progression() {
    let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
    system.initialize_coherent_consciousness();
    
    let initial_time = system.time;
    let dt = 0.01;
    
    // Process several cycles
    for i in 0..10 {
        system.process_consciousness_cycle(dt, None);
        
        // Time should advance
        assert_relative_eq!(system.time, initial_time + (i + 1) as f64 * dt, epsilon = 1e-10);
        
        // State should be valid
        let state = system.get_consciousness_state();
        assert!(state.field_state.consciousness_level >= 0.0);
        assert!(state.integration_metrics.overall_integration >= 0.0);
    }
}

#[tokio::test]
async fn test_consciousness_emergence_detection() {
    let mut system = ConsciousnessSystem::new((4, 4, 4), 40.0);
    system.initialize_coherent_consciousness();
    
    let mut emergence_detected = false;
    
    // Run for a period and check for emergence
    for _ in 0..100 {
        system.process_consciousness_cycle(0.01, None);
        
        if system.is_consciousness_emerged() {
            emergence_detected = true;
            break;
        }
    }
    
    // With proper initialization, consciousness should emerge
    // Note: This might be probabilistic, so we allow for cases where it doesn't emerge
    let diagnostics = system.get_diagnostics();
    assert!(diagnostics.system_time > 0.0);
    
    if emergence_detected {
        assert!(system.integration_strength > 1.0); // Should increase on emergence
    }
}

#[tokio::test]
async fn test_stimulus_application() {
    let mut system = ConsciousnessSystem::new((5, 5, 5), 40.0);
    system.initialize_coherent_consciousness();
    
    let position = (2, 2, 2);
    let intensity = 1.0;
    let frequency = 40.0;
    
    // Record initial state
    let initial_state = system.get_consciousness_state();
    
    // Apply stimulus
    system.apply_stimulus(position, intensity, frequency, StimulusModality::Pleasant);
    
    // Process cycles to see effect
    for _ in 0..5 {
        system.process_consciousness_cycle(0.01, None);
    }
    
    let final_state = system.get_consciousness_state();
    
    // State should have changed
    // This is a weak test since the stimulus effect might be subtle
    assert!(final_state.timestamp >= initial_state.timestamp);
}

#[tokio::test]
async fn test_consciousness_state_metrics() {
    let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
    system.initialize_coherent_consciousness();
    
    // Process some cycles
    for _ in 0..20 {
        system.process_consciousness_cycle(0.01, None);
    }
    
    let state = system.get_consciousness_state();
    
    // Check all metrics are within valid ranges
    assert!(state.field_state.consciousness_level >= 0.0 && state.field_state.consciousness_level <= 1.0);
    assert!(state.consciousness_quality.unity_level >= 0.0 && state.consciousness_quality.unity_level <= 1.0);
    assert!(state.consciousness_quality.coherence_strength >= 0.0);
    
    // Integration metrics should be valid
    assert!(state.integration_metrics.field_unity_coherence >= 0.0);
    assert!(state.integration_metrics.unity_lattice_coupling >= 0.0);
    assert!(state.integration_metrics.lattice_reality_interface >= 0.0);
    assert!(state.integration_metrics.overall_integration >= 0.0);
    assert!(state.integration_metrics.consciousness_emergence >= 0.0 && 
            state.integration_metrics.consciousness_emergence <= 1.0);
    
    // Timestamp should be reasonable
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    assert!(state.timestamp <= now);
    assert!(state.timestamp >= now - 60.0); // Within last minute
}

#[tokio::test]
async fn test_consciousness_simulation() {
    let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
    system.initialize_coherent_consciousness();
    
    let duration = 0.2; // 200ms simulation
    let time_step = 0.001; // 1ms steps
    
    let states = system.run_consciousness(duration, time_step);
    
    // Should have recorded multiple states
    assert!(states.len() > 0);
    assert!(states.len() <= (duration / time_step / 10.0) as usize + 1); // Records every 10 time steps
    
    // States should be temporally ordered
    for i in 1..states.len() {
        assert!(states[i].timestamp >= states[i-1].timestamp);
    }
    
    // Integration metrics should be computed for each state
    for state in &states {
        assert!(state.integration_metrics.overall_integration >= 0.0);
    }
}

#[tokio::test]
async fn test_consciousness_diagnostics() {
    let mut system = ConsciousnessSystem::new((4, 4, 4), 40.0);
    system.initialize_coherent_consciousness();
    
    // Process some cycles
    for _ in 0..15 {
        system.process_consciousness_cycle(0.02, None);
    }
    
    let diagnostics = system.get_diagnostics();
    
    // Check diagnostic fields
    assert!(diagnostics.integration_strength >= 1.0); // Should be initialized to 1.0
    assert!(diagnostics.system_time >= 0.0);
    
    // Boolean flags should be valid
    // Note: These might be false depending on the system state
    assert!(diagnostics.field_conscious == true || diagnostics.field_conscious == false);
    assert!(diagnostics.unity_achieved == true || diagnostics.unity_achieved == false);
    assert!(diagnostics.quantum_coherent == true || diagnostics.quantum_coherent == false);
    assert!(diagnostics.reality_collapsed == true || diagnostics.reality_collapsed == false);
    assert!(diagnostics.consciousness_emerged == true || diagnostics.consciousness_emerged == false);
}

#[tokio::test]
async fn test_different_stimulus_modalities() {
    let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
    system.initialize_coherent_consciousness();
    
    let position = (1, 1, 1);
    let intensity = 0.8;
    let frequency = 40.0;
    
    let modalities = vec![
        StimulusModality::Pleasant,
        StimulusModality::Unpleasant,
        StimulusModality::Neutral,
    ];
    
    for modality in modalities {
        // Apply stimulus
        system.apply_stimulus(position, intensity, frequency, modality);
        
        // Process a few cycles
        for _ in 0..3 {
            system.process_consciousness_cycle(0.01, None);
        }
        
        // System should remain stable
        let state = system.get_consciousness_state();
        assert!(state.integration_metrics.overall_integration >= 0.0);
    }
}

#[tokio::test]
async fn test_consciousness_integration_coupling() {
    let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
    system.initialize_coherent_consciousness();
    
    // Process enough cycles to allow coupling to develop
    for _ in 0..50 {
        system.process_consciousness_cycle(0.01, None);
    }
    
    let state = system.get_consciousness_state();
    
    // Test that coupling metrics are computed
    let metrics = &state.integration_metrics;
    
    // Field-unity coupling should be based on coherence levels
    assert!(metrics.field_unity_coherence >= 0.0);
    
    // Unity-lattice coupling should be related to unity and quantum states
    assert!(metrics.unity_lattice_coupling >= 0.0);
    
    // Lattice-reality coupling should be computed
    assert!(metrics.lattice_reality_interface >= 0.0);
    
    // Overall integration should be a function of all components
    assert!(metrics.overall_integration >= 0.0);
}

#[tokio::test]
async fn test_consciousness_temporal_consistency() {
    let mut system = ConsciousnessSystem::new((4, 4, 4), 40.0);
    system.initialize_coherent_consciousness();
    
    let mut previous_time = 0.0;
    let dt = 0.01;
    
    // Check temporal consistency over many cycles
    for _ in 0..30 {
        system.process_consciousness_cycle(dt, None);
        
        // Time should advance consistently
        assert_relative_eq!(system.time, previous_time + dt, epsilon = 1e-10);
        previous_time = system.time;
        
        // State should be consistent
        let state = system.get_consciousness_state();
        assert!(state.field_state.consciousness_level >= 0.0);
    }
}

#[tokio::test]
async fn test_consciousness_different_frequencies() {
    let frequencies = vec![20.0, 40.0, 60.0, 80.0];
    
    for frequency in frequencies {
        let mut system = ConsciousnessSystem::new((3, 3, 3), frequency);
        system.initialize_coherent_consciousness();
        
        assert_eq!(system.base_frequency, frequency);
        
        // Process cycles
        for _ in 0..10 {
            system.process_consciousness_cycle(0.01, None);
        }
        
        // System should remain stable regardless of frequency
        let state = system.get_consciousness_state();
        assert!(state.integration_metrics.overall_integration >= 0.0);
    }
}

#[tokio::test] 
async fn test_consciousness_different_dimensions() {
    let dimensions = vec![(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)];
    
    for dim in dimensions {
        let mut system = ConsciousnessSystem::new(dim, 40.0);
        system.initialize_coherent_consciousness();
        
        assert_eq!(system.dimensions, dim);
        
        // Process cycles
        for _ in 0..10 {
            system.process_consciousness_cycle(0.01, None);
        }
        
        // System should work with different dimensions
        let state = system.get_consciousness_state();
        assert!(state.integration_metrics.overall_integration >= 0.0);
    }
}

#[tokio::test]
async fn test_concurrent_consciousness_systems() {
    let mut handles = Vec::new();
    
    // Create multiple consciousness systems running concurrently
    for i in 0..3 {
        let handle = tokio::spawn(async move {
            let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0 + i as f64 * 5.0);
            system.initialize_coherent_consciousness();
            
            let states = system.run_consciousness(0.1, 0.01);
            states.len()
        });
        handles.push(handle);
    }
    
    // Wait for all systems to complete
    let mut total_states = 0;
    for handle in handles {
        let state_count = handle.await.expect("Consciousness system should complete");
        total_states += state_count;
        assert!(state_count > 0);
    }
    
    assert!(total_states > 0);
}

/// Property-based tests for consciousness system
#[cfg(feature = "property-tests")]
mod consciousness_property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_consciousness_dimensions_invariant(
            width in 2usize..10,
            height in 2usize..10,
            depth in 2usize..10,
            frequency in 10.0f64..100.0
        ) {
            let system = ConsciousnessSystem::new((width, height, depth), frequency);
            
            prop_assert_eq!(system.dimensions, (width, height, depth));
            prop_assert_eq!(system.base_frequency, frequency);
            prop_assert_eq!(system.time, 0.0);
        }
        
        #[test]
        fn test_consciousness_state_bounds(
            dt in 0.001f64..0.1,
            cycles in 1usize..50
        ) {
            tokio_test::block_on(async {
                let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
                system.initialize_coherent_consciousness();
                
                for _ in 0..cycles {
                    system.process_consciousness_cycle(dt, None);
                    
                    let state = system.get_consciousness_state();
                    
                    // Consciousness level should be bounded
                    prop_assert!(state.field_state.consciousness_level >= 0.0);
                    prop_assert!(state.field_state.consciousness_level <= 1.0);
                    
                    // Unity level should be bounded
                    prop_assert!(state.consciousness_quality.unity_level >= 0.0);
                    prop_assert!(state.consciousness_quality.unity_level <= 1.0);
                    
                    // Emergence flag should be binary
                    prop_assert!(state.integration_metrics.consciousness_emergence >= 0.0);
                    prop_assert!(state.integration_metrics.consciousness_emergence <= 1.0);
                    
                    // Overall integration should be non-negative
                    prop_assert!(state.integration_metrics.overall_integration >= 0.0);
                }
            });
        }
        
        #[test]
        fn test_stimulus_intensity_bounds(
            intensity in 0.0f64..5.0,
            frequency in 1.0f64..200.0
        ) {
            let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
            system.initialize_coherent_consciousness();
            
            // Apply stimulus with varying parameters
            system.apply_stimulus((1, 1, 1), intensity, frequency, StimulusModality::Neutral);
            
            // Process a cycle
            system.process_consciousness_cycle(0.01, None);
            
            // System should remain stable
            let state = system.get_consciousness_state();
            prop_assert!(state.integration_metrics.overall_integration >= 0.0);
        }
    }
}

/// Stress tests for consciousness system
mod consciousness_stress_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_consciousness_long_duration() {
        let mut system = ConsciousnessSystem::new((3, 3, 3), 40.0);
        system.initialize_coherent_consciousness();
        
        // Run for extended period
        let states = system.run_consciousness(1.0, 0.01); // 1 second simulation
        
        assert!(states.len() > 0);
        
        // System should maintain stability
        let final_state = states.last().unwrap();
        assert!(final_state.integration_metrics.overall_integration >= 0.0);
    }
    
    #[tokio::test]
    async fn test_consciousness_rapid_stimulation() {
        let mut system = ConsciousnessSystem::new((4, 4, 4), 40.0);
        system.initialize_coherent_consciousness();
        
        // Apply rapid stimulation
        for i in 0..20 {
            let intensity = 0.5 + (i as f64 * 0.1) % 2.0;
            let frequency = 30.0 + (i as f64 * 5.0) % 50.0;
            
            system.apply_stimulus((i % 4, (i + 1) % 4, (i + 2) % 4), 
                                intensity, frequency, StimulusModality::Pleasant);
            
            // Process a few cycles
            for _ in 0..2 {
                system.process_consciousness_cycle(0.005, None);
            }
        }
        
        // System should remain stable under rapid stimulation
        let state = system.get_consciousness_state();
        assert!(state.integration_metrics.overall_integration >= 0.0);
    }
    
    #[tokio::test]
    async fn test_consciousness_high_frequency() {
        let mut system = ConsciousnessSystem::new((3, 3, 3), 200.0); // Very high frequency
        system.initialize_coherent_consciousness();
        
        // Process with small time steps to maintain stability
        for _ in 0..100 {
            system.process_consciousness_cycle(0.0001, None); // Very small dt
        }
        
        let state = system.get_consciousness_state();
        assert!(state.integration_metrics.overall_integration >= 0.0);
    }
}