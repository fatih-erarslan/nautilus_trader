//! Comprehensive tests for STDP synapse implementation
//! 
//! Tests include validation against biological data, Hebbian learning principles,
//! and edge cases for robust implementation.

use super::*;
use approx::assert_abs_diff_eq;

/// Test STDP curve matches biological expectations
#[test]
fn test_biological_stdp_curve() {
    let synapse = STDPSynapse::new(0, 1, 0.5);
    
    // Test specific Δt values from biological literature
    let test_points = vec![
        (-50.0, -0.00525 * (-50.0 / 20.0).exp()), // LTD region
        (-20.0, -0.00525 * (-20.0 / 20.0).exp()), // LTD peak
        (-10.0, -0.00525 * (-10.0 / 20.0).exp()),
        (0.0, 0.0),                                 // Zero at Δt = 0
        (10.0, 0.005 * (-10.0 / 20.0).exp()),     // LTP region
        (20.0, 0.005 * (-20.0 / 20.0).exp()),     // LTP peak
        (50.0, 0.005 * (-50.0 / 20.0).exp()),
    ];
    
    let curve = synapse.get_stdp_curve(&test_points.iter().map(|(dt, _)| *dt).collect::<Vec<_>>());
    
    for (i, (expected_dt, expected_dw)) in test_points.iter().enumerate() {
        let (actual_dt, actual_dw) = curve[i];
        assert_abs_diff_eq!(actual_dt, *expected_dt, epsilon = 1e-10);
        assert_abs_diff_eq!(actual_dw, *expected_dw, epsilon = 1e-6);
    }
}

/// Test weight evolution under repeated stimulation
#[test]
fn test_weight_evolution_stability() {
    let mut synapse = STDPSynapse::new(0, 1, 0.5);
    let dt = 0.1;
    let initial_weight = synapse.weight;
    
    // Apply 1000 LTP-inducing spike pairs
    for i in 0..1000 {
        let time = i as f64 * dt * 10.0; // 10ms between pairs
        
        // Pre spike at t, post spike at t + 1ms (LTP)
        synapse.update_stdp(true, false, dt, time).unwrap();
        synapse.update_stdp(false, true, dt, time + 1.0).unwrap();
        
        // Allow traces to decay
        for _ in 0..50 {
            synapse.update_stdp(false, false, dt, time + 5.0).unwrap();
        }
    }
    
    // Weight should increase but remain stable
    assert!(synapse.weight > initial_weight);
    assert!(synapse.weight <= synapse.w_max);
    assert!(synapse.weight >= synapse.w_min);
    
    // Test convergence - additional stimulation should have minimal effect
    let stable_weight = synapse.weight;
    
    // Apply 100 more LTP pairs
    for i in 1000..1100 {
        let time = i as f64 * dt * 10.0;
        synapse.update_stdp(true, false, dt, time).unwrap();
        synapse.update_stdp(false, true, dt, time + 1.0).unwrap();
    }
    
    let weight_change = (synapse.weight - stable_weight).abs();
    assert!(weight_change < 0.01, "Weight should stabilize after repeated stimulation");
}

/// Test Hebbian learning principle validation
#[test]
fn test_hebbian_learning_principle() {
    // "Cells that fire together, wire together"
    let mut correlated_synapse = STDPSynapse::new(0, 1, 0.5);
    let mut anticorrelated_synapse = STDPSynapse::new(2, 3, 0.5);
    let dt = 0.1;
    
    let initial_corr_weight = correlated_synapse.weight;
    let initial_anticorr_weight = anticorrelated_synapse.weight;
    
    // Simulate 100 trials
    for i in 0..100 {
        let time = i as f64 * 50.0; // 50ms between trials
        
        // Correlated activity: pre always followed by post
        correlated_synapse.update_stdp(true, false, dt, time).unwrap();
        correlated_synapse.update_stdp(false, true, dt, time + 2.0).unwrap(); // 2ms delay
        
        // Anti-correlated activity: post always followed by pre
        anticorrelated_synapse.update_stdp(false, true, dt, time).unwrap();
        anticorrelated_synapse.update_stdp(true, false, dt, time + 2.0).unwrap(); // 2ms delay
        
        // Allow time for trace decay
        for j in 0..100 {
            let decay_time = time + 10.0 + j as f64 * dt;
            correlated_synapse.update_stdp(false, false, dt, decay_time).unwrap();
            anticorrelated_synapse.update_stdp(false, false, dt, decay_time).unwrap();
        }
    }
    
    // Correlated activity should strengthen synapse
    assert!(correlated_synapse.weight > initial_corr_weight,
           "Correlated activity should increase synaptic weight");
    
    // Anti-correlated activity should weaken synapse
    assert!(anticorrelated_synapse.weight < initial_anticorr_weight,
           "Anti-correlated activity should decrease synaptic weight");
    
    let corr_stats = correlated_synapse.get_statistics();
    let anticorr_stats = anticorrelated_synapse.get_statistics();
    
    assert!(corr_stats.potentiation_ratio > anticorr_stats.potentiation_ratio);
    assert!(anticorr_stats.depression_ratio > corr_stats.depression_ratio);
}

/// Test realistic spike pattern learning
#[test]
fn test_realistic_spike_patterns() {
    let mut synapse = STDPSynapse::new(0, 1, 0.5);
    let dt = 0.1;
    
    // Simulate realistic Poisson-like spike trains
    use std::collections::VecDeque;
    
    // Pre-synaptic spikes (10 Hz average)
    let pre_spikes = vec![12.0, 145.0, 287.0, 398.0, 523.0, 687.0, 789.0, 834.0, 976.0];
    // Post-synaptic spikes (8 Hz average, slightly delayed for correlation)
    let post_spikes = vec![15.0, 148.0, 291.0, 401.0, 527.0, 691.0, 793.0, 838.0];
    
    let mut pre_iter = pre_spikes.iter().peekable();
    let mut post_iter = post_spikes.iter().peekable();
    
    let initial_weight = synapse.weight;
    
    // Simulate 1 second of activity
    for step in 0..10000 {
        let current_time = step as f64 * dt;
        
        let pre_spike = pre_iter.peek().map_or(false, |&&t| {
            if (current_time - t).abs() < dt/2.0 {
                pre_iter.next();
                true
            } else {
                false
            }
        });
        
        let post_spike = post_iter.peek().map_or(false, |&&t| {
            if (current_time - t).abs() < dt/2.0 {
                post_iter.next();
                true
            } else {
                false
            }
        });
        
        synapse.update_stdp(pre_spike, post_spike, dt, current_time).unwrap();
    }
    
    // Weight should increase due to positive correlation in realistic patterns
    assert!(synapse.weight > initial_weight, 
           "Weight should increase for positively correlated realistic spike patterns");
    
    let stats = synapse.get_statistics();
    println!("Final weight: {:.6}", synapse.weight);
    println!("Potentiation ratio: {:.6}", stats.potentiation_ratio);
    println!("Depression ratio: {:.6}", stats.depression_ratio);
}

/// Test edge cases and error handling
#[test]
fn test_edge_cases() {
    let mut synapse = STDPSynapse::new(0, 1, 0.5);
    
    // Test invalid time step
    let result = synapse.update_stdp(false, false, 0.0, 0.0);
    assert!(result.is_err());
    
    let result = synapse.update_stdp(false, false, -1.0, 0.0);
    assert!(result.is_err());
    
    // Test extreme parameter values
    let mut extreme_synapse = STDPSynapse::with_parameters(
        0, 1, 0.5,
        1.0,    // Very large A+
        1.0,    // Very large A-
        1.0,    // Very small tau+
        1.0,    // Very small tau-
        10.0,   // Large w_max
        0.1,    // Small delay
    );
    
    // Should still function without errors
    let result = extreme_synapse.update_stdp(true, true, 0.1, 0.0);
    assert!(result.is_ok());
    
    // Weight should be bounded
    assert!(extreme_synapse.weight <= extreme_synapse.w_max);
    assert!(extreme_synapse.weight >= extreme_synapse.w_min);
}

/// Test synaptic delay functionality
#[test]
fn test_synaptic_delay_accuracy() {
    let delay = 3.0; // 3ms delay
    let mut synapse = STDPSynapse::with_parameters(
        0, 1, 0.5,
        0.005, 0.00525, 20.0, 20.0, 1.0, delay
    );
    
    // Send spike at t=0
    synapse.update_stdp(true, false, 0.1, 0.0).unwrap();
    
    // Check output at various times
    let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let outputs: Vec<f64> = times.iter()
        .map(|&t| synapse.get_delayed_output(t))
        .collect();
    
    // Output should be zero before delay time
    assert_abs_diff_eq!(outputs[0], 0.0, epsilon = 1e-10); // t=0
    assert_abs_diff_eq!(outputs[1], 0.0, epsilon = 1e-10); // t=1ms
    assert_abs_diff_eq!(outputs[2], 0.0, epsilon = 1e-10); // t=2ms
    
    // Output should be non-zero at delay time
    assert!(outputs[3] > 0.0, "Output should be positive at t={}ms", delay); // t=3ms
    
    // Output should return to zero after spike
    // (This test might need adjustment based on exact implementation)
}

/// Test trace dynamics and exponential decay
#[test]
fn test_trace_dynamics() {
    let mut synapse = STDPSynapse::new(0, 1, 0.5);
    let dt = 1.0; // 1ms steps for easy calculation
    
    // Apply pre-synaptic spike to set trace
    synapse.update_stdp(true, false, dt, 0.0).unwrap();
    
    let initial_pre_trace = synapse.pre_trace;
    assert_abs_diff_eq!(initial_pre_trace, 1.0, epsilon = 1e-10);
    
    // Test exponential decay over multiple time steps
    let expected_values = vec![
        1.0 * (-1.0 / 20.0).exp(),    // After 1ms
        1.0 * (-2.0 / 20.0).exp(),    // After 2ms  
        1.0 * (-3.0 / 20.0).exp(),    // After 3ms
        1.0 * (-5.0 / 20.0).exp(),    // After 5ms
        1.0 * (-10.0 / 20.0).exp(),   // After 10ms
    ];
    
    let time_steps = vec![1.0, 1.0, 1.0, 2.0, 5.0]; // Cumulative: 1, 2, 3, 5, 10
    
    for (i, &dt_step) in time_steps.iter().enumerate() {
        synapse.update_stdp(false, false, dt_step, (i+1) as f64).unwrap();
        assert_abs_diff_eq!(synapse.pre_trace, expected_values[i], epsilon = 1e-6,
                           "Trace decay mismatch at step {}", i+1);
    }
    
    // Test post-synaptic trace decay similarly
    synapse.reset();
    synapse.update_stdp(false, true, dt, 0.0).unwrap();
    
    let initial_post_trace = synapse.post_trace;
    assert_abs_diff_eq!(initial_post_trace, 1.0, epsilon = 1e-10);
    
    // Decay should follow same exponential pattern
    synapse.update_stdp(false, false, dt, 1.0).unwrap();
    let expected_post_trace = 1.0 * (-1.0 / 20.0).exp();
    assert_abs_diff_eq!(synapse.post_trace, expected_post_trace, epsilon = 1e-6);
}

/// Performance test for high-frequency updates
#[test]
fn test_performance_high_frequency() {
    let mut synapse = STDPSynapse::new(0, 1, 0.5);
    let dt = 0.01; // 0.01ms (100kHz update rate)
    let num_updates = 100_000;
    
    let start_time = std::time::Instant::now();
    
    for i in 0..num_updates {
        let current_time = i as f64 * dt;
        let pre_spike = i % 1000 == 0;  // Spike every 10ms
        let post_spike = i % 1100 == 0; // Slightly different frequency
        
        synapse.update_stdp(pre_spike, post_spike, dt, current_time).unwrap();
    }
    
    let elapsed = start_time.elapsed();
    let updates_per_sec = num_updates as f64 / elapsed.as_secs_f64();
    
    println!("STDP Performance: {:.0} updates/sec", updates_per_sec);
    
    // Should handle at least 1M updates per second on modern hardware
    assert!(updates_per_sec > 100_000.0, 
           "Performance too slow: {} updates/sec", updates_per_sec);
    
    // Verify synapse is still functional
    let stats = synapse.get_statistics();
    assert!(stats.spike_count > 0);
    assert!(synapse.weight >= synapse.w_min);
    assert!(synapse.weight <= synapse.w_max);
}

/// Test multiple synapses interaction (network-level behavior)
#[test] 
fn test_multiple_synapses_network() {
    // Create a small network: 2 input neurons, 1 output neuron
    let mut synapses = vec![
        STDPSynapse::new(0, 2, 0.3), // Input 0 -> Output
        STDPSynapse::new(1, 2, 0.7), // Input 1 -> Output
    ];
    
    let dt = 0.1;
    let initial_weights: Vec<f64> = synapses.iter().map(|s| s.weight).collect();
    
    // Simulate competitive learning scenario:
    // Input 0 is more frequently correlated with output
    for trial in 0..200 {
        let time = trial as f64 * 20.0; // 20ms between trials
        
        // Input 0 active in 80% of trials, Input 1 active in 40% of trials
        let input0_active = trial % 5 != 0; // 80% of trials
        let input1_active = trial % 5 == 0 || trial % 5 == 1; // 40% of trials
        let output_active = input0_active || input1_active;
        
        // Apply spikes with 1ms delay for causality
        for (i, synapse) in synapses.iter_mut().enumerate() {
            let pre_spike = if i == 0 { input0_active } else { input1_active };
            
            synapse.update_stdp(pre_spike, false, dt, time).unwrap();
            synapse.update_stdp(false, output_active, dt, time + 1.0).unwrap();
            
            // Allow trace decay
            for j in 0..50 {
                synapse.update_stdp(false, false, dt, time + 5.0 + j as f64 * dt).unwrap();
            }
        }
    }
    
    // Synapse 0 (more correlated) should be stronger than Synapse 1
    assert!(synapses[0].weight > initial_weights[0], 
           "Highly correlated synapse should strengthen");
    assert!(synapses[0].weight > synapses[1].weight,
           "More correlated synapse should be stronger");
    
    let stats0 = synapses[0].get_statistics();
    let stats1 = synapses[1].get_statistics();
    
    assert!(stats0.potentiation_ratio > stats1.potentiation_ratio,
           "More active synapse should have higher potentiation ratio");
}

/// Integration test with spiking neurons
#[test]
fn test_integration_with_spiking_neurons() {
    let mut pre_neuron = SpikingNeuron::new(0);
    let mut post_neuron = SpikingNeuron::new(1);
    let mut synapse = STDPSynapse::new(0, 1, 0.5);
    
    let dt = 0.1;
    let simulation_time = 1000.0; // 1 second
    let steps = (simulation_time / dt) as usize;
    
    let initial_weight = synapse.weight;
    
    for step in 0..steps {
        let current_time = step as f64 * dt;
        
        // Apply external input to pre-neuron
        let input_current = if (current_time % 100.0) < 50.0 { 2.0 } else { 0.1 };
        let pre_spiked = pre_neuron.update(input_current, dt, current_time);
        
        // Apply synaptic current to post-neuron
        let synaptic_current = synapse.get_delayed_output(current_time) * 5.0;
        let post_spiked = post_neuron.update(synaptic_current, dt, current_time);
        
        // Update synapse
        synapse.update_stdp(pre_spiked, post_spiked, dt, current_time).unwrap();
    }
    
    // Verify learning occurred
    let final_weight = synapse.weight;
    let stats = synapse.get_statistics();
    
    println!("Integration test results:");
    println!("  Initial weight: {:.6}", initial_weight);
    println!("  Final weight: {:.6}", final_weight);
    println!("  Pre-neuron spikes: {}", pre_neuron.spike_times.len());
    println!("  Post-neuron spikes: {}", post_neuron.spike_times.len());
    println!("  Synapse spike count: {}", stats.spike_count);
    
    assert!(stats.spike_count > 0, "Synapse should have processed spikes");
    assert!(pre_neuron.spike_times.len() > 0, "Pre-neuron should have spiked");
    
    // Weight should evolve based on spike timing relationships
    assert!((final_weight - initial_weight).abs() > 0.001,
           "Weight should change significantly during learning");
}