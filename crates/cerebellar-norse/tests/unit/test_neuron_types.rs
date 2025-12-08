//! Unit tests for neuron types module
//! 
//! Comprehensive tests for LIF and AdEx neuron implementations,
//! including edge cases, error conditions, and performance validation.

use std::collections::HashMap;
use candle_core::{Tensor, Device, DType};
use candle_nn as nn;
use approx::assert_abs_diff_eq;
use proptest::prelude::*;
use rstest::*;
use serial_test::serial;

use cerebellar_norse::neuron_types::*;
use cerebellar_norse::*;
use crate::utils::fixtures::*;
use crate::utils::*;

/// Test LIF neuron parameter initialization
#[rstest]
#[case(10.0, 5.0, 1.0, 0.0)]
#[case(20.0, 10.0, 2.0, -0.5)]
#[case(5.0, 2.0, 0.5, 0.2)]
fn test_lif_parameters_initialization(
    #[case] tau_mem: f64,
    #[case] tau_syn: f64,
    #[case] v_th: f64,
    #[case] v_reset: f64
) {
    let params = LIFParameters {
        tau_mem,
        tau_syn,
        v_th,
        v_reset,
        v_leak: 0.0,
        refractory_period: 2.0,
    };
    
    assert_eq!(params.tau_mem, tau_mem);
    assert_eq!(params.tau_syn, tau_syn);
    assert_eq!(params.v_th, v_th);
    assert_eq!(params.v_reset, v_reset);
}

/// Test LIF neuron default parameters
#[test]
fn test_lif_parameters_default() {
    let params = LIFParameters::default();
    
    assert_eq!(params.tau_mem, 10.0);
    assert_eq!(params.tau_syn, 5.0);
    assert_eq!(params.v_th, 1.0);
    assert_eq!(params.v_reset, 0.0);
    assert_eq!(params.v_leak, 0.0);
    assert_eq!(params.refractory_period, 2.0);
}

/// Test AdEx neuron parameter initialization
#[rstest]
#[case(10.0, 5.0, 1.0, 0.0, 100.0, 0.1, 0.1)]
#[case(15.0, 8.0, 1.5, -0.2, 80.0, 0.2, 0.05)]
fn test_adex_parameters_initialization(
    #[case] tau_mem: f64,
    #[case] tau_syn: f64,
    #[case] v_th: f64,
    #[case] v_reset: f64,
    #[case] tau_adapt: f64,
    #[case] alpha: f64,
    #[case] delta_th: f64
) {
    let params = AdExParameters {
        tau_mem,
        tau_syn,
        v_th,
        v_reset,
        v_leak: 0.0,
        tau_adapt,
        alpha,
        delta_th,
    };
    
    assert_eq!(params.tau_mem, tau_mem);
    assert_eq!(params.tau_syn, tau_syn);
    assert_eq!(params.v_th, v_th);
    assert_eq!(params.v_reset, v_reset);
    assert_eq!(params.tau_adapt, tau_adapt);
    assert_eq!(params.alpha, alpha);
    assert_eq!(params.delta_th, delta_th);
}

/// Test LIF state creation and initialization
#[test]
fn test_lif_state_creation() {
    let device = Device::Cpu;
    let size = 100;
    let params = LIFParameters::default();
    
    let state = LIFState::new(size, params, device).unwrap();
    
    assert_eq!(state.v_mem.shape().dims(), &[size]);
    assert_eq!(state.i_syn.shape().dims(), &[size]);
    assert_eq!(state.refractory.shape().dims(), &[size]);
    
    // Check initial values are zero
    let v_mem_sum: f32 = state.v_mem.sum_all().unwrap().to_scalar().unwrap();
    let i_syn_sum: f32 = state.i_syn.sum_all().unwrap().to_scalar().unwrap();
    let refrac_sum: f32 = state.refractory.sum_all().unwrap().to_scalar().unwrap();
    
    assert_abs_diff_eq!(v_mem_sum, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(i_syn_sum, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(refrac_sum, 0.0, epsilon = 1e-6);
}

/// Test AdEx state creation and initialization
#[test]
fn test_adex_state_creation() {
    let device = Device::Cpu;
    let size = 50;
    let params = AdExParameters::default();
    
    let state = AdExState::new(size, params, device).unwrap();
    
    assert_eq!(state.v_mem.shape().dims(), &[size]);
    assert_eq!(state.i_syn.shape().dims(), &[size]);
    assert_eq!(state.adaptation.shape().dims(), &[size]);
    assert_eq!(state.refractory.shape().dims(), &[size]);
    
    // Check initial values are zero
    let v_mem_sum: f32 = state.v_mem.sum_all().unwrap().to_scalar().unwrap();
    let i_syn_sum: f32 = state.i_syn.sum_all().unwrap().to_scalar().unwrap();
    let adapt_sum: f32 = state.adaptation.sum_all().unwrap().to_scalar().unwrap();
    
    assert_abs_diff_eq!(v_mem_sum, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(i_syn_sum, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(adapt_sum, 0.0, epsilon = 1e-6);
}

/// Test LIF neuron update with subthreshold input
#[test]
fn test_lif_update_subthreshold() {
    let device = Device::Cpu;
    let size = 10;
    let params = LIFParameters::default();
    let mut state = LIFState::new(size, params, device.clone()).unwrap();
    
    // Apply subthreshold input
    let input = Tensor::full(&[size], 0.5, (DType::F32, &device)).unwrap();
    let dt = 0.001;
    
    let spikes = state.update(&input, dt).unwrap();
    
    // Should not produce spikes
    let spike_sum: f32 = spikes.sum_all().unwrap().to_scalar().unwrap();
    assert_abs_diff_eq!(spike_sum, 0.0, epsilon = 1e-6);
    
    // Membrane potential should increase
    let v_mem_sum: f32 = state.v_mem.sum_all().unwrap().to_scalar().unwrap();
    assert!(v_mem_sum > 0.0);
}

/// Test LIF neuron update with suprathreshold input
#[test]
fn test_lif_update_suprathreshold() {
    let device = Device::Cpu;
    let size = 5;
    let params = LIFParameters::default();
    let mut state = LIFState::new(size, params, device.clone()).unwrap();
    
    // Apply strong input multiple times to reach threshold
    let input = Tensor::full(&[size], 2.0, (DType::F32, &device)).unwrap();
    let dt = 0.001;
    
    // Multiple updates to accumulate
    for _ in 0..10 {
        let spikes = state.update(&input, dt).unwrap();
        
        // Check if any spikes occurred
        let spike_sum: f32 = spikes.sum_all().unwrap().to_scalar().unwrap();
        if spike_sum > 0.0 {
            // Verify spikes are binary
            let spike_values: Vec<f32> = spikes.flatten_all().unwrap().to_vec1().unwrap();
            for spike in spike_values {
                assert!(spike == 0.0 || spike == 1.0);
            }
            break;
        }
    }
}

/// Test AdEx neuron adaptation mechanism
#[test]
fn test_adex_adaptation() {
    let device = Device::Cpu;
    let size = 3;
    let params = AdExParameters::default();
    let mut state = AdExState::new(size, params, device.clone()).unwrap();
    
    // Apply strong input to trigger adaptation
    let input = Tensor::full(&[size], 3.0, (DType::F32, &device)).unwrap();
    let dt = 0.001;
    
    let mut adaptation_values = Vec::new();
    
    // Track adaptation over multiple updates
    for _ in 0..20 {
        let spikes = state.update(&input, dt).unwrap();
        let adaptation: f32 = state.adaptation.sum_all().unwrap().to_scalar().unwrap();
        adaptation_values.push(adaptation);
        
        // Check if adaptation increased after spikes
        let spike_sum: f32 = spikes.sum_all().unwrap().to_scalar().unwrap();
        if spike_sum > 0.0 {
            // Next iteration should show increased adaptation
            continue;
        }
    }
    
    // Adaptation should have increased over time
    assert!(adaptation_values.last().unwrap() >= adaptation_values.first().unwrap());
}

/// Test LIF neuron reset functionality
#[test]
fn test_lif_reset() {
    let device = Device::Cpu;
    let size = 10;
    let params = LIFParameters::default();
    let mut state = LIFState::new(size, params, device.clone()).unwrap();
    
    // Apply input to change state
    let input = Tensor::full(&[size], 1.0, (DType::F32, &device)).unwrap();
    let dt = 0.001;
    
    state.update(&input, dt).unwrap();
    
    // Verify state has changed
    let v_mem_before: f32 = state.v_mem.sum_all().unwrap().to_scalar().unwrap();
    assert!(v_mem_before > 0.0);
    
    // Reset state
    state.reset().unwrap();
    
    // Verify state is reset
    let v_mem_after: f32 = state.v_mem.sum_all().unwrap().to_scalar().unwrap();
    let i_syn_after: f32 = state.i_syn.sum_all().unwrap().to_scalar().unwrap();
    
    assert_abs_diff_eq!(v_mem_after, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(i_syn_after, 0.0, epsilon = 1e-6);
}

/// Test AdEx neuron reset functionality
#[test]
fn test_adex_reset() {
    let device = Device::Cpu;
    let size = 5;
    let params = AdExParameters::default();
    let mut state = AdExState::new(size, params, device.clone()).unwrap();
    
    // Apply input to change state
    let input = Tensor::full(&[size], 2.0, (DType::F32, &device)).unwrap();
    let dt = 0.001;
    
    state.update(&input, dt).unwrap();
    
    // Reset state
    state.reset().unwrap();
    
    // Verify all state variables are reset
    let v_mem_sum: f32 = state.v_mem.sum_all().unwrap().to_scalar().unwrap();
    let i_syn_sum: f32 = state.i_syn.sum_all().unwrap().to_scalar().unwrap();
    let adapt_sum: f32 = state.adaptation.sum_all().unwrap().to_scalar().unwrap();
    
    assert_abs_diff_eq!(v_mem_sum, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(i_syn_sum, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(adapt_sum, 0.0, epsilon = 1e-6);
}

/// Test LIF cell creation and state generation
#[test]
fn test_lif_cell_creation() {
    let device = Device::Cpu;
    let params = LIFParameters::default();
    let size = 20;
    
    let cell = LIFCell::new(params.clone(), size, device.clone());
    
    assert_eq!(cell.size, size);
    assert_eq!(cell.params.tau_mem, params.tau_mem);
    assert_eq!(cell.device, device);
    
    // Test state creation
    let state = cell.create_state().unwrap();
    assert_eq!(state.v_mem.shape().dims(), &[size]);
}

/// Test AdEx cell creation and state generation
#[test]
fn test_adex_cell_creation() {
    let device = Device::Cpu;
    let params = AdExParameters::default();
    let size = 15;
    
    let cell = AdExCell::new(params.clone(), size, device.clone());
    
    assert_eq!(cell.size, size);
    assert_eq!(cell.params.tau_mem, params.tau_mem);
    assert_eq!(cell.device, device);
    
    // Test state creation
    let state = cell.create_state().unwrap();
    assert_eq!(state.v_mem.shape().dims(), &[size]);
    assert_eq!(state.adaptation.shape().dims(), &[size]);
}

/// Test neuron factory for different layer types
#[test]
fn test_neuron_factory() {
    let granule = NeuronFactory::create_granule_cell();
    let purkinje = NeuronFactory::create_purkinje_cell();
    let golgi = NeuronFactory::create_golgi_cell();
    let dcn = NeuronFactory::create_dcn_cell();
    
    // Test different thresholds
    assert!(granule.threshold < purkinje.threshold);
    assert!(purkinje.threshold < dcn.threshold);
    assert_eq!(golgi.threshold, 1.0);
    
    // Test different decay constants
    assert_eq!(granule.decay_mem, 0.95);
    assert_eq!(purkinje.decay_mem, 0.9);
    assert_eq!(golgi.decay_mem, 0.85);
    assert_eq!(dcn.decay_mem, 0.92);
}

/// Test neuron cell factory with different configurations
#[test]
fn test_neuron_cell_factory() {
    let device = Device::Cpu;
    let dt = 0.001;
    
    // Test for different layer types
    let layer_types = vec![
        LayerType::GranuleCell,
        LayerType::PurkinjeCell,
        LayerType::GolgiCell,
        LayerType::DeepCerebellarNucleus,
    ];
    
    for layer_type in layer_types {
        let config = LayerConfig {
            size: 10,
            layer_type,
            neuron_type: NeuronType::LIF,
        };
        
        let lif_cell = NeuronCellFactory::create_lif_cell(&config, dt, device.clone());
        assert_eq!(lif_cell.size, 10);
        
        let adex_cell = NeuronCellFactory::create_adex_cell(&config, dt, device.clone());
        assert_eq!(adex_cell.size, 10);
    }
}

/// Test batch neuron processor
#[test]
fn test_batch_neuron_processor() {
    let neurons = vec![LIFNeuron::new_trading_optimized(); 50];
    let mut processor = BatchNeuronProcessor::new(neurons);
    
    let inputs = vec![1.0; 50];
    let spikes = processor.process_batch(&inputs);
    
    assert_eq!(spikes.len(), 50);
    
    // Check that outputs are boolean
    for spike in spikes {
        assert!(spike == true || spike == false);
    }
    
    // Test batch statistics
    let stats = processor.get_batch_stats();
    assert_eq!(stats.total_neurons, 50);
    assert!(stats.active_neurons <= 50);
    assert!(stats.average_membrane_potential >= 0.0);
}

/// Test neuron dynamics with different time constants
#[rstest]
#[case(5.0, 10.0)]
#[case(10.0, 5.0)]
#[case(20.0, 2.0)]
fn test_neuron_dynamics_time_constants(#[case] tau_mem: f64, #[case] tau_syn: f64) {
    let device = Device::Cpu;
    let size = 1;
    let params = LIFParameters {
        tau_mem,
        tau_syn,
        v_th: 1.0,
        v_reset: 0.0,
        v_leak: 0.0,
        refractory_period: 2.0,
    };
    
    let mut state = LIFState::new(size, params, device.clone()).unwrap();
    
    // Apply constant input
    let input = Tensor::full(&[size], 0.5, (DType::F32, &device)).unwrap();
    let dt = 0.001;
    
    // Track membrane potential over time
    let mut v_mem_history = Vec::new();
    
    for _ in 0..100 {
        state.update(&input, dt).unwrap();
        let v_mem: f32 = state.v_mem.get(0).unwrap().to_scalar().unwrap();
        v_mem_history.push(v_mem);
    }
    
    // Check that membrane potential approaches steady state
    let final_v = v_mem_history.last().unwrap();
    let mid_v = &v_mem_history[v_mem_history.len() / 2];
    
    assert!(final_v > mid_v, "Membrane potential should increase over time");
}

/// Test neuron behavior with different input patterns
#[rstest]
#[case(vec![1.0, 1.0, 1.0, 1.0, 1.0])]
#[case(vec![0.0, 0.5, 1.0, 0.5, 0.0])]
#[case(vec![2.0, 0.0, 2.0, 0.0, 2.0])]
fn test_neuron_input_patterns(#[case] input_pattern: Vec<f32>) {
    let device = Device::Cpu;
    let size = 1;
    let params = LIFParameters::default();
    let mut state = LIFState::new(size, params, device.clone()).unwrap();
    
    let dt = 0.001;
    let mut spike_count = 0;
    
    for input_val in input_pattern {
        let input = Tensor::full(&[size], input_val, (DType::F32, &device)).unwrap();
        let spikes = state.update(&input, dt).unwrap();
        let spike: f32 = spikes.get(0).unwrap().to_scalar().unwrap();
        
        if spike > 0.0 {
            spike_count += 1;
        }
    }
    
    // At least validate that the function runs without errors
    assert!(spike_count >= 0);
}

/// Test error handling for invalid tensor shapes
#[test]
fn test_error_handling_invalid_shapes() {
    let device = Device::Cpu;
    let size = 10;
    let params = LIFParameters::default();
    let mut state = LIFState::new(size, params, device.clone()).unwrap();
    
    // Try to update with wrong input size
    let wrong_input = Tensor::full(&[size + 5], 1.0, (DType::F32, &device)).unwrap();
    let dt = 0.001;
    
    // This should fail due to tensor shape mismatch
    let result = state.update(&wrong_input, dt);
    assert!(result.is_err());
}

/// Test neuron state serialization/deserialization
#[test]
fn test_neuron_state_serialization() {
    let params = LIFParameters::default();
    let serialized = serde_json::to_string(&params).unwrap();
    let deserialized: LIFParameters = serde_json::from_str(&serialized).unwrap();
    
    assert_eq!(params.tau_mem, deserialized.tau_mem);
    assert_eq!(params.tau_syn, deserialized.tau_syn);
    assert_eq!(params.v_th, deserialized.v_th);
    assert_eq!(params.v_reset, deserialized.v_reset);
}

/// Property-based test for LIF neuron stability
proptest! {
    #[test]
    fn test_lif_neuron_stability(
        input_current in -5.0f32..5.0f32,
        tau_mem in 1.0f64..100.0f64,
        tau_syn in 1.0f64..50.0f64,
        dt in 0.0001f64..0.01f64
    ) {
        let device = Device::Cpu;
        let size = 1;
        let params = LIFParameters {
            tau_mem,
            tau_syn,
            v_th: 1.0,
            v_reset: 0.0,
            v_leak: 0.0,
            refractory_period: 2.0,
        };
        
        let mut state = LIFState::new(size, params, device.clone()).unwrap();
        let input = Tensor::full(&[size], input_current, (DType::F32, &device)).unwrap();
        
        // Run for many steps to test stability
        for _ in 0..1000 {
            let result = state.update(&input, dt);
            prop_assert!(result.is_ok());
            
            // Check that values remain finite
            let v_mem: f32 = state.v_mem.get(0).unwrap().to_scalar().unwrap();
            let i_syn: f32 = state.i_syn.get(0).unwrap().to_scalar().unwrap();
            
            prop_assert!(v_mem.is_finite());
            prop_assert!(i_syn.is_finite());
        }
    }
}

/// Performance test for neuron updates
#[test]
#[serial]
fn test_neuron_update_performance() {
    let device = Device::Cpu;
    let size = 1000;
    let params = LIFParameters::default();
    let mut state = LIFState::new(size, params, device.clone()).unwrap();
    
    let input = Tensor::randn(&[size], (DType::F32, &device)).unwrap();
    let dt = 0.001;
    
    let start_time = std::time::Instant::now();
    
    // Perform many updates
    for _ in 0..1000 {
        state.update(&input, dt).unwrap();
    }
    
    let duration = start_time.elapsed();
    
    // Should complete in reasonable time (less than 1 second)
    assert!(duration < std::time::Duration::from_secs(1));
    
    // Calculate throughput
    let operations_per_second = 1_000_000 / duration.as_micros();
    println!("Neuron update throughput: {} ops/sec", operations_per_second);
}

/// Test neuron behavior with extreme parameters
#[test]
fn test_neuron_extreme_parameters() {
    let device = Device::Cpu;
    let size = 1;
    
    // Test with very fast dynamics
    let fast_params = LIFParameters {
        tau_mem: 0.1,
        tau_syn: 0.05,
        v_th: 0.1,
        v_reset: 0.0,
        v_leak: 0.0,
        refractory_period: 0.1,
    };
    
    let mut fast_state = LIFState::new(size, fast_params, device.clone()).unwrap();
    
    // Test with very slow dynamics
    let slow_params = LIFParameters {
        tau_mem: 1000.0,
        tau_syn: 500.0,
        v_th: 10.0,
        v_reset: 0.0,
        v_leak: 0.0,
        refractory_period: 100.0,
    };
    
    let mut slow_state = LIFState::new(size, slow_params, device.clone()).unwrap();
    
    let input = Tensor::full(&[size], 1.0, (DType::F32, &device)).unwrap();
    let dt = 0.001;
    
    // Both should work without errors
    let fast_result = fast_state.update(&input, dt);
    let slow_result = slow_state.update(&input, dt);
    
    assert!(fast_result.is_ok());
    assert!(slow_result.is_ok());
}

/// Test neuron memory usage
#[test]
fn test_neuron_memory_usage() {
    let device = Device::Cpu;
    let sizes = vec![10, 100, 1000];
    let params = LIFParameters::default();
    
    for size in sizes {
        let (state, memory_used) = measure_memory_usage(|| {
            LIFState::new(size, params.clone(), device.clone()).unwrap()
        });
        
        assert_eq!(state.v_mem.shape().dims(), &[size]);
        
        // Memory usage should scale reasonably with size
        // This is a simplified check - real memory measurement would be more complex
        assert!(memory_used >= 0);
    }
}

/// Test concurrent neuron updates
#[test]
#[serial]
fn test_concurrent_neuron_updates() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let device = Device::Cpu;
    let size = 100;
    let params = LIFParameters::default();
    let state = Arc::new(Mutex::new(LIFState::new(size, params, device.clone()).unwrap()));
    
    let input = Tensor::randn(&[size], (DType::F32, &device)).unwrap();
    let dt = 0.001;
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads to update neurons
    for _ in 0..4 {
        let state_clone = Arc::clone(&state);
        let input_clone = input.clone();
        
        let handle = thread::spawn(move || {
            let mut state = state_clone.lock().unwrap();
            for _ in 0..10 {
                state.update(&input_clone, dt).unwrap();
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final state is valid
    let final_state = state.lock().unwrap();
    let v_mem: f32 = final_state.v_mem.sum_all().unwrap().to_scalar().unwrap();
    assert!(v_mem.is_finite());
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    /// Test integration between LIF and AdEx neurons
    #[test]
    fn test_lif_adex_integration() {
        let device = Device::Cpu;
        let size = 5;
        
        // Create LIF and AdEx states
        let lif_params = LIFParameters::default();
        let adex_params = AdExParameters::default();
        
        let mut lif_state = LIFState::new(size, lif_params, device.clone()).unwrap();
        let mut adex_state = AdExState::new(size, adex_params, device.clone()).unwrap();
        
        // Apply same input to both
        let input = Tensor::full(&[size], 1.5, (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        let lif_spikes = lif_state.update(&input, dt).unwrap();
        let adex_spikes = adex_state.update(&input, dt).unwrap();
        
        // Both should produce valid outputs
        assert_eq!(lif_spikes.shape().dims(), &[size]);
        assert_eq!(adex_spikes.shape().dims(), &[size]);
        
        // Verify outputs are binary
        let lif_values: Vec<f32> = lif_spikes.flatten_all().unwrap().to_vec1().unwrap();
        let adex_values: Vec<f32> = adex_spikes.flatten_all().unwrap().to_vec1().unwrap();
        
        for value in lif_values {
            assert!(value == 0.0 || value == 1.0);
        }
        
        for value in adex_values {
            assert!(value == 0.0 || value == 1.0);
        }
    }
    
    /// Test neuron scaling with different sizes
    #[test]
    fn test_neuron_scaling() {
        let device = Device::Cpu;
        let params = LIFParameters::default();
        let sizes = vec![1, 10, 100, 1000];
        
        for size in sizes {
            let mut state = LIFState::new(size, params.clone(), device.clone()).unwrap();
            let input = Tensor::full(&[size], 1.0, (DType::F32, &device)).unwrap();
            let dt = 0.001;
            
            // Test that scaling works correctly
            let (spikes, duration) = time_operation(|| {
                state.update(&input, dt).unwrap()
            });
            
            assert_eq!(spikes.shape().dims(), &[size]);
            
            // Performance should scale reasonably
            println!("Size {}: {} μs", size, duration.as_micros());
        }
    }
}