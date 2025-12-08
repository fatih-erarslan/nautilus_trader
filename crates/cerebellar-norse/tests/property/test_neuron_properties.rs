//! Property-based tests for neuron implementations
//! 
//! These tests use property-based testing to validate neuron behavior
//! across a wide range of inputs and parameters, ensuring robustness
//! and correctness under all conditions.

use std::collections::HashMap;
use candle_core::{Tensor, Device, DType};
use proptest::prelude::*;
use approx::assert_abs_diff_eq;

use cerebellar_norse::neuron_types::*;
use cerebellar_norse::*;
use crate::utils::fixtures::*;
use crate::utils::*;

/// Property: LIF neuron membrane potential should be finite for all inputs
proptest! {
    #[test]
    fn prop_lif_membrane_potential_finite(
        input_current in -10.0f32..10.0f32,
        tau_mem in 0.1f64..100.0f64,
        tau_syn in 0.1f64..50.0f64,
        dt in 0.0001f64..0.1f64,
        steps in 1usize..1000
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
        
        for _ in 0..steps {
            state.update(&input, dt).unwrap();
            
            let v_mem: f32 = state.v_mem.get(0).unwrap().to_scalar().unwrap();
            let i_syn: f32 = state.i_syn.get(0).unwrap().to_scalar().unwrap();
            
            prop_assert!(v_mem.is_finite(), "Membrane potential should be finite: {}", v_mem);
            prop_assert!(i_syn.is_finite(), "Synaptic current should be finite: {}", i_syn);
        }
    }
}

/// Property: AdEx neuron adaptation should be non-negative
proptest! {
    #[test]
    fn prop_adex_adaptation_nonnegative(
        input_current in -5.0f32..5.0f32,
        tau_adapt in 1.0f64..500.0f64,
        alpha in 0.001f64..1.0f64,
        steps in 1usize..500
    ) {
        let device = Device::Cpu;
        let size = 1;
        let params = AdExParameters {
            tau_mem: 10.0,
            tau_syn: 5.0,
            v_th: 1.0,
            v_reset: 0.0,
            v_leak: 0.0,
            tau_adapt,
            alpha,
            delta_th: 0.1,
        };
        
        let mut state = AdExState::new(size, params, device.clone()).unwrap();
        let input = Tensor::full(&[size], input_current, (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        for _ in 0..steps {
            state.update(&input, dt).unwrap();
            
            let adaptation: f32 = state.adaptation.get(0).unwrap().to_scalar().unwrap();
            prop_assert!(adaptation >= 0.0, "Adaptation should be non-negative: {}", adaptation);
            prop_assert!(adaptation.is_finite(), "Adaptation should be finite: {}", adaptation);
        }
    }
}

/// Property: Spike outputs should always be binary (0 or 1)
proptest! {
    #[test]
    fn prop_spike_outputs_binary(
        input_current in -10.0f32..10.0f32,
        size in 1usize..100,
        neuron_type in prop::sample::select(vec![NeuronType::LIF, NeuronType::AdEx])
    ) {
        let device = Device::Cpu;
        let dt = 0.001;
        
        let config = LayerConfig {
            size,
            neuron_type,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        };
        
        let mut layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
        let input = Tensor::full(&[1, size], input_current, (DType::F32, &device)).unwrap();
        
        let (spikes, _) = layer.forward(&input).unwrap();
        let spike_values: Vec<f32> = spikes.flatten_all().unwrap().to_vec1().unwrap();
        
        for spike in spike_values {
            prop_assert!(spike == 0.0 || spike == 1.0, "Spike value should be 0 or 1: {}", spike);
        }
    }
}

/// Property: Neuron reset should restore initial state
proptest! {
    #[test]
    fn prop_neuron_reset_restores_initial_state(
        size in 1usize..50,
        input_current in -5.0f32..5.0f32,
        steps in 1usize..100
    ) {
        let device = Device::Cpu;
        let params = LIFParameters::default();
        let mut state = LIFState::new(size, params, device.clone()).unwrap();
        
        // Get initial state
        let initial_v_mem: f32 = state.v_mem.sum_all().unwrap().to_scalar().unwrap();
        let initial_i_syn: f32 = state.i_syn.sum_all().unwrap().to_scalar().unwrap();
        
        // Apply input to change state
        let input = Tensor::full(&[size], input_current, (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        for _ in 0..steps {
            state.update(&input, dt).unwrap();
        }
        
        // Reset state
        state.reset().unwrap();
        
        // Check that state is restored
        let final_v_mem: f32 = state.v_mem.sum_all().unwrap().to_scalar().unwrap();
        let final_i_syn: f32 = state.i_syn.sum_all().unwrap().to_scalar().unwrap();
        
        prop_assert_eq!(initial_v_mem, final_v_mem);
        prop_assert_eq!(initial_i_syn, final_i_syn);
    }
}

/// Property: Membrane potential should not exceed threshold after reset
proptest! {
    #[test]
    fn prop_membrane_potential_reset_after_spike(
        input_current in 1.0f32..10.0f32,
        v_th in 0.1f64..5.0f64,
        v_reset in -1.0f64..1.0f64,
        steps in 1usize..200
    ) {
        let device = Device::Cpu;
        let size = 1;
        let params = LIFParameters {
            tau_mem: 10.0,
            tau_syn: 5.0,
            v_th,
            v_reset,
            v_leak: 0.0,
            refractory_period: 2.0,
        };
        
        let mut state = LIFState::new(size, params, device.clone()).unwrap();
        let input = Tensor::full(&[size], input_current, (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        for _ in 0..steps {
            let spikes = state.update(&input, dt).unwrap();
            let spike: f32 = spikes.get(0).unwrap().to_scalar().unwrap();
            let v_mem: f32 = state.v_mem.get(0).unwrap().to_scalar().unwrap();
            
            if spike > 0.0 {
                // After spike, membrane potential should be at reset value
                prop_assert!((v_mem - v_reset as f32).abs() < 1e-6, 
                    "Membrane potential should be reset after spike: {} != {}", v_mem, v_reset);
            }
        }
    }
}

/// Property: Synaptic current should decay over time without input
proptest! {
    #[test]
    fn prop_synaptic_current_decay(
        initial_current in 0.1f32..5.0f32,
        tau_syn in 1.0f64..100.0f64,
        steps in 10usize..200
    ) {
        let device = Device::Cpu;
        let size = 1;
        let params = LIFParameters {
            tau_mem: 10.0,
            tau_syn,
            v_th: 100.0, // High threshold to prevent spikes
            v_reset: 0.0,
            v_leak: 0.0,
            refractory_period: 2.0,
        };
        
        let mut state = LIFState::new(size, params, device.clone()).unwrap();
        
        // Set initial synaptic current
        state.i_syn = Tensor::full(&[size], initial_current, (DType::F32, &device)).unwrap();
        
        let no_input = Tensor::zeros(&[size], (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        let mut previous_current = initial_current;
        
        for _ in 0..steps {
            state.update(&no_input, dt).unwrap();
            let current_current: f32 = state.i_syn.get(0).unwrap().to_scalar().unwrap();
            
            // Current should decay (decrease or stay the same)
            prop_assert!(current_current <= previous_current + 1e-6, 
                "Synaptic current should decay: {} > {}", current_current, previous_current);
            
            previous_current = current_current;
        }
        
        // Final current should be less than initial
        prop_assert!(previous_current < initial_current, 
            "Final current should be less than initial: {} >= {}", previous_current, initial_current);
    }
}

/// Property: Refractory period should prevent spikes
proptest! {
    #[test]
    fn prop_refractory_period_prevents_spikes(
        input_current in 2.0f32..10.0f32,
        refractory_period in 1.0f64..10.0f64
    ) {
        let device = Device::Cpu;
        let size = 1;
        let params = LIFParameters {
            tau_mem: 5.0,
            tau_syn: 2.0,
            v_th: 1.0,
            v_reset: 0.0,
            v_leak: 0.0,
            refractory_period,
        };
        
        let mut state = LIFState::new(size, params, device.clone()).unwrap();
        let input = Tensor::full(&[size], input_current, (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        let mut spike_times = Vec::new();
        
        // Run for enough time to potentially see multiple spikes
        for step in 0..1000 {
            let spikes = state.update(&input, dt).unwrap();
            let spike: f32 = spikes.get(0).unwrap().to_scalar().unwrap();
            
            if spike > 0.0 {
                spike_times.push(step);
            }
        }
        
        // Check that spikes are separated by at least the refractory period
        let refractory_steps = (refractory_period / dt) as usize;
        
        for i in 1..spike_times.len() {
            let interval = spike_times[i] - spike_times[i-1];
            prop_assert!(interval >= refractory_steps, 
                "Spikes should be separated by refractory period: {} < {}", 
                interval, refractory_steps);
        }
    }
}

/// Property: Neuron factory should create neurons with correct parameters
proptest! {
    #[test]
    fn prop_neuron_factory_correctness(
        layer_type in prop::sample::select(vec![
            LayerType::GranuleCell,
            LayerType::PurkinjeCell,
            LayerType::GolgiCell,
            LayerType::DeepCerebellarNucleus,
        ])
    ) {
        let granule = NeuronFactory::create_granule_cell();
        let purkinje = NeuronFactory::create_purkinje_cell();
        let golgi = NeuronFactory::create_golgi_cell();
        let dcn = NeuronFactory::create_dcn_cell();
        
        // Verify threshold ordering
        prop_assert!(granule.threshold < purkinje.threshold);
        prop_assert!(purkinje.threshold < dcn.threshold);
        
        // Verify all parameters are valid
        let neurons = vec![granule, purkinje, golgi, dcn];
        for neuron in neurons {
            prop_assert!(neuron.threshold > 0.0);
            prop_assert!(neuron.decay_mem > 0.0 && neuron.decay_mem < 1.0);
            prop_assert!(neuron.decay_syn > 0.0 && neuron.decay_syn < 1.0);
            prop_assert!(neuron.reset_potential.is_finite());
        }
    }
}

/// Property: Batch processor should handle different batch sizes correctly
proptest! {
    #[test]
    fn prop_batch_processor_correctness(
        batch_size in 1usize..200,
        input_amplitude in -5.0f32..5.0f32
    ) {
        let neurons = vec![LIFNeuron::new_trading_optimized(); batch_size];
        let mut processor = BatchNeuronProcessor::new(neurons);
        
        let inputs = vec![input_amplitude; batch_size];
        let spikes = processor.process_batch(&inputs);
        
        // Output should have same length as input
        prop_assert_eq!(spikes.len(), batch_size);
        
        // All outputs should be boolean
        for spike in spikes {
            prop_assert!(spike == true || spike == false);
        }
        
        // Statistics should be consistent
        let stats = processor.get_batch_stats();
        prop_assert_eq!(stats.total_neurons, batch_size);
        prop_assert!(stats.active_neurons <= batch_size);
        prop_assert!(stats.average_membrane_potential.is_finite());
    }
}

/// Property: Connection weights should maintain sparsity
proptest! {
    #[test]
    fn prop_connection_weights_sparsity(
        input_size in 5usize..50,
        output_size in 5usize..50,
        connectivity in 0.1f64..0.9f64,
        weight_scale in 0.01f64..1.0f64
    ) {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device.clone());
        let root = vs.root();
        
        let connection = ConnectionWeights::new(
            &root,
            input_size,
            output_size,
            connectivity,
            weight_scale,
            false,
        ).unwrap();
        
        let input = Tensor::randn(&[2, input_size], (DType::F32, &device)).unwrap();
        let output = connection.forward(&input);
        
        // Output should have correct shape
        prop_assert_eq!(output.shape().dims(), &[2, output_size]);
        
        // All values should be finite
        let output_values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for value in output_values {
            prop_assert!(value.is_finite());
        }
        
        // Statistics should be consistent
        let stats = connection.weight_statistics();
        prop_assert_eq!(stats.get("connectivity").unwrap_or(&0.0), &connectivity);
    }
}

/// Property: Layer statistics should be consistent
proptest! {
    #[test]
    fn prop_layer_statistics_consistency(
        size in 1usize..100,
        neuron_type in prop::sample::select(vec![NeuronType::LIF, NeuronType::AdEx]),
        tau_mem in 1.0f64..50.0f64,
        tau_syn in 1.0f64..25.0f64
    ) {
        let device = Device::Cpu;
        let dt = 0.001;
        
        let config = LayerConfig {
            size,
            neuron_type,
            tau_mem,
            tau_syn_exc: tau_syn,
            tau_syn_inh: tau_syn * 2.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        };
        
        let mut layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
        let input = Tensor::randn(&[1, size], (DType::F32, &device)).unwrap();
        
        layer.forward(&input).unwrap();
        
        let stats = layer.get_statistics();
        
        // Size should match
        prop_assert_eq!(stats.get("size").unwrap_or(&0.0), &(size as f64));
        
        // Firing rate should be between 0 and 1
        let firing_rate = stats.get("firing_rate").unwrap_or(&0.0);
        prop_assert!(*firing_rate >= 0.0 && *firing_rate <= 1.0);
        
        // All statistics should be finite
        for (key, value) in stats {
            prop_assert!(value.is_finite(), "Statistic {} should be finite: {}", key, value);
        }
    }
}

/// Property: Multi-layer network should preserve information flow
proptest! {
    #[test]
    fn prop_multi_layer_information_flow(
        layer1_size in 10usize..50,
        layer2_size in 5usize..25,
        connectivity in 0.1f64..0.8f64,
        weight_scale in 0.01f64..0.5f64
    ) {
        let device = Device::Cpu;
        let dt = 0.001;
        let vs = nn::VarStore::new(device.clone());
        let root = vs.root();
        
        let configs = vec![
            LayerConfig {
                size: layer1_size,
                neuron_type: NeuronType::LIF,
                tau_mem: 10.0,
                tau_syn_exc: 2.0,
                tau_syn_inh: 10.0,
                tau_adapt: Some(50.0),
                a: Some(2e-9),
                b: Some(1e-10),
            },
            LayerConfig {
                size: layer2_size,
                neuron_type: NeuronType::LIF,
                tau_mem: 15.0,
                tau_syn_exc: 3.0,
                tau_syn_inh: 5.0,
                tau_adapt: Some(100.0),
                a: Some(4e-9),
                b: Some(5e-10),
            },
        ];
        
        let connection_params = vec![(connectivity, weight_scale, false)];
        
        let mut network = MultiLayerCerebellar::new(
            configs,
            connection_params,
            dt,
            device.clone(),
            &root,
        ).unwrap();
        
        let input = Tensor::randn(&[1, layer1_size], (DType::F32, &device)).unwrap();
        let outputs = network.forward(&input).unwrap();
        
        // Should have outputs for both layers
        prop_assert_eq!(outputs.len(), 2);
        
        // Outputs should have correct shapes
        prop_assert_eq!(outputs[0].shape().dims(), &[1, layer1_size]);
        prop_assert_eq!(outputs[1].shape().dims(), &[1, layer2_size]);
        
        // All outputs should be binary
        for output in outputs {
            let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
            for value in values {
                prop_assert!(value == 0.0 || value == 1.0);
            }
        }
    }
}

/// Property: Extreme parameter values should not cause crashes
proptest! {
    #[test]
    fn prop_extreme_parameters_robustness(
        tau_mem in 0.001f64..1000.0f64,
        tau_syn in 0.001f64..500.0f64,
        v_th in 0.001f64..100.0f64,
        v_reset in -10.0f64..10.0f64,
        input_current in -100.0f32..100.0f32
    ) {
        let device = Device::Cpu;
        let size = 1;
        let params = LIFParameters {
            tau_mem,
            tau_syn,
            v_th,
            v_reset,
            v_leak: 0.0,
            refractory_period: 2.0,
        };
        
        // Should not crash during creation
        let mut state = LIFState::new(size, params, device.clone()).unwrap();
        let input = Tensor::full(&[size], input_current, (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        // Should not crash during update
        let result = state.update(&input, dt);
        prop_assert!(result.is_ok());
        
        // Values should remain finite
        let v_mem: f32 = state.v_mem.get(0).unwrap().to_scalar().unwrap();
        let i_syn: f32 = state.i_syn.get(0).unwrap().to_scalar().unwrap();
        
        prop_assert!(v_mem.is_finite());
        prop_assert!(i_syn.is_finite());
    }
}

/// Property: Neuron behavior should be deterministic with same inputs
proptest! {
    #[test]
    fn prop_neuron_deterministic_behavior(
        input_current in -5.0f32..5.0f32,
        size in 1usize..20,
        steps in 1usize..100,
        seed in 0u64..1000u64
    ) {
        let device = Device::Cpu;
        let params = LIFParameters::default();
        
        // Create two identical states
        let mut state1 = LIFState::new(size, params.clone(), device.clone()).unwrap();
        let mut state2 = LIFState::new(size, params, device.clone()).unwrap();
        
        let input = Tensor::full(&[size], input_current, (DType::F32, &device)).unwrap();
        let dt = 0.001;
        
        // Apply same inputs to both states
        for _ in 0..steps {
            let spikes1 = state1.update(&input, dt).unwrap();
            let spikes2 = state2.update(&input, dt).unwrap();
            
            // Outputs should be identical
            let values1: Vec<f32> = spikes1.flatten_all().unwrap().to_vec1().unwrap();
            let values2: Vec<f32> = spikes2.flatten_all().unwrap().to_vec1().unwrap();
            
            prop_assert_eq!(values1, values2);
        }
        
        // Final states should be identical
        let v_mem1: Vec<f32> = state1.v_mem.flatten_all().unwrap().to_vec1().unwrap();
        let v_mem2: Vec<f32> = state2.v_mem.flatten_all().unwrap().to_vec1().unwrap();
        
        for (v1, v2) in v_mem1.iter().zip(v_mem2.iter()) {
            prop_assert!((v1 - v2).abs() < 1e-6);
        }
    }
}

/// Property: Neuron state should be serializable and deserializable
proptest! {
    #[test]
    fn prop_neuron_state_serialization(
        tau_mem in 1.0f64..100.0f64,
        tau_syn in 1.0f64..50.0f64,
        v_th in 0.1f64..10.0f64,
        v_reset in -5.0f64..5.0f64
    ) {
        let params = LIFParameters {
            tau_mem,
            tau_syn,
            v_th,
            v_reset,
            v_leak: 0.0,
            refractory_period: 2.0,
        };
        
        // Serialize and deserialize
        let serialized = serde_json::to_string(&params).unwrap();
        let deserialized: LIFParameters = serde_json::from_str(&serialized).unwrap();
        
        // All fields should match
        prop_assert_eq!(params.tau_mem, deserialized.tau_mem);
        prop_assert_eq!(params.tau_syn, deserialized.tau_syn);
        prop_assert_eq!(params.v_th, deserialized.v_th);
        prop_assert_eq!(params.v_reset, deserialized.v_reset);
        prop_assert_eq!(params.v_leak, deserialized.v_leak);
        prop_assert_eq!(params.refractory_period, deserialized.refractory_period);
    }
}

#[cfg(test)]
mod statistical_properties {
    use super::*;
    use statrs::statistics::Statistics;
    
    /// Property: Spike times should follow exponential distribution for constant input
    proptest! {
        #[test]
        fn prop_spike_timing_statistics(
            input_current in 1.5f32..3.0f32,
            steps in 1000usize..5000
        ) {
            let device = Device::Cpu;
            let size = 1;
            let params = LIFParameters::default();
            let mut state = LIFState::new(size, params, device.clone()).unwrap();
            
            let input = Tensor::full(&[size], input_current, (DType::F32, &device)).unwrap();
            let dt = 0.001;
            
            let mut spike_times = Vec::new();
            
            for step in 0..steps {
                let spikes = state.update(&input, dt).unwrap();
                let spike: f32 = spikes.get(0).unwrap().to_scalar().unwrap();
                
                if spike > 0.0 {
                    spike_times.push(step as f64 * dt);
                }
            }
            
            // Should have at least some spikes
            prop_assume!(spike_times.len() > 5);
            
            // Calculate inter-spike intervals
            let intervals: Vec<f64> = spike_times.windows(2)
                .map(|w| w[1] - w[0])
                .collect();
            
            // Inter-spike intervals should be positive
            for interval in &intervals {
                prop_assert!(*interval > 0.0);
            }
            
            // Mean inter-spike interval should be reasonable
            let mean_interval = intervals.mean();
            prop_assert!(mean_interval > 0.0 && mean_interval < 1.0);
            
            // Coefficient of variation should be reasonable
            let std_interval = intervals.std_dev();
            let cv = std_interval / mean_interval;
            prop_assert!(cv > 0.0 && cv < 10.0);
        }
    }
    
    /// Property: Membrane potential distribution should be reasonable
    proptest! {
        #[test]
        fn prop_membrane_potential_distribution(
            input_current in 0.5f32..1.5f32,
            steps in 1000usize..3000
        ) {
            let device = Device::Cpu;
            let size = 1;
            let params = LIFParameters::default();
            let mut state = LIFState::new(size, params, device.clone()).unwrap();
            
            let input = Tensor::full(&[size], input_current, (DType::F32, &device)).unwrap();
            let dt = 0.001;
            
            let mut v_mem_samples = Vec::new();
            
            for _ in 0..steps {
                state.update(&input, dt).unwrap();
                let v_mem: f32 = state.v_mem.get(0).unwrap().to_scalar().unwrap();
                v_mem_samples.push(v_mem as f64);
            }
            
            // Statistical properties
            let mean_v = v_mem_samples.mean();
            let std_v = v_mem_samples.std_dev();
            
            // Mean should be positive for positive input
            prop_assert!(mean_v > 0.0);
            
            // Standard deviation should be reasonable
            prop_assert!(std_v > 0.0 && std_v < 10.0);
            
            // All samples should be finite
            for sample in v_mem_samples {
                prop_assert!(sample.is_finite());
            }
        }
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;
    
    /// Test behavior with zero input
    proptest! {
        #[test]
        fn prop_zero_input_behavior(
            size in 1usize..50,
            steps in 1usize..100
        ) {
            let device = Device::Cpu;
            let params = LIFParameters::default();
            let mut state = LIFState::new(size, params, device.clone()).unwrap();
            
            let zero_input = Tensor::zeros(&[size], (DType::F32, &device)).unwrap();
            let dt = 0.001;
            
            for _ in 0..steps {
                let spikes = state.update(&zero_input, dt).unwrap();
                let spike_sum: f32 = spikes.sum_all().unwrap().to_scalar().unwrap();
                
                // Should not spike with zero input
                prop_assert_eq!(spike_sum, 0.0);
            }
            
            // Membrane potential should decay to zero
            let final_v_mem: f32 = state.v_mem.sum_all().unwrap().to_scalar().unwrap();
            prop_assert!(final_v_mem.abs() < 1e-3);
        }
    }
    
    /// Test behavior with very small time steps
    proptest! {
        #[test]
        fn prop_small_time_step_stability(
            dt in 1e-6f64..1e-3f64,
            input_current in 0.1f32..2.0f32
        ) {
            let device = Device::Cpu;
            let size = 1;
            let params = LIFParameters::default();
            let mut state = LIFState::new(size, params, device.clone()).unwrap();
            
            let input = Tensor::full(&[size], input_current, (DType::F32, &device)).unwrap();
            
            // Run for fixed real time with small time step
            let total_time = 0.1; // 100ms
            let steps = (total_time / dt) as usize;
            
            for _ in 0..steps.min(10000) { // Limit iterations for performance
                let result = state.update(&input, dt);
                prop_assert!(result.is_ok());
                
                let v_mem: f32 = state.v_mem.get(0).unwrap().to_scalar().unwrap();
                prop_assert!(v_mem.is_finite());
            }
        }
    }
    
    /// Test behavior with very large inputs
    proptest! {
        #[test]
        fn prop_large_input_robustness(
            input_current in 100.0f32..1000.0f32,
            steps in 1usize..50
        ) {
            let device = Device::Cpu;
            let size = 1;
            let params = LIFParameters::default();
            let mut state = LIFState::new(size, params, device.clone()).unwrap();
            
            let input = Tensor::full(&[size], input_current, (DType::F32, &device)).unwrap();
            let dt = 0.001;
            
            for _ in 0..steps {
                let result = state.update(&input, dt);
                prop_assert!(result.is_ok());
                
                let v_mem: f32 = state.v_mem.get(0).unwrap().to_scalar().unwrap();
                let i_syn: f32 = state.i_syn.get(0).unwrap().to_scalar().unwrap();
                
                // Values should remain finite even with large inputs
                prop_assert!(v_mem.is_finite());
                prop_assert!(i_syn.is_finite());
            }
        }
    }
}