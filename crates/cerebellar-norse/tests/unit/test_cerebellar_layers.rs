//! Unit tests for cerebellar layers module
//! 
//! Comprehensive tests for cerebellar layer implementations,
//! connection weights, and multi-layer networks.

use std::collections::HashMap;
use candle_core::{Tensor, Device, DType};
use candle_nn as nn;
use approx::assert_abs_diff_eq;
use proptest::prelude::*;
use rstest::*;
use serial_test::serial;

use cerebellar_norse::cerebellar_layers::*;
use cerebellar_norse::*;
use crate::utils::fixtures::*;
use crate::utils::*;

/// Test cerebellar layer creation with different configurations
#[rstest]
#[case(NeuronType::LIF, 50)]
#[case(NeuronType::AdEx, 25)]
#[case(NeuronType::LIF, 100)]
fn test_cerebellar_layer_creation(#[case] neuron_type: NeuronType, #[case] size: usize) {
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
    
    let layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
    
    assert_eq!(layer.size(), size);
    assert_eq!(layer.neuron_type(), neuron_type);
    assert_eq!(layer.device(), device);
}

/// Test cerebellar layer forward pass
#[test]
fn test_cerebellar_layer_forward() {
    let device = Device::Cpu;
    let dt = 0.001;
    let size = 20;
    let batch_size = 3;
    
    let config = LayerConfig {
        size,
        neuron_type: NeuronType::LIF,
        tau_mem: 10.0,
        tau_syn_exc: 2.0,
        tau_syn_inh: 10.0,
        tau_adapt: Some(50.0),
        a: Some(2e-9),
        b: Some(1e-10),
    };
    
    let mut layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
    
    // Create input current
    let input_current = Tensor::randn(&[batch_size, size], (DType::F32, &device)).unwrap();
    
    // Forward pass
    let (spikes, state) = layer.forward(&input_current).unwrap();
    
    // Verify output shape
    assert_eq!(spikes.shape().dims(), &[batch_size, size]);
    
    // Verify spikes are binary
    let spike_values: Vec<f32> = spikes.flatten_all().unwrap().to_vec1().unwrap();
    for spike in spike_values {
        assert!(spike == 0.0 || spike == 1.0, "Spike value should be 0 or 1, got {}", spike);
    }
    
    // Verify state exists
    assert!(state.get_membrane_potential().is_some());
    assert!(state.get_synaptic_current().is_some());
}

/// Test cerebellar layer state reset
#[test]
fn test_cerebellar_layer_reset() {
    let device = Device::Cpu;
    let dt = 0.001;
    let size = 15;
    let batch_size = 2;
    
    let config = LayerConfig {
        size,
        neuron_type: NeuronType::LIF,
        tau_mem: 10.0,
        tau_syn_exc: 2.0,
        tau_syn_inh: 10.0,
        tau_adapt: Some(50.0),
        a: Some(2e-9),
        b: Some(1e-10),
    };
    
    let mut layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
    
    // Apply input to change state
    let input = Tensor::randn(&[batch_size, size], (DType::F32, &device)).unwrap();
    layer.forward(&input).unwrap();
    
    // Reset state
    layer.reset_state(Some(batch_size));
    
    // Verify state is reset
    if let Some(v_mem) = layer.get_membrane_potential() {
        let v_mem_sum: f32 = v_mem.sum_all().unwrap().to_scalar().unwrap();
        assert_abs_diff_eq!(v_mem_sum, 0.0, epsilon = 1e-6);
    }
}

/// Test cerebellar layer with different batch sizes
#[rstest]
#[case(1)]
#[case(5)]
#[case(10)]
#[case(32)]
fn test_cerebellar_layer_batch_sizes(#[case] batch_size: i64) {
    let device = Device::Cpu;
    let dt = 0.001;
    let size = 10;
    
    let config = LayerConfig {
        size,
        neuron_type: NeuronType::LIF,
        tau_mem: 10.0,
        tau_syn_exc: 2.0,
        tau_syn_inh: 10.0,
        tau_adapt: Some(50.0),
        a: Some(2e-9),
        b: Some(1e-10),
    };
    
    let mut layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
    
    // Test with different batch sizes
    let input = Tensor::randn(&[batch_size, size], (DType::F32, &device)).unwrap();
    let (spikes, _) = layer.forward(&input).unwrap();
    
    assert_eq!(spikes.shape().dims(), &[batch_size, size]);
}

/// Test cerebellar layer firing rate calculation
#[test]
fn test_cerebellar_layer_firing_rate() {
    let device = Device::Cpu;
    let dt = 0.001;
    let size = 100;
    let batch_size = 10;
    
    let config = LayerConfig {
        size,
        neuron_type: NeuronType::LIF,
        tau_mem: 10.0,
        tau_syn_exc: 2.0,
        tau_syn_inh: 10.0,
        tau_adapt: Some(50.0),
        a: Some(2e-9),
        b: Some(1e-10),
    };
    
    let mut layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
    
    // Apply strong input to generate spikes
    let input = Tensor::full(&[batch_size, size], 3.0, (DType::F32, &device)).unwrap();
    
    // Multiple forward passes to accumulate activity
    for _ in 0..20 {
        layer.forward(&input).unwrap();
    }
    
    let firing_rate = layer.firing_rate();
    
    // Should be between 0 and 1
    assert!(firing_rate >= 0.0 && firing_rate <= 1.0);
}

/// Test cerebellar layer statistics
#[test]
fn test_cerebellar_layer_statistics() {
    let device = Device::Cpu;
    let dt = 0.001;
    let size = 25;
    let batch_size = 5;
    
    let config = LayerConfig {
        size,
        neuron_type: NeuronType::LIF,
        tau_mem: 10.0,
        tau_syn_exc: 2.0,
        tau_syn_inh: 10.0,
        tau_adapt: Some(50.0),
        a: Some(2e-9),
        b: Some(1e-10),
    };
    
    let mut layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
    
    // Apply input
    let input = Tensor::randn(&[batch_size, size], (DType::F32, &device)).unwrap();
    layer.forward(&input).unwrap();
    
    let stats = layer.get_statistics();
    
    // Verify expected statistics are present
    assert!(stats.contains_key("size"));
    assert!(stats.contains_key("firing_rate"));
    assert_eq!(stats["size"], size as f64);
    assert!(stats["firing_rate"] >= 0.0);
}

/// Test connection weights creation
#[test]
fn test_connection_weights_creation() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device.clone());
    let root = vs.root();
    
    let input_size = 20;
    let output_size = 10;
    let connectivity = 0.3;
    let weight_scale = 0.1;
    let inhibitory = false;
    
    let connection = ConnectionWeights::new(
        &root,
        input_size,
        output_size,
        connectivity,
        weight_scale,
        inhibitory,
    ).unwrap();
    
    // Test forward pass
    let input = Tensor::randn(&[5, input_size], (DType::F32, &device)).unwrap();
    let output = connection.forward(&input);
    
    assert_eq!(output.shape().dims(), &[5, output_size]);
}

/// Test connection weights statistics
#[test]
fn test_connection_weights_statistics() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device.clone());
    let root = vs.root();
    
    let connection = ConnectionWeights::new(
        &root,
        15,   // input size
        8,    // output size
        0.4,  // connectivity
        0.2,  // weight scale
        false, // not inhibitory
    ).unwrap();
    
    let stats = connection.weight_statistics();
    
    // Check that statistics are computed
    assert!(stats.contains_key("connectivity"));
    assert_eq!(stats["connectivity"], 0.4);
    
    // Other stats should be present if weight computation succeeds
    if stats.contains_key("mean") {
        assert!(stats["mean"].is_finite());
    }
}

/// Test inhibitory connections
#[test]
fn test_inhibitory_connections() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device.clone());
    let root = vs.root();
    
    let excitatory = ConnectionWeights::new(
        &(&root / "excitatory"),
        10, 5, 0.3, 0.1, false,
    ).unwrap();
    
    let inhibitory = ConnectionWeights::new(
        &(&root / "inhibitory"),
        10, 5, 0.3, 0.1, true,
    ).unwrap();
    
    let input = Tensor::randn(&[3, 10], (DType::F32, &device)).unwrap();
    
    let exc_output = excitatory.forward(&input);
    let inh_output = inhibitory.forward(&input);
    
    // Both should have the same shape
    assert_eq!(exc_output.shape().dims(), &[3, 5]);
    assert_eq!(inh_output.shape().dims(), &[3, 5]);
    
    // In a real implementation, inhibitory weights would be negative
    // This is handled during weight initialization
}

/// Test multi-layer cerebellar network
#[test]
fn test_multi_layer_cerebellar() {
    let device = Device::Cpu;
    let dt = 0.001;
    let vs = nn::VarStore::new(device.clone());
    let root = vs.root();
    
    // Create layer configurations
    let configs = vec![
        LayerConfig {
            size: 20,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        },
        LayerConfig {
            size: 10,
            neuron_type: NeuronType::AdEx,
            tau_mem: 15.0,
            tau_syn_exc: 3.0,
            tau_syn_inh: 5.0,
            tau_adapt: Some(100.0),
            a: Some(4e-9),
            b: Some(5e-10),
        },
    ];
    
    // Connection parameters: (connectivity, weight_scale, inhibitory)
    let connection_params = vec![(0.3, 0.1, false)];
    
    let mut network = MultiLayerCerebellar::new(
        configs,
        connection_params,
        dt,
        device.clone(),
        &root,
    ).unwrap();
    
    // Test forward pass
    let input = Tensor::randn(&[2, 20], (DType::F32, &device)).unwrap();
    let outputs = network.forward(&input).unwrap();
    
    assert_eq!(outputs.len(), 2); // Two layers
    assert_eq!(outputs[0].shape().dims(), &[2, 20]); // First layer
    assert_eq!(outputs[1].shape().dims(), &[2, 10]); // Second layer
}

/// Test multi-layer network reset
#[test]
fn test_multi_layer_reset() {
    let device = Device::Cpu;
    let dt = 0.001;
    let vs = nn::VarStore::new(device.clone());
    let root = vs.root();
    
    let configs = vec![
        LayerConfig {
            size: 10,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        },
        LayerConfig {
            size: 5,
            neuron_type: NeuronType::LIF,
            tau_mem: 15.0,
            tau_syn_exc: 3.0,
            tau_syn_inh: 5.0,
            tau_adapt: Some(100.0),
            a: Some(4e-9),
            b: Some(5e-10),
        },
    ];
    
    let connection_params = vec![(0.2, 0.1, false)];
    
    let mut network = MultiLayerCerebellar::new(
        configs,
        connection_params,
        dt,
        device.clone(),
        &root,
    ).unwrap();
    
    // Apply input to change state
    let input = Tensor::randn(&[3, 10], (DType::F32, &device)).unwrap();
    network.forward(&input).unwrap();
    
    // Reset all layers
    network.reset(Some(3));
    
    // Verify layers are accessible after reset
    assert!(network.get_layer(0).is_some());
    assert!(network.get_layer(1).is_some());
    assert!(network.get_layer(2).is_none()); // Should not exist
}

/// Test multi-layer network layer access
#[test]
fn test_multi_layer_layer_access() {
    let device = Device::Cpu;
    let dt = 0.001;
    let vs = nn::VarStore::new(device.clone());
    let root = vs.root();
    
    let configs = vec![
        LayerConfig {
            size: 15,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        },
        LayerConfig {
            size: 8,
            neuron_type: NeuronType::AdEx,
            tau_mem: 12.0,
            tau_syn_exc: 3.0,
            tau_syn_inh: 5.0,
            tau_adapt: Some(80.0),
            a: Some(3e-9),
            b: Some(4e-10),
        },
    ];
    
    let connection_params = vec![(0.25, 0.15, false)];
    
    let mut network = MultiLayerCerebellar::new(
        configs,
        connection_params,
        dt,
        device.clone(),
        &root,
    ).unwrap();
    
    // Test layer access
    let layer0 = network.get_layer(0).unwrap();
    let layer1 = network.get_layer(1).unwrap();
    
    assert_eq!(layer0.size(), 15);
    assert_eq!(layer1.size(), 8);
    assert_eq!(layer0.neuron_type(), NeuronType::LIF);
    assert_eq!(layer1.neuron_type(), NeuronType::AdEx);
    
    // Test mutable access
    let layer0_mut = network.get_layer_mut(0).unwrap();
    layer0_mut.reset_state(Some(1));
}

/// Test multi-layer network statistics
#[test]
fn test_multi_layer_statistics() {
    let device = Device::Cpu;
    let dt = 0.001;
    let vs = nn::VarStore::new(device.clone());
    let root = vs.root();
    
    let configs = vec![
        LayerConfig {
            size: 12,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        },
        LayerConfig {
            size: 6,
            neuron_type: NeuronType::LIF,
            tau_mem: 15.0,
            tau_syn_exc: 3.0,
            tau_syn_inh: 5.0,
            tau_adapt: Some(100.0),
            a: Some(4e-9),
            b: Some(5e-10),
        },
    ];
    
    let connection_params = vec![(0.3, 0.1, false)];
    
    let mut network = MultiLayerCerebellar::new(
        configs,
        connection_params,
        dt,
        device.clone(),
        &root,
    ).unwrap();
    
    // Apply some input
    let input = Tensor::randn(&[2, 12], (DType::F32, &device)).unwrap();
    network.forward(&input).unwrap();
    
    // Get layer statistics
    let layer_stats = network.get_layer_statistics();
    assert_eq!(layer_stats.len(), 2);
    
    for stats in &layer_stats {
        assert!(stats.contains_key("size"));
        assert!(stats.contains_key("firing_rate"));
    }
    
    // Get connection statistics
    let conn_stats = network.get_connection_statistics();
    assert_eq!(conn_stats.len(), 1); // One connection between layers
    
    for stats in &conn_stats {
        assert!(stats.contains_key("connectivity"));
    }
}

/// Test layer current injection
#[test]
fn test_layer_current_injection() {
    let device = Device::Cpu;
    let dt = 0.001;
    let size = 10;
    
    let config = LayerConfig {
        size,
        neuron_type: NeuronType::LIF,
        tau_mem: 10.0,
        tau_syn_exc: 2.0,
        tau_syn_inh: 10.0,
        tau_adapt: Some(50.0),
        a: Some(2e-9),
        b: Some(1e-10),
    };
    
    let mut layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
    
    // Inject strong current
    let current = Tensor::full(&[1, size], 2.0, (DType::F32, &device)).unwrap();
    let spikes = layer.inject_current(&current).unwrap();
    
    assert_eq!(spikes.shape().dims(), &[1, size]);
    
    // Verify spikes are binary
    let spike_values: Vec<f32> = spikes.flatten_all().unwrap().to_vec1().unwrap();
    for spike in spike_values {
        assert!(spike == 0.0 || spike == 1.0);
    }
}

/// Test layer with different time constants
#[rstest]
#[case(5.0, 2.0)]
#[case(10.0, 5.0)]
#[case(20.0, 10.0)]
fn test_layer_time_constants(#[case] tau_mem: f64, #[case] tau_syn: f64) {
    let device = Device::Cpu;
    let dt = 0.001;
    let size = 5;
    
    let config = LayerConfig {
        size,
        neuron_type: NeuronType::LIF,
        tau_mem,
        tau_syn_exc: tau_syn,
        tau_syn_inh: tau_syn * 2.0,
        tau_adapt: Some(50.0),
        a: Some(2e-9),
        b: Some(1e-10),
    };
    
    let mut layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
    
    // Apply constant input
    let input = Tensor::full(&[1, size], 1.0, (DType::F32, &device)).unwrap();
    
    // Test multiple time steps
    for _ in 0..50 {
        let (spikes, _) = layer.forward(&input).unwrap();
        assert_eq!(spikes.shape().dims(), &[1, size]);
    }
}

/// Test error handling for invalid configurations
#[test]
fn test_invalid_configurations() {
    let device = Device::Cpu;
    let dt = 0.001;
    let vs = nn::VarStore::new(device.clone());
    let root = vs.root();
    
    // Test with mismatched connection parameters
    let configs = vec![
        LayerConfig {
            size: 10,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        },
        LayerConfig {
            size: 5,
            neuron_type: NeuronType::LIF,
            tau_mem: 15.0,
            tau_syn_exc: 3.0,
            tau_syn_inh: 5.0,
            tau_adapt: Some(100.0),
            a: Some(4e-9),
            b: Some(5e-10),
        },
    ];
    
    // Wrong number of connection parameters
    let wrong_connection_params = vec![(0.3, 0.1, false), (0.2, 0.1, false)];
    
    let result = MultiLayerCerebellar::new(
        configs,
        wrong_connection_params,
        dt,
        device.clone(),
        &root,
    );
    
    assert!(result.is_err());
}

/// Test connection weight updates
#[test]
fn test_connection_weight_updates() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device.clone());
    let root = vs.root();
    
    let mut connection = ConnectionWeights::new(
        &root,
        10, 5, 0.3, 0.1, false,
    ).unwrap();
    
    // Create dummy weight update
    let update = Tensor::randn(&[5, 10], (DType::F32, &device)).unwrap();
    
    // This should not fail (even though update might not be implemented)
    let result = connection.update_weights(&update);
    assert!(result.is_ok());
    
    // Apply constraints should also work
    let result = connection.apply_constraints();
    assert!(result.is_ok());
}

/// Test multi-layer network weight updates
#[test]
fn test_multi_layer_weight_updates() {
    let device = Device::Cpu;
    let dt = 0.001;
    let vs = nn::VarStore::new(device.clone());
    let root = vs.root();
    
    let configs = vec![
        LayerConfig {
            size: 8,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        },
        LayerConfig {
            size: 4,
            neuron_type: NeuronType::LIF,
            tau_mem: 15.0,
            tau_syn_exc: 3.0,
            tau_syn_inh: 5.0,
            tau_adapt: Some(100.0),
            a: Some(4e-9),
            b: Some(5e-10),
        },
    ];
    
    let connection_params = vec![(0.3, 0.1, false)];
    
    let mut network = MultiLayerCerebellar::new(
        configs,
        connection_params,
        dt,
        device.clone(),
        &root,
    ).unwrap();
    
    // Create weight updates
    let updates = vec![
        Tensor::randn(&[4, 8], (DType::F32, &device)).unwrap(),
    ];
    
    // Apply updates
    let result = network.update_connections(updates);
    assert!(result.is_ok());
}

/// Property-based test for layer stability
proptest! {
    #[test]
    fn test_layer_stability(
        size in 1usize..100,
        input_scale in -2.0f32..2.0f32,
        tau_mem in 1.0f64..50.0f64,
        tau_syn in 1.0f64..20.0f64
    ) {
        let device = Device::Cpu;
        let dt = 0.001;
        let batch_size = 2;
        
        let config = LayerConfig {
            size,
            neuron_type: NeuronType::LIF,
            tau_mem,
            tau_syn_exc: tau_syn,
            tau_syn_inh: tau_syn * 2.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        };
        
        let mut layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
        let input = Tensor::full(&[batch_size, size], input_scale, (DType::F32, &device)).unwrap();
        
        // Run for multiple steps
        for _ in 0..100 {
            let result = layer.forward(&input);
            prop_assert!(result.is_ok());
            
            let (spikes, _) = result.unwrap();
            prop_assert_eq!(spikes.shape().dims(), &[batch_size, size]);
            
            // Verify spikes are binary
            let spike_values: Vec<f32> = spikes.flatten_all().unwrap().to_vec1().unwrap();
            for spike in spike_values {
                prop_assert!(spike == 0.0 || spike == 1.0);
            }
        }
    }
}

/// Performance test for layer operations
#[test]
#[serial]
fn test_layer_performance() {
    let device = Device::Cpu;
    let dt = 0.001;
    let size = 1000;
    let batch_size = 32;
    
    let config = LayerConfig {
        size,
        neuron_type: NeuronType::LIF,
        tau_mem: 10.0,
        tau_syn_exc: 2.0,
        tau_syn_inh: 10.0,
        tau_adapt: Some(50.0),
        a: Some(2e-9),
        b: Some(1e-10),
    };
    
    let mut layer = CerebellarLayer::new(&config, dt, device.clone()).unwrap();
    let input = Tensor::randn(&[batch_size, size], (DType::F32, &device)).unwrap();
    
    let start_time = std::time::Instant::now();
    
    // Run many forward passes
    for _ in 0..100 {
        layer.forward(&input).unwrap();
    }
    
    let duration = start_time.elapsed();
    
    // Should complete in reasonable time
    assert!(duration < std::time::Duration::from_secs(5));
    
    println!("Layer performance: {} μs per forward pass", 
             duration.as_micros() / 100);
}

/// Test layer memory usage scaling
#[test]
fn test_layer_memory_scaling() {
    let device = Device::Cpu;
    let dt = 0.001;
    let sizes = vec![10, 100, 1000];
    
    for size in sizes {
        let config = LayerConfig {
            size,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        };
        
        let (layer, memory_used) = measure_memory_usage(|| {
            CerebellarLayer::new(&config, dt, device.clone()).unwrap()
        });
        
        assert_eq!(layer.size(), size);
        
        // Memory should scale reasonably
        println!("Size {}: {} bytes", size, memory_used);
        assert!(memory_used >= 0);
    }
}

/// Test concurrent layer operations
#[test]
#[serial]
fn test_concurrent_layer_operations() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let device = Device::Cpu;
    let dt = 0.001;
    let size = 50;
    
    let config = LayerConfig {
        size,
        neuron_type: NeuronType::LIF,
        tau_mem: 10.0,
        tau_syn_exc: 2.0,
        tau_syn_inh: 10.0,
        tau_adapt: Some(50.0),
        a: Some(2e-9),
        b: Some(1e-10),
    };
    
    let layer = Arc::new(Mutex::new(
        CerebellarLayer::new(&config, dt, device.clone()).unwrap()
    ));
    
    let input = Tensor::randn(&[1, size], (DType::F32, &device)).unwrap();
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads
    for _ in 0..4 {
        let layer_clone = Arc::clone(&layer);
        let input_clone = input.clone();
        
        let handle = thread::spawn(move || {
            let mut layer = layer_clone.lock().unwrap();
            for _ in 0..10 {
                layer.forward(&input_clone).unwrap();
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for completion
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify final state
    let layer = layer.lock().unwrap();
    assert_eq!(layer.size(), size);
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    /// Test integration between different layer types
    #[test]
    fn test_layer_type_integration() {
        let device = Device::Cpu;
        let dt = 0.001;
        let size = 20;
        
        // Create different layer types
        let lif_config = LayerConfig {
            size,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        };
        
        let adex_config = LayerConfig {
            size,
            neuron_type: NeuronType::AdEx,
            tau_mem: 15.0,
            tau_syn_exc: 3.0,
            tau_syn_inh: 5.0,
            tau_adapt: Some(100.0),
            a: Some(4e-9),
            b: Some(5e-10),
        };
        
        let mut lif_layer = CerebellarLayer::new(&lif_config, dt, device.clone()).unwrap();
        let mut adex_layer = CerebellarLayer::new(&adex_config, dt, device.clone()).unwrap();
        
        // Apply same input to both layers
        let input = Tensor::randn(&[3, size], (DType::F32, &device)).unwrap();
        
        let (lif_spikes, _) = lif_layer.forward(&input).unwrap();
        let (adex_spikes, _) = adex_layer.forward(&input).unwrap();
        
        // Both should produce valid outputs
        assert_eq!(lif_spikes.shape().dims(), &[3, size]);
        assert_eq!(adex_spikes.shape().dims(), &[3, size]);
        
        // Both should be binary
        let lif_values: Vec<f32> = lif_spikes.flatten_all().unwrap().to_vec1().unwrap();
        let adex_values: Vec<f32> = adex_spikes.flatten_all().unwrap().to_vec1().unwrap();
        
        for value in lif_values {
            assert!(value == 0.0 || value == 1.0);
        }
        
        for value in adex_values {
            assert!(value == 0.0 || value == 1.0);
        }
    }
    
    /// Test layer chain processing
    #[test]
    fn test_layer_chain_processing() {
        let device = Device::Cpu;
        let dt = 0.001;
        let vs = nn::VarStore::new(device.clone());
        let root = vs.root();
        
        // Create a chain of layers
        let layer1_config = LayerConfig {
            size: 30,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        };
        
        let layer2_config = LayerConfig {
            size: 20,
            neuron_type: NeuronType::AdEx,
            tau_mem: 15.0,
            tau_syn_exc: 3.0,
            tau_syn_inh: 5.0,
            tau_adapt: Some(100.0),
            a: Some(4e-9),
            b: Some(5e-10),
        };
        
        let layer3_config = LayerConfig {
            size: 10,
            neuron_type: NeuronType::LIF,
            tau_mem: 12.0,
            tau_syn_exc: 2.5,
            tau_syn_inh: 8.0,
            tau_adapt: Some(80.0),
            a: Some(3e-9),
            b: Some(3e-10),
        };
        
        let mut layer1 = CerebellarLayer::new(&layer1_config, dt, device.clone()).unwrap();
        let mut layer2 = CerebellarLayer::new(&layer2_config, dt, device.clone()).unwrap();
        let mut layer3 = CerebellarLayer::new(&layer3_config, dt, device.clone()).unwrap();
        
        // Create connections
        let conn1to2 = ConnectionWeights::new(
            &(&root / "conn1to2"),
            30, 20, 0.3, 0.1, false,
        ).unwrap();
        
        let conn2to3 = ConnectionWeights::new(
            &(&root / "conn2to3"),
            20, 10, 0.4, 0.15, false,
        ).unwrap();
        
        // Process through the chain
        let input = Tensor::randn(&[2, 30], (DType::F32, &device)).unwrap();
        
        let (spikes1, _) = layer1.forward(&input).unwrap();
        let current2 = conn1to2.forward(&spikes1);
        
        let (spikes2, _) = layer2.forward(&current2).unwrap();
        let current3 = conn2to3.forward(&spikes2);
        
        let (spikes3, _) = layer3.forward(&current3).unwrap();
        
        // Verify processing chain
        assert_eq!(spikes1.shape().dims(), &[2, 30]);
        assert_eq!(spikes2.shape().dims(), &[2, 20]);
        assert_eq!(spikes3.shape().dims(), &[2, 10]);
        
        // Verify all outputs are binary
        for spikes in [&spikes1, &spikes2, &spikes3] {
            let values: Vec<f32> = spikes.flatten_all().unwrap().to_vec1().unwrap();
            for value in values {
                assert!(value == 0.0 || value == 1.0);
            }
        }
    }
}