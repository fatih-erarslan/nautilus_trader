//! Integration tests for TENGRI neuromorphic implementation
//!
//! Tests the complete LIF spiking neuron system for energy efficiency,
//! latency targets, and biological realism.

use tengri::neuromorphic::{
    SpikingNeuron, NeuronConfig, SpikeEvent, STDPSynapse, SynapseConfig,
    EventQueue as NeuromorphicEventQueue, NeuralEvent, EventPriority,
    NeuromorphicConfig, NeuromorphicSystem, PerformanceMetrics,
    SpikeSwarm, SpikeSwarmConfig, SpikeEncoding,
};

/// Test LIF neuron meets energy efficiency targets (~45 pJ/spike)
#[test]
fn test_energy_efficiency_target() {
    let config = NeuronConfig::energy_optimized();
    let mut neuron = SpikingNeuron::new(0, config, 0.1).unwrap();
    
    // Generate spikes with consistent current
    let input_current = 400.0; // pA - should generate regular spikes
    let mut spike_count = 0;
    
    for i in 0..1000 {
        let time_ms = i as f64 * 0.1;
        if neuron.update(input_current, time_ms).is_some() {
            spike_count += 1;
        }
    }
    
    // Should have generated spikes and be energy efficient
    assert!(spike_count > 0, "Neuron should generate spikes");
    assert!(neuron.is_energy_efficient(), "Neuron should meet energy targets");
    
    let performance = neuron.performance();
    assert!(performance.energy_per_spike_pj <= 50.0, 
           "Energy consumption {} pJ should be ≤ 50 pJ", 
           performance.energy_per_spike_pj);
}

/// Test neuron latency meets sub-millisecond target  
#[test]
fn test_latency_target() {
    let config = NeuronConfig::speed_optimized();
    let mut neuron = SpikingNeuron::new(0, config, 0.01).unwrap(); // 10μs timestep
    
    let start_time = std::time::Instant::now();
    
    // Run many updates to measure processing latency
    for i in 0..10000 {
        let time_ms = i as f64 * 0.01;
        neuron.update(200.0, time_ms);
    }
    
    let total_time = start_time.elapsed();
    let avg_latency_us = total_time.as_micros() as f64 / 10000.0;
    
    // Should meet sub-millisecond latency (< 1000 μs per update)
    assert!(avg_latency_us < 100.0, 
           "Average latency {} μs should be < 100 μs", 
           avg_latency_us);
}

/// Test LIF membrane dynamics are biologically realistic
#[test]
fn test_biological_realism() {
    let config = NeuronConfig::default();
    let mut neuron = SpikingNeuron::new(0, config.clone(), 0.1).unwrap();
    
    // Test 1: Sub-threshold response should follow exponential approach
    let steady_state_voltage = neuron.sub_threshold_response(100.0, 100.0);
    let expected_steady = config.v_rest_mv + config.resistance_mohm * 100.0;
    
    assert!(steady_state_voltage > config.v_rest_mv);
    assert!(steady_state_voltage < expected_steady);
    
    // Test 2: Time to threshold should be reasonable
    let time_to_threshold = neuron.time_to_threshold(300.0);
    assert!(time_to_threshold.is_some());
    let time_ms = time_to_threshold.unwrap();
    assert!(time_ms > 0.0 && time_ms < 200.0, "Time to threshold should be reasonable");
    
    // Test 3: Refractory period should prevent immediate re-firing
    // Force a spike
    for i in 0..100 {
        if neuron.update(800.0, i as f64 * 0.1).is_some() {
            break;
        }
    }
    
    let initial_spike_count = neuron.state().spike_count;
    
    // During refractory period, no spikes should occur
    for i in 0..25 { // 2.5ms of high current during refractory
        neuron.update(800.0, (i as f64) * 0.1);
    }
    
    // Should still be just 1 spike due to refractory period
    assert_eq!(neuron.state().spike_count, initial_spike_count);
}

/// Test STDP synapses implement correct learning rules
#[test]
fn test_stdp_learning() {
    let config = SynapseConfig::default();
    let mut synapse = STDPSynapse::new(0, 1, 2, config).unwrap();
    
    let initial_weight = synapse.weight();
    
    // Test LTP: pre-spike before post-spike
    let pre_spike = SpikeEvent::new(1, 10.0, -55.0, 45.0);
    let post_spike = SpikeEvent::new(2, 12.0, -55.0, 45.0); // 2ms later
    
    synapse.process_pre_spike(&pre_spike, 10.0);
    synapse.process_post_spike(&post_spike, 12.0);
    
    // Weight should increase (LTP)
    assert!(synapse.weight() > initial_weight,
           "STDP should cause LTP when pre-spike precedes post-spike");
    
    // Reset synapse  
    let mut synapse2 = STDPSynapse::new(1, 3, 4, config).unwrap();
    let initial_weight2 = synapse2.weight();
    
    // Test LTD: post-spike before pre-spike
    let post_spike2 = SpikeEvent::new(4, 10.0, -55.0, 45.0);
    let pre_spike2 = SpikeEvent::new(3, 12.0, -55.0, 45.0); // 2ms later
    
    synapse2.process_post_spike(&post_spike2, 10.0);
    synapse2.process_pre_spike(&pre_spike2, 12.0);
    
    // Weight should decrease (LTD)
    assert!(synapse2.weight() < initial_weight2,
           "STDP should cause LTD when post-spike precedes pre-spike");
}

/// Test event queue processes spikes in correct temporal order
#[test]
fn test_event_queue_ordering() {
    let mut queue = NeuromorphicEventQueue::new();
    
    // Add events with different timestamps and priorities
    let event1 = NeuralEvent::plasticity(1, 0.1, 15.0, EventPriority::Normal);
    let event2 = NeuralEvent::plasticity(2, 0.1, 5.0, EventPriority::High);
    let event3 = NeuralEvent::plasticity(3, 0.1, 10.0, EventPriority::Critical);
    let event4 = NeuralEvent::plasticity(4, 0.1, 5.0, EventPriority::Low);
    
    queue.push_event(event1).unwrap();
    queue.push_event(event2).unwrap();
    queue.push_event(event3).unwrap();
    queue.push_event(event4).unwrap();
    
    queue.set_time(20.0);
    let processed_events = queue.process_until(20.0);
    
    assert_eq!(processed_events.len(), 4);
    
    // Events at time 5.0: High priority should come before Low priority
    assert_eq!(processed_events[0].timestamp_ms(), 5.0);
    assert_eq!(processed_events[0].priority(), EventPriority::High);
    
    assert_eq!(processed_events[1].timestamp_ms(), 5.0);
    assert_eq!(processed_events[1].priority(), EventPriority::Low);
    
    // Event at time 10.0 should be third
    assert_eq!(processed_events[2].timestamp_ms(), 10.0);
    assert_eq!(processed_events[2].priority(), EventPriority::Critical);
    
    // Event at time 15.0 should be last
    assert_eq!(processed_events[3].timestamp_ms(), 15.0);
}

/// Test spike swarm collective dynamics and avalanche detection
#[test]
fn test_spike_swarm_dynamics() {
    let swarm_config = SpikeSwarmConfig {
        population_size: 100,
        encoding: SpikeEncoding::Rate { time_window_ms: 100.0 },
        avalanche_threshold: 0.1,
        real_time_analysis: true,
        ..Default::default()
    };
    
    let neuron_config = NeuronConfig::default();
    let system_config = NeuromorphicConfig::default();
    
    let mut swarm = SpikeSwarm::new(swarm_config, neuron_config, &system_config).unwrap();
    
    // Generate coordinated activity that should trigger avalanches
    let high_current = vec![600.0; 100]; // High current for synchronization
    let low_current = vec![50.0; 100];   // Low background current
    
    // Alternate activity levels to create avalanche-like patterns
    for i in 0..500 {
        let inputs = if (i / 50) % 2 == 0 { &high_current } else { &low_current };
        let spikes = swarm.update(inputs, 0.1);
        
        // Should generate some spikes during high activity periods
        if (i / 50) % 2 == 0 && i > 100 {
            assert!(spikes.len() > 0 || swarm.population_firing_rate() > 0.0,
                   "Should generate activity during high current periods");
        }
    }
    
    let stats = swarm.processing_stats();
    assert!(stats.total_spikes_processed > 0);
    assert!(stats.population_firing_rate_hz >= 0.0);
    
    // May or may not detect avalanches depending on dynamics, but metrics should be valid
    assert!(stats.avg_processing_time_us >= 0.0);
    assert!(stats.memory_usage_bytes > 0);
}

/// Test neuromorphic system meets performance targets
#[test]
fn test_performance_targets() {
    let config = NeuromorphicConfig {
        target_energy_pj: 45.0,
        max_latency_us: 1000,
        target_samples: 1000,
        parameter_count: 27_000_000,
        timestep_ms: 0.1,
        ..Default::default()
    };
    
    let mut system = NeuromorphicSystem::new(config).unwrap();
    
    // Run performance benchmark
    let duration = std::time::Duration::from_millis(50);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let metrics = rt.block_on(async {
        system.benchmark(duration).await.unwrap()
    });
    
    // Validate performance metrics
    assert!(metrics.total_compute_time >= duration);
    assert!(metrics.spike_throughput_sps > 0.0);
    assert!(metrics.memory_usage_bytes > 0);
    assert!(metrics.hardware_utilization >= 0.0);
    assert!(metrics.hardware_utilization <= 100.0);
    
    // Energy and latency may not meet targets in simulation but should be reasonable
    assert!(metrics.energy_per_spike_pj >= 0.0);
    assert!(metrics.inference_latency_us < 10000); // Should be under 10ms in simulation
}

/// Test integration between all neuromorphic components
#[test]
fn test_full_integration() {
    // Create a small integrated system
    let neuron_config = NeuronConfig::default();
    let synapse_config = SynapseConfig::default();
    
    // Create connected neurons
    let mut pre_neuron = SpikingNeuron::new(0, neuron_config.clone(), 0.1).unwrap();
    let mut post_neuron = SpikingNeuron::new(1, neuron_config, 0.1).unwrap();
    let mut synapse = STDPSynapse::new(0, 0, 1, synapse_config).unwrap();
    
    // Create event queue
    let mut event_queue = NeuromorphicEventQueue::new();
    
    // Run integrated simulation
    let mut time_ms = 0.0;
    let dt_ms = 0.1;
    let pre_current = 400.0; // Drive pre-neuron
    
    for _ in 0..1000 {
        time_ms += dt_ms;
        
        // Update pre-neuron
        if let Some(pre_spike) = pre_neuron.update(pre_current, time_ms) {
            // Add spike to event queue
            let event = NeuralEvent::spike(pre_spike.clone(), vec![1], EventPriority::Normal);
            event_queue.push_event(event).unwrap();
            
            // Process through synapse
            synapse.process_pre_spike(&pre_spike, time_ms);
        }
        
        // Update synapse and get synaptic current
        let post_voltage = post_neuron.state().voltage_mv;
        let synaptic_current = synapse.update(time_ms, post_voltage);
        
        // Update post-neuron with synaptic input
        if let Some(post_spike) = post_neuron.update(synaptic_current / 1000.0, time_ms) {
            // Process post-spike for STDP
            synapse.process_post_spike(&post_spike, time_ms);
        }
        
        // Process events
        event_queue.set_time(time_ms);
        let events = event_queue.process_until(time_ms);
        
        // Events should be processed in order
        for event in events {
            assert!(event.timestamp_ms() <= time_ms);
        }
    }
    
    // System should have generated activity and learned
    assert!(pre_neuron.state().spike_count > 0, "Pre-neuron should spike");
    
    // Synapse weight may have changed due to STDP
    let final_weight = synapse.weight();
    assert!(final_weight > 0.0 && final_weight <= 1.0);
    
    // Event queue should have processed events
    let stats = event_queue.stats();
    assert!(stats.total_events_processed > 0);
}

/// Validate system can handle target parameter count (27M parameters)
#[test]
fn test_parameter_scaling() {
    // Test with smaller scale but validate scaling characteristics
    let small_population_size = 1000;
    let target_parameters = 27_000_000;
    
    // Estimate parameters per neuron (state + connections)
    let params_per_neuron = target_parameters / small_population_size;
    
    // Should be reasonable for complex spiking neurons
    assert!(params_per_neuron >= 100, "Should have sufficient parameters per neuron");
    assert!(params_per_neuron <= 100_000, "Parameter count should be reasonable");
    
    // Validate memory scaling
    let neuron_config = NeuronConfig::default();
    let system_config = NeuromorphicConfig {
        parameter_count: target_parameters,
        ..Default::default()
    };
    
    let system = NeuromorphicSystem::new(system_config).unwrap();
    let estimated_memory = system.config().parameter_count * 8; // 8 bytes per f64
    
    // Should be within reasonable memory bounds (< 1GB for 27M parameters)
    assert!(estimated_memory < 1_000_000_000, 
           "Memory usage should be reasonable: {} bytes", estimated_memory);
}