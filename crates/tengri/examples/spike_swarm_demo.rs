//! # Spike Swarm Demo for TENGRI
//! 
//! Demonstrates the spike swarm neural network implementation
//! with critical dynamics, avalanche detection, and trading signal processing.

use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;

use tengri::{
    SpikeSwarm, SpikeSwarmConfig, SpikeEncoding,
    types::{TradingSignal, SignalType},
};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("debug")
        .init();
    
    println!("🧠 TENGRI Spike Swarm Neural Network Demo");
    println!("==========================================\n");
    
    // Demo configuration
    run_basic_demo()?;
    run_trading_signal_demo()?;
    run_performance_benchmark()?;
    run_critical_dynamics_demo()?;
    
    Ok(())
}

/// Basic spike swarm functionality demo
fn run_basic_demo() -> Result<()> {
    println!("🔬 Basic Spike Swarm Demo");
    println!("--------------------------");
    
    // Create spike swarm configuration
    let config = SpikeSwarmConfig {
        num_neurons: 10000, // Smaller for demo
        connectivity_prob: 0.002,
        parallel_processing: true,
        excitatory_ratio: 0.8,
        dt: 0.1,
        ..Default::default()
    };
    
    let start_time = Instant::now();
    let mut swarm = SpikeSwarm::new(config)?;
    let creation_time = start_time.elapsed();
    
    println!("✅ Created swarm in {:.2}ms", creation_time.as_secs_f64() * 1000.0);
    
    // Get initial status
    let status = swarm.get_status()?;
    println!("📊 Swarm Status:");
    println!("   - Neurons: {}", status.active_neurons);
    println!("   - Synapses: {}", status.total_synapses);
    println!("   - Populations: {}", status.population_activities.len());
    println!("   - Memory usage: {:.1}MB", status.performance_stats.memory_usage);
    
    // Run basic simulation
    println!("\n🚀 Running simulation (100 steps)...");
    let mut total_spikes = 0;
    
    for step in 0..100 {
        let spikes = swarm.step()?;
        total_spikes += spikes.len();
        
        if step % 20 == 0 {
            println!("   Step {}: {} spikes", step, spikes.len());
        }
    }
    
    let final_status = swarm.get_status()?;
    println!("✅ Simulation complete:");
    println!("   - Total spikes: {}", total_spikes);
    println!("   - Avalanches: {}", final_status.avalanche_count);
    println!("   - Global synchrony: {:.3}", final_status.sync_metrics.global_synchrony);
    println!("   - Simulation speed: {:.1} steps/sec", final_status.performance_stats.simulation_speed);
    
    Ok(())
}

/// Trading signal processing demo
fn run_trading_signal_demo() -> Result<()> {
    println!("\n💹 Trading Signal Processing Demo");
    println!("----------------------------------");
    
    let config = SpikeSwarmConfig {
        num_neurons: 20000,
        connectivity_prob: 0.001,
        ..Default::default()
    };
    
    let mut swarm = SpikeSwarm::new(config)?;
    
    // Create test trading signals
    let signals = vec![
        TradingSignal {
            id: uuid::Uuid::new_v4(),
            symbol: "BTCUSDT".to_string(),
            signal_type: SignalType::Buy,
            strength: 0.85,
            confidence: 0.92,
            timestamp: chrono::Utc::now(),
            source: "DEMO".to_string(),
            metadata: {
                let mut map = HashMap::new();
                map.insert("rsi".to_string(), 28.0);
                map.insert("macd".to_string(), 0.7);
                map.insert("volume".to_string(), 1.5);
                map
            },
            expires_at: None,
        },
        TradingSignal {
            id: uuid::Uuid::new_v4(),
            symbol: "ETHUSDT".to_string(),
            signal_type: SignalType::Sell,
            strength: 0.72,
            confidence: 0.81,
            timestamp: chrono::Utc::now(),
            source: "DEMO".to_string(),
            metadata: {
                let mut map = HashMap::new();
                map.insert("rsi".to_string(), 78.0);
                map.insert("macd".to_string(), -0.4);
                map.insert("volume".to_string(), 0.9);
                map
            },
            expires_at: None,
        },
    ];
    
    // Encode signals using different encoding schemes
    println!("📡 Encoding trading signals...");
    for (i, signal) in signals.iter().enumerate() {
        let encoding = match i % 3 {
            0 => SpikeEncoding::Rate,
            1 => SpikeEncoding::Temporal,
            _ => SpikeEncoding::Phase,
        };
        
        println!("   Signal {}: {} {:?} -> {:?} encoding", 
                 i + 1, signal.symbol, signal.signal_type, encoding);
        
        swarm.encode_signal(signal, encoding)?;
    }
    
    // Run simulation to process signals
    println!("\n🧠 Processing signals through spike swarm...");
    let mut step_spikes = Vec::new();
    
    for step in 0..500 {
        let spikes = swarm.step()?;
        step_spikes.push(spikes.len());
        
        if step % 100 == 0 && !spikes.is_empty() {
            println!("   Step {}: {} spikes", step, spikes.len());
        }
    }
    
    // Attempt to decode from populations
    println!("\n🔍 Decoding population activity...");
    let population_types = [
        ("rate_pop_0", SpikeEncoding::Rate),
        ("temporal_pop_0", SpikeEncoding::Temporal),
        ("phase_pop_0", SpikeEncoding::Phase),
    ];
    
    for (pop_id, encoding) in population_types {
        if let Some(decoded) = swarm.decode_population(pop_id, encoding)? {
            println!("   {}: {:?} signal (strength={:.2}, confidence={:.2})",
                     pop_id, decoded.signal_type, decoded.strength, decoded.confidence);
        } else {
            println!("   {}: No activity detected", pop_id);
        }
    }
    
    let final_status = swarm.get_status()?;
    println!("\n📈 Processing Results:");
    println!("   - Total steps with spikes: {}", step_spikes.iter().filter(|&&x| x > 0).count());
    println!("   - Max spikes in step: {}", step_spikes.iter().max().unwrap_or(&0));
    println!("   - Population activities:");
    
    for (pop_name, activity) in &final_status.population_activities {
        if *activity > 0.001 {
            println!("     {}: {:.3}", pop_name, activity);
        }
    }
    
    Ok(())
}

/// Performance benchmark demo
fn run_performance_benchmark() -> Result<()> {
    println!("\n⚡ Performance Benchmark");
    println!("------------------------");
    
    let sizes = [1000, 5000, 10000];
    
    for &size in &sizes {
        println!("\n🧪 Testing {} neurons:", size);
        
        let config = SpikeSwarmConfig {
            num_neurons: size,
            connectivity_prob: 0.002,
            parallel_processing: true,
            ..Default::default()
        };
        
        // Creation benchmark
        let start = Instant::now();
        let mut swarm = SpikeSwarm::new(config)?;
        let creation_time = start.elapsed();
        
        // Inject some activity
        for i in 0..(size / 100).max(1) {
            // This is a conceptual injection - actual implementation would differ
            // as neurons are private. In real usage, signals would be encoded.
        }
        
        // Simulation benchmark
        let start = Instant::now();
        let mut total_spikes = 0;
        
        for _ in 0..100 {
            let spikes = swarm.step()?;
            total_spikes += spikes.len();
        }
        
        let sim_time = start.elapsed();
        let steps_per_sec = 100.0 / sim_time.as_secs_f64();
        
        let status = swarm.get_status()?;
        
        println!("   📊 Results:");
        println!("      Creation: {:.1}ms", creation_time.as_secs_f64() * 1000.0);
        println!("      Simulation: {:.1} steps/sec", steps_per_sec);
        println!("      Memory: {:.1}MB", status.performance_stats.memory_usage);
        println!("      Parallel efficiency: {:.1}%", 
                 status.performance_stats.parallel_efficiency * 100.0);
        println!("      Spikes generated: {}", total_spikes);
    }
    
    Ok(())
}

/// Critical dynamics demonstration
fn run_critical_dynamics_demo() -> Result<()> {
    println!("\n🌪️  Critical Dynamics Demo");
    println!("---------------------------");
    
    let config = SpikeSwarmConfig {
        num_neurons: 30000,
        connectivity_prob: 0.001, // Critical connectivity
        recording: tengri::spike_swarm::RecordingConfig {
            record_avalanches: true,
            record_spikes: true,
            max_duration: 30.0,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let mut swarm = SpikeSwarm::new(config)?;
    
    // Create a strong signal to trigger activity
    let trigger_signal = TradingSignal {
        id: uuid::Uuid::new_v4(),
        symbol: "TRIGGER".to_string(),
        signal_type: SignalType::StrongBuy,
        strength: 0.95,
        confidence: 0.98,
        timestamp: chrono::Utc::now(),
        source: "CRITICAL_DEMO".to_string(),
        metadata: HashMap::new(),
        expires_at: None,
    };
    
    // Encode trigger signal
    swarm.encode_signal(&trigger_signal, SpikeEncoding::Rate)?;
    swarm.encode_signal(&trigger_signal, SpikeEncoding::Temporal)?;
    
    println!("🎯 Triggering critical dynamics...");
    
    // Run extended simulation to observe critical behavior
    let results = swarm.run(5000.0)?; // 5 seconds
    
    println!("🔬 Critical Dynamics Analysis:");
    println!("   - Total spikes: {}", results.total_spikes);
    println!("   - Avalanches detected: {}", results.avalanche_count);
    println!("   - Simulation time: {:.2}s", results.simulation_time.as_secs_f64());
    
    let dynamics = &results.final_status.critical_dynamics;
    println!("   - Power-law exponent: {:.2} (target: 1.5)", dynamics.power_law_exponent);
    println!("   - Criticality index: {:.2}", dynamics.criticality_index);
    println!("   - Branching parameter: {:.2}", dynamics.branching_parameter);
    
    // Analyze avalanche sizes
    if !dynamics.avalanche_sizes.is_empty() {
        println!("   - Avalanche size distribution:");
        let mut sizes: Vec<_> = dynamics.avalanche_sizes.iter().collect();
        sizes.sort_by_key(|(size, _)| *size);
        
        for (size, count) in sizes.iter().take(10) {
            println!("     Size {}: {} occurrences", size, count);
        }
    }
    
    // Synchrony analysis
    let sync = &results.final_status.sync_metrics;
    println!("   - Global synchrony: {:.4}", sync.global_synchrony);
    println!("   - Local synchrony patterns:");
    
    for (pop, sync_val) in &sync.local_synchrony {
        if *sync_val > 0.01 {
            println!("     {}: {:.3}", pop, sync_val);
        }
    }
    
    // Performance summary
    let perf = &results.final_status.performance_stats;
    println!("\n⚡ Performance Summary:");
    println!("   - Average simulation speed: {:.1} steps/sec", perf.simulation_speed);
    println!("   - Memory usage: {:.1}MB", perf.memory_usage);
    println!("   - Spike processing rate: {:.0} spikes/sec", perf.spike_rate);
    println!("   - Parallel efficiency: {:.1}%", perf.parallel_efficiency * 100.0);
    
    Ok(())
}