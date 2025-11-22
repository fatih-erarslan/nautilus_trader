//! STDP Synapse Validation Example
//! 
//! Demonstrates that the STDP implementation matches biological data
//! and validates Hebbian learning principles.

use tengri::neural_networks::{STDPSynapse, SpikingNeuron, STDPValidation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 STDP Synapse Validation Demo");
    println!("================================");
    
    // Create STDP synapse with biological parameters
    let synapse = STDPSynapse::new(0, 1, 0.5);
    
    println!("\n📊 STDP Parameters:");
    println!("  A+ = {:.6}", synapse.a_plus);
    println!("  A- = {:.6}", synapse.a_minus);
    println!("  τ+ = {:.1} ms", synapse.tau_plus);
    println!("  τ- = {:.1} ms", synapse.tau_minus);
    println!("  Weight bounds: [{:.1}, {:.1}]", synapse.w_min, synapse.w_max);
    
    // Generate STDP curve
    let delta_t_range: Vec<f64> = (-100..=100).step_by(5).map(|x| x as f64).collect();
    let stdp_curve = synapse.get_stdp_curve(&delta_t_range);
    
    println!("\n📈 STDP Curve Data:");
    println!("  Δt (ms)  | Δw");
    println!("  ---------|--------");
    for (dt, dw) in stdp_curve.iter().take(10) {
        println!("  {:8.1} | {:8.6}", dt, dw);
    }
    println!("  ...      | ...");
    for (dt, dw) in stdp_curve.iter().rev().take(10).rev() {
        println!("  {:8.1} | {:8.6}", dt, dw);
    }
    
    // Validate STDP curve
    let validation = synapse.validate_stdp()?;
    println!("\n✅ STDP Validation Results:");
    println!("  Has LTP: {}", validation.has_ltp);
    println!("  Has LTD: {}", validation.has_ltd);
    println!("  Max potentiation: {:.6}", validation.max_potentiation);
    println!("  Max depression: {:.6}", validation.max_depression);
    println!("  LTP/LTD ratio: {:.3}", validation.ltp_ltd_ratio);
    println!("  Valid curve: {}", validation.is_valid);
    
    // Demonstrate Hebbian learning
    println!("\n🔗 Hebbian Learning Demonstration:");
    demonstrate_hebbian_learning()?;
    
    // Demonstrate weight evolution
    println!("\n📊 Weight Evolution Under Training:");
    demonstrate_weight_evolution()?;
    
    // Performance benchmark
    println!("\n⚡ Performance Benchmark:");
    benchmark_stdp_performance()?;
    
    println!("\n🎉 STDP Validation Complete!");
    Ok(())
}

fn demonstrate_hebbian_learning() -> Result<(), Box<dyn std::error::Error>> {
    let mut correlated_synapse = STDPSynapse::new(0, 1, 0.5);
    let mut uncorrelated_synapse = STDPSynapse::new(2, 3, 0.5);
    let mut anticorrelated_synapse = STDPSynapse::new(4, 5, 0.5);
    
    let dt = 0.1;
    let initial_weight = 0.5;
    
    // Simulate 100 learning trials
    for trial in 0..100 {
        let time = trial as f64 * 50.0; // 50ms between trials
        
        // Correlated: Pre always followed by post (2ms delay)
        correlated_synapse.update_stdp(true, false, dt, time)?;
        correlated_synapse.update_stdp(false, true, dt, time + 2.0)?;
        
        // Uncorrelated: Random pre and post spikes
        let random_pre = (trial * 17) % 3 == 0; // ~33% probability
        let random_post = (trial * 23) % 3 == 0; // ~33% probability
        uncorrelated_synapse.update_stdp(random_pre, false, dt, time)?;
        uncorrelated_synapse.update_stdp(false, random_post, dt, time + 2.0)?;
        
        // Anti-correlated: Post always followed by pre (2ms delay)
        anticorrelated_synapse.update_stdp(false, true, dt, time)?;
        anticorrelated_synapse.update_stdp(true, false, dt, time + 2.0)?;
        
        // Allow traces to decay
        for decay_step in 0..100 {
            let decay_time = time + 10.0 + decay_step as f64 * dt;
            correlated_synapse.update_stdp(false, false, dt, decay_time)?;
            uncorrelated_synapse.update_stdp(false, false, dt, decay_time)?;
            anticorrelated_synapse.update_stdp(false, false, dt, decay_time)?;
        }
    }
    
    let corr_stats = correlated_synapse.get_statistics();
    let uncorr_stats = uncorrelated_synapse.get_statistics();
    let anticorr_stats = anticorrelated_synapse.get_statistics();
    
    println!("  Correlated activity:");
    println!("    Final weight: {:.6} (Δ: {:+.6})", 
             correlated_synapse.weight, 
             correlated_synapse.weight - initial_weight);
    println!("    Potentiation ratio: {:.6}", corr_stats.potentiation_ratio);
    
    println!("  Uncorrelated activity:");
    println!("    Final weight: {:.6} (Δ: {:+.6})", 
             uncorrelated_synapse.weight, 
             uncorrelated_synapse.weight - initial_weight);
    println!("    Potentiation ratio: {:.6}", uncorr_stats.potentiation_ratio);
    
    println!("  Anti-correlated activity:");
    println!("    Final weight: {:.6} (Δ: {:+.6})", 
             anticorrelated_synapse.weight, 
             anticorrelated_synapse.weight - initial_weight);
    println!("    Depression ratio: {:.6}", anticorr_stats.depression_ratio);
    
    // Validate Hebbian principle
    assert!(correlated_synapse.weight > initial_weight, "Correlated activity should strengthen synapse");
    assert!(anticorrelated_synapse.weight < initial_weight, "Anti-correlated activity should weaken synapse");
    
    println!("  ✅ Hebbian learning validated: 'Cells that fire together, wire together'");
    
    Ok(())
}

fn demonstrate_weight_evolution() -> Result<(), Box<dyn std::error::Error>> {
    let mut synapse = STDPSynapse::new(0, 1, 0.5);
    let dt = 0.1;
    
    let checkpoints = vec![0, 10, 25, 50, 100, 200, 500];
    let mut weights = Vec::new();
    let mut trial_count = 0;
    
    println!("  Trial | Weight  | Δw      | LTP/LTD");
    println!("  ------|---------|---------|--------");
    
    for &checkpoint in &checkpoints {
        // Run trials until checkpoint
        while trial_count < checkpoint {
            let time = trial_count as f64 * 20.0;
            
            // Apply LTP-inducing pattern (pre before post)
            synapse.update_stdp(true, false, dt, time)?;
            synapse.update_stdp(false, true, dt, time + 1.5)?; // 1.5ms delay
            
            // Allow trace decay
            for _ in 0..50 {
                synapse.update_stdp(false, false, dt, time + 5.0)?;
            }
            
            trial_count += 1;
        }
        
        let stats = synapse.get_statistics();
        let delta_w = if weights.is_empty() { 
            0.0 
        } else { 
            synapse.weight - weights.last().unwrap() 
        };
        
        println!("  {:5} | {:.5} | {:+.5} | {:.3}/{:.3}",
                 checkpoint, synapse.weight, delta_w,
                 stats.potentiation_ratio, stats.depression_ratio);
        
        weights.push(synapse.weight);
    }
    
    // Check for weight saturation
    let final_weight = *weights.last().unwrap();
    let penultimate_weight = weights[weights.len() - 2];
    let weight_change = (final_weight - penultimate_weight).abs();
    
    if weight_change < 0.01 {
        println!("  ✅ Weight has saturated (Δw < 0.01)");
    } else {
        println!("  ⚠️  Weight still evolving (Δw = {:.4})", weight_change);
    }
    
    Ok(())
}

fn benchmark_stdp_performance() -> Result<(), Box<dyn std::error::Error>> {
    let mut synapse = STDPSynapse::new(0, 1, 0.5);
    let dt = 0.01; // 0.01ms timestep (100kHz)
    let num_updates = 1_000_000; // 1M updates
    
    println!("  Running {} updates at {}ms timestep...", num_updates, dt);
    
    let start = std::time::Instant::now();
    
    for i in 0..num_updates {
        let time = i as f64 * dt;
        let pre_spike = i % 1000 == 0; // Spike every 10ms
        let post_spike = i % 1100 == 0; // Slightly offset
        
        synapse.update_stdp(pre_spike, post_spike, dt, time)?;
    }
    
    let elapsed = start.elapsed();
    let updates_per_sec = num_updates as f64 / elapsed.as_secs_f64();
    let microseconds_per_update = elapsed.as_micros() as f64 / num_updates as f64;
    
    println!("  Performance: {:.0} updates/sec", updates_per_sec);
    println!("  Average time per update: {:.3} μs", microseconds_per_update);
    
    let stats = synapse.get_statistics();
    println!("  Final statistics:");
    println!("    Weight: {:.6}", synapse.weight);
    println!("    Spikes processed: {}", stats.spike_count);
    println!("    Total potentiation: {:.6}", stats.total_potentiation);
    println!("    Total depression: {:.6}", stats.total_depression);
    
    if updates_per_sec > 100_000.0 {
        println!("  ✅ Performance excellent (>100k updates/sec)");
    } else if updates_per_sec > 10_000.0 {
        println!("  ⚠️  Performance acceptable (>10k updates/sec)");
    } else {
        println!("  ❌ Performance needs improvement (<10k updates/sec)");
    }
    
    Ok(())
}