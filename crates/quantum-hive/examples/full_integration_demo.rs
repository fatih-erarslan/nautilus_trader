//! Full Integration Demo - Quantum Hive with All Systems Active
//! 
//! This example demonstrates how the complete quantum-hive system operates with:
//! - QAR as the Supreme Sovereign Queen
//! - Autopoietic Hyperbolic Lattice with 1000+ nodes
//! - Neuromorphic modules empowering QAR
//! - Neural ecosystem integration (Cognition Engine + ruv-FANN)
//! - ATS-Core calibration for reliable predictions
//! - Dynamic agent deployment based on market conditions
//! - ruv-swarm ephemeral agents on lattice nodes

use quantum_hive::*;
use anyhow::Result;
use tracing::{info, warn};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .with_level(true)
        .init();

    info!("🚀 Starting Full Quantum-Hive Integration Demo");
    
    // 1. Create the Autopoietic Hive
    info!("\n📡 Phase 1: Initializing Autopoietic Hive...");
    let mut hive = AutopoieticHive::with_config(HiveConfig {
        node_count: 100, // Reduced for demo
        checkpoint_interval: Duration::from_secs(30),
        quantum_job_batch_size: 32,
        enable_gpu: true,
        neuromorphic_config: NeuromorphicHiveConfig {
            enable_ceflann_elm: true,
            enable_quantum_snn: true,
            enable_cerflann_norse: true,
            enable_cerebellar_jax: true,
            adaptive_fusion: true,
            neuromorphic_learning_rate: 0.001,
        },
    });

    // 2. Initialize Neural Ecosystem Integration
    info!("\n🧠 Phase 2: Integrating Neural Ecosystem...");
    let neural_ecosystem = NeuralEcosystemCoordinator::new().await?;
    
    // 3. Initialize Neural Calibration System
    info!("\n🎯 Phase 3: Setting up Neural Calibration...");
    let calibration_system = NeuralCalibrationSystem::new().await?;
    
    // 4. Initialize Dynamic Agent Deployment
    info!("\n🤖 Phase 4: Initializing Dynamic Agent Deployment...");
    let agent_orchestrator = DynamicAgentOrchestrator::new().await?;
    
    // 5. Demonstrate the complete flow
    info!("\n⚡ Phase 5: Running Integrated Trading Cycle...");
    
    // Simulate market data
    let market_data = MarketData {
        price: 50000.0,
        volume: 5000.0,
        volatility: 0.025, // 2.5% volatility
    };
    
    // A. Process through neuromorphic modules
    info!("\n🔬 Step A: Processing through Neuromorphic Modules...");
    let neuromorphic_signal = process_neuromorphic_pipeline(&hive, &market_data).await?;
    info!("  ✓ Neuromorphic signal generated - Confidence: {:.2}%", 
          neuromorphic_signal.confidence * 100.0);
    
    // B. Process through neural ecosystem
    info!("\n🌊 Step B: Processing through Neural Ecosystem...");
    let comprehensive_signal = neural_ecosystem.process_market_data(&market_data).await?;
    info!("  ✓ Comprehensive neural signal - Prediction: {:.4}, Consensus: {:.2}%",
          comprehensive_signal.prediction,
          comprehensive_signal.consensus_score * 100.0);
    
    // C. Calibrate all neural predictions
    info!("\n🎯 Step C: Calibrating Neural Predictions...");
    let mut calibrated_neuromorphic = neuromorphic_signal.clone();
    let calibration_result = calibration_system
        .calibrate_neuromorphic_signal(&mut calibrated_neuromorphic)
        .await?;
    info!("  ✓ Calibration complete - Original: {:.2}%, Calibrated: {:.2}%",
          neuromorphic_signal.confidence * 100.0,
          calibration_result.calibrated_confidence * 100.0);
    
    // D. Integrate calibrated signals into QAR
    info!("\n👑 Step D: Integrating into Quantum Queen's QAR...");
    hive.queen.integrate_neuromorphic_signal(calibrated_neuromorphic).await?;
    info!("  ✓ QAR updated with calibrated neuromorphic insights");
    
    // E. Deploy agents based on market conditions
    info!("\n🐝 Step E: Deploying Specialized Agents...");
    let deployment_report = agent_orchestrator
        .analyze_and_deploy(&market_data, &hive.nodes[..10], &hive.swarm_intelligence)
        .await?;
    info!("  ✓ Deployed {} agents in {}ms", 
          deployment_report.deployed_count,
          deployment_report.deployment_time_ms);
    
    for agent_type in &deployment_report.agent_types[..3] { // Show first 3
        info!("    - {:?}", agent_type);
    }
    
    // F. Get deployment recommendations
    info!("\n📊 Step F: Getting Agent Deployment Recommendations...");
    let recommendations = agent_orchestrator
        .get_deployment_recommendations(&market_data)
        .await?;
    
    for rec in &recommendations[..3] { // Show top 3
        info!("  📌 Recommend {:?} - Priority: {:.1} - Reason: {}",
              rec.agent_type, rec.priority, rec.reason);
    }
    
    // G. Demonstrate swarm intelligence with neuromorphic insights
    info!("\n🌐 Step G: Swarm Intelligence with Neuromorphic Insights...");
    demonstrate_swarm_intelligence(&mut hive).await?;
    
    // H. Show performance metrics
    info!("\n📈 Step H: Performance Metrics...");
    let (ops, avg_latency, _) = hive.performance_tracker.get_performance_stats();
    info!("  Operations: {}", ops);
    info!("  Avg Latency: {} ns", avg_latency);
    info!("  Neuromorphic Metrics:");
    info!("    - ELM Accuracy: {:.2}%", hive.performance_tracker.neuromorphic_metrics.elm_accuracy * 100.0);
    info!("    - SNN Efficiency: {:.2}%", hive.performance_tracker.neuromorphic_metrics.snn_spike_efficiency * 100.0);
    
    // Clean up ephemeral agents
    info!("\n🧹 Cleaning up ephemeral agents...");
    agent_orchestrator.cleanup_terminated_agents().await?;
    
    info!("\n✅ Full Integration Demo Complete!");
    info!("The quantum-hive is now fully operational with:");
    info!("  - QAR empowered by neuromorphic modules ✓");
    info!("  - Neural ecosystem integrated (Cognition Engine + ruv-FANN) ✓");
    info!("  - ATS-Core calibration active ✓");
    info!("  - Dynamic agent deployment ready ✓");
    info!("  - ruv-swarm ephemeral agents on hyperbolic lattice ✓");
    
    Ok(())
}

/// Process market data through neuromorphic pipeline
async fn process_neuromorphic_pipeline(
    hive: &AutopoieticHive,
    market_data: &MarketData,
) -> Result<NeuromorphicSignal> {
    // Simulate neuromorphic processing
    use std::collections::HashMap;
    
    let mut module_contributions = HashMap::new();
    
    // CEFLANN-ELM contribution
    module_contributions.insert("ceflann_elm".to_string(), ModuleContribution {
        module_name: "ceflann_elm".to_string(),
        prediction: 0.75,
        confidence: 0.85,
        processing_time_us: 50,
    });
    
    // Quantum Cerebellar SNN contribution
    module_contributions.insert("quantum_snn".to_string(), ModuleContribution {
        module_name: "quantum_snn".to_string(),
        prediction: 0.72,
        confidence: 0.88,
        processing_time_us: 45,
    });
    
    // CERFLANN Norse contribution
    module_contributions.insert("cerflann_norse".to_string(), ModuleContribution {
        module_name: "cerflann_norse".to_string(),
        prediction: 0.78,
        confidence: 0.82,
        processing_time_us: 60,
    });
    
    // CERFLANN JAX contribution
    module_contributions.insert("cerebellar_jax".to_string(), ModuleContribution {
        module_name: "cerebellar_jax".to_string(),
        prediction: 0.76,
        confidence: 0.90,
        processing_time_us: 40,
    });
    
    Ok(NeuromorphicSignal {
        prediction: 0.75,
        confidence: 0.86,
        module_contributions,
        spike_patterns: vec![
            vec![1.0, 0.0, 1.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0, 0.0, 1.0],
        ],
        temporal_coherence: 0.92,
        functional_optimization: 0.88,
    })
}

/// Demonstrate swarm intelligence capabilities
async fn demonstrate_swarm_intelligence(hive: &mut AutopoieticHive) -> Result<()> {
    // Simulate emergence pattern detection
    let pattern_count = hive.swarm_intelligence.emergence_patterns.read().len();
    info!("  Detected {} emergence patterns", pattern_count);
    
    // Show pheromone trail example
    let node1_id = 0;
    let node2_id = 1;
    let pheromone_strength = hive.swarm_intelligence
        .get_pheromone_strength(node1_id, node2_id);
    info!("  Pheromone strength between nodes {} and {}: {:.3}", 
          node1_id, node2_id, pheromone_strength);
    
    // Simulate collective decision
    let collective_signal = 0.73;
    info!("  Collective swarm signal: {:.4}", collective_signal);
    
    Ok(())
}

// Additional helper types for the demo
#[derive(Debug, Clone)]
pub struct ModuleContribution {
    pub module_name: String,
    pub prediction: f64,
    pub confidence: f64,
    pub processing_time_us: u64,
}

#[derive(Debug, Clone)]
pub struct NeuromorphicSignal {
    pub prediction: f64,
    pub confidence: f64,
    pub module_contributions: HashMap<String, ModuleContribution>,
    pub spike_patterns: Vec<Vec<f64>>,
    pub temporal_coherence: f64,
    pub functional_optimization: f64,
}