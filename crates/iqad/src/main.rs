//! IQAD Command-line Interface
//! 
//! Demonstrates the Immune-inspired Quantum Anomaly Detection system.

use iqad::{ImmuneQuantumAnomalyDetector, IqadConfig};
use std::collections::HashMap;
use std::env;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let scenario = args.get(1).map(String::as_str).unwrap_or("network");
    
    println!("🧬 Immune-inspired Quantum Anomaly Detector v{}", env!("CARGO_PKG_VERSION"));
    println!("Scenario: {}", scenario);
    println!();
    
    // Create configuration
    let config = IqadConfig {
        num_detectors: 100,
        quantum_dimension: 4,
        sensitivity: 0.85,
        negative_selection_threshold: 0.7,
        mutation_rate: 0.1,
        use_simd: true,
        use_gpu: false, // Set to false for CPU-only demonstration
        log_level: "INFO".to_string(),
        ..Default::default()
    };
    
    // Initialize detector
    println!("Initializing IQAD...");
    let detector = ImmuneQuantumAnomalyDetector::new(config.clone()).await?;
    
    // Generate test data based on scenario
    let (normal_data, anomaly_data) = match scenario {
        "network" => {
            println!("Detecting network traffic anomalies...");
            (
                generate_network_traffic(false),
                generate_network_traffic(true)
            )
        }
        "financial" => {
            println!("Detecting financial transaction anomalies...");
            (
                generate_financial_data(false),
                generate_financial_data(true)
            )
        }
        "sensor" => {
            println!("Detecting sensor reading anomalies...");
            (
                generate_sensor_data(false),
                generate_sensor_data(true)
            )
        }
        "quantum" => {
            println!("Detecting quantum state anomalies...");
            (
                generate_quantum_data(false),
                generate_quantum_data(true)
            )
        }
        _ => {
            println!("Unknown scenario. Using network traffic.");
            (
                generate_network_traffic(false),
                generate_network_traffic(true)
            )
        }
    };
    
    // Test normal data
    println!("\n📊 Testing Normal Data:");
    let normal_result = detector.detect_anomalies(
        normal_data.clone(),
        None,
        None
    ).await?;
    
    println!("  Anomaly Score: {:.4}", normal_result.score);
    println!("  Classification: {}", if normal_result.detected { "ANOMALY ⚠️" } else { "NORMAL ✅" });
    println!("  Confidence: {:.2}%", normal_result.confidence * 100.0);
    println!("  Execution Time: {:.2}ms", normal_result.execution_time_ms);
    
    // Test anomaly data
    println!("\n📊 Testing Anomaly Data:");
    let anomaly_result = detector.detect_anomalies(
        anomaly_data.clone(),
        None,
        None
    ).await?;
    
    println!("  Anomaly Score: {:.4}", anomaly_result.score);
    println!("  Classification: {}", if anomaly_result.detected { "ANOMALY ⚠️" } else { "NORMAL ✅" });
    println!("  Confidence: {:.2}%", anomaly_result.confidence * 100.0);
    println!("  Execution Time: {:.2}ms", anomaly_result.execution_time_ms);
    
    // Batch detection
    println!("\n📊 Batch Detection Test:");
    let mut batch_data = Vec::new();
    for i in 0..10 {
        if i % 3 == 0 {
            batch_data.push(generate_network_traffic(true)); // Anomaly
        } else {
            batch_data.push(generate_network_traffic(false)); // Normal
        }
    }
    
    let mut batch_results = Vec::new();
    for data in batch_data {
        let result = detector.detect_anomalies(data, None, None).await?;
        batch_results.push(result);
    }
    
    let anomaly_count = batch_results.iter().filter(|r| r.detected).count();
    println!("  Total samples: {}", batch_results.len());
    println!("  Anomalies detected: {} ({:.1}%)", 
        anomaly_count, 
        anomaly_count as f64 / batch_results.len() as f64 * 100.0
    );
    
    // Display detector information
    println!("\n🧬 Detector Configuration:");
    println!("  Number of Detectors: {}", config.num_detectors);
    println!("  Quantum Dimension: {} qubits", config.quantum_dimension);
    println!("  Sensitivity: {:.2}%", config.sensitivity * 100.0);
    println!("  Mutation Rate: {:.2}%", config.mutation_rate * 100.0);
    
    println!("\nAvailable scenarios: network, financial, sensor, quantum");
    
    Ok(())
}

// Generate network traffic features
fn generate_network_traffic(is_anomaly: bool) -> HashMap<String, f64> {
    let mut features = HashMap::new();
    
    if is_anomaly {
        // DDoS-like pattern
        features.insert("packet_rate".to_string(), 5000.0);
        features.insert("bytes_per_second".to_string(), 10_000_000.0);
        features.insert("unique_sources".to_string(), 1.0);
        features.insert("port_diversity".to_string(), 0.1);
        features.insert("protocol_tcp".to_string(), 0.1);
        features.insert("protocol_udp".to_string(), 0.9);
        features.insert("avg_packet_size".to_string(), 64.0);
        features.insert("connection_duration".to_string(), 0.1);
    } else {
        // Normal traffic
        features.insert("packet_rate".to_string(), 100.0);
        features.insert("bytes_per_second".to_string(), 50_000.0);
        features.insert("unique_sources".to_string(), 25.0);
        features.insert("port_diversity".to_string(), 0.7);
        features.insert("protocol_tcp".to_string(), 0.8);
        features.insert("protocol_udp".to_string(), 0.2);
        features.insert("avg_packet_size".to_string(), 512.0);
        features.insert("connection_duration".to_string(), 45.0);
    }
    
    features
}

// Generate financial transaction features
fn generate_financial_data(is_anomaly: bool) -> HashMap<String, f64> {
    let mut features = HashMap::new();
    
    if is_anomaly {
        // Fraudulent pattern
        features.insert("amount".to_string(), 9999.99);
        features.insert("time_since_last".to_string(), 0.5); // Quick succession
        features.insert("merchant_risk_score".to_string(), 0.9);
        features.insert("location_change".to_string(), 1.0); // Different location
        features.insert("unusual_time".to_string(), 1.0); // Odd hours
        features.insert("velocity_check".to_string(), 10.0); // Many transactions
    } else {
        // Normal transaction
        features.insert("amount".to_string(), 45.50);
        features.insert("time_since_last".to_string(), 48.0); // Hours
        features.insert("merchant_risk_score".to_string(), 0.1);
        features.insert("location_change".to_string(), 0.0);
        features.insert("unusual_time".to_string(), 0.0);
        features.insert("velocity_check".to_string(), 1.0);
    }
    
    features
}

// Generate sensor reading features
fn generate_sensor_data(is_anomaly: bool) -> HashMap<String, f64> {
    let mut features = HashMap::new();
    
    if is_anomaly {
        // Equipment failure pattern
        features.insert("temperature".to_string(), 95.0); // Overheating
        features.insert("vibration".to_string(), 8.5); // High vibration
        features.insert("pressure".to_string(), 180.0); // Over pressure
        features.insert("rpm".to_string(), 4500.0); // Over speed
        features.insert("noise_level".to_string(), 95.0); // Very loud
    } else {
        // Normal operation
        features.insert("temperature".to_string(), 65.0);
        features.insert("vibration".to_string(), 2.0);
        features.insert("pressure".to_string(), 100.0);
        features.insert("rpm".to_string(), 3000.0);
        features.insert("noise_level".to_string(), 70.0);
    }
    
    features
}

// Generate quantum state features
fn generate_quantum_data(is_anomaly: bool) -> HashMap<String, f64> {
    let mut features = HashMap::new();
    
    if is_anomaly {
        // Decoherence pattern
        features.insert("fidelity".to_string(), 0.6); // Low fidelity
        features.insert("entanglement".to_string(), 0.2); // Lost entanglement
        features.insert("phase_error".to_string(), 0.3); // High error
        features.insert("bit_flip_rate".to_string(), 0.15); // High flip rate
        features.insert("measurement_confidence".to_string(), 0.7);
    } else {
        // Stable quantum state
        features.insert("fidelity".to_string(), 0.98);
        features.insert("entanglement".to_string(), 0.95);
        features.insert("phase_error".to_string(), 0.02);
        features.insert("bit_flip_rate".to_string(), 0.001);
        features.insert("measurement_confidence".to_string(), 0.99);
    }
    
    features
}