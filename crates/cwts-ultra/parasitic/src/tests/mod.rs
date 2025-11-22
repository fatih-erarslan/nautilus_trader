//! Integration tests for the parasitic MCP server

pub mod cordyceps_tests;

use std::time::Duration;
use tokio::time::timeout;
use uuid::Uuid;

use crate::{
    ParasiticEngineInner, ParasiticConfig, ParasiticMCPServer,
    mcp_server::{MCPMessage, ParasiticEvent},
    organisms::*,
};

#[tokio::test]
async fn test_mcp_server_initialization() {
    let config = ParasiticConfig::default();
    let engine = ParasiticEngineInner::new(config.clone());
    
    // Initialize engine
    engine.initialize().await.expect("Engine initialization failed");
    
    // Verify organisms are spawned
    assert!(!engine.organisms.is_empty(), "No organisms spawned");
    
    // Check organism types
    let organism_types: Vec<String> = engine.organisms
        .iter()
        .map(|entry| entry.value().organism_type().to_string())
        .collect();
    
    assert!(organism_types.contains(&"cuckoo".to_string()));
    assert!(organism_types.contains(&"wasp".to_string()));
    assert!(organism_types.contains(&"virus".to_string()));
    assert!(organism_types.contains(&"bacteria".to_string()));
    assert!(organism_types.contains(&"Cordyceps".to_string()));
}

#[tokio::test]
async fn test_mcp_server_creation() {
    let config = ParasiticConfig::default();
    let mut mcp_server = ParasiticMCPServer::new(&config.mcp_config).await
        .expect("MCP server creation failed");
    
    let engine = ParasiticEngineInner::new(config);
    engine.initialize().await.expect("Engine initialization failed");
    
    mcp_server.set_engine(engine);
    
    // Verify server configuration
    assert_eq!(mcp_server.config.port, 3001);
    assert_eq!(mcp_server.config.bind_address, "127.0.0.1");
    assert_eq!(mcp_server.clients.len(), 0);
}

#[tokio::test] 
async fn test_pair_infection_workflow() {
    let config = ParasiticConfig::default();
    let engine = ParasiticEngineInner::new(config);
    
    engine.initialize().await.expect("Engine initialization failed");
    
    // Get first organism
    let organism_id = engine.organisms.iter().next().unwrap().key().clone();
    
    // Test pair infection
    let result = engine.infect_pair("BTCUSD".to_string(), organism_id).await;
    assert!(result.is_ok(), "Pair infection failed: {:?}", result.err());
    
    let infection = result.unwrap();
    assert_eq!(infection.pair_id, "BTCUSD");
    assert_eq!(infection.organism_id, organism_id);
    assert!(infection.vulnerability_score > 0.0);
    
    // Check infected pairs list
    let infected_pairs = engine.get_infected_pairs().await;
    assert_eq!(infected_pairs.len(), 1);
    assert_eq!(infected_pairs[0].pair_id, "BTCUSD");
}

#[tokio::test]
async fn test_mcp_protocol_messages() {
    // Test MCP message serialization/deserialization
    let message = MCPMessage {
        jsonrpc: "2.0".to_string(),
        id: Some(serde_json::json!(1)),
        method: Some("parasitic_infect".to_string()),
        params: Some(serde_json::json!({
            "pair_id": "BTCUSD",
            "organism_id": "550e8400-e29b-41d4-a716-446655440000"
        })),
        result: None,
        error: None,
    };
    
    let serialized = serde_json::to_string(&message).unwrap();
    let deserialized: MCPMessage = serde_json::from_str(&serialized).unwrap();
    
    assert_eq!(message.jsonrpc, deserialized.jsonrpc);
    assert_eq!(message.method, deserialized.method);
}

#[tokio::test]
async fn test_parasitic_event_broadcasting() {
    let event = ParasiticEvent::PairInfected {
        pair_id: "ETHUSD".to_string(),
        organism_id: Uuid::new_v4(),
        infection_strength: 0.85,
        timestamp: chrono::Utc::now(),
    };
    
    let serialized = serde_json::to_string(&event).unwrap();
    let deserialized: ParasiticEvent = serde_json::from_str(&serialized).unwrap();
    
    match (event, deserialized) {
        (ParasiticEvent::PairInfected { pair_id: p1, .. },
         ParasiticEvent::PairInfected { pair_id: p2, .. }) => {
            assert_eq!(p1, p2);
        }
        _ => panic!("Event deserialization mismatch"),
    }
}

#[tokio::test]
async fn test_organism_genetics() {
    let genetics = OrganismGenetics::random();
    
    // All genetic traits should be in valid range [0.0, 1.0]
    assert!(genetics.aggression >= 0.0 && genetics.aggression <= 1.0);
    assert!(genetics.adaptability >= 0.0 && genetics.adaptability <= 1.0);
    assert!(genetics.efficiency >= 0.0 && genetics.efficiency <= 1.0);
    assert!(genetics.resilience >= 0.0 && genetics.resilience <= 1.0);
    assert!(genetics.reaction_speed >= 0.0 && genetics.reaction_speed <= 1.0);
    assert!(genetics.risk_tolerance >= 0.0 && genetics.risk_tolerance <= 1.0);
    assert!(genetics.cooperation >= 0.0 && genetics.cooperation <= 1.0);
    assert!(genetics.stealth >= 0.0 && genetics.stealth <= 1.0);
}

#[tokio::test]
async fn test_organism_specific_behaviors() {
    // Test Cuckoo organism
    let cuckoo = CuckooOrganism::new();
    assert_eq!(cuckoo.organism_type(), "cuckoo");
    assert!(cuckoo.get_genetics().stealth > 0.7, "Cuckoo should have high stealth");
    
    // Test Wasp organism  
    let wasp = WaspOrganism::new();
    assert_eq!(wasp.organism_type(), "wasp");
    assert!(wasp.get_genetics().aggression > 0.8, "Wasp should have high aggression");
    
    // Test Cordyceps organism
    use crate::organisms::cordyceps::{CordycepsOrganism, CordycepsConfig, SIMDLevel, StealthConfig};
    let cordyceps_config = CordycepsConfig {
        max_infections: 10,
        spore_production_rate: 2.0,
        neural_control_strength: 1.5,
        quantum_enabled: false,
        simd_level: SIMDLevel::Basic,
        infection_radius: 5.0,
        min_host_fitness: 0.3,
        stealth_mode: StealthConfig {
            pattern_camouflage: true,
            behavior_mimicry: true,
            temporal_jittering: true,
            operation_fragmentation: false,
        },
    };
    let cordyceps = CordycepsOrganism::new(cordyceps_config).unwrap();
    assert_eq!(cordyceps.organism_type(), "Cordyceps");
    
    // Test resource consumption differences
    let cuckoo_resources = cuckoo.resource_consumption();
    let wasp_resources = wasp.resource_consumption();
    let cordyceps_resources = cordyceps.resource_consumption();
    
    assert!(cuckoo_resources.cpu_usage < wasp_resources.cpu_usage, 
            "Cuckoo should use less CPU than Wasp");
    assert!(cuckoo_resources.latency_overhead_ns > wasp_resources.latency_overhead_ns,
            "Cuckoo should have higher latency due to stealth");
    assert!(cordyceps_resources.latency_overhead_ns <= 100_000,
            "Cordyceps should meet sub-100μs requirement");
}

#[tokio::test]
async fn test_evolution_engine() {
    let config = ParasiticConfig::default();
    let engine = ParasiticEngineInner::new(config);
    
    engine.initialize().await.expect("Engine initialization failed");
    
    // Get evolution status
    let status = engine.get_evolution_status().await;
    assert_eq!(status.current_generation, 0);
    assert!(status.next_evolution > status.last_evolution);
}

#[tokio::test]
async fn test_performance_requirements() {
    // Test MCP server startup time
    let start = std::time::Instant::now();
    
    let config = ParasiticConfig::default();
    let mcp_server = ParasiticMCPServer::new(&config.mcp_config).await
        .expect("MCP server creation failed");
    
    let startup_time = start.elapsed();
    
    // Should start in under 10ms for ultra-low latency
    assert!(startup_time.as_millis() < 10, 
            "MCP server startup too slow: {:?}", startup_time);
    
    // Test infection calculation speed
    let cuckoo = CuckooOrganism::new();
    let start = std::time::Instant::now();
    
    for _ in 0..1000 {
        let _ = cuckoo.calculate_infection_strength(0.75);
    }
    
    let calc_time = start.elapsed();
    
    // 1000 calculations should complete in under 1ms
    assert!(calc_time.as_millis() < 1,
            "Infection strength calculation too slow: {:?}", calc_time);
}

#[tokio::test]
async fn test_latency_requirements() {
    let config = ParasiticConfig::default();
    let engine = ParasiticEngineInner::new(config);
    engine.initialize().await.expect("Engine initialization failed");
    
    let organism_id = engine.organisms.iter().next().unwrap().key().clone();
    
    // Measure pair infection latency
    let start = std::time::Instant::now();
    let result = engine.infect_pair("BTCUSD".to_string(), organism_id).await;
    let infection_latency = start.elapsed();
    
    assert!(result.is_ok());
    
    // Should complete in under 100µs for ultra-low latency trading
    assert!(infection_latency.as_micros() < 100,
            "Pair infection too slow: {:?}", infection_latency);
}

/// Integration test to verify the complete MCP workflow
#[tokio::test]
async fn test_complete_mcp_workflow() {
    let config = ParasiticConfig::default();
    
    // 1. Initialize engine
    let engine = ParasiticEngineInner::new(config.clone());
    engine.initialize().await.expect("Engine initialization failed");
    
    // 2. Create MCP server
    let mut mcp_server = ParasiticMCPServer::new(&config.mcp_config).await
        .expect("MCP server creation failed");
    mcp_server.set_engine(engine.clone());
    
    // 3. Test resources
    let organisms_resource = mcp_server.get_organisms_resource().await;
    assert!(organisms_resource.is_ok());
    
    let infected_pairs_resource = mcp_server.get_infected_pairs_resource().await;
    assert!(infected_pairs_resource.is_ok());
    
    let evolution_status_resource = mcp_server.get_evolution_status_resource().await;
    assert!(evolution_status_resource.is_ok());
    
    // 4. Test pair infection via engine
    let organism_id = engine.organisms.iter().next().unwrap().key().clone();
    let infection_result = engine.infect_pair("ETHUSD".to_string(), organism_id).await;
    assert!(infection_result.is_ok());
    
    // 5. Verify infection is reflected in resources
    let infected_pairs = engine.get_infected_pairs().await;
    assert_eq!(infected_pairs.len(), 1);
    assert_eq!(infected_pairs[0].pair_id, "ETHUSD");
    
    println!("✅ Complete MCP workflow test passed!");
    println!("   - Engine initialized with {} organisms", engine.organisms.len());
    println!("   - MCP server created successfully");
    println!("   - All resources accessible");
    println!("   - Pair infection workflow functional");
}