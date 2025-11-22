//! Integration tests for swarm dynamics and chaos engineering
//! Tests emergent behaviors, system resilience, and complex dynamics

use autopoiesis::dynamics::*;
use autopoiesis::emergence::*;
use autopoiesis::core::*;
use std::collections::{HashMap, VecDeque};
use tokio::time::{sleep, Duration};
#[cfg(feature = "test-utils")]
use approx::assert_relative_eq;

#[cfg(feature = "property-tests")]
use proptest::prelude::*;
use rand::Rng;

/// Chaos engineering test utilities
pub mod chaos_utils {
    use super::*;
    use rand::Rng;
    
    pub enum ChaosEvent {
        NetworkPartition,
        ComponentFailure(String),
        ResourceExhaustion,
        LatencySpike(f64),
        DataCorruption,
        ExternalShock(f64),
    }
    
    pub struct ChaosManager {
        pub events: Vec<ChaosEvent>,
        pub probability: f64,
        pub severity: f64,
    }
    
    impl ChaosManager {
        pub fn new(probability: f64, severity: f64) -> Self {
            Self {
                events: Vec::new(),
                probability,
                severity,
            }
        }
        
        pub fn inject_chaos(&mut self) -> Option<ChaosEvent> {
            let mut rng = rand::thread_rng();
            
            if rng.gen::<f64>() < self.probability {
                let event = match rng.gen_range(0..6) {
                    0 => ChaosEvent::NetworkPartition,
                    1 => ChaosEvent::ComponentFailure(format!("component_{}", rng.gen::<u32>())),
                    2 => ChaosEvent::ResourceExhaustion,
                    3 => ChaosEvent::LatencySpike(rng.gen_range(0.1..2.0)),
                    4 => ChaosEvent::DataCorruption,
                    5 => ChaosEvent::ExternalShock(rng.gen_range(0.1..self.severity)),
                    _ => unreachable!(),
                };
                
                self.events.push(event);
                self.events.last().cloned()
            } else {
                None
            }
        }
        
        pub fn apply_chaos_to_system(&self, event: &ChaosEvent, system: &mut TestSwarmSystem) {
            match event {
                ChaosEvent::NetworkPartition => {
                    system.network_available = false;
                },
                ChaosEvent::ComponentFailure(component) => {
                    system.failed_components.insert(component.clone());
                },
                ChaosEvent::ResourceExhaustion => {
                    system.resource_multiplier *= 0.1; // Severe resource reduction
                },
                ChaosEvent::LatencySpike(multiplier) => {
                    system.latency_multiplier = *multiplier;
                },
                ChaosEvent::DataCorruption => {
                    system.data_corruption_rate = 0.1;
                },
                ChaosEvent::ExternalShock(magnitude) => {
                    system.external_perturbation = *magnitude;
                },
            }
        }
        
        pub fn recover_from_chaos(&self, system: &mut TestSwarmSystem) {
            // Gradual recovery simulation
            system.network_available = true;
            system.failed_components.clear();
            system.resource_multiplier = (system.resource_multiplier * 1.1).min(1.0);
            system.latency_multiplier = (system.latency_multiplier * 0.9).max(1.0);
            system.data_corruption_rate *= 0.5;
            system.external_perturbation *= 0.8;
        }
    }
}

/// Test swarm system for chaos engineering
#[derive(Clone, Debug)]
pub struct TestSwarmSystem {
    pub agents: Vec<TestAgent>,
    pub network_available: bool,
    pub failed_components: std::collections::HashSet<String>,
    pub resource_multiplier: f64,
    pub latency_multiplier: f64,
    pub data_corruption_rate: f64,
    pub external_perturbation: f64,
    pub system_health: f64,
    pub emergence_detector: EmergenceDetector,
    pub time: f64,
}

#[derive(Clone, Debug)]
pub struct TestAgent {
    pub id: String,
    pub position: (f64, f64, f64),
    pub velocity: (f64, f64, f64),
    pub state: AgentState,
    pub connections: Vec<String>,
    pub performance: f64,
    pub resilience: f64,
}

#[derive(Clone, Debug)]
pub enum AgentState {
    Active,
    Degraded,
    Failed,
    Recovering,
}

impl TestSwarmSystem {
    pub fn new(num_agents: usize) -> Self {
        let mut agents = Vec::new();
        let mut rng = rand::thread_rng();
        
        for i in 0..num_agents {
            agents.push(TestAgent {
                id: format!("agent_{}", i),
                position: (
                    rng.gen_range(-10.0..10.0),
                    rng.gen_range(-10.0..10.0),
                    rng.gen_range(-10.0..10.0),
                ),
                velocity: (0.0, 0.0, 0.0),
                state: AgentState::Active,
                connections: Vec::new(),
                performance: 1.0,
                resilience: rng.gen_range(0.5..1.0),
            });
        }
        
        // Create some connections
        for i in 0..num_agents {
            let num_connections = rng.gen_range(1..5.min(num_agents));
            for _ in 0..num_connections {
                let target = rng.gen_range(0..num_agents);
                if target != i {
                    agents[i].connections.push(format!("agent_{}", target));
                }
            }
        }
        
        Self {
            agents,
            network_available: true,
            failed_components: std::collections::HashSet::new(),
            resource_multiplier: 1.0,
            latency_multiplier: 1.0,
            data_corruption_rate: 0.0,
            external_perturbation: 0.0,
            system_health: 1.0,
            emergence_detector: EmergenceDetector::new(DetectionParameters::default()),
            time: 0.0,
        }
    }
    
    pub fn update(&mut self, dt: f64) {
        self.time += dt;
        
        // Update agents
        for agent in &mut self.agents {
            self.update_agent(agent, dt);
        }
        
        // Update system health
        self.calculate_system_health();
        
        // Apply external perturbation
        if self.external_perturbation > 0.0 {
            self.apply_external_perturbation();
        }
    }
    
    fn update_agent(&mut self, agent: &mut TestAgent, dt: f64) {
        // Apply chaos effects
        let mut performance_modifier = 1.0;
        
        if !self.network_available {
            performance_modifier *= 0.3; // Severe degradation without network
        }
        
        if self.failed_components.contains(&agent.id) {
            agent.state = AgentState::Failed;
            performance_modifier = 0.0;
        }
        
        performance_modifier *= self.resource_multiplier;
        performance_modifier /= self.latency_multiplier;
        
        // Data corruption affects performance
        if self.data_corruption_rate > 0.0 {
            let mut rng = rand::thread_rng();
            if rng.gen::<f64>() < self.data_corruption_rate {
                performance_modifier *= 0.5;
            }
        }
        
        // Update agent state based on performance
        agent.performance = (agent.performance * 0.9 + performance_modifier * 0.1)
            .max(0.0).min(1.0);
        
        // State transitions
        match agent.state {
            AgentState::Active => {
                if agent.performance < 0.3 {
                    agent.state = AgentState::Degraded;
                }
            },
            AgentState::Degraded => {
                if agent.performance < 0.1 {
                    agent.state = AgentState::Failed;
                } else if agent.performance > 0.7 {
                    agent.state = AgentState::Active;
                }
            },
            AgentState::Failed => {
                if agent.performance > 0.2 && agent.resilience > 0.7 {
                    agent.state = AgentState::Recovering;
                }
            },
            AgentState::Recovering => {
                if agent.performance > 0.6 {
                    agent.state = AgentState::Active;
                } else if agent.performance < 0.1 {
                    agent.state = AgentState::Failed;
                }
            },
        }
        
        // Simple flocking behavior
        self.update_agent_movement(agent, dt);
    }
    
    fn update_agent_movement(&mut self, agent: &mut TestAgent, dt: f64) {
        let mut separation = (0.0, 0.0, 0.0);
        let mut alignment = (0.0, 0.0, 0.0);
        let mut cohesion = (0.0, 0.0, 0.0);
        let mut neighbors = 0;
        
        // Find neighbors and calculate forces
        for other in &self.agents {
            if other.id == agent.id {
                continue;
            }
            
            let dx = other.position.0 - agent.position.0;
            let dy = other.position.1 - agent.position.1;
            let dz = other.position.2 - agent.position.2;
            let distance = (dx*dx + dy*dy + dz*dz).sqrt();
            
            if distance < 5.0 { // Neighbor radius
                neighbors += 1;
                
                // Separation (avoid crowding)
                if distance > 0.0 {
                    separation.0 -= dx / distance;
                    separation.1 -= dy / distance;
                    separation.2 -= dz / distance;
                }
                
                // Alignment (match velocity)
                alignment.0 += other.velocity.0;
                alignment.1 += other.velocity.1;
                alignment.2 += other.velocity.2;
                
                // Cohesion (move toward center)
                cohesion.0 += dx;
                cohesion.1 += dy;
                cohesion.2 += dz;
            }
        }
        
        if neighbors > 0 {
            // Average the forces
            alignment.0 /= neighbors as f64;
            alignment.1 /= neighbors as f64;
            alignment.2 /= neighbors as f64;
            
            cohesion.0 /= neighbors as f64;
            cohesion.1 /= neighbors as f64;
            cohesion.2 /= neighbors as f64;
        }
        
        // Apply forces to velocity
        let force_strength = 0.1 * agent.performance; // Performance affects responsiveness
        agent.velocity.0 += (separation.0 + alignment.0 + cohesion.0) * force_strength * dt;
        agent.velocity.1 += (separation.1 + alignment.1 + cohesion.1) * force_strength * dt;
        agent.velocity.2 += (separation.2 + alignment.2 + cohesion.2) * force_strength * dt;
        
        // Apply velocity damping
        agent.velocity.0 *= 0.95;
        agent.velocity.1 *= 0.95;
        agent.velocity.2 *= 0.95;
        
        // Update position
        agent.position.0 += agent.velocity.0 * dt;
        agent.position.1 += agent.velocity.1 * dt;
        agent.position.2 += agent.velocity.2 * dt;
        
        // Boundary conditions
        let boundary = 20.0;
        if agent.position.0.abs() > boundary {
            agent.position.0 = agent.position.0.signum() * boundary;
            agent.velocity.0 *= -0.5;
        }
        if agent.position.1.abs() > boundary {
            agent.position.1 = agent.position.1.signum() * boundary;
            agent.velocity.1 *= -0.5;
        }
        if agent.position.2.abs() > boundary {
            agent.position.2 = agent.position.2.signum() * boundary;
            agent.velocity.2 *= -0.5;
        }
    }
    
    fn calculate_system_health(&mut self) {
        let active_agents = self.agents.iter()
            .filter(|a| matches!(a.state, AgentState::Active))
            .count();
        
        let total_performance: f64 = self.agents.iter()
            .map(|a| a.performance)
            .sum();
        
        let avg_performance = if self.agents.is_empty() {
            0.0
        } else {
            total_performance / self.agents.len() as f64
        };
        
        let connectivity = self.calculate_connectivity();
        
        self.system_health = (
            (active_agents as f64 / self.agents.len() as f64) * 0.4 +
            avg_performance * 0.4 +
            connectivity * 0.2
        ).max(0.0).min(1.0);
    }
    
    fn calculate_connectivity(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        
        let total_connections: usize = self.agents.iter()
            .map(|a| a.connections.len())
            .sum();
        
        let max_possible_connections = self.agents.len() * (self.agents.len() - 1);
        
        if max_possible_connections == 0 {
            0.0
        } else {
            total_connections as f64 / max_possible_connections as f64
        }
    }
    
    fn apply_external_perturbation(&mut self) {
        let mut rng = rand::thread_rng();
        
        for agent in &mut self.agents {
            if rng.gen::<f64>() < self.external_perturbation {
                // Random perturbation to position and velocity
                agent.position.0 += rng.gen_range(-1.0..1.0);
                agent.position.1 += rng.gen_range(-1.0..1.0);
                agent.position.2 += rng.gen_range(-1.0..1.0);
                
                agent.velocity.0 += rng.gen_range(-0.5..0.5);
                agent.velocity.1 += rng.gen_range(-0.5..0.5);
                agent.velocity.2 += rng.gen_range(-0.5..0.5);
                
                agent.performance *= rng.gen_range(0.8..1.0);
            }
        }
    }
    
    pub fn get_emergence_metrics(&self) -> EmergenceMetrics {
        let center = self.calculate_center_of_mass();
        let spread = self.calculate_spatial_spread();
        let velocity_coherence = self.calculate_velocity_coherence();
        let state_diversity = self.calculate_state_diversity();
        
        EmergenceMetrics {
            center_of_mass: center,
            spatial_spread: spread,
            velocity_coherence,
            state_diversity,
            system_health: self.system_health,
            connectivity: self.calculate_connectivity(),
        }
    }
    
    fn calculate_center_of_mass(&self) -> (f64, f64, f64) {
        if self.agents.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        
        let sum_x: f64 = self.agents.iter().map(|a| a.position.0).sum();
        let sum_y: f64 = self.agents.iter().map(|a| a.position.1).sum();
        let sum_z: f64 = self.agents.iter().map(|a| a.position.2).sum();
        
        let n = self.agents.len() as f64;
        (sum_x / n, sum_y / n, sum_z / n)
    }
    
    fn calculate_spatial_spread(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        
        let center = self.calculate_center_of_mass();
        
        let variance: f64 = self.agents.iter()
            .map(|a| {
                let dx = a.position.0 - center.0;
                let dy = a.position.1 - center.1;
                let dz = a.position.2 - center.2;
                dx*dx + dy*dy + dz*dz
            })
            .sum::<f64>() / self.agents.len() as f64;
        
        variance.sqrt()
    }
    
    fn calculate_velocity_coherence(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        
        let avg_velocity = {
            let sum_vx: f64 = self.agents.iter().map(|a| a.velocity.0).sum();
            let sum_vy: f64 = self.agents.iter().map(|a| a.velocity.1).sum();
            let sum_vz: f64 = self.agents.iter().map(|a| a.velocity.2).sum();
            let n = self.agents.len() as f64;
            (sum_vx / n, sum_vy / n, sum_vz / n)
        };
        
        let variance: f64 = self.agents.iter()
            .map(|a| {
                let dvx = a.velocity.0 - avg_velocity.0;
                let dvy = a.velocity.1 - avg_velocity.1;
                let dvz = a.velocity.2 - avg_velocity.2;
                dvx*dvx + dvy*dvy + dvz*dvz
            })
            .sum::<f64>() / self.agents.len() as f64;
        
        // Coherence is inverse of variance (normalized)
        1.0 / (1.0 + variance)
    }
    
    fn calculate_state_diversity(&self) -> f64 {
        let mut state_counts = HashMap::new();
        
        for agent in &self.agents {
            let state_key = match agent.state {
                AgentState::Active => "active",
                AgentState::Degraded => "degraded",
                AgentState::Failed => "failed",
                AgentState::Recovering => "recovering",
            };
            *state_counts.entry(state_key).or_insert(0) += 1;
        }
        
        if self.agents.is_empty() {
            return 0.0;
        }
        
        // Shannon entropy of state distribution
        let total = self.agents.len() as f64;
        let entropy: f64 = state_counts.values()
            .map(|&count| {
                let p = count as f64 / total;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum();
        
        // Normalize by maximum possible entropy
        let max_entropy = (state_counts.len() as f64).ln();
        if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 }
    }
}

#[derive(Clone, Debug)]
pub struct EmergenceMetrics {
    pub center_of_mass: (f64, f64, f64),
    pub spatial_spread: f64,
    pub velocity_coherence: f64,
    pub state_diversity: f64,
    pub system_health: f64,
    pub connectivity: f64,
}

#[tokio::test]
async fn test_swarm_initialization() {
    let system = TestSwarmSystem::new(20);
    
    assert_eq!(system.agents.len(), 20);
    assert!(system.network_available);
    assert!(system.failed_components.is_empty());
    assert_relative_eq!(system.resource_multiplier, 1.0, epsilon = 1e-10);
    assert_relative_eq!(system.system_health, 1.0, epsilon = 0.1);
    
    // All agents should start active
    let active_count = system.agents.iter()
        .filter(|a| matches!(a.state, AgentState::Active))
        .count();
    assert_eq!(active_count, 20);
}

#[tokio::test]
async fn test_swarm_basic_dynamics() {
    let mut system = TestSwarmSystem::new(10);
    let dt = 0.1;
    
    // Record initial positions
    let initial_positions: Vec<_> = system.agents.iter()
        .map(|a| a.position)
        .collect();
    
    // Update system multiple times
    for _ in 0..50 {
        system.update(dt);
    }
    
    // Positions should have changed (movement occurred)
    let final_positions: Vec<_> = system.agents.iter()
        .map(|a| a.position)
        .collect();
    
    let movement_detected = initial_positions.iter()
        .zip(final_positions.iter())
        .any(|(initial, final)| {
            let dx = initial.0 - final.0;
            let dy = initial.1 - final.1;
            let dz = initial.2 - final.2;
            (dx*dx + dy*dy + dz*dz).sqrt() > 0.1
        });
    
    assert!(movement_detected);
    assert!(system.time > 0.0);
}

#[tokio::test]
async fn test_chaos_network_partition() {
    let mut system = TestSwarmSystem::new(15);
    let mut chaos_manager = chaos_utils::ChaosManager::new(1.0, 0.5); // Always inject chaos
    
    // Record initial health
    let initial_health = system.system_health;
    
    // Inject network partition
    if let Some(event) = chaos_manager.inject_chaos() {
        chaos_manager.apply_chaos_to_system(&event, &mut system);
    }
    
    // Update system under chaos
    for _ in 0..20 {
        system.update(0.1);
    }
    
    // System health should degrade under network partition
    assert!(system.system_health < initial_health);
    assert!(!system.network_available);
    
    // Allow recovery
    chaos_manager.recover_from_chaos(&mut system);
    
    // Update system during recovery
    for _ in 0..50 {
        system.update(0.1);
    }
    
    // Health should improve during recovery
    assert!(system.system_health > 0.2); // Should recover somewhat
}

#[tokio::test]
async fn test_chaos_component_failures() {
    let mut system = TestSwarmSystem::new(20);
    let mut chaos_manager = chaos_utils::ChaosManager::new(0.8, 0.7);
    
    let initial_active = system.agents.iter()
        .filter(|a| matches!(a.state, AgentState::Active))
        .count();
    
    // Inject multiple chaos events
    for _ in 0..10 {
        if let Some(event) = chaos_manager.inject_chaos() {
            chaos_manager.apply_chaos_to_system(&event, &mut system);
        }
        
        // Update system
        for _ in 0..5 {
            system.update(0.05);
        }
    }
    
    let final_active = system.agents.iter()
        .filter(|a| matches!(a.state, AgentState::Active))
        .count();
    
    // Some agents should be affected by chaos
    assert!(final_active < initial_active || system.system_health < 0.8);
    
    // System should not completely collapse
    assert!(system.system_health > 0.1);
}

#[tokio::test]
async fn test_chaos_resource_exhaustion() {
    let mut system = TestSwarmSystem::new(12);
    
    // Simulate severe resource exhaustion
    system.resource_multiplier = 0.1;
    
    let initial_performance: f64 = system.agents.iter()
        .map(|a| a.performance)
        .sum::<f64>() / system.agents.len() as f64;
    
    // Update under resource stress
    for _ in 0..30 {
        system.update(0.1);
    }
    
    let final_performance: f64 = system.agents.iter()
        .map(|a| a.performance)
        .sum::<f64>() / system.agents.len() as f64;
    
    // Performance should degrade
    assert!(final_performance < initial_performance);
    
    // But system should maintain some functionality
    let failed_agents = system.agents.iter()
        .filter(|a| matches!(a.state, AgentState::Failed))
        .count();
    
    assert!(failed_agents < system.agents.len()); // Not all agents should fail
}

#[tokio::test]
async fn test_emergent_flocking_behavior() {
    let mut system = TestSwarmSystem::new(25);
    
    // Scatter agents initially
    let mut rng = rand::thread_rng();
    for agent in &mut system.agents {
        agent.position = (
            rng.gen_range(-15.0..15.0),
            rng.gen_range(-15.0..15.0),
            rng.gen_range(-15.0..15.0),
        );
    }
    
    let initial_spread = system.calculate_spatial_spread();
    
    // Let system evolve
    for _ in 0..100 {
        system.update(0.1);
    }
    
    let final_spread = system.calculate_spatial_spread();
    let velocity_coherence = system.calculate_velocity_coherence();
    
    // Flocking should increase coherence and potentially reduce spread
    assert!(velocity_coherence > 0.1); // Some level of velocity alignment
    
    // System should maintain reasonable organization
    let metrics = system.get_emergence_metrics();
    assert!(metrics.connectivity > 0.0);
    assert!(metrics.system_health > 0.5);
}

#[tokio::test]
async fn test_emergence_under_perturbation() {
    let mut system = TestSwarmSystem::new(18);
    
    // Apply external perturbation
    system.external_perturbation = 0.3;
    
    let mut emergence_metrics = Vec::new();
    
    // Collect emergence metrics over time
    for _ in 0..80 {
        system.update(0.1);
        emergence_metrics.push(system.get_emergence_metrics());
    }
    
    // Analyze emergence patterns
    let avg_health: f64 = emergence_metrics.iter()
        .map(|m| m.system_health)
        .sum::<f64>() / emergence_metrics.len() as f64;
    
    let avg_coherence: f64 = emergence_metrics.iter()
        .map(|m| m.velocity_coherence)
        .sum::<f64>() / emergence_metrics.len() as f64;
    
    // System should maintain reasonable performance under perturbation
    assert!(avg_health > 0.3);
    assert!(avg_coherence > 0.05);
    
    // Check for emergence of organized behavior
    let final_metrics = emergence_metrics.last().unwrap();
    assert!(final_metrics.connectivity > 0.0);
}

#[tokio::test]
async fn test_system_resilience_recovery() {
    let mut system = TestSwarmSystem::new(16);
    let mut chaos_manager = chaos_utils::ChaosManager::new(0.6, 0.8);
    
    // Apply sustained chaos
    for _ in 0..20 {
        if let Some(event) = chaos_manager.inject_chaos() {
            chaos_manager.apply_chaos_to_system(&event, &mut system);
        }
        system.update(0.1);
    }
    
    let stressed_health = system.system_health;
    
    // Begin recovery phase
    chaos_manager.recover_from_chaos(&mut system);
    
    // Allow recovery time
    for _ in 0..50 {
        system.update(0.1);
    }
    
    let recovered_health = system.system_health;
    
    // System should show recovery
    assert!(recovered_health >= stressed_health);
    
    // Check agent state recovery
    let recovering_agents = system.agents.iter()
        .filter(|a| matches!(a.state, AgentState::Recovering | AgentState::Active))
        .count();
    
    assert!(recovering_agents > 0);
}

#[tokio::test]
async fn test_cascade_failure_resistance() {
    let mut system = TestSwarmSystem::new(30);
    
    // Deliberately fail a subset of highly connected agents
    let mut failed_count = 0;
    for agent in &mut system.agents {
        if agent.connections.len() > 3 && failed_count < 5 {
            system.failed_components.insert(agent.id.clone());
            failed_count += 1;
        }
    }
    
    let initial_failed = failed_count;
    
    // Update system to see if failure cascades
    for _ in 0..40 {
        system.update(0.1);
    }
    
    let final_failed = system.agents.iter()
        .filter(|a| matches!(a.state, AgentState::Failed))
        .count();
    
    // Cascade should be limited by agent resilience
    assert!(final_failed <= system.agents.len() / 2); // No more than half should fail
    
    // System should maintain some functionality
    assert!(system.system_health > 0.2);
}

#[tokio::test]
async fn test_adaptive_topology_formation() {
    let mut system = TestSwarmSystem::new(20);
    
    // Start with sparse connections
    for agent in &mut system.agents {
        agent.connections.clear();
    }
    
    let initial_connectivity = system.calculate_connectivity();
    
    // Let system evolve and form connections organically
    // In a real implementation, agents would form connections based on proximity and performance
    for i in 0..100 {
        system.update(0.1);
        
        // Simulate adaptive connection formation every 10 steps
        if i % 10 == 0 {
            let mut rng = rand::thread_rng();
            for agent_idx in 0..system.agents.len() {
                let agent = &system.agents[agent_idx];
                
                // Find nearby agents
                for other_idx in 0..system.agents.len() {
                    if agent_idx == other_idx {
                        continue;
                    }
                    
                    let other = &system.agents[other_idx];
                    let dx = agent.position.0 - other.position.0;
                    let dy = agent.position.1 - other.position.1;
                    let dz = agent.position.2 - other.position.2;
                    let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                    
                    // Form connection if close and both performing well
                    if distance < 3.0 && 
                       agent.performance > 0.7 && 
                       other.performance > 0.7 &&
                       !agent.connections.contains(&other.id) &&
                       agent.connections.len() < 5 {
                        
                        if rng.gen::<f64>() < 0.3 { // 30% chance to connect
                            system.agents[agent_idx].connections.push(other.id.clone());
                        }
                    }
                }
            }
        }
    }
    
    let final_connectivity = system.calculate_connectivity();
    
    // Connectivity should increase through adaptive formation
    assert!(final_connectivity >= initial_connectivity);
    
    // System should show improved organization
    let metrics = system.get_emergence_metrics();
    assert!(metrics.system_health > 0.6);
}

/// Property-based tests for swarm dynamics
#[cfg(feature = "property-tests")]
mod swarm_property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_system_health_bounds(
            num_agents in 1usize..50,
            chaos_probability in 0.0f64..1.0,
            steps in 1usize..100
        ) {
            let mut system = TestSwarmSystem::new(num_agents);
            let mut chaos_manager = chaos_utils::ChaosManager::new(chaos_probability, 0.5);
            
            for _ in 0..steps {
                if let Some(event) = chaos_manager.inject_chaos() {
                    chaos_manager.apply_chaos_to_system(&event, &mut system);
                }
                system.update(0.1);
                
                // System health should always be bounded
                prop_assert!(system.system_health >= 0.0);
                prop_assert!(system.system_health <= 1.0);
                
                // Agent performance should be bounded
                for agent in &system.agents {
                    prop_assert!(agent.performance >= 0.0);
                    prop_assert!(agent.performance <= 1.0);
                }
            }
        }
        
        #[test]
        fn test_emergence_metrics_validity(
            positions in prop::collection::vec(
                ((-20.0f64..20.0), (-20.0f64..20.0), (-20.0f64..20.0)),
                1..30
            )
        ) {
            let mut system = TestSwarmSystem::new(positions.len());
            
            // Set positions
            for (i, &pos) in positions.iter().enumerate() {
                if i < system.agents.len() {
                    system.agents[i].position = pos;
                }
            }
            
            let metrics = system.get_emergence_metrics();
            
            // All metrics should be valid
            prop_assert!(metrics.spatial_spread >= 0.0);
            prop_assert!(metrics.velocity_coherence >= 0.0);
            prop_assert!(metrics.velocity_coherence <= 1.0);
            prop_assert!(metrics.state_diversity >= 0.0);
            prop_assert!(metrics.state_diversity <= 1.0);
            prop_assert!(metrics.system_health >= 0.0);
            prop_assert!(metrics.system_health <= 1.0);
            prop_assert!(metrics.connectivity >= 0.0);
            prop_assert!(metrics.connectivity <= 1.0);
        }
    }
}

/// Performance benchmarks for swarm operations
#[tokio::test]
async fn benchmark_swarm_update() {
    let mut system = TestSwarmSystem::new(100); // Large swarm
    
    let start = std::time::Instant::now();
    
    // Benchmark 1000 updates
    for _ in 0..1000 {
        system.update(0.01);
    }
    
    let duration = start.elapsed();
    println!("Swarm update benchmark: {:?} for 1000 updates of 100 agents", duration);
    
    // Should complete efficiently
    assert!(duration.as_millis() < 5000); // Less than 5 seconds
    
    // System should remain stable
    assert!(system.system_health > 0.3);
}

#[tokio::test]
async fn benchmark_chaos_injection() {
    let mut system = TestSwarmSystem::new(50);
    let mut chaos_manager = chaos_utils::ChaosManager::new(0.1, 0.5);
    
    let start = std::time::Instant::now();
    
    // Benchmark chaos injection with updates
    for _ in 0..500 {
        if let Some(event) = chaos_manager.inject_chaos() {
            chaos_manager.apply_chaos_to_system(&event, &mut system);
        }
        system.update(0.02);
    }
    
    let duration = start.elapsed();
    println!("Chaos engineering benchmark: {:?} for 500 iterations", duration);
    
    // Should handle chaos efficiently
    assert!(duration.as_millis() < 3000); // Less than 3 seconds
    
    // System should survive chaos testing
    assert!(system.system_health > 0.1);
}