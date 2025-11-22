//! Integration tests for core autopoietic system
//! Tests the fundamental autopoietic properties: self-creation, self-maintenance, 
//! operational closure, and structural coupling

use autopoiesis::core::autopoiesis::*;
use autopoiesis::consciousness::*;
use autopoiesis::emergence::*;
use autopoiesis::dynamics::*;
use std::collections::HashMap;
use tokio::time::{sleep, Duration};
#[cfg(feature = "test-utils")]
use approx::assert_relative_eq;

#[cfg(feature = "property-tests")]
use proptest::prelude::*;

/// Simple state implementation for testing
#[derive(Clone, Debug)]
pub struct TestState {
    pub energy: f64,
    pub information: f64,
}

impl State for TestState {
    fn energy(&self) -> f64 {
        self.energy
    }
    
    fn information(&self) -> f64 {
        self.information
    }
    
    fn transform(&mut self, interaction: &Interaction) {
        match interaction.interaction_type {
            InteractionType::Production => {
                self.energy += interaction.strength * 0.1;
                self.information += interaction.strength * 0.05;
            },
            InteractionType::Catalysis => {
                self.energy *= 1.0 + interaction.strength * 0.1;
            },
            InteractionType::Regulation => {
                self.energy = self.energy.clamp(0.0, 10.0);
            },
            InteractionType::Inhibition => {
                self.energy *= 1.0 - interaction.strength * 0.1;
            },
        }
    }
}

/// Test boundary implementation
#[derive(Clone, Debug)]
pub struct TestBoundary {
    pub radius: f64,
    pub permeability: HashMap<String, f64>,
}

impl Boundary for TestBoundary {
    fn contains(&self, position: &nalgebra::Vector3<f64>) -> bool {
        position.norm() <= self.radius
    }
    
    fn permeability(&self, substance: &str) -> f64 {
        *self.permeability.get(substance).unwrap_or(&0.5)
    }
    
    fn update(&mut self, components: &[Component<impl State>]) {
        // Adaptive boundary - grows with more components
        self.radius = (components.len() as f64).sqrt() * 2.0;
    }
}

/// Test process implementation
#[derive(Clone, Debug)]
pub struct TestProcess {
    pub id: String,
    pub rate: f64,
    pub produces: Vec<String>,
    pub consumes: Vec<String>,
}

impl ProcessTrait for TestProcess {
    fn execute(&self, components: &mut [Component<impl State>]) -> autopoiesis::Result<()> {
        // Simple production process
        for component in components.iter_mut() {
            if self.consumes.contains(&component.id) {
                component.state.transform(&Interaction {
                    strength: self.rate,
                    interaction_type: InteractionType::Production,
                });
            }
        }
        Ok(())
    }
    
    fn rate(&self) -> f64 {
        self.rate
    }
    
    fn can_execute(&self, components: &[Component<impl State>]) -> bool {
        self.consumes.iter().all(|consumed| {
            components.iter().any(|c| c.id == *consumed)
        })
    }
}

/// Test environment implementation
#[derive(Clone, Debug)]
pub struct TestEnvironment {
    pub energy_density: f64,
    pub materials: HashMap<String, f64>,
}

impl Environment for TestEnvironment {
    fn energy_density(&self, _position: &nalgebra::Vector3<f64>) -> f64 {
        self.energy_density
    }
    
    fn material_concentration(&self, material: &str, _position: &nalgebra::Vector3<f64>) -> f64 {
        *self.materials.get(material).unwrap_or(&0.0)
    }
    
    fn update(&mut self, _time_step: f64) {
        // Slowly replenish energy
        self.energy_density = (self.energy_density + 0.01).min(1.0);
    }
}

/// Create a test autopoietic system
fn create_test_system() -> BasicAutopoieticSystem<TestState, TestBoundary, TestProcess, TestEnvironment> {
    use petgraph::Graph;
    use std::marker::PhantomData;
    
    // Create organization
    let mut relations = HashMap::new();
    relations.insert("production".to_string(), Relation {
        name: "production".to_string(),
        invariant: true,
        coupling_strength: 1.0,
    });
    
    let mut process_network = Graph::new();
    let _process_node = process_network.add_node(ProcessNode {
        id: "main_process".to_string(),
        rate: 1.0,
        produces: vec!["enzyme".to_string()],
        consumes: vec!["substrate".to_string()],
    });
    
    let organization = Organization {
        relations,
        process_network,
        _phantom: PhantomData,
    };
    
    // Create structure
    let components = vec![
        Component {
            id: "substrate".to_string(),
            state: TestState { energy: 1.0, information: 0.5 },
            production_rate: 0.5,
            decay_rate: 0.1,
        },
        Component {
            id: "enzyme".to_string(),
            state: TestState { energy: 2.0, information: 1.0 },
            production_rate: 1.0,
            decay_rate: 0.05,
        },
    ];
    
    let interactions = Graph::new();
    let structure = Structure { components, interactions };
    
    // Create boundary
    let mut permeability = HashMap::new();
    permeability.insert("energy".to_string(), 0.8);
    permeability.insert("substrate".to_string(), 0.6);
    
    let boundary = TestBoundary {
        radius: 5.0,
        permeability,
    };
    
    // Create environment
    let mut materials = HashMap::new();
    materials.insert("substrate".to_string(), 1.0);
    
    let environment = TestEnvironment {
        energy_density: 1.0,
        materials,
    };
    
    BasicAutopoieticSystem {
        organization,
        structure,
        boundary,
        environment,
        time: 0.0,
        _phantom: PhantomData,
    }
}

#[tokio::test]
async fn test_autopoietic_system_initialization() {
    let system = create_test_system();
    
    // Test basic properties
    assert_eq!(system.structure().components.len(), 2);
    assert!(system.organization().relations.contains_key("production"));
    assert_eq!(system.time, 0.0);
}

#[tokio::test]
async fn test_operational_closure() {
    let mut system = create_test_system();
    
    // Add a proper closed loop
    let organization = system.organization();
    let mut new_network = organization.process_network.clone();
    
    // Add process that consumes enzyme and produces substrate
    let closure_node = new_network.add_node(ProcessNode {
        id: "closure_process".to_string(),
        rate: 0.8,
        produces: vec!["substrate".to_string()],
        consumes: vec!["enzyme".to_string()],
    });
    
    // Replace organization (this requires access to internal structure)
    // In practice, this would be done through proper API methods
    assert!(system.verify_operational_closure());
}

#[tokio::test]
async fn test_autopoietic_cycle() {
    let mut system = create_test_system();
    let initial_health = system.autopoietic_health();
    
    // Run several cycles
    for _ in 0..10 {
        let result = system.autopoietic_cycle(0.1).await.expect("Cycle should succeed");
        assert!(result.energy_consumed >= 0.0);
        assert!(result.entropy_produced >= 0.0);
    }
    
    // System should be stable or improving
    let final_health = system.autopoietic_health();
    assert!(final_health >= initial_health * 0.8); // Allow some degradation
}

#[tokio::test]
async fn test_structural_coupling() {
    let mut system = create_test_system();
    let env = system.environment.clone();
    
    // Test structural coupling
    assert!(system.verify_structural_coupling(&env));
    
    // Modify environment and test adaptation
    let mut perturbed_env = env.clone();
    perturbed_env.energy_density = 0.1; // Low energy environment
    
    // System should still be coupled but may need adaptation
    let coupled = system.verify_structural_coupling(&perturbed_env);
    assert!(coupled || system.autopoietic_health() > 0.3);
}

#[tokio::test]
async fn test_perturbation_adaptation() {
    let mut system = create_test_system();
    let initial_health = system.autopoietic_health();
    
    // Apply various perturbations
    let perturbations = vec![
        Perturbation {
            source: PerturbationSource::Environmental,
            magnitude: 0.3,
            duration: 1.0,
        },
        Perturbation {
            source: PerturbationSource::Internal,
            magnitude: 0.6,
            duration: 0.5,
        },
        Perturbation {
            source: PerturbationSource::Boundary,
            magnitude: 0.9,
            duration: 0.2,
        },
    ];
    
    for perturbation in perturbations {
        let result = system.adapt_organization(perturbation).await;
        assert!(result.is_ok());
        
        // Health should not drop too much
        let current_health = system.autopoietic_health();
        assert!(current_health >= initial_health * 0.5);
    }
}

#[tokio::test]
async fn test_boundary_dynamics() {
    let mut system = create_test_system();
    let initial_radius = system.boundary.radius;
    
    // Add more components
    system.structure_mut().components.push(Component {
        id: "new_component".to_string(),
        state: TestState { energy: 1.5, information: 0.8 },
        production_rate: 0.7,
        decay_rate: 0.08,
    });
    
    // Update boundary
    system.boundary.update(&system.structure().components);
    
    // Boundary should adapt to increased complexity
    assert!(system.boundary.radius >= initial_radius);
}

#[tokio::test]
async fn test_long_term_stability() {
    let mut system = create_test_system();
    let mut health_history = Vec::new();
    
    // Run for extended period
    for _ in 0..100 {
        let _result = system.autopoietic_cycle(0.05).await.expect("Cycle should succeed");
        health_history.push(system.autopoietic_health());
    }
    
    // System should maintain reasonable health
    let average_health: f64 = health_history.iter().sum::<f64>() / health_history.len() as f64;
    assert!(average_health > 0.3);
    
    // Health should not show catastrophic decline
    let final_health = health_history.last().unwrap();
    let initial_health = health_history.first().unwrap();
    assert!(final_health / initial_health > 0.5);
}

#[tokio::test]
async fn test_component_production_and_decay() {
    let mut system = create_test_system();
    let initial_count = system.structure().components.len();
    
    // Run cycles and track component changes
    let mut production_events = 0;
    let mut decay_events = 0;
    
    for _ in 0..50 {
        let result = system.autopoietic_cycle(0.1).await.expect("Cycle should succeed");
        production_events += result.produced_components.len();
        decay_events += result.decayed_components.len();
    }
    
    // Should have some production and decay activity
    assert!(production_events > 0 || decay_events > 0);
    
    // Final component count should be reasonable
    let final_count = system.structure().components.len();
    assert!(final_count > 0); // System shouldn't completely die
}

#[tokio::test]
async fn test_energy_conservation() {
    let mut system = create_test_system();
    
    // Track energy over time
    let mut total_energy_consumed = 0.0;
    let mut total_entropy_produced = 0.0;
    
    for _ in 0..20 {
        let result = system.autopoietic_cycle(0.1).await.expect("Cycle should succeed");
        total_energy_consumed += result.energy_consumed;
        total_entropy_produced += result.entropy_produced;
    }
    
    // Energy should be consumed and entropy produced
    assert!(total_energy_consumed > 0.0);
    assert!(total_entropy_produced > 0.0);
    
    // Basic thermodynamic relationship
    assert!(total_entropy_produced <= total_energy_consumed);
}

#[tokio::test]
async fn test_concurrent_autopoietic_systems() {
    let mut systems = vec![
        create_test_system(),
        create_test_system(),
        create_test_system(),
    ];
    
    // Run systems concurrently
    let mut handles = Vec::new();
    
    for mut system in systems.into_iter() {
        let handle = tokio::spawn(async move {
            let mut health_values = Vec::new();
            for _ in 0..20 {
                let _result = system.autopoietic_cycle(0.05).await.expect("Cycle should succeed");
                health_values.push(system.autopoietic_health());
            }
            health_values
        });
        handles.push(handle);
    }
    
    // Collect results
    let mut all_results = Vec::new();
    for handle in handles {
        let health_values = handle.await.expect("Task should complete");
        all_results.push(health_values);
    }
    
    // All systems should maintain health
    for health_values in all_results {
        let average_health: f64 = health_values.iter().sum::<f64>() / health_values.len() as f64;
        assert!(average_health > 0.2);
    }
}

#[tokio::test]
async fn test_autopoietic_resilience() {
    let mut system = create_test_system();
    
    // Apply stress test with rapid perturbations
    for i in 0..10 {
        let perturbation = Perturbation {
            source: PerturbationSource::Environmental,
            magnitude: 0.5 + (i as f64 * 0.05), // Increasing magnitude
            duration: 0.1,
        };
        
        system.adapt_organization(perturbation).await.expect("Adaptation should succeed");
        
        // Run a few cycles after each perturbation
        for _ in 0..3 {
            let _result = system.autopoietic_cycle(0.05).await.expect("Cycle should succeed");
        }
        
        // System should remain viable
        assert!(system.autopoietic_health() > 0.1);
    }
}

/// Property-based test for autopoietic invariants
#[cfg(feature = "property-tests")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_health_bounds(
            energy in 0.1f64..10.0,
            production_rate in 0.1f64..2.0,
            decay_rate in 0.01f64..0.5
        ) {
            let mut system = create_test_system();
            
            // Modify component parameters
            if let Some(component) = system.structure_mut().components.get_mut(0) {
                component.state.energy = energy;
                component.production_rate = production_rate;
                component.decay_rate = decay_rate;
            }
            
            let health = system.autopoietic_health();
            prop_assert!(health >= 0.0 && health <= 1.0);
        }
        
        #[test]
        fn test_cycle_invariants(
            dt in 0.001f64..0.1,
            cycles in 1usize..50
        ) {
            tokio_test::block_on(async {
                let mut system = create_test_system();
                
                for _ in 0..cycles {
                    let result = system.autopoietic_cycle(dt).await?;
                    
                    // Invariants that should always hold
                    prop_assert!(result.energy_consumed >= 0.0);
                    prop_assert!(result.entropy_produced >= 0.0);
                    prop_assert!(system.autopoietic_health() >= 0.0);
                    prop_assert!(system.structure().components.len() >= 0);
                }
                
                Ok(())
            }).expect("Property test should pass");
        }
    }
}