//! Core autopoietic system trait based on Maturana & Varela's theory
//! This trait defines the fundamental properties of self-creating, self-maintaining systems

use async_trait::async_trait;
use std::collections::HashMap;
use std::marker::PhantomData;
use nalgebra as na;
use petgraph::graph::{Graph, NodeIndex};
use serde::{Deserialize, Serialize};

use crate::Result;

/// Represents the organizational structure of an autopoietic system
#[derive(Clone, Debug)]
pub struct Organization<S> {
    /// The invariant relationships that define the system's identity
    pub relations: HashMap<String, Relation>,
    /// The process network that maintains the organization
    pub process_network: Graph<ProcessNode, f64>,
    _phantom: PhantomData<S>,
}

/// Represents the concrete realization of the organization
#[derive(Clone, Debug)]
pub struct Structure<S> {
    /// Current components of the system
    pub components: Vec<Component<S>>,
    /// Actual interactions between components
    pub interactions: Graph<usize, Interaction>,
}

/// A relation that defines part of the organizational identity
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Relation {
    pub name: String,
    pub invariant: bool,
    pub coupling_strength: f64,
}

/// A process that produces components
#[derive(Clone, Debug)]
pub struct ProcessNode {
    pub id: String,
    pub rate: f64,
    pub produces: Vec<String>,
    pub consumes: Vec<String>,
}

/// A component of the autopoietic system
#[derive(Clone, Debug)]
pub struct Component<S> {
    pub id: String,
    pub state: S,
    pub production_rate: f64,
    pub decay_rate: f64,
}

/// An interaction between components
#[derive(Clone, Debug)]
pub struct Interaction {
    pub strength: f64,
    pub interaction_type: InteractionType,
}

#[derive(Clone, Debug)]
pub enum InteractionType {
    Production,
    Regulation,
    Catalysis,
    Inhibition,
}

/// The boundary that separates the system from its environment
pub trait Boundary: Send + Sync {
    /// Check if a component is inside the boundary
    fn contains(&self, position: &na::Vector3<f64>) -> bool;
    
    /// Get the permeability for a given substance
    fn permeability(&self, substance: &str) -> f64;
    
    /// Update boundary based on internal dynamics  
    fn update(&mut self, components: &[Component<Box<dyn State>>]);
}

/// The state of an autopoietic component
pub trait State: Clone + Send + Sync {
    /// Get the energy content of this state
    fn energy(&self) -> f64;
    
    /// Get the information content
    fn information(&self) -> f64;
    
    /// Transform the state based on interactions
    fn transform(&mut self, interaction: &Interaction);
}

/// A process that transforms components
pub trait ProcessTrait: Send + Sync {
    /// Execute the process given current components
    fn execute(&self, components: &mut [Component<Box<dyn State>>]) -> Result<()>;
    
    /// Get the rate of this process
    fn rate(&self) -> f64;
    
    /// Check if process conditions are met
    fn can_execute(&self, components: &[Component<Box<dyn State>>]) -> bool;
}

/// The environment that provides energy and materials
pub trait Environment: Send + Sync {
    /// Get available energy at a position
    fn energy_density(&self, position: &na::Vector3<f64>) -> f64;
    
    /// Get material concentration
    fn material_concentration(&self, material: &str, position: &na::Vector3<f64>) -> f64;
    
    /// Apply environmental changes
    fn update(&mut self, time_step: f64);
}

/// Core autopoietic system trait
#[async_trait]
pub trait AutopoieticSystem: Send + Sync {
    type State: State;
    type Boundary: Boundary;
    type Process: ProcessTrait;
    type Environment: Environment;
    
    /// Get the current organization (invariant structure)
    fn organization(&self) -> &Organization<Self::State>;
    
    /// Get the current structure (variable realization)
    fn structure(&self) -> &Structure<Self::State>;
    
    /// Get mutable access to structure
    fn structure_mut(&mut self) -> &mut Structure<Self::State>;
    
    /// Perform one autopoietic cycle
    async fn autopoietic_cycle(&mut self, dt: f64) -> Result<AutopoieticResult>;
    
    /// Verify operational closure
    fn verify_operational_closure(&self) -> bool {
        // Check that all processes produce components that enable other processes
        let org = self.organization();
        let mut produced = std::collections::HashSet::new();
        let mut consumed = std::collections::HashSet::new();
        
        for node in org.process_network.node_indices() {
            if let Some(process) = org.process_network.node_weight(node) {
                for product in &process.produces {
                    produced.insert(product.clone());
                }
                for consumable in &process.consumes {
                    consumed.insert(consumable.clone());
                }
            }
        }
        
        // Operational closure means everything consumed is also produced
        consumed.is_subset(&produced)
    }
    
    /// Verify structural coupling with environment
    fn verify_structural_coupling(&self, env: &Self::Environment) -> bool;
    
    /// Calculate autopoietic health (0.0 to 1.0)
    fn autopoietic_health(&self) -> f64 {
        let closure = if self.verify_operational_closure() { 1.0 } else { 0.0 };
        let structure = self.structure();
        let component_health = structure.components.iter()
            .map(|c| (c.production_rate / (c.production_rate + c.decay_rate)).min(1.0))
            .sum::<f64>() / structure.components.len() as f64;
        
        (closure * 0.5 + component_health * 0.5).min(1.0).max(0.0)
    }
    
    /// Adapt organization based on perturbations
    async fn adapt_organization(&mut self, perturbation: Perturbation) -> Result<()>;
}

/// Result of an autopoietic cycle
#[derive(Clone, Debug)]
pub struct AutopoieticResult {
    /// Components produced in this cycle
    pub produced_components: Vec<String>,
    /// Components that decayed
    pub decayed_components: Vec<String>,
    /// Energy consumed
    pub energy_consumed: f64,
    /// Entropy produced
    pub entropy_produced: f64,
    /// Organizational changes
    pub organizational_changes: Vec<OrganizationalChange>,
}

/// A perturbation to the system
#[derive(Clone, Debug)]
pub struct Perturbation {
    pub source: PerturbationSource,
    pub magnitude: f64,
    pub duration: f64,
}

#[derive(Clone, Debug)]
pub enum PerturbationSource {
    Environmental,
    Internal,
    Boundary,
}

#[derive(Clone, Debug)]
pub enum OrganizationalChange {
    ProcessAdded(String),
    ProcessRemoved(String),
    RelationModified(String, f64),
    BoundaryAdjusted,
}

/// Implementation of basic autopoietic dynamics
pub struct BasicAutopoieticSystem<S, B, P, E>
where
    S: State,
    B: Boundary,
    P: ProcessTrait,
    E: Environment,
{
    organization: Organization<S>,
    structure: Structure<S>,
    boundary: B,
    environment: E,
    time: f64,
    _phantom: PhantomData<P>,
}

#[async_trait]
impl<S, B, P, E> AutopoieticSystem for BasicAutopoieticSystem<S, B, P, E>
where
    S: State + Send + Sync + 'static,
    B: Boundary + Send + Sync + 'static,
    P: ProcessTrait + Send + Sync + 'static,
    E: Environment + Send + Sync + 'static,
{
    type State = S;
    type Boundary = B;
    type Process = P;
    type Environment = E;
    
    fn organization(&self) -> &Organization<Self::State> {
        &self.organization
    }
    
    fn structure(&self) -> &Structure<Self::State> {
        &self.structure
    }
    
    fn structure_mut(&mut self) -> &mut Structure<Self::State> {
        &mut self.structure
    }
    
    async fn autopoietic_cycle(&mut self, dt: f64) -> Result<AutopoieticResult> {
        let mut result = AutopoieticResult {
            produced_components: Vec::new(),
            decayed_components: Vec::new(),
            energy_consumed: 0.0,
            entropy_produced: 0.0,
            organizational_changes: Vec::new(),
        };
        
        // Execute all processes
        for node in self.organization.process_network.node_indices() {
            if let Some(process) = self.organization.process_network.node_weight(node) {
                // Check if we can execute this process
                let can_execute = process.consumes.iter().all(|comp_type| {
                    self.structure.components.iter().any(|c| c.id == *comp_type)
                });
                
                if can_execute {
                    // Produce components
                    for product in &process.produces {
                        result.produced_components.push(product.clone());
                    }
                    
                    // Consume energy
                    result.energy_consumed += process.rate * dt;
                }
            }
        }
        
        // Apply decay
        let mut indices_to_remove = Vec::new();
        for (idx, component) in self.structure.components.iter_mut().enumerate() {
            let decay_prob = component.decay_rate * dt;
            if decay_prob > rand::random::<f64>() {
                result.decayed_components.push(component.id.clone());
                indices_to_remove.push(idx);
            }
        }
        
        // Remove decayed components
        for idx in indices_to_remove.into_iter().rev() {
            self.structure.components.remove(idx);
        }
        
        // Calculate entropy production (simplified)
        result.entropy_produced = result.energy_consumed * 0.1; // 10% conversion to entropy
        
        // Update boundary based on components
        self.boundary.update(&self.structure.components);
        
        self.time += dt;
        
        Ok(result)
    }
    
    fn verify_structural_coupling(&self, _env: &Self::Environment) -> bool {
        // Check if boundary allows necessary exchanges with environment
        true // Simplified implementation
    }
    
    async fn adapt_organization(&mut self, perturbation: Perturbation) -> Result<()> {
        // Adapt organization based on perturbation magnitude
        if perturbation.magnitude > 0.5 {
            // Modify process rates
            for node in self.organization.process_network.node_indices() {
                if let Some(process) = self.organization.process_network.node_weight_mut(node) {
                    process.rate *= 1.0 + perturbation.magnitude * 0.1;
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_autopoietic_health() {
        // Test health calculation
    }
    
    #[test]
    fn test_operational_closure() {
        // Test closure verification
    }
}