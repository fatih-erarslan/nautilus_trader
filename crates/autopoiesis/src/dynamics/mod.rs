//! Dynamics module for swarm intelligence and temporal dynamics
//! 
//! This module provides core implementations for various dynamical systems
//! that exhibit emergent behavior, including self-organized criticality,
//! complex adaptive systems, phase space reconstruction, and lattice field dynamics.

pub mod soc;
pub mod cas;
pub mod phase_space;
pub mod lattice_dynamics;

// Re-export key types for convenient access
pub use soc::{
    SelfOrganizedCriticality, 
    SocParameters, 
    AvalancheEvent, 
    PowerLawAnalysis,
    SocState,
};

pub use cas::{
    ComplexAdaptiveSystem,
    CasParameters,
    AdaptiveAgent,
    AgentType,
    FitnessSnapshot,
    CasState,
    InteractionNetwork,
    NetworkTopology,
};

pub use phase_space::{
    PhaseSpaceReconstructor,
    PhaseSpaceParameters,
    AttractorAnalysis,
    AttractorType,
};

pub use lattice_dynamics::{
    LatticeFieldDynamics,
    LatticeParameters,
    AdaptiveLattice,
    LatticeCoordinate,
    FieldState,
    LatticeState,
    InteractionRules,
};

/// Combined dynamics system that integrates all dynamical components
/// for comprehensive swarm intelligence analysis
pub struct IntegratedDynamicsSystem {
    /// Self-organized criticality component
    pub soc: SelfOrganizedCriticality,
    /// Complex adaptive system component
    pub cas: ComplexAdaptiveSystem,
    /// Phase space reconstructor
    pub phase_space: PhaseSpaceReconstructor,
    /// Lattice field dynamics
    pub lattice: LatticeFieldDynamics,
    /// Integration parameters
    params: IntegrationParameters,
    /// Current time step
    time_step: usize,
}

#[derive(Clone, Debug)]
pub struct IntegrationParameters {
    /// Time step size
    pub dt: f64,
    /// Coupling strength between systems
    pub coupling_strength: f64,
    /// Data exchange interval
    pub exchange_interval: usize,
    /// Synchronization method
    pub sync_method: SynchronizationMethod,
}

#[derive(Clone, Debug)]
pub enum SynchronizationMethod {
    /// Loose coupling - systems evolve independently
    Loose,
    /// Tight coupling - systems exchange data every step
    Tight,
    /// Adaptive coupling - coupling strength varies based on coherence
    Adaptive,
    /// Master-slave configuration
    MasterSlave { master: SystemType },
}

#[derive(Clone, Debug)]
pub enum SystemType {
    SOC,
    CAS,
    PhaseSpace,
    Lattice,
}

impl Default for IntegrationParameters {
    fn default() -> Self {
        Self {
            dt: 0.01,
            coupling_strength: 0.1,
            exchange_interval: 10,
            sync_method: SynchronizationMethod::Adaptive,
        }
    }
}

impl IntegratedDynamicsSystem {
    /// Create new integrated dynamics system
    pub fn new(
        soc_params: SocParameters,
        cas_params: CasParameters,
        phase_params: PhaseSpaceParameters,
        lattice_params: LatticeParameters,
        integration_params: IntegrationParameters,
    ) -> Self {
        let soc = SelfOrganizedCriticality::new(soc_params);
        let cas = ComplexAdaptiveSystem::new(cas_params);
        let phase_space = PhaseSpaceReconstructor::new(phase_params);
        
        let interaction_rules = InteractionRules::default();
        let lattice = LatticeFieldDynamics::new(lattice_params, interaction_rules);

        Self {
            soc,
            cas,
            phase_space,
            lattice,
            params: integration_params,
            time_step: 0,
        }
    }

    /// Initialize all systems with coordinated initial conditions
    pub fn initialize(&mut self) {
        // Initialize SOC with random state
        self.soc.initialize_random();
        
        // Initialize lattice with soliton configuration
        let center = vec![25.0, 25.0];
        self.lattice.initialize_soliton(&center, 5.0, 1.0);
        
        // CAS is initialized automatically during construction
        // Phase space starts empty and builds up over time
    }

    /// Evolve all systems by one integrated time step
    pub fn evolve_step(&mut self) {
        let current_time = self.time_step as f64 * self.params.dt;

        // Evolve individual systems
        if let Some(avalanche) = self.soc.step(current_time) {
            // Feed avalanche data to phase space reconstructor
            self.phase_space.add_data_point(avalanche.size as f64);
        }

        self.cas.evolve_generation();
        self.lattice.evolve_step();

        // Data exchange and coupling
        if self.time_step % self.params.exchange_interval == 0 {
            self.exchange_data();
        }

        // Update phase space with system metrics
        let complexity_metric = self.calculate_system_complexity();
        self.phase_space.add_data_point(complexity_metric);

        self.time_step += 1;
    }

    /// Exchange data between systems based on coupling method
    fn exchange_data(&mut self) {
        match &self.params.sync_method {
            SynchronizationMethod::Loose => {
                // Minimal data exchange - just phase space updates
                let soc_energy = self.soc.get_state().total_energy;
                self.phase_space.add_data_point(soc_energy);
            },
            
            SynchronizationMethod::Tight => {
                // Full data exchange every step
                self.tight_coupling_exchange();
            },
            
            SynchronizationMethod::Adaptive => {
                // Adaptive coupling based on system coherence
                let coherence = self.calculate_system_coherence();
                if coherence > 0.7 {
                    self.tight_coupling_exchange();
                } else {
                    // Light coupling when systems are incoherent
                    let avg_fitness = self.cas.get_state().fitness_stats
                        .as_ref()
                        .map(|s| s.mean_fitness)
                        .unwrap_or(0.0);
                    self.phase_space.add_data_point(avg_fitness);
                }
            },
            
            SynchronizationMethod::MasterSlave { master } => {
                // One system drives the others
                match master {
                    SystemType::SOC => {
                        let soc_state = self.soc.get_state();
                        // Use SOC criticality to influence CAS mutation rate
                        // This would require modifying CAS to accept external parameters
                    },
                    SystemType::CAS => {
                        let cas_state = self.cas.get_state();
                        // Use CAS diversity to influence SOC drive rate
                        // This would require modifying SOC parameters
                    },
                    _ => {
                        // Other master types would be implemented similarly
                    }
                }
            },
        }
    }

    /// Perform tight coupling data exchange
    fn tight_coupling_exchange(&mut self) {
        // SOC -> CAS: Use avalanche patterns to influence agent interactions
        let soc_state = self.soc.get_state();
        let criticality_ratio = soc_state.critical_sites as f64 / soc_state.grid_snapshot.len() as f64;
        
        // CAS -> Lattice: Use agent diversity to influence field dynamics
        let cas_state = self.cas.get_state();
        let diversity = cas_state.fitness_stats
            .as_ref()
            .map(|s| s.diversity_index)
            .unwrap_or(0.0);
        
        // Lattice -> SOC: Use field energy to influence drive rate
        let lattice_state = self.lattice.get_state();
        let field_energy_ratio = lattice_state.total_energy / (lattice_state.num_sites as f64 + 1.0);
        
        // Update phase space with cross-system metrics
        self.phase_space.add_data_point(criticality_ratio);
        self.phase_space.add_data_point(diversity);
        self.phase_space.add_data_point(field_energy_ratio);
    }

    /// Calculate overall system complexity
    fn calculate_system_complexity(&self) -> f64 {
        let soc_complexity = {
            let state = self.soc.get_state();
            state.critical_sites as f64 / state.grid_snapshot.len() as f64
        };

        let cas_complexity = self.cas.get_state().fitness_stats
            .as_ref()
            .map(|s| s.diversity_index)
            .unwrap_or(0.0);

        let lattice_complexity = {
            let state = self.lattice.get_state();
            state.topology_changes as f64 / (state.time_step as f64 + 1.0)
        };

        (soc_complexity + cas_complexity + lattice_complexity) / 3.0
    }

    /// Calculate system coherence
    fn calculate_system_coherence(&self) -> f64 {
        // Measure how synchronized the different systems are
        // This is a simplified measure - could be made more sophisticated
        
        let soc_energy = self.soc.get_state().total_energy;
        let cas_fitness = self.cas.get_state().fitness_stats
            .as_ref()
            .map(|s| s.mean_fitness)
            .unwrap_or(0.0);
        let lattice_energy = self.lattice.get_state().total_energy;

        // Normalize values to [0,1] range
        let soc_norm = soc_energy.tanh();
        let cas_norm = cas_fitness.tanh();
        let lattice_norm = (lattice_energy / 100.0).tanh();

        // Calculate coherence as inverse of variance
        let mean = (soc_norm + cas_norm + lattice_norm) / 3.0;
        let variance = ((soc_norm - mean).powi(2) + 
                       (cas_norm - mean).powi(2) + 
                       (lattice_norm - mean).powi(2)) / 3.0;

        1.0 / (1.0 + variance)
    }

    /// Get comprehensive system state
    pub fn get_integrated_state(&self) -> IntegratedSystemState {
        IntegratedSystemState {
            time_step: self.time_step,
            soc_state: self.soc.get_state(),
            cas_state: self.cas.get_state(),
            lattice_state: self.lattice.get_state(),
            phase_space_points: self.phase_space.get_phase_space().len(),
            system_complexity: self.calculate_system_complexity(),
            system_coherence: self.calculate_system_coherence(),
            coupling_strength: self.params.coupling_strength,
        }
    }

    /// Get individual system references
    pub fn get_soc(&self) -> &SelfOrganizedCriticality { &self.soc }
    pub fn get_cas(&self) -> &ComplexAdaptiveSystem { &self.cas }
    pub fn get_phase_space(&self) -> &PhaseSpaceReconstructor { &self.phase_space }
    pub fn get_lattice(&self) -> &LatticeFieldDynamics { &self.lattice }

    /// Get mutable references (for external updates)
    pub fn get_soc_mut(&mut self) -> &mut SelfOrganizedCriticality { &mut self.soc }
    pub fn get_cas_mut(&mut self) -> &mut ComplexAdaptiveSystem { &mut self.cas }
    pub fn get_phase_space_mut(&mut self) -> &mut PhaseSpaceReconstructor { &mut self.phase_space }
    pub fn get_lattice_mut(&mut self) -> &mut LatticeFieldDynamics { &mut self.lattice }
}

#[derive(Clone, Debug)]
pub struct IntegratedSystemState {
    pub time_step: usize,
    pub soc_state: SocState,
    pub cas_state: CasState,
    pub lattice_state: LatticeState,
    pub phase_space_points: usize,
    pub system_complexity: f64,
    pub system_coherence: f64,
    pub coupling_strength: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrated_system_creation() {
        let soc_params = SocParameters::default();
        let cas_params = CasParameters::default();
        let phase_params = PhaseSpaceParameters::default();
        let lattice_params = LatticeParameters::default();
        let integration_params = IntegrationParameters::default();

        let system = IntegratedDynamicsSystem::new(
            soc_params,
            cas_params,
            phase_params,
            lattice_params,
            integration_params,
        );

        assert_eq!(system.time_step, 0);
    }

    #[test]
    fn test_system_evolution() {
        let soc_params = SocParameters::default();
        let cas_params = CasParameters::default();
        let phase_params = PhaseSpaceParameters::default();
        let lattice_params = LatticeParameters::default();
        let integration_params = IntegrationParameters::default();

        let mut system = IntegratedDynamicsSystem::new(
            soc_params,
            cas_params,
            phase_params,
            lattice_params,
            integration_params,
        );

        system.initialize();
        let initial_time = system.time_step;
        
        system.evolve_step();
        
        assert_eq!(system.time_step, initial_time + 1);
    }

    #[test]
    fn test_complexity_calculation() {
        let soc_params = SocParameters::default();
        let cas_params = CasParameters::default();
        let phase_params = PhaseSpaceParameters::default();
        let lattice_params = LatticeParameters::default();
        let integration_params = IntegrationParameters::default();

        let mut system = IntegratedDynamicsSystem::new(
            soc_params,
            cas_params,
            phase_params,
            lattice_params,
            integration_params,
        );

        system.initialize();
        let complexity = system.calculate_system_complexity();
        
        assert!(complexity >= 0.0);
        assert!(complexity <= 1.0);
    }
}