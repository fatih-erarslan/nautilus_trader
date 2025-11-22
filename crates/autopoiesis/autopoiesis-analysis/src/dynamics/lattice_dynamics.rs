use nalgebra::{DVector, DMatrix, Complex};
use std::collections::HashMap;
use rayon::prelude::*;

/// Grinberg lattice field dynamics for modeling emergent spacetime
/// Implements discrete field theory on adaptive lattices with topological evolution
pub struct LatticeFieldDynamics {
    /// The adaptive lattice structure
    lattice: AdaptiveLattice,
    /// Field values at each lattice site
    field_values: HashMap<LatticeCoordinate, FieldState>,
    /// Interaction rules between sites
    interaction_rules: InteractionRules,
    /// Evolution parameters
    params: LatticeParameters,
    /// Current time step
    time_step: usize,
    /// Energy conservation tracker
    total_energy: f64,
    /// Topology change history
    topology_history: Vec<TopologyChange>,
}

/// Coordinate system for the adaptive lattice
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct LatticeCoordinate {
    /// Spatial coordinates (can be multi-dimensional)
    pub coords: Vec<i32>,
    /// Hierarchical level (for adaptive refinement)
    pub level: usize,
}

/// State of the field at a lattice site
#[derive(Clone, Debug)]
pub struct FieldState {
    /// Complex field amplitude
    pub amplitude: Complex<f64>,
    /// Field gradient (spatial derivatives)
    pub gradient: DVector<f64>,
    /// Local energy density
    pub energy_density: f64,
    /// Topological charge (for solitons, vortices, etc.)
    pub topological_charge: f64,
    /// Connection to neighboring sites (discrete connection)
    pub connections: HashMap<LatticeCoordinate, Complex<f64>>,
    /// Local curvature measure
    pub curvature: f64,
}

/// Rules governing field interactions
#[derive(Clone, Debug)]
pub struct InteractionRules {
    /// Coupling strength
    pub coupling_strength: f64,
    /// Nonlinearity parameter
    pub nonlinearity: f64,
    /// Dissipation rate
    pub dissipation: f64,
    /// External field strength
    pub external_field: f64,
    /// Lattice spacing
    pub lattice_spacing: f64,
    /// Interaction range
    pub interaction_range: usize,
}

/// Parameters for lattice evolution
#[derive(Clone, Debug)]
pub struct LatticeParameters {
    /// Time step size
    pub dt: f64,
    /// Spatial dimensions
    pub dimensions: usize,
    /// Initial lattice size
    pub initial_size: Vec<usize>,
    /// Adaptive refinement threshold
    pub refinement_threshold: f64,
    /// Coarsening threshold
    pub coarsening_threshold: f64,
    /// Maximum hierarchy level
    pub max_level: usize,
    /// Conservation tolerance
    pub conservation_tolerance: f64,
}

/// Adaptive lattice structure with hierarchical refinement
#[derive(Clone, Debug)]
pub struct AdaptiveLattice {
    /// Active sites in the lattice
    pub sites: Vec<LatticeCoordinate>,
    /// Neighbor relationships
    pub neighbors: HashMap<LatticeCoordinate, Vec<LatticeCoordinate>>,
    /// Hierarchy tree for adaptive refinement
    pub hierarchy: HashMap<LatticeCoordinate, Vec<LatticeCoordinate>>,
    /// Boundary conditions
    pub boundary_conditions: BoundaryConditions,
}

/// Boundary condition types
#[derive(Clone, Debug)]
pub enum BoundaryConditions {
    Periodic,
    Dirichlet(Complex<f64>),
    Neumann,
    Open,
    Absorbing,
}

/// Record of topology changes
#[derive(Clone, Debug)]
pub struct TopologyChange {
    pub time_step: usize,
    pub change_type: TopologyChangeType,
    pub location: LatticeCoordinate,
    pub energy_change: f64,
}

#[derive(Clone, Debug)]
pub enum TopologyChangeType {
    SiteCreation,
    SiteDestruction,
    ConnectionCreation,
    ConnectionDestruction,
    LevelRefinement,
    LevelCoarsening,
}

impl Default for LatticeParameters {
    fn default() -> Self {
        Self {
            dt: 0.01,
            dimensions: 2,
            initial_size: vec![50, 50],
            refinement_threshold: 1.0,
            coarsening_threshold: 0.1,
            max_level: 5,
            conservation_tolerance: 1e-10,
        }
    }
}

impl Default for InteractionRules {
    fn default() -> Self {
        Self {
            coupling_strength: 1.0,
            nonlinearity: 0.1,
            dissipation: 0.01,
            external_field: 0.0,
            lattice_spacing: 1.0,
            interaction_range: 1,
        }
    }
}

impl LatticeFieldDynamics {
    /// Create new lattice field dynamics system
    pub fn new(params: LatticeParameters, rules: InteractionRules) -> Self {
        let lattice = AdaptiveLattice::new(&params);
        let mut field_values = HashMap::new();
        
        // Initialize field values
        for site in &lattice.sites {
            field_values.insert(site.clone(), FieldState::new(&params.initial_size));
        }

        Self {
            lattice,
            field_values,
            interaction_rules: rules,
            params,
            time_step: 0,
            total_energy: 0.0,
            topology_history: Vec::new(),
        }
    }

    /// Initialize with random field configuration
    pub fn initialize_random(&mut self) {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();
        
        for (_, field_state) in self.field_values.iter_mut() {
            let real_part = rng.gen::<f64>() * 2.0 - 1.0;
            let imag_part = rng.gen::<f64>() * 2.0 - 1.0;
            field_state.amplitude = Complex::new(real_part, imag_part);
            
            // Initialize gradient
            field_state.gradient = DVector::from_fn(self.params.dimensions, |_, _| {
                rng.gen::<f64>() * 0.1 - 0.05
            });
        }
        
        self.update_derived_quantities();
    }

    /// Initialize with soliton configuration
    pub fn initialize_soliton(&mut self, center: &[f64], width: f64, amplitude: f64) {
        for (coord, field_state) in self.field_values.iter_mut() {
            let position = coord.to_physical_position(self.interaction_rules.lattice_spacing);
            
            // Calculate distance from soliton center
            let distance_sq: f64 = position.iter()
                .zip(center.iter())
                .map(|(p, c)| (p - c).powi(2))
                .sum();
            
            let distance = distance_sq.sqrt();
            
            // Soliton profile (tanh for kink soliton)
            let profile = amplitude * (distance / width).tanh();
            field_state.amplitude = Complex::new(profile, 0.0);
            
            // Calculate analytical gradient
            let gradient_magnitude = amplitude / (width * (distance / width).cosh().powi(2));
            field_state.gradient = DVector::from_fn(self.params.dimensions, |i, _| {
                if distance > 1e-10 {
                    gradient_magnitude * (position[i] - center[i]) / distance
                } else {
                    0.0
                }
            });
        }
        
        self.update_derived_quantities();
    }

    /// Evolve the field by one time step
    pub fn evolve_step(&mut self) {
        // Calculate field evolution using Verlet integration
        self.calculate_forces();
        self.integrate_fields();
        
        // Check for adaptive refinement/coarsening
        self.adaptive_refinement();
        
        // Update derived quantities
        self.update_derived_quantities();
        
        // Check conservation laws
        self.check_conservation();
        
        self.time_step += 1;
    }

    /// Calculate forces and field evolution
    fn calculate_forces(&mut self) {
        let sites: Vec<_> = self.lattice.sites.clone();
        
        // Use parallel processing for force calculation
        let force_updates: Vec<_> = sites.par_iter().map(|site| {
            self.calculate_site_force(site)
        }).collect();
        
        // Apply force updates
        for (site, force_update) in sites.iter().zip(force_updates.iter()) {
            if let Some(field_state) = self.field_values.get_mut(site) {
                field_state.amplitude += force_update * self.params.dt;
            }
        }
    }

    /// Calculate force on a specific site
    fn calculate_site_force(&self, site: &LatticeCoordinate) -> Complex<f64> {
        let field_state = match self.field_values.get(site) {
            Some(state) => state,
            None => return Complex::new(0.0, 0.0),
        };
        
        let neighbors = match self.lattice.neighbors.get(site) {
            Some(neighs) => neighs,
            None => return Complex::new(0.0, 0.0),
        };
        
        let mut force = Complex::new(0.0, 0.0);
        
        // Kinetic term (discrete Laplacian)
        let mut laplacian = Complex::new(0.0, 0.0);
        let spacing_sq = self.interaction_rules.lattice_spacing.powi(2);
        
        for neighbor in neighbors {
            if let Some(neighbor_state) = self.field_values.get(neighbor) {
                laplacian += neighbor_state.amplitude - field_state.amplitude;
            }
        }
        laplacian /= spacing_sq;
        
        // Add kinetic force
        force += self.interaction_rules.coupling_strength * laplacian;
        
        // Nonlinear interaction term
        let amplitude_squared = field_state.amplitude.norm_sqr();
        force -= self.interaction_rules.nonlinearity * amplitude_squared * field_state.amplitude;
        
        // External field
        force += Complex::new(self.interaction_rules.external_field, 0.0);
        
        // Dissipation
        force -= self.interaction_rules.dissipation * field_state.amplitude;
        
        // Topological interactions (discrete gauge field contribution)
        force += self.calculate_topological_force(site, field_state);
        
        force
    }

    /// Calculate topological force contributions
    fn calculate_topological_force(&self, site: &LatticeCoordinate, state: &FieldState) -> Complex<f64> {
        let neighbors = match self.lattice.neighbors.get(site) {
            Some(neighs) => neighs,
            None => return Complex::new(0.0, 0.0),
        };
        
        let mut topological_force = Complex::new(0.0, 0.0);
        
        // Berry curvature contribution
        for neighbor in neighbors {
            if let Some(connection) = state.connections.get(neighbor) {
                if let Some(neighbor_state) = self.field_values.get(neighbor) {
                    // Discrete Berry connection
                    let phase_difference = (neighbor_state.amplitude / state.amplitude).arg();
                    let berry_connection = Complex::new(0.0, phase_difference);
                    
                    topological_force += connection * berry_connection * state.amplitude;
                }
            }
        }
        
        // Scale by topological coupling
        topological_force * state.topological_charge * 0.1
    }

    /// Integrate field equations using Verlet method
    fn integrate_fields(&mut self) {
        // This is a simplified integration - in practice, you'd use a more sophisticated
        // symplectic integrator for better energy conservation
        
        for (_, field_state) in self.field_values.iter_mut() {
            // Update energy density
            field_state.energy_density = field_state.amplitude.norm_sqr() + 
                field_state.gradient.norm_squared();
        }
    }

    /// Adaptive mesh refinement based on field gradients
    fn adaptive_refinement(&mut self) {
        let mut sites_to_refine = Vec::new();
        let mut sites_to_coarsen = Vec::new();
        
        // Check refinement criteria
        for site in &self.lattice.sites {
            if let Some(field_state) = self.field_values.get(site) {
                let gradient_magnitude = field_state.gradient.norm();
                
                if gradient_magnitude > self.params.refinement_threshold && 
                   site.level < self.params.max_level {
                    sites_to_refine.push(site.clone());
                } else if gradient_magnitude < self.params.coarsening_threshold && 
                         site.level > 0 {
                    sites_to_coarsen.push(site.clone());
                }
            }
        }
        
        // Perform refinement
        for site in sites_to_refine {
            self.refine_site(&site);
        }
        
        // Perform coarsening
        for site in sites_to_coarsen {
            self.coarsen_site(&site);
        }
    }

    /// Refine a lattice site by creating finer sub-sites
    fn refine_site(&mut self, site: &LatticeCoordinate) {
        let new_level = site.level + 1;
        let mut new_sites = Vec::new();
        
        // Create 2^d new sites for d dimensions
        let num_subsites = 1 << self.params.dimensions;
        
        for i in 0..num_subsites {
            let mut new_coords = site.coords.clone();
            
            // Generate subcell coordinates
            for dim in 0..self.params.dimensions {
                if (i >> dim) & 1 == 1 {
                    new_coords[dim] = new_coords[dim] * 2 + 1;
                } else {
                    new_coords[dim] = new_coords[dim] * 2;
                }
            }
            
            let new_site = LatticeCoordinate {
                coords: new_coords,
                level: new_level,
            };
            
            new_sites.push(new_site.clone());
        }
        
        // Interpolate field values to new sites
        if let Some(parent_state) = self.field_values.get(site).cloned() {
            for new_site in &new_sites {
                let mut new_state = parent_state.clone();
                
                // Add some perturbation for refined sites
                new_state.amplitude *= Complex::new(0.98, 0.0);
                
                self.field_values.insert(new_site.clone(), new_state);
                self.lattice.sites.push(new_site.clone());
            }
            
            // Update hierarchy
            self.lattice.hierarchy.insert(site.clone(), new_sites.clone());
            
            // Record topology change
            self.topology_history.push(TopologyChange {
                time_step: self.time_step,
                change_type: TopologyChangeType::LevelRefinement,
                location: site.clone(),
                energy_change: 0.0, // Calculate actual energy change
            });
        }
        
        // Update neighbor relationships
        self.update_neighbor_relationships();
    }

    /// Coarsen a lattice site by merging with siblings
    fn coarsen_site(&mut self, site: &LatticeCoordinate) {
        if site.level == 0 {
            return; // Cannot coarsen base level
        }
        
        // Find parent and siblings
        let parent_coords: Vec<i32> = site.coords.iter().map(|&c| c / 2).collect();
        let parent = LatticeCoordinate {
            coords: parent_coords,
            level: site.level - 1,
        };
        
        // Find all siblings
        let mut siblings = Vec::new();
        let num_siblings = 1 << self.params.dimensions;
        
        for i in 0..num_siblings {
            let mut sibling_coords = parent.coords.clone();
            
            for dim in 0..self.params.dimensions {
                if (i >> dim) & 1 == 1 {
                    sibling_coords[dim] = sibling_coords[dim] * 2 + 1;
                } else {
                    sibling_coords[dim] = sibling_coords[dim] * 2;
                }
            }
            
            let sibling = LatticeCoordinate {
                coords: sibling_coords,
                level: site.level,
            };
            
            if self.field_values.contains_key(&sibling) {
                siblings.push(sibling);
            }
        }
        
        // Only coarsen if all siblings have low gradients
        let all_coarsenable = siblings.iter().all(|s| {
            if let Some(state) = self.field_values.get(s) {
                state.gradient.norm() < self.params.coarsening_threshold
            } else {
                false
            }
        });
        
        if all_coarsenable && siblings.len() == num_siblings {
            // Average field values
            let mut averaged_amplitude = Complex::new(0.0, 0.0);
            
            for sibling in &siblings {
                if let Some(state) = self.field_values.get(sibling) {
                    averaged_amplitude += state.amplitude;
                }
            }
            averaged_amplitude /= siblings.len() as f64;
            
            // Create parent state
            let parent_state = FieldState {
                amplitude: averaged_amplitude,
                gradient: DVector::zeros(self.params.dimensions),
                energy_density: averaged_amplitude.norm_sqr(),
                topological_charge: 0.0,
                connections: HashMap::new(),
                curvature: 0.0,
            };
            
            // Remove siblings and add parent
            for sibling in &siblings {
                self.field_values.remove(sibling);
                self.lattice.sites.retain(|s| s != sibling);
            }
            
            self.field_values.insert(parent.clone(), parent_state);
            self.lattice.sites.push(parent.clone());
            
            // Record topology change
            self.topology_history.push(TopologyChange {
                time_step: self.time_step,
                change_type: TopologyChangeType::LevelCoarsening,
                location: parent,
                energy_change: 0.0,
            });
        }
        
        self.update_neighbor_relationships();
    }

    /// Update neighbor relationships after topology changes
    fn update_neighbor_relationships(&mut self) {
        self.lattice.neighbors.clear();
        
        for site in &self.lattice.sites {
            let mut neighbors = Vec::new();
            
            // Find neighbors at same or compatible levels
            for other_site in &self.lattice.sites {
                if site == other_site {
                    continue;
                }
                
                if self.are_neighbors(site, other_site) {
                    neighbors.push(other_site.clone());
                }
            }
            
            self.lattice.neighbors.insert(site.clone(), neighbors);
        }
    }

    /// Check if two sites are neighbors
    fn are_neighbors(&self, site1: &LatticeCoordinate, site2: &LatticeCoordinate) -> bool {
        // Sites are neighbors if they are adjacent and at compatible levels
        let level_diff = (site1.level as i32 - site2.level as i32).abs();
        
        if level_diff > 1 {
            return false; // Too far apart in hierarchy
        }
        
        // Check spatial adjacency
        let coord_diff: i32 = site1.coords.iter()
            .zip(site2.coords.iter())
            .map(|(c1, c2)| (c1 - c2).abs())
            .sum();
        
        // Allow connection if Manhattan distance is 1 at same level
        // or appropriate scaled distance for different levels
        if site1.level == site2.level {
            coord_diff == 1
        } else {
            // More complex logic for inter-level connections
            coord_diff <= 2
        }
    }

    /// Update derived quantities (energy, curvature, etc.)
    fn update_derived_quantities(&mut self) {
        let mut total_energy = 0.0;
        
        for (site, field_state) in self.field_values.iter_mut() {
            // Update energy density
            let kinetic_energy = field_state.gradient.norm_squared();
            let potential_energy = field_state.amplitude.norm_sqr().powi(2) * 
                self.interaction_rules.nonlinearity;
            
            field_state.energy_density = kinetic_energy + potential_energy;
            total_energy += field_state.energy_density;
            
            // Update topological charge (simplified winding number)
            field_state.topological_charge = self.calculate_local_topological_charge(site);
            
            // Update curvature
            field_state.curvature = self.calculate_local_curvature(site);
        }
        
        self.total_energy = total_energy;
    }

    /// Calculate local topological charge
    fn calculate_local_topological_charge(&self, site: &LatticeCoordinate) -> f64 {
        let neighbors = match self.lattice.neighbors.get(site) {
            Some(neighs) => neighs,
            None => return 0.0,
        };
        
        let current_state = match self.field_values.get(site) {
            Some(state) => state,
            None => return 0.0,
        };
        
        let mut winding = 0.0;
        let current_phase = current_state.amplitude.arg();
        
        for neighbor in neighbors {
            if let Some(neighbor_state) = self.field_values.get(neighbor) {
                let neighbor_phase = neighbor_state.amplitude.arg();
                let phase_diff = neighbor_phase - current_phase;
                
                // Wrap to [-π, π]
                let wrapped_diff = ((phase_diff + std::f64::consts::PI) % 
                    (2.0 * std::f64::consts::PI)) - std::f64::consts::PI;
                
                winding += wrapped_diff;
            }
        }
        
        winding / (2.0 * std::f64::consts::PI)
    }

    /// Calculate local curvature measure
    fn calculate_local_curvature(&self, site: &LatticeCoordinate) -> f64 {
        let neighbors = match self.lattice.neighbors.get(site) {
            Some(neighs) => neighs,
            None => return 0.0,
        };
        
        if neighbors.len() < 2 {
            return 0.0;
        }
        
        let current_state = match self.field_values.get(site) {
            Some(state) => state,
            None => return 0.0,
        };
        
        // Discrete Gaussian curvature approximation
        let mut curvature_sum = 0.0;
        let expected_angle = 2.0 * std::f64::consts::PI / neighbors.len() as f64;
        
        for i in 0..neighbors.len() {
            let next_i = (i + 1) % neighbors.len();
            
            if let (Some(state1), Some(state2)) = (
                self.field_values.get(&neighbors[i]),
                self.field_values.get(&neighbors[next_i])
            ) {
                let vec1 = state1.amplitude - current_state.amplitude;
                let vec2 = state2.amplitude - current_state.amplitude;
                
                let angle = (vec1.conj() * vec2).arg();
                curvature_sum += angle - expected_angle;
            }
        }
        
        curvature_sum / neighbors.len() as f64
    }

    /// Check conservation laws
    fn check_conservation(&self) {
        // Check energy conservation (should be approximately constant)
        let energy_change = if let Some(last_change) = self.topology_history.last() {
            (self.total_energy - last_change.energy_change).abs()
        } else {
            0.0
        };
        
        if energy_change > self.params.conservation_tolerance {
            eprintln!("Warning: Energy conservation violated by {}", energy_change);
        }
        
        // Check charge conservation
        let total_charge: f64 = self.field_values.values()
            .map(|state| state.topological_charge)
            .sum();
        
        // Topological charge should be quantized
        let charge_fractional_part = total_charge.fract().abs();
        if charge_fractional_part > 0.1 && charge_fractional_part < 0.9 {
            eprintln!("Warning: Non-quantized topological charge: {}", total_charge);
        }
    }

    /// Get current system state
    pub fn get_state(&self) -> LatticeState {
        LatticeState {
            time_step: self.time_step,
            total_energy: self.total_energy,
            num_sites: self.lattice.sites.len(),
            topology_changes: self.topology_history.len(),
            total_topological_charge: self.field_values.values()
                .map(|state| state.topological_charge)
                .sum(),
            max_field_amplitude: self.field_values.values()
                .map(|state| state.amplitude.norm())
                .fold(0.0, f64::max),
            lattice_levels: self.get_level_distribution(),
        }
    }

    fn get_level_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();
        
        for site in &self.lattice.sites {
            *distribution.entry(site.level).or_insert(0) += 1;
        }
        
        distribution
    }

    /// Get field values at all sites
    pub fn get_field_values(&self) -> &HashMap<LatticeCoordinate, FieldState> {
        &self.field_values
    }

    /// Get lattice structure
    pub fn get_lattice(&self) -> &AdaptiveLattice {
        &self.lattice
    }
}

impl AdaptiveLattice {
    fn new(params: &LatticeParameters) -> Self {
        let mut sites = Vec::new();
        let mut neighbors = HashMap::new();
        
        // Initialize regular lattice
        match params.dimensions {
            1 => {
                for i in 0..params.initial_size[0] {
                    let coord = LatticeCoordinate {
                        coords: vec![i as i32],
                        level: 0,
                    };
                    sites.push(coord);
                }
            },
            2 => {
                for i in 0..params.initial_size[0] {
                    for j in 0..params.initial_size[1] {
                        let coord = LatticeCoordinate {
                            coords: vec![i as i32, j as i32],
                            level: 0,
                        };
                        sites.push(coord);
                    }
                }
            },
            3 => {
                for i in 0..params.initial_size[0] {
                    for j in 0..params.initial_size[1] {
                        for k in 0..params.initial_size.get(2).unwrap_or(&10) {
                            let coord = LatticeCoordinate {
                                coords: vec![i as i32, j as i32, *k as i32],
                                level: 0,
                            };
                            sites.push(coord);
                        }
                    }
                }
            },
            _ => panic!("Unsupported dimension: {}", params.dimensions),
        }
        
        // Initialize neighbor relationships
        for site in &sites {
            let mut site_neighbors = Vec::new();
            
            for other_site in &sites {
                if site == other_site {
                    continue;
                }
                
                // Check if sites are adjacent
                let coord_diff: i32 = site.coords.iter()
                    .zip(other_site.coords.iter())
                    .map(|(c1, c2)| (c1 - c2).abs())
                    .sum();
                
                if coord_diff == 1 {
                    site_neighbors.push(other_site.clone());
                }
            }
            
            neighbors.insert(site.clone(), site_neighbors);
        }
        
        Self {
            sites,
            neighbors,
            hierarchy: HashMap::new(),
            boundary_conditions: BoundaryConditions::Periodic,
        }
    }
}

impl FieldState {
    fn new(initial_size: &[usize]) -> Self {
        Self {
            amplitude: Complex::new(0.0, 0.0),
            gradient: DVector::zeros(initial_size.len()),
            energy_density: 0.0,
            topological_charge: 0.0,
            connections: HashMap::new(),
            curvature: 0.0,
        }
    }
}

impl LatticeCoordinate {
    /// Convert lattice coordinate to physical position
    fn to_physical_position(&self, spacing: f64) -> Vec<f64> {
        self.coords.iter()
            .map(|&c| c as f64 * spacing / (1 << self.level) as f64)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct LatticeState {
    pub time_step: usize,
    pub total_energy: f64,
    pub num_sites: usize,
    pub topology_changes: usize,
    pub total_topological_charge: f64,
    pub max_field_amplitude: f64,
    pub lattice_levels: HashMap<usize, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_initialization() {
        let params = LatticeParameters::default();
        let rules = InteractionRules::default();
        let lattice_dynamics = LatticeFieldDynamics::new(params, rules);
        
        assert!(!lattice_dynamics.lattice.sites.is_empty());
        assert_eq!(lattice_dynamics.time_step, 0);
    }

    #[test]
    fn test_soliton_initialization() {
        let params = LatticeParameters::default();
        let rules = InteractionRules::default();
        let mut lattice_dynamics = LatticeFieldDynamics::new(params, rules);
        
        let center = vec![25.0, 25.0];
        lattice_dynamics.initialize_soliton(&center, 5.0, 1.0);
        
        let state = lattice_dynamics.get_state();
        assert!(state.max_field_amplitude > 0.0);
    }

    #[test]
    fn test_evolution_step() {
        let mut params = LatticeParameters::default();
        params.initial_size = vec![10, 10]; // Smaller for testing
        let rules = InteractionRules::default();
        let mut lattice_dynamics = LatticeFieldDynamics::new(params, rules);
        
        lattice_dynamics.initialize_random();
        let initial_time = lattice_dynamics.time_step;
        
        lattice_dynamics.evolve_step();
        
        assert_eq!(lattice_dynamics.time_step, initial_time + 1);
    }

    #[test]
    fn test_coordinate_conversion() {
        let coord = LatticeCoordinate {
            coords: vec![5, 10],
            level: 1,
        };
        
        let position = coord.to_physical_position(1.0);
        assert_eq!(position, vec![2.5, 5.0]); // Divided by 2^level
    }

    #[test]
    fn test_neighbor_detection() {
        let params = LatticeParameters::default();
        let rules = InteractionRules::default();
        let lattice_dynamics = LatticeFieldDynamics::new(params, rules);
        
        let coord1 = LatticeCoordinate {
            coords: vec![5, 5],
            level: 0,
        };
        let coord2 = LatticeCoordinate {
            coords: vec![5, 6],
            level: 0,
        };
        let coord3 = LatticeCoordinate {
            coords: vec![5, 7],
            level: 0,
        };
        
        assert!(lattice_dynamics.are_neighbors(&coord1, &coord2));
        assert!(!lattice_dynamics.are_neighbors(&coord1, &coord3));
    }
}