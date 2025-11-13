//! Temperature calculations and thermodynamic relationships
//!
//! Implements temperature-dependent physics including thermal noise,
//! Boltzmann distributions, and temperature-energy relationships.

use hyperphysics_pbit::{Result, PBitError};

/// Physical constants for temperature calculations
pub mod constants {
    /// Boltzmann constant in eV/K
    pub const BOLTZMANN_EV: f64 = 8.617333262e-5;

    /// Boltzmann constant in J/K
    pub const BOLTZMANN_J: f64 = 1.380649e-23;

    /// Room temperature in Kelvin
    pub const ROOM_TEMPERATURE: f64 = 300.0;
}

/// Temperature state and dynamics
#[derive(Debug, Clone)]
pub struct Temperature {
    /// Current temperature in Kelvin
    pub kelvin: f64,

    /// Inverse temperature β = 1/(k_B T)
    pub beta: f64,

    /// Temperature in energy units (k_B T)
    pub energy_scale: f64,
}

impl Temperature {
    /// Create new temperature from Kelvin
    pub fn from_kelvin(kelvin: f64) -> Result<Self> {
        if kelvin <= 0.0 {
            return Err(PBitError::InvalidTemperature { temp: kelvin });
        }

        let beta = 1.0 / (constants::BOLTZMANN_EV * kelvin);
        let energy_scale = constants::BOLTZMANN_EV * kelvin;

        Ok(Self {
            kelvin,
            beta,
            energy_scale,
        })
    }

    /// Create from inverse temperature β
    pub fn from_beta(beta: f64) -> Result<Self> {
        if beta <= 0.0 {
            return Err(PBitError::InvalidTemperature { temp: 1.0 / beta });
        }

        let kelvin = 1.0 / (constants::BOLTZMANN_EV * beta);
        let energy_scale = constants::BOLTZMANN_EV * kelvin;

        Ok(Self {
            kelvin,
            beta,
            energy_scale,
        })
    }

    /// Create from dimensionless temperature (in units of coupling strength)
    pub fn from_dimensionless(t: f64, coupling_scale: f64) -> Result<Self> {
        let energy = t * coupling_scale;
        let kelvin = energy / constants::BOLTZMANN_EV;
        Self::from_kelvin(kelvin)
    }

    /// Room temperature (300K)
    /// 
    /// # Panics
    /// Never panics - room temperature is always valid
    pub fn room_temperature() -> Self {
        // Safe: Room temperature is always positive and valid
        Self::from_kelvin(constants::ROOM_TEMPERATURE)
            .expect("Room temperature constant is always valid")
    }

    /// Zero temperature limit (practical minimum)
    /// 
    /// # Panics
    /// Never panics - minimum temperature is always valid
    pub fn zero_limit() -> Self {
        // Safe: 1e-6 K is always positive and valid
        Self::from_kelvin(1e-6)
            .expect("Zero limit temperature is always valid")
    }

    /// Infinite temperature limit (practical maximum)
    /// 
    /// # Panics
    /// Never panics - maximum temperature is always valid
    pub fn infinite_limit() -> Self {
        // Safe: 1e6 K is always positive and valid
        Self::from_kelvin(1e6)
            .expect("Infinite limit temperature is always valid")
    }

    /// Compute Boltzmann factor exp(-β E)
    pub fn boltzmann_factor(&self, energy: f64) -> f64 {
        (-self.beta * energy).exp()
    }

    /// Compute partition function for two-level system
    ///
    /// Z = exp(-β E₀) + exp(-β E₁)
    pub fn partition_function_two_level(&self, e0: f64, e1: f64) -> f64 {
        self.boltzmann_factor(e0) + self.boltzmann_factor(e1)
    }

    /// Compute thermal occupation probability for two-level system
    ///
    /// P(E₁) = exp(-β E₁) / Z
    pub fn thermal_occupation(&self, e0: f64, e1: f64) -> f64 {
        let z = self.partition_function_two_level(e0, e1);
        self.boltzmann_factor(e1) / z
    }

    /// Compute thermal energy from occupation probability
    pub fn thermal_energy(&self, e0: f64, e1: f64) -> f64 {
        let p1 = self.thermal_occupation(e0, e1);
        let p0 = 1.0 - p1;
        p0 * e0 + p1 * e1
    }

    /// Thermal fluctuation magnitude (k_B T)
    pub fn thermal_fluctuation(&self) -> f64 {
        self.energy_scale
    }

    /// Generate thermal noise sample from Boltzmann distribution
    pub fn thermal_noise(&self) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Box-Muller transform for Gaussian noise
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let noise = (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        noise * self.thermal_fluctuation().sqrt()
    }

    /// Compute specific heat for two-level system
    ///
    /// C_v = (∂⟨E⟩/∂T) = β² ⟨(E - ⟨E⟩)²⟩
    pub fn specific_heat_two_level(&self, e0: f64, e1: f64) -> f64 {
        let avg_e = self.thermal_energy(e0, e1);
        let p1 = self.thermal_occupation(e0, e1);
        let p0 = 1.0 - p1;

        let variance = p0 * (e0 - avg_e).powi(2) + p1 * (e1 - avg_e).powi(2);
        self.beta.powi(2) * variance
    }

    /// Landauer limit: minimum energy dissipation for bit erasure
    ///
    /// E_min = k_B T ln(2)
    pub fn landauer_limit(&self) -> f64 {
        self.energy_scale * 2.0_f64.ln()
    }

    /// Check if system is in quantum regime (k_B T < ħω)
    pub fn is_quantum_regime(&self, frequency: f64) -> bool {
        let hbar = 6.582119569e-16; // eV⋅s
        let quantum_energy = hbar * frequency;
        self.energy_scale < quantum_energy
    }

    /// Check if system is in classical regime (k_B T >> ħω)
    pub fn is_classical_regime(&self, frequency: f64) -> bool {
        let hbar = 6.582119569e-16; // eV⋅s
        let quantum_energy = hbar * frequency;
        self.energy_scale > 10.0 * quantum_energy
    }
}

/// Temperature ramping for annealing schedules
#[derive(Debug, Clone)]
pub struct TemperatureSchedule {
    /// Initial temperature
    pub t_initial: Temperature,

    /// Final temperature
    pub t_final: Temperature,

    /// Current time step
    pub step: usize,

    /// Total steps
    pub total_steps: usize,

    /// Schedule type
    pub schedule_type: ScheduleType,
}

/// Annealing schedule types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleType {
    /// Linear: T(t) = T₀ + (T_f - T₀) × t/t_max
    Linear,

    /// Exponential: T(t) = T₀ × (T_f/T₀)^(t/t_max)
    Exponential,

    /// Logarithmic: T(t) = T₀ / ln(1 + t)
    Logarithmic,

    /// Inverse: T(t) = T₀ / (1 + α t)
    Inverse,
}

impl TemperatureSchedule {
    /// Create new temperature schedule
    pub fn new(
        t_initial: Temperature,
        t_final: Temperature,
        total_steps: usize,
        schedule_type: ScheduleType,
    ) -> Self {
        Self {
            t_initial,
            t_final,
            step: 0,
            total_steps,
            schedule_type,
        }
    }

    /// Get temperature at current step
    pub fn current_temperature(&self) -> Temperature {
        self.temperature_at_step(self.step)
    }

    /// Get temperature at specific step
    pub fn temperature_at_step(&self, step: usize) -> Temperature {
        let progress = (step as f64) / (self.total_steps as f64);
        let t0 = self.t_initial.kelvin;
        let tf = self.t_final.kelvin;

        let kelvin = match self.schedule_type {
            ScheduleType::Linear => {
                t0 + (tf - t0) * progress
            }
            ScheduleType::Exponential => {
                t0 * (tf / t0).powf(progress)
            }
            ScheduleType::Logarithmic => {
                t0 / (1.0 + progress).ln()
            }
            ScheduleType::Inverse => {
                t0 / (1.0 + progress * (t0 / tf - 1.0))
            }
        };

        Temperature::from_kelvin(kelvin).unwrap()
    }

    /// Advance to next step
    pub fn step(&mut self) -> Temperature {
        self.step += 1;
        if self.step > self.total_steps {
            self.step = self.total_steps;
        }
        self.current_temperature()
    }

    /// Reset schedule
    pub fn reset(&mut self) {
        self.step = 0;
    }

    /// Check if schedule is complete
    pub fn is_complete(&self) -> bool {
        self.step >= self.total_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_creation() {
        let t = Temperature::from_kelvin(300.0).expect("Valid temperature");
        assert!((t.kelvin - 300.0).abs() < 1e-10);
        assert!(t.beta > 0.0);
        assert!(t.energy_scale > 0.0);
    }

    #[test]
    fn test_beta_consistency() {
        let t1 = Temperature::from_kelvin(300.0).expect("Valid temperature");
        let t2 = Temperature::from_beta(t1.beta).expect("Valid beta");
        assert!((t1.kelvin - t2.kelvin).abs() < 1e-6);
    }

    #[test]
    fn test_boltzmann_factor() {
        let t = Temperature::from_kelvin(300.0).expect("Valid temperature");
        let factor = t.boltzmann_factor(0.0);
        assert!((factor - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_thermal_occupation() {
        let t = Temperature::from_kelvin(300.0).expect("Valid temperature");
        let e0 = 0.0;
        let e1 = 0.1; // 0.1 eV gap

        let p1 = t.thermal_occupation(e0, e1);
        assert!(p1 >= 0.0 && p1 <= 1.0);

        // At room temperature, 0.1 eV gap should strongly favor ground state
        assert!(p1 < 0.1);
    }

    #[test]
    fn test_landauer_limit() {
        let t = Temperature::room_temperature();
        let limit = t.landauer_limit();

        // At room temperature, should be ~0.018 eV
        assert!((limit - 0.018).abs() < 0.001);
    }

    #[test]
    fn test_linear_schedule() {
        let t0 = Temperature::from_kelvin(1000.0).expect("Valid temperature");
        let tf = Temperature::from_kelvin(100.0).expect("Valid temperature");
        let schedule = TemperatureSchedule::new(t0, tf, 10, ScheduleType::Linear);

        let t_mid = schedule.temperature_at_step(5);
        assert!((t_mid.kelvin - 550.0).abs() < 1.0);
    }

    #[test]
    fn test_exponential_schedule() {
        let t0 = Temperature::from_kelvin(1000.0).expect("Valid temperature");
        let tf = Temperature::from_kelvin(100.0).expect("Valid temperature");
        let schedule = TemperatureSchedule::new(t0, tf, 10, ScheduleType::Exponential);

        let t_mid = schedule.temperature_at_step(5);
        let expected = 1000.0 * (0.1_f64.powf(0.5));
        assert!((t_mid.kelvin - expected).abs() < 1.0);
    }

    #[test]
    fn test_schedule_stepping() {
        let t0 = Temperature::from_kelvin(1000.0).expect("Valid temperature");
        let tf = Temperature::from_kelvin(100.0).expect("Valid temperature");
        let mut schedule = TemperatureSchedule::new(t0, tf, 10, ScheduleType::Linear);

        assert!(!schedule.is_complete());

        for _ in 0..10 {
            schedule.step();
        }

        assert!(schedule.is_complete());
        let final_t = schedule.current_temperature();
        assert!((final_t.kelvin - 100.0).abs() < 1.0);
    }
}
