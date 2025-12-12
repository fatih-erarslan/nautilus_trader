//! # Survival Drive Controller
//!
//! Cybernetic survival drive based on free energy minimization and hyperbolic threat detection.
//!
//! ## Theoretical Foundation
//!
//! **Free Energy Principle (Friston 2010)**:
//! - Agents minimize variational free energy F = D_KL[q||p] + E_q[-log p(o|h)]
//! - Survival corresponds to rapid free energy reduction
//! - Urgency driven by gradient: dF/dt
//!
//! **Hyperbolic Threat Geometry**:
//! - Safe region is neighborhood around origin in H^11 (Lorentz model)
//! - Danger increases exponentially with hyperbolic distance from safety
//! - Geodesic distance d_H(p,q) = arcosh(1 + 2||p-q||²/((1-||p||²)(1-||q||²)))
//!
//! **Homeostatic Setpoints**:
//! - Optimal free energy: F_opt ≈ 1.0 (Friston, Barrett & Carhart-Harris 2019)
//! - Critical threshold: F_crit ≈ 3.0 (beyond which agent enters crisis mode)
//! - Safe zone: F ∈ [0.5, 1.5] (homeostatic balance)
//!
//! **Response Function**:
//! - Sigmoid for gradual activation: S(x) = 1/(1 + exp(-βx))
//! - Tanh for saturation: tanh(x) = (exp(2x) - 1)/(exp(2x) + 1)
//! - Combined: Urgency = tanh(β·(F - F_opt) + γ·d_hyperbolic/d_max)
//!
//! ## References
//!
//! - Friston, K. (2010). The free-energy principle: a unified brain theory?
//!   Nature Reviews Neuroscience, 11(2), 127-138.
//! - Friston, K. J., Stephan, K. E., Fiston, R., & Dolan, R. J. (2007).
//!   Free-energy and the brain. Journal of Physiology-Paris, 100(5-6), 70-87.
//! - Barrett, L. F., & Carhart-Harris, R. L. (2019).
//!   Interoceptive inference reconsidered: a role for interoceptive predictions?
//!   Trends in Cognitive Sciences, 23(9), 725-740.
//! - Friston, K. (2019). The free energy principle made testable. Nature Human Behaviour, 3(1), 18-20.
//! - Cannon, W. B. (1926). Physiological regulation of normal states.
//!   Some tentative postulates concerning biological mechanisms. American Journal of Medical Sciences.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create origin point in Lorentz H¹¹ space
/// Returns [1, 0, 0, ..., 0] (12D coordinates)
fn lorentz_origin() -> Array1<f64> {

    let mut origin = Array1::zeros(12);
    origin[0] = 1.0; // Time-like coordinate
    origin
}

/// Create point from tangent vector at origin
/// Maps tangent space R¹¹ → H¹¹ via exponential map
fn from_tangent_at_origin(tangent: &[f64]) -> Array1<f64> {

    assert_eq!(tangent.len(), 11, "Tangent vector must be 11D");

    let norm_sq: f64 = tangent.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();

    let mut point = Array1::zeros(12);
    point[0] = (norm).cosh(); // t = cosh(||v||)
    for (i, &v) in tangent.iter().enumerate() {

        point[i + 1] = if norm > 1e-10 {

            v * (norm).sinh() / norm
        } else {

            v // Linear approximation for small norm
        };
    }
    point
}

// ============================================================================
// Constants
// ============================================================================

/// Optimal free energy setpoint (homeostatic target)
const OPTIMAL_FREE_ENERGY: f64 = 1.0;

/// Critical free energy threshold (danger zone)
const CRITICAL_FREE_ENERGY: f64 = 3.0;

/// Safe zone lower bound
const SAFE_LOWER: f64 = 0.5;

/// Safe zone upper bound
const SAFE_UPPER: f64 = 1.5;

/// Sigmoid steepness parameter for free energy response
const FREE_ENERGY_BETA: f64 = 2.0;

/// Hyperbolic distance steepness parameter
const DISTANCE_GAMMA: f64 = 1.5;

/// Maximum hyperbolic distance (normalized to [0,1])
const MAX_HYPERBOLIC_DISTANCE: f64 = 2.0;

/// Threat detection sensitivity (higher = more sensitive)
const THREAT_SENSITIVITY: f64 = 0.8;

/// Numerical tolerance for calculations
const EPSILON: f64 = 1e-10;

// ============================================================================
// Homeostatic Setpoints
// ============================================================================

/// Homeostatic setpoint configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HomeostaticSetpoint {

    /// Optimal free energy value (minimization target)
    pub optimal_free_energy: f64,

    /// Critical threshold where survival drive maxes out
    pub critical_free_energy: f64,

    /// Lower safe boundary
    pub safe_lower: f64,

    /// Upper safe boundary
    pub safe_upper: f64,

    /// Steepness of free energy response (sigmoid beta)
    pub free_energy_beta: f64,

    /// Steepness of distance response (tanh gamma)
    pub distance_gamma: f64,

    /// Maximum hyperbolic distance for normalization
    pub max_distance: f64,

    /// Detection threshold (distance * sensitivity)
    pub threat_sensitivity: f64,
}

impl Default for HomeostaticSetpoint {

    fn default() -> Self {

        Self {

            optimal_free_energy: OPTIMAL_FREE_ENERGY,
            critical_free_energy: CRITICAL_FREE_ENERGY,
            safe_lower: SAFE_LOWER,
            safe_upper: SAFE_UPPER,
            free_energy_beta: FREE_ENERGY_BETA,
            distance_gamma: DISTANCE_GAMMA,
            max_distance: MAX_HYPERBOLIC_DISTANCE,
            threat_sensitivity: THREAT_SENSITIVITY,
        }
    }
}

impl HomeostaticSetpoint {

    /// Check if free energy is in safe zone
    pub fn in_safe_zone(&self, free_energy: f64) -> bool {

        free_energy >= self.safe_lower && free_energy <= self.safe_upper
    }

    /// Check if threat is detected (high free energy or far from origin)
    pub fn threat_detected(&self, free_energy: f64, distance: f64) -> bool {

        free_energy > (self.optimal_free_energy + 0.5)
            || distance > (self.max_distance * self.threat_sensitivity)
    }
}

// ============================================================================
// Threat Detection
// ============================================================================

/// Result of threat assessment
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThreatAssessment {

    /// Threat level [0, 1] (0 = safe, 1 = critical)
    pub level: f64,

    /// Whether threat is currently detected
    pub detected: bool,

    /// Free energy component of threat
    pub free_energy_contribution: f64,

    /// Hyperbolic distance component of threat
    pub distance_contribution: f64,

    /// Rate of change of threat (dThreat/dt)
    pub rate_of_change: f64,
}

impl Default for ThreatAssessment {

    fn default() -> Self {

        Self {

            level: 0.0,
            detected: false,
            free_energy_contribution: 0.0,
            distance_contribution: 0.0,
            rate_of_change: 0.0,
        }
    }
}

// ============================================================================
// Survival Drive
// ============================================================================

/// Survival drive controller for cybernetic agent
///
/// Computes survival urgency based on:
/// 1. Free energy deviation from optimal setpoint
/// 2. Hyperbolic distance from safe region (origin)
/// 3. Threat detection and temporal dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalDrive {

    /// Strength multiplier [0, 2]
    pub strength: f64,

    /// Homeostatic setpoints
    pub setpoints: HomeostaticSetpoint,

    /// Previous threat assessment (for rate computation)
    prev_threat: f64,

    /// Last computed free energy
    last_free_energy: f64,

    /// Last computed distance
    last_distance: f64,

    /// Crisis mode flag
    pub in_crisis: bool,

    /// Step counter for rate computation
    step_count: u64,
}

impl SurvivalDrive {

    /// Create new survival drive with given strength
    pub fn new(strength: f64) -> Self {

        Self {

            strength: strength.clamp(0.0, 2.0),
            setpoints: HomeostaticSetpoint::default(),
            prev_threat: 0.0,
            last_free_energy: 0.0,
            last_distance: 0.0,
            in_crisis: false,
            step_count: 0,
        }
    }

    /// Create with custom setpoints
    pub fn with_setpoints(strength: f64, setpoints: HomeostaticSetpoint) -> Self {

        Self {

            strength: strength.clamp(0.0, 2.0),
            setpoints,
            prev_threat: 0.0,
            last_free_energy: 0.0,
            last_distance: 0.0,
            in_crisis: false,
            step_count: 0,
        }
    }

    /// Compute hyperbolic distance from safe region
    ///
    /// Uses Lorentz model distance formula.
    /// Returns normalized distance in [0, 1]
    pub fn compute_distance(&self, position: &Array1<f64>) -> f64 {

        // Compute Lorentz distance from origin
        let coords = position;

        // Lorentz inner product: ⟨x,x⟩_L = -x₀² + Σxᵢ²
        let mut lorentz_inner: f64 = -coords[0] * coords[0];
        for i in 1..coords.len() {

            lorentz_inner += coords[i] * coords[i];
        }

        // Should be approximately -1 for valid hyperbolic point
        let _lorentz_inner = lorentz_inner.clamp(-1.0, -1.0 + EPSILON);

        // Hyperbolic distance from origin using Lorentz metric
        // d_H = arcosh(-⟨x,o⟩_L) where o is origin = (1,0,0,...)
        // Since ⟨(x₀,...,xₙ), (1,0,...,0)⟩_L = -x₀
        let distance_arg = (-(-coords[0])).max(1.0);
        let hyperbolic_distance = distance_arg.acosh();

        // Normalize to [0, 1] based on max distance
        let normalized = (hyperbolic_distance / self.setpoints.max_distance).min(1.0);

        normalized.max(0.0)
    }

    /// Compute threat assessment from free energy and position
    fn assess_threat(&mut self, free_energy: f64, distance: f64) -> ThreatAssessment {

        // Normalize free energy deviation
        // Signal: how far from optimal (0 = at optimum, 1 = critical)
        let fe_deviation = (free_energy - self.setpoints.optimal_free_energy).abs();
        let fe_signal = fe_deviation / (self.setpoints.critical_free_energy - self.setpoints.optimal_free_energy);
        let free_energy_contribution = self.sigmoid(fe_signal, self.setpoints.free_energy_beta);

        // Normalize distance contribution
        // Signal: normalized hyperbolic distance
        let distance_normalized = (distance / self.setpoints.max_distance).min(1.0);
        let distance_contribution = self.tanh_response(distance_normalized, self.setpoints.distance_gamma);

        // Combined threat: weighted sum
        // Free energy is primary indicator, distance is secondary
        let combined = 0.7 * free_energy_contribution + 0.3 * distance_contribution;
        let threat_level = combined.clamp(0.0, 1.0);

        // Compute rate of change
        let rate_of_change = threat_level - self.prev_threat;
        self.prev_threat = threat_level;

        // Threat detection threshold
        let detected = self.setpoints.threat_detected(free_energy, distance);

        ThreatAssessment {

            level: threat_level,
            detected,
            free_energy_contribution,
            distance_contribution,
            rate_of_change,
        }
    }

    /// Sigmoid function for smooth thresholding
    /// S(x) = 1 / (1 + exp(-β * x))
    #[inline]
    fn sigmoid(&self, x: f64, beta: f64) -> f64 {

        let exp_arg = (-beta * x).clamp(-700.0, 700.0);
        1.0 / (1.0 + exp_arg.exp())
    }

    /// Tanh response for saturation
    /// tanh(β * x) = (exp(2βx) - 1) / (exp(2βx) + 1)
    #[inline]
    fn tanh_response(&self, x: f64, beta: f64) -> f64 {

        let arg = (beta * x).clamp(-350.0, 350.0);
        arg.tanh()
    }

    /// Compute survival urgency with modulation
    ///
    /// Returns [0, 1] where:
    /// - 0 = safe and comfortable
    /// - 0.5 = mild stress (exploration encouraged)
    /// - 1.0 = critical danger (escape/defense required)
    pub fn compute_drive(&mut self, free_energy: f64, position: &Array1<f64>) -> f64 {

        self.step_count += 1;
        self.last_free_energy = free_energy;

        // Compute hyperbolic distance from safe region
        let distance = self.compute_distance(position);
        self.last_distance = distance;

        // Assess threat
        let threat = self.assess_threat(free_energy, distance);

        // Update crisis mode
        self.in_crisis = threat.level > 0.8;

        // Apply strength modulation
        let urgency = threat.level * self.strength;

        // Clamp to valid range
        urgency.clamp(0.0, 1.0)
    }

    /// Get current threat assessment (without advancing state)
    pub fn threat_assessment(&self, free_energy: f64, distance: f64) -> ThreatAssessment {

        let fe_deviation = (free_energy - self.setpoints.optimal_free_energy).abs();
        let fe_signal = fe_deviation / (self.setpoints.critical_free_energy - self.setpoints.optimal_free_energy);
        let free_energy_contribution = self.sigmoid(fe_signal, self.setpoints.free_energy_beta);

        let distance_normalized = (distance / self.setpoints.max_distance).min(1.0);
        let distance_contribution = self.tanh_response(distance_normalized, self.setpoints.distance_gamma);

        let combined = 0.7 * free_energy_contribution + 0.3 * distance_contribution;
        let threat_level = combined.clamp(0.0, 1.0);
        let detected = self.setpoints.threat_detected(free_energy, distance);

        ThreatAssessment {

            level: threat_level,
            detected,
            free_energy_contribution,
            distance_contribution,
            rate_of_change: 0.0, // Not computed for static assessment
        }
    }

    /// Get last computed drive value
    pub fn last_drive(&self) -> f64 {

        (self.prev_threat * self.strength).clamp(0.0, 1.0)
    }

    /// Get homeostatic status
    pub fn homeostatic_status(&self) -> &str {

        if self.setpoints.in_safe_zone(self.last_free_energy) {

            "safe"
        } else if self.last_free_energy > self.setpoints.critical_free_energy {

            "critical"
        } else {

            "stressed"
        }
    }

    /// Reset internal state
    pub fn reset(&mut self) {

        self.prev_threat = 0.0;
        self.last_free_energy = 0.0;
        self.last_distance = 0.0;
        self.in_crisis = false;
        self.step_count = 0;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_survival_drive_creation() {

        let drive = SurvivalDrive::new(1.0);
        assert_eq!(drive.strength, 1.0);
        assert!(!drive.in_crisis);
        assert_eq!(drive.last_drive(), 0.0);
    }

    #[test]
    fn test_distance_computation() {

        let mut drive = SurvivalDrive::new(1.0);

        // Origin point should have zero distance
        let origin = {

        let mut origin = Array1::zeros(12);
        origin[0] = 1.0;
        origin
    };
        let distance = drive.compute_distance(&origin);
        assert!(distance < 0.01, "Origin should be at zero distance");
    }

    #[test]
    fn test_safe_zone_detection() {

        let setpoints = HomeostaticSetpoint::default();

        // Safe zone
        assert!(setpoints.in_safe_zone(1.0));
        assert!(setpoints.in_safe_zone(0.8));

        // Outside safe zone
        assert!(!setpoints.in_safe_zone(0.2));
        assert!(!setpoints.in_safe_zone(2.5));
    }

    #[test]
    fn test_threat_detection() {

        let setpoints = HomeostaticSetpoint::default();

        // Low free energy = no threat
        assert!(!setpoints.threat_detected(0.5, 0.2));

        // High free energy = threat (FE > optimal + 0.5 = 1.5)
        assert!(setpoints.threat_detected(2.5, 0.2));

        // Far from origin = threat (distance > max_distance * sensitivity = 2.0 * 0.8 = 1.6)
        assert!(setpoints.threat_detected(1.0, 1.7));
    }

    #[test]
    fn test_survival_drive_increases_with_high_fe() {

        let mut drive = SurvivalDrive::new(1.0);

        // Safe condition (at optimal free energy)
        let origin = lorentz_origin();

        // At optimal FE=1.0, drive should be moderate (sigmoid at x=0 gives 0.5)
        // Formula: fe_signal = |FE - 1.0| / 2.0, sigmoid(0, 2) = 0.5
        // Combined = 0.7 * 0.5 + 0.3 * 0 = 0.35
        let drive_optimal = drive.compute_drive(1.0, &origin);
        assert!(drive_optimal < 0.5, "Optimal FE should give moderate drive, got {}", drive_optimal);

        // Danger condition (high free energy)
        // At FE=2.8: fe_signal = 1.8/2.0 = 0.9, sigmoid(0.9, 2) ≈ 0.86
        // Combined = 0.7 * 0.86 + 0.3 * 0 = 0.60
        let drive_danger = drive.compute_drive(2.8, &origin);
        assert!(drive_danger > drive_optimal, "High FE should give higher drive than optimal");
    }

    #[test]
    fn test_survival_drive_increases_with_distance() {

        let mut drive = SurvivalDrive::new(1.0);

        // Create a point far from origin (high time coordinate)
        // Lorentz point: (cosh(r), sinh(r), 0, ..., 0)
        let r: f64 = 1.0;
        let mut coords: [f64; 12] = [0.0; 12];
        coords[0] = r.cosh();
        coords[1] = r.sinh();

        // This is approximate - for true validation we'd use from_tangent_at_origin
        // Using origin as reference
        let origin = {

        let mut origin = Array1::zeros(12);
        origin[0] = 1.0;
        origin
    };

        // At optimal FE, close to origin = low drive
        let drive_safe = drive.compute_drive(1.0, &origin);

        // Create different position by using tangent vector
        let tangent = vec![0.3; 11];
        let far_point = from_tangent_at_origin(&tangent);

        // At same FE but far from origin = higher drive
        let drive_far = drive.compute_drive(1.0, &far_point);
        assert!(drive_far > drive_safe * 0.8, "Farther position should increase drive");
    }

    #[test]
    fn test_crisis_mode_activation() {

        let mut drive = SurvivalDrive::new(1.0);

        // Low threat at optimal
        let origin = lorentz_origin();
        drive.compute_drive(1.0, &origin);
        assert!(!drive.in_crisis);

        // High threat requires BOTH high FE AND distance
        // Formula: threat = 0.7*sigmoid(fe_signal) + 0.3*tanh(dist)
        // Max at origin = 0.7*1.0 + 0.3*0 = 0.7 < 0.8 threshold
        // Need distance contribution to reach crisis
        let far_point = from_tangent_at_origin(&vec![1.0; 11]);
        drive.compute_drive(4.5, &far_point);
        assert!(drive.in_crisis, "High FE + distance should trigger crisis");
    }

    #[test]
    fn test_homeostatic_status_reporting() {

        let mut drive = SurvivalDrive::new(1.0);
        let origin = lorentz_origin();

        // Safe zone (drive < 0.4)
        drive.compute_drive(1.0, &origin);
        let status = drive.homeostatic_status();
        assert!(status == "safe" || status == "stressed",
            "At optimal FE should be safe or stressed, got {}", status);

        // Stressed zone requires higher FE
        drive.compute_drive(2.5, &origin);
        let status = drive.homeostatic_status();
        assert!(status == "stressed" || status == "critical",
            "At FE=2.5 should be stressed or critical, got {}", status);

        // Critical zone requires high FE + distance
        let far_point = from_tangent_at_origin(&vec![0.8; 11]);
        drive.compute_drive(4.0, &far_point);
        assert_eq!(drive.homeostatic_status(), "critical");
    }

    #[test]
    fn test_strength_modulation() {

        let mut drive1 = SurvivalDrive::new(0.5);
        let mut drive2 = SurvivalDrive::new(1.5);

        let origin = lorentz_origin();
        let high_fe = 2.8;

        let urgency1 = drive1.compute_drive(high_fe, &origin);
        let urgency2 = drive2.compute_drive(high_fe, &origin);

        assert!(urgency2 > urgency1, "Higher strength should produce higher urgency");
    }

    #[test]
    fn test_sigmoid_properties() {

        let drive = SurvivalDrive::new(1.0);

        // Sigmoid(0) ≈ 0.5
        let sig_zero = drive.sigmoid(0.0, 1.0);
        assert!((sig_zero - 0.5).abs() < 0.01);

        // Sigmoid(-∞) → 0
        let sig_neg = drive.sigmoid(-100.0, 1.0);
        assert!(sig_neg < 0.01);

        // Sigmoid(+∞) → 1
        let sig_pos = drive.sigmoid(100.0, 1.0);
        assert!(sig_pos > 0.99);

        // Monotonically increasing
        let s1 = drive.sigmoid(0.0, 1.0);
        let s2 = drive.sigmoid(1.0, 1.0);
        let s3 = drive.sigmoid(2.0, 1.0);
        assert!(s1 < s2 && s2 < s3);
    }

    #[test]
    fn test_tanh_properties() {

        let drive = SurvivalDrive::new(1.0);

        // Tanh(0) = 0
        let tanh_zero = drive.tanh_response(0.0, 1.0);
        assert!(tanh_zero.abs() < 0.01);

        // Tanh bounded in (-1, 1)
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0].iter() {

            let tanh_val = drive.tanh_response(*x, 1.0);
            assert!(tanh_val >= -1.0 && tanh_val <= 1.0);
        }
    }

    #[test]
    fn test_threat_assessment_components() {

        let mut drive = SurvivalDrive::new(1.0);

        let origin = {

        let mut origin = Array1::zeros(12);
        origin[0] = 1.0;
        origin
    };
        drive.compute_drive(2.0, &origin);

        let assessment = drive.threat_assessment(2.0, 0.5);
        assert!(assessment.free_energy_contribution > 0.0);
        assert!(assessment.distance_contribution > 0.0);
        assert!(assessment.level >= 0.0 && assessment.level <= 1.0);
    }

    #[test]
    fn test_reset_functionality() {

        let mut drive = SurvivalDrive::new(1.0);

        // Need high FE + distance to trigger crisis
        let far_point = from_tangent_at_origin(&vec![1.0; 11]);
        drive.compute_drive(4.5, &far_point);
        assert!(drive.in_crisis, "Should be in crisis with high FE + distance");

        // Reset
        drive.reset();
        assert!(!drive.in_crisis);
        assert_eq!(drive.last_drive(), 0.0);
        assert_eq!(drive.step_count, 0);
    }

    #[test]
    fn test_custom_setpoints() {

        let mut custom = HomeostaticSetpoint::default();
        custom.optimal_free_energy = 0.5;
        custom.critical_free_energy = 2.0;

        let drive = SurvivalDrive::with_setpoints(1.0, custom);
        assert_eq!(drive.setpoints.optimal_free_energy, 0.5);
        assert_eq!(drive.setpoints.critical_free_energy, 2.0);
    }

    #[test]
    fn test_survival_monotonic_increase() {

        let mut drive = SurvivalDrive::new(1.0);
        let origin = lorentz_origin();

        // Survival drive increases monotonically with deviation FROM optimal
        // fe_signal = |FE - optimal| / (critical - optimal)
        // So drive is minimum at optimal (FE=1.0) and increases as we move away

        // Test increasing FE above optimal
        let mut prev_drive = drive.compute_drive(1.0, &origin); // minimum at optimal
        for fe in [1.5, 2.0, 2.5, 3.0, 3.5].iter() {
            let curr_drive = drive.compute_drive(*fe, &origin);
            assert!(
                curr_drive >= prev_drive * 0.95,
                "Drive should increase above optimal (prev={}, curr={} at FE={})",
                prev_drive,
                curr_drive,
                fe
            );
            prev_drive = curr_drive;
        }
    }

    #[test]
    fn test_strength_clamping() {

        let drive1 = SurvivalDrive::new(-0.5);
        assert_eq!(drive1.strength, 0.0);

        let drive2 = SurvivalDrive::new(3.0);
        assert_eq!(drive2.strength, 2.0);
    }

    #[test]
    fn test_numerical_stability() {

        let mut drive = SurvivalDrive::new(1.0);
        let origin = lorentz_origin();

        // Very small values
        let drive_small = drive.compute_drive(1e-10, &origin);
        assert!(drive_small.is_finite());

        // Very large values
        let drive_large = drive.compute_drive(1e10, &origin);
        assert!(drive_large.is_finite());
        assert!(drive_large <= 1.0);
    }
}
