//! # Systems Dynamics Tracker Demo
//!
//! Demonstrates the comprehensive agency dynamics tracking system with:
//! - Real-time state recording
//! - Criticality metrics (branching ratio)
//! - Spectral analysis
//! - Temporal statistics
//! - CSV/JSON export
//! - Emergence detection

use hyperphysics_agency::{
    AgencyDynamics, AgentState, TemporalStats, CriticalitySummary, DynamicsStats,
};
use ndarray::Array1;
use std::f64::consts::PI;

/// Create an AgentState for testing
fn create_state(phi: f64, free_energy: f64, control: f64, survival: f64, model_accuracy: f64) -> AgentState {
    let mut position = Array1::zeros(12);
    position[0] = 1.0; // Lorentz origin (timelike component)

    AgentState {
        beliefs: Array1::from_elem(32, 0.5),
        phi,
        free_energy,
        survival,
        control,
        model_accuracy,
        precision: Array1::from_elem(32, 1.0),
        position,
        prediction_errors: std::collections::VecDeque::new(),
    }
}

fn main() {
    println!("=============================================================================");
    println!("          SYSTEMS DYNAMICS TRACKER - AGENCY EMERGENCE ANALYSIS");
    println!("=============================================================================\n");

    // ========================================================================
    // Demo 1: Basic State Recording and History
    // ========================================================================

    println!("Demo 1: Basic State Recording");
    println!("-----");

    let mut dynamics = AgencyDynamics::new();

    // Record 50 states
    for i in 0..50 {
        let phi = 2.0 + (i as f64 / 50.0) * 3.0;
        let free_energy = 2.5 - (i as f64 / 50.0) * 1.5;
        let control = 0.2 + (i as f64 / 50.0) * 0.6;
        let survival = 0.5;
        let model_accuracy = 0.5 + (i as f64 / 50.0) * 0.3;

        let state = create_state(phi, free_energy, control, survival, model_accuracy);
        dynamics.record_state(&state);
    }

    println!("Recorded {} state snapshots", dynamics.len());
    println!("History capacity: {}\n", dynamics.len());

    // ========================================================================
    // Demo 2: Temporal Statistics
    // ========================================================================

    println!("Demo 2: Temporal Statistics");
    println!("-----");

    let phi_series = dynamics.get_series("phi");
    let control_series = dynamics.get_series("control");
    let fe_series = dynamics.get_series("free_energy");

    let phi_stats = TemporalStats::from_series(&phi_series);
    let control_stats = TemporalStats::from_series(&control_series);
    let fe_stats = TemporalStats::from_series(&fe_series);

    println!("Integrated Information (Phi):");
    println!("  Mean: {:.4}", phi_stats.mean);
    println!("  Std Dev: {:.4}", phi_stats.std);
    println!("  Min: {:.4}, Max: {:.4}", phi_stats.min, phi_stats.max);
    println!("  Autocorrelation (lag-1): {:.4}", phi_stats.autocorr_lag1);
    println!("  Volatility: {:.4}", phi_stats.volatility);
    println!("  Skewness: {:.4}", phi_stats.skewness);
    println!("  Kurtosis: {:.4}\n", phi_stats.kurtosis);

    println!("Control Authority:");
    println!("  Mean: {:.4}", control_stats.mean);
    println!("  Std Dev: {:.4}", control_stats.std);
    println!("  Autocorr (lag-1): {:.4}", control_stats.autocorr_lag1);
    println!("  Volatility: {:.4}\n", control_stats.volatility);

    println!("Free Energy:");
    println!("  Mean: {:.4}", fe_stats.mean);
    println!("  Std Dev: {:.4}", fe_stats.std);
    println!("  Volatility: {:.4}\n", fe_stats.volatility);

    // ========================================================================
    // Demo 3: Criticality Metrics
    // ========================================================================

    println!("Demo 3: Criticality Analysis");
    println!("-----");

    // Create longer trajectory for criticality analysis
    let mut dynamics2 = AgencyDynamics::with_capacity(1000);

    for t in 0..600 {
        let normalized_t = t as f64 / 600.0;

        // Smoothly increase toward criticality
        let control = 0.1 + 0.7 * normalized_t;
        let phi = 2.0 + 4.0 * normalized_t;
        let free_energy = 3.0 * (1.0 - 0.5 * normalized_t);

        // Add oscillations
        let freq = 2.0 * PI * 0.02;
        let phi_osc = phi + 0.5 * (freq * t as f64).sin();

        let state = create_state(phi_osc, free_energy, control, 0.6, 0.7);
        dynamics2.record_state(&state);
    }

    let criticality: CriticalitySummary = dynamics2.compute_criticality();

    println!("Branching Ratio (sigma):");
    if let Some(br) = criticality.branching_ratio {
        println!("  sigma = {:.4} (critical point at sigma ~ 1.0)", br);
        let criticality_distance = (br - 1.0).abs();
        if criticality_distance < 0.1 {
            println!("  -> System is NEAR CRITICALITY");
        } else if br < 1.0 {
            println!("  -> System is SUB-CRITICAL (subcritical cascades)");
        } else {
            println!("  -> System is SUPER-CRITICAL (runaway cascades)");
        }
    }

    println!("\nOther Criticality Metrics:");
    println!("  Lyapunov Exponent: {:?}", criticality.lyapunov_exponent);
    println!("  Hurst Exponent: {:?}", criticality.hurst_exponent);
    println!("  Entropy Rate: {:?}\n", criticality.entropy_rate);

    // ========================================================================
    // Demo 4: Spectral Analysis
    // ========================================================================

    println!("Demo 4: Spectral Analysis");
    println!("-----");

    // Create signal with clear oscillations
    let mut dynamics3 = AgencyDynamics::with_capacity(512);

    for t in 0..512 {
        let t_norm = t as f64 / 512.0;

        // Alpha oscillation
        let alpha = 2.0 * (2.0 * PI * 0.1 * t_norm).sin();

        // Theta modulation
        let theta = 1.0 * (2.0 * PI * 0.05 * t_norm).sin();

        let phi = 3.0 + alpha + theta;

        let state = create_state(phi, 1.5, 0.6, 0.5, 0.7);
        dynamics3.record_state(&state);
    }

    let spectral = dynamics3.analyze_spectral();

    println!("Power Spectral Density (PSD) Analysis:");
    if let Some(freq) = spectral.peak_frequency {
        println!("  Peak Frequency (normalized): {:.4}", freq);
    }
    if let Some(power) = spectral.peak_power {
        println!("  Peak Power: {:.4}", power);
    }
    if let Some(entropy) = spectral.spectral_entropy {
        println!("  Spectral Entropy: {:.4}", entropy);
    }
    println!("  Harmonics: {:?}\n", spectral.harmonics);

    // ========================================================================
    // Demo 5: Export Functionality
    // ========================================================================

    println!("Demo 5: Data Export");
    println!("-----");

    // Create sample data
    let mut export_dynamics = AgencyDynamics::new();
    for i in 0..10 {
        let phi = 2.0 + (i as f64 / 10.0);
        let fe = 2.0 - (i as f64 / 20.0);
        let state = create_state(phi, fe, 0.5, 0.6, 0.7);
        export_dynamics.record_state(&state);
    }

    // CSV Export
    let csv = export_dynamics.export_csv();
    let lines: Vec<&str> = csv.lines().collect();
    println!("CSV Export (first 3 rows):");
    for (i, line) in lines.iter().take(3).enumerate() {
        if i == 0 {
            println!("  Header: {}", line);
        } else {
            println!("  Data[{}]: {}", i - 1, line);
        }
    }
    println!("  ... ({} total rows)\n", lines.len());

    // JSON Export
    match export_dynamics.export_json() {
        Ok(json) => {
            println!("JSON Export: {} bytes", json.len());
            println!("  First 100 chars: {}\n", &json.chars().take(100).collect::<String>());
        }
        Err(e) => println!("JSON export error: {}\n", e),
    }

    // ========================================================================
    // Demo 6: Emergence Indicators
    // ========================================================================

    println!("Demo 6: Agency Emergence Indicators");
    println!("-----");

    let stats: Option<DynamicsStats> = dynamics.get_stats();
    if let Some(s) = stats {
        let emergence = s.emergence_indicator();
        let robustness = s.robustness_score();

        println!("Emergence Indicator: {:.4}", emergence);
        println!("  Range: [0, 1] (0 = no agency, 1 = full agency)");
        if emergence < 0.3 {
            println!("  Status: Minimal emergence");
        } else if emergence < 0.6 {
            println!("  Status: Moderate emergence");
        } else {
            println!("  Status: Strong emergence");
        }

        println!("\nRobustness Score: {:.4}", robustness);
        println!("  Range: [0, 1] (0 = fragile, 1 = robust)");
        if robustness < 0.3 {
            println!("  Status: Fragile system");
        } else if robustness < 0.7 {
            println!("  Status: Moderately robust");
        } else {
            println!("  Status: Highly robust");
        }
    }

    // ========================================================================
    // Demo 7: Time Series Analysis
    // ========================================================================

    println!("\n\nDemo 7: Advanced Time Series Analysis");
    println!("-----");

    // Analyze autocorrelation patterns
    println!("Autocorrelation Lag-1 Analysis (temporal persistence):");
    println!("  Phi autocorr:       {:.4} (persistence of consciousness)", phi_stats.autocorr_lag1);
    println!("  Control autocorr: {:.4} (persistence of agency)", control_stats.autocorr_lag1);
    println!("  FE autocorr:      {:.4} (persistence of surprise)\n", fe_stats.autocorr_lag1);

    println!("Interpretation:");
    if phi_stats.autocorr_lag1 > 0.7 {
        println!("  -> Consciousness shows high temporal coherence");
    }
    if control_stats.autocorr_lag1 > 0.6 {
        println!("  -> Control authority is persistent and directed");
    }
    if fe_stats.volatility < 0.5 {
        println!("  -> Free energy is stabilizing");
    }

    // ========================================================================
    // Summary
    // ========================================================================

    println!("\n=============================================================================");
    println!("                           ANALYSIS SUMMARY");
    println!("=============================================================================");
    println!("Total snapshots analyzed: {}", dynamics.len());
    println!("\nKey Metrics:");
    println!("  * Consciousness (Phi): {:.3} +/- {:.3}", phi_stats.mean, phi_stats.std);
    println!("  * Control: {:.3} +/- {:.3}", control_stats.mean, control_stats.std);
    println!("  * Free Energy: {:.3} +/- {:.3}", fe_stats.mean, fe_stats.std);

    if let Some(s) = dynamics.get_stats() {
        println!("\nEmergence Metrics:");
        println!("  * Emergence Indicator: {:.3}", s.emergence_indicator());
        println!("  * Robustness Score: {:.3}", s.robustness_score());
    }

    println!("\nConclusion:");
    println!("The systems dynamics tracker successfully captures:");
    println!("  * Real-time state evolution");
    println!("  * Criticality detection via branching ratio");
    println!("  * Spectral properties of consciousness oscillations");
    println!("  * Emergence of agency through temporal integration");
    println!("  * Data export for visualization and further analysis");
    println!("\n");
}
