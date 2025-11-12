#![no_main]

use libfuzzer_sys::fuzz_target;
use hyperphysics_pbit::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fuzz_target!(|data: &[u8]| {
    // Need at least 16 bytes for temperature and steps
    if data.len() < 16 {
        return;
    }

    // Extract fuzzing parameters
    let temperature = f64::from_le_bytes(data[0..8].try_into().unwrap());
    let steps = usize::from_le_bytes(data[8..16].try_into().unwrap());

    // Validate inputs
    if temperature.is_nan() || temperature <= 0.0 || temperature > 1e6 || steps == 0 || steps > 10000 {
        return;
    }

    // Try to create lattice and simulate
    if let Ok(lattice) = PBitLattice::roi_48(1.0) {
        let mut sim = match GillespieSimulator::new(lattice, temperature) {
            Ok(s) => s,
            Err(_) => return,
        };

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Run simulation - should never panic
        let _ = sim.simulate(steps.min(1000), &mut rng);

        // Verify invariants after simulation
        assert!(sim.total_rate() >= 0.0, "Negative transition rate detected");
        assert!(sim.current_time().is_finite(), "Non-finite time detected");
    }
});
