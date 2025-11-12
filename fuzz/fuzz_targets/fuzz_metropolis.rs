#![no_main]

use libfuzzer_sys::fuzz_target;
use hyperphysics_pbit::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    let temperature = f64::from_le_bytes(data[0..8].try_into().unwrap());
    let steps = usize::from_le_bytes(data[8..16].try_into().unwrap());

    // Validate inputs
    if temperature.is_nan() || temperature <= 0.0 || temperature > 1e6 || steps == 0 || steps > 10000 {
        return;
    }

    // Try to create lattice and run Metropolis
    if let Ok(lattice) = PBitLattice::roi_48(1.0) {
        let mut sim = match MetropolisSimulator::new(lattice, temperature) {
            Ok(s) => s,
            Err(_) => return,
        };

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Run simulation
        for _ in 0..steps.min(1000) {
            let _ = sim.step(&mut rng);
        }

        // Verify invariants
        let energy = sim.total_energy();
        assert!(energy.is_finite(), "Non-finite energy detected");

        let magnetization = sim.lattice().magnetization();
        assert!(magnetization.is_finite(), "Non-finite magnetization detected");
    }
});
