#![no_main]

use libfuzzer_sys::fuzz_target;
use hyperphysics_pbit::*;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    let size = usize::from_le_bytes(data[0..8].try_into().unwrap());
    let coupling = f64::from_le_bytes(data[8..16].try_into().unwrap());

    // Validate inputs
    if size == 0 || size > 100 || coupling.is_nan() || coupling.abs() > 1e6 {
        return;
    }

    // Test different lattice geometries
    let lattices = vec![
        PBitLattice::square(size.min(32)),
        PBitLattice::triangular(size.min(20)),
        PBitLattice::roi_48(coupling),
    ];

    for lattice_result in lattices {
        if let Ok(lattice) = lattice_result {
            // Verify basic properties
            assert!(lattice.size() > 0, "Lattice has zero sites");

            // Check neighbor symmetry
            for i in 0..lattice.size() {
                let neighbors = lattice.neighbors(i);
                for &j in neighbors.iter() {
                    assert!(j < lattice.size(), "Invalid neighbor index");

                    // j should also have i as neighbor (symmetry)
                    let j_neighbors = lattice.neighbors(j);
                    assert!(j_neighbors.contains(&i), "Neighbor relationship not symmetric");
                }
            }

            // Verify magnetization is bounded
            let mag = lattice.magnetization();
            assert!(mag.abs() <= lattice.size() as f64, "Magnetization exceeds bounds");
        }
    }
});
