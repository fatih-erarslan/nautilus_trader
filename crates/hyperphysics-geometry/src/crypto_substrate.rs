//! Cryptographic Substrate Integration
//!
//! Integrates the {7,3} hyperbolic tessellation with Dilithium post-quantum cryptography
//! to create a secure, verifiable cryptographic substrate for consciousness metrics.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │   DILITHIUM-CRYSTAL LATTICE CRYPTOGRAPHY            │
//! ├─────────────────────────────────────────────────────┤
//! │                                                     │
//! │  {7,3} Tessellation    ←→    Dilithium Crypto     │
//! │  ─────────────────          ──────────────────     │
//! │  • Heptagonal tiles          • Per-tile keypairs  │
//! │  • Hyperbolic geometry       • Post-quantum sigs  │
//! │  • Fuchsian symmetry         • ZK proofs          │
//! │  • 3-tiles-per-vertex        • Secure channels   │
//! │                                                     │
//! │  pBit Lattice:                                     │
//! │  • Map pBits → Tiles                               │
//! │  • Sign state updates                              │
//! │  • Verify neighborhoods                            │
//! │  • Secure consciousness metrics                    │
//! │                                                     │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # References
//!
//! - pbRTCA v3.1: "Dilithium-Crystal Lattice Cryptography"
//! - NIST FIPS 204: Module-Lattice-Based Digital Signature Standard

use crate::{
    tessellation_73::{HeptagonalTessellation, TileId},
    poincare::PoincarePoint,
    GeometryError,
    Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cryptographic substrate mapping pBits to heptagonal tiles
///
/// Each tile in the {7,3} tessellation can hold cryptographic state
/// and be signed with Dilithium post-quantum signatures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoSubstrate {
    /// Underlying {7,3} tessellation
    tessellation: HeptagonalTessellation,

    /// Map from pBit indices to tile IDs
    pbit_to_tile: HashMap<usize, TileId>,

    /// Map from tile IDs to assigned pBit groups
    tile_to_pbits: HashMap<TileId, Vec<usize>>,

    /// Cryptographic state per tile
    tile_crypto_state: HashMap<TileId, TileCryptoState>,

    /// Maximum pBits per tile (7 for heptagonal)
    pbits_per_tile: usize,
}

/// Cryptographic state associated with a tile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileCryptoState {
    /// Tile identifier
    pub tile_id: TileId,

    /// pBit indices assigned to this tile
    pub pbits: Vec<usize>,

    /// Position in hyperbolic space
    pub position: PoincarePoint,

    /// Dilithium keypair identifier (references external key storage)
    pub keypair_id: Option<String>,

    /// Last signature timestamp
    pub last_signed: Option<u64>,

    /// Signature generation counter
    pub signature_count: u64,

    /// Neighbors for verification purposes
    pub neighbor_tiles: Vec<TileId>,
}

impl CryptoSubstrate {
    /// Create new cryptographic substrate from {7,3} tessellation
    ///
    /// # Arguments
    ///
    /// * `tessellation` - The underlying {7,3} hyperbolic tessellation
    /// * `total_pbits` - Total number of pBits to map onto tessellation
    ///
    /// # Examples
    ///
    /// ```
    /// use hyperphysics_geometry::tessellation_73::HeptagonalTessellation;
    /// use hyperphysics_geometry::crypto_substrate::CryptoSubstrate;
    ///
    /// let tessellation = HeptagonalTessellation::new(2)?;
    /// let substrate = CryptoSubstrate::new(tessellation, 48)?;
    ///
    /// assert_eq!(substrate.num_tiles(), substrate.tessellation().num_tiles());
    /// # Ok::<(), hyperphysics_geometry::GeometryError>(())
    /// ```
    pub fn new(tessellation: HeptagonalTessellation, total_pbits: usize) -> Result<Self> {
        let num_tiles = tessellation.num_tiles();
        let pbits_per_tile = (total_pbits + num_tiles - 1) / num_tiles; // Ceiling division

        let mut substrate = Self {
            tessellation,
            pbit_to_tile: HashMap::new(),
            tile_to_pbits: HashMap::new(),
            tile_crypto_state: HashMap::new(),
            pbits_per_tile,
        };

        // Distribute pBits across tiles
        substrate.distribute_pbits(total_pbits)?;

        // Initialize crypto state for each tile
        substrate.initialize_crypto_states()?;

        Ok(substrate)
    }

    /// Distribute pBits evenly across tiles
    fn distribute_pbits(&mut self, total_pbits: usize) -> Result<()> {
        let tiles = self.tessellation.tiles();

        for (pbit_idx, tile_idx) in (0..total_pbits).zip((0..tiles.len()).cycle()) {
            let tile_id = TileId(tile_idx);

            // Map pBit to tile
            self.pbit_to_tile.insert(pbit_idx, tile_id);

            // Track pBits assigned to this tile
            self.tile_to_pbits
                .entry(tile_id)
                .or_insert_with(Vec::new)
                .push(pbit_idx);
        }

        Ok(())
    }

    /// Initialize cryptographic state for all tiles
    fn initialize_crypto_states(&mut self) -> Result<()> {
        for tile in self.tessellation.tiles() {
            let pbits = self.tile_to_pbits
                .get(&tile.id)
                .cloned()
                .unwrap_or_default();

            let neighbor_tiles: Vec<TileId> = tile.neighbors
                .iter()
                .filter_map(|opt_id| *opt_id)
                .collect();

            let crypto_state = TileCryptoState {
                tile_id: tile.id,
                pbits,
                position: tile.center,
                keypair_id: None, // Will be assigned when Dilithium keypairs are generated
                last_signed: None,
                signature_count: 0,
                neighbor_tiles,
            };

            self.tile_crypto_state.insert(tile.id, crypto_state);
        }

        Ok(())
    }

    /// Get tile ID for a given pBit
    ///
    /// # Arguments
    ///
    /// * `pbit_idx` - Index of the pBit
    ///
    /// # Returns
    ///
    /// The TileId containing this pBit, or None if not mapped
    pub fn get_tile_for_pbit(&self, pbit_idx: usize) -> Option<TileId> {
        self.pbit_to_tile.get(&pbit_idx).copied()
    }

    /// Get all pBits assigned to a tile
    ///
    /// # Arguments
    ///
    /// * `tile_id` - The tile identifier
    ///
    /// # Returns
    ///
    /// Vector of pBit indices assigned to this tile
    pub fn get_pbits_for_tile(&self, tile_id: TileId) -> Vec<usize> {
        self.tile_to_pbits
            .get(&tile_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get cryptographic state for a tile
    pub fn get_crypto_state(&self, tile_id: TileId) -> Option<&TileCryptoState> {
        self.tile_crypto_state.get(&tile_id)
    }

    /// Get mutable cryptographic state for a tile
    pub fn get_crypto_state_mut(&mut self, tile_id: TileId) -> Option<&mut TileCryptoState> {
        self.tile_crypto_state.get_mut(&tile_id)
    }

    /// Assign a Dilithium keypair to a tile
    ///
    /// # Arguments
    ///
    /// * `tile_id` - The tile to assign a keypair to
    /// * `keypair_id` - Identifier for the Dilithium keypair (external storage)
    pub fn assign_keypair(&mut self, tile_id: TileId, keypair_id: String) -> Result<()> {
        if let Some(state) = self.tile_crypto_state.get_mut(&tile_id) {
            state.keypair_id = Some(keypair_id);
            Ok(())
        } else {
            Err(GeometryError::InvalidTessellation {
                message: format!("Tile {:?} not found", tile_id),
            })
        }
    }

    /// Record that a tile was signed (for tracking purposes)
    ///
    /// # Arguments
    ///
    /// * `tile_id` - The tile that was signed
    /// * `timestamp` - Unix timestamp of signature
    pub fn record_signature(&mut self, tile_id: TileId, timestamp: u64) -> Result<()> {
        if let Some(state) = self.tile_crypto_state.get_mut(&tile_id) {
            state.last_signed = Some(timestamp);
            state.signature_count += 1;
            Ok(())
        } else {
            Err(GeometryError::InvalidTessellation {
                message: format!("Tile {:?} not found", tile_id),
            })
        }
    }

    /// Get neighbors of a tile for neighborhood verification
    ///
    /// # Arguments
    ///
    /// * `tile_id` - The tile whose neighbors to retrieve
    ///
    /// # Returns
    ///
    /// Vector of neighboring tile IDs
    pub fn get_tile_neighbors(&self, tile_id: TileId) -> Vec<TileId> {
        self.tile_crypto_state
            .get(&tile_id)
            .map(|state| state.neighbor_tiles.clone())
            .unwrap_or_default()
    }

    /// Get the underlying tessellation
    pub fn tessellation(&self) -> &HeptagonalTessellation {
        &self.tessellation
    }

    /// Get number of tiles
    pub fn num_tiles(&self) -> usize {
        self.tessellation.num_tiles()
    }

    /// Get number of pBits mapped
    pub fn num_pbits(&self) -> usize {
        self.pbit_to_tile.len()
    }

    /// Get statistics about the substrate
    pub fn stats(&self) -> SubstrateStats {
        let tiles_with_keypairs = self.tile_crypto_state
            .values()
            .filter(|s| s.keypair_id.is_some())
            .count();

        let total_signatures: u64 = self.tile_crypto_state
            .values()
            .map(|s| s.signature_count)
            .sum();

        let avg_pbits_per_tile = if self.num_tiles() > 0 {
            self.num_pbits() as f64 / self.num_tiles() as f64
        } else {
            0.0
        };

        SubstrateStats {
            total_tiles: self.num_tiles(),
            total_pbits: self.num_pbits(),
            tiles_with_keypairs,
            total_signatures,
            avg_pbits_per_tile,
        }
    }
}

/// Statistics about the cryptographic substrate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateStats {
    /// Total number of tiles
    pub total_tiles: usize,

    /// Total number of pBits mapped
    pub total_pbits: usize,

    /// Number of tiles with assigned keypairs
    pub tiles_with_keypairs: usize,

    /// Total signatures across all tiles
    pub total_signatures: u64,

    /// Average pBits per tile
    pub avg_pbits_per_tile: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tessellation_73::HeptagonalTessellation;

    #[test]
    fn test_create_substrate() -> Result<()> {
        let tess = HeptagonalTessellation::new(0)?;
        let substrate = CryptoSubstrate::new(tess, 48)?;

        assert_eq!(substrate.num_pbits(), 48);
        assert_eq!(substrate.num_tiles(), 1); // Central tile only

        Ok(())
    }

    #[test]
    fn test_pbit_to_tile_mapping() -> Result<()> {
        let tess = HeptagonalTessellation::new(0)?;
        let substrate = CryptoSubstrate::new(tess, 48)?;

        // All pBits should be mapped to the central tile
        for pbit_idx in 0..48 {
            let tile_id = substrate.get_tile_for_pbit(pbit_idx);
            assert!(tile_id.is_some());
            assert_eq!(tile_id.unwrap(), TileId(0));
        }

        Ok(())
    }

    #[test]
    fn test_tile_crypto_state() -> Result<()> {
        let tess = HeptagonalTessellation::new(0)?;
        let substrate = CryptoSubstrate::new(tess, 7)?;

        let state = substrate.get_crypto_state(TileId(0));
        assert!(state.is_some());

        let state = state.unwrap();
        assert_eq!(state.tile_id, TileId(0));
        assert_eq!(state.pbits.len(), 7);
        assert_eq!(state.signature_count, 0);
        assert!(state.keypair_id.is_none());

        Ok(())
    }

    #[test]
    fn test_assign_keypair() -> Result<()> {
        let tess = HeptagonalTessellation::new(0)?;
        let mut substrate = CryptoSubstrate::new(tess, 7)?;

        substrate.assign_keypair(TileId(0), "keypair_123".to_string())?;

        let state = substrate.get_crypto_state(TileId(0)).unwrap();
        assert_eq!(state.keypair_id, Some("keypair_123".to_string()));

        Ok(())
    }

    #[test]
    fn test_record_signature() -> Result<()> {
        let tess = HeptagonalTessellation::new(0)?;
        let mut substrate = CryptoSubstrate::new(tess, 7)?;

        substrate.record_signature(TileId(0), 1234567890)?;

        let state = substrate.get_crypto_state(TileId(0)).unwrap();
        assert_eq!(state.last_signed, Some(1234567890));
        assert_eq!(state.signature_count, 1);

        // Record another signature
        substrate.record_signature(TileId(0), 1234567900)?;
        let state = substrate.get_crypto_state(TileId(0)).unwrap();
        assert_eq!(state.signature_count, 2);

        Ok(())
    }

    #[test]
    fn test_substrate_stats() -> Result<()> {
        let tess = HeptagonalTessellation::new(0)?;
        let mut substrate = CryptoSubstrate::new(tess, 48)?;

        substrate.assign_keypair(TileId(0), "key_1".to_string())?;
        substrate.record_signature(TileId(0), 1000)?;
        substrate.record_signature(TileId(0), 2000)?;

        let stats = substrate.stats();
        assert_eq!(stats.total_tiles, 1);
        assert_eq!(stats.total_pbits, 48);
        assert_eq!(stats.tiles_with_keypairs, 1);
        assert_eq!(stats.total_signatures, 2);
        assert_eq!(stats.avg_pbits_per_tile, 48.0);

        Ok(())
    }

    #[test]
    fn test_get_pbits_for_tile() -> Result<()> {
        let tess = HeptagonalTessellation::new(0)?;
        let substrate = CryptoSubstrate::new(tess, 7)?;

        let pbits = substrate.get_pbits_for_tile(TileId(0));
        assert_eq!(pbits.len(), 7);

        // Check that all pBits 0-6 are in the list
        for i in 0..7 {
            assert!(pbits.contains(&i));
        }

        Ok(())
    }
}
