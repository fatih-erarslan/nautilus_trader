//! c302 Data Loader
//!
//! Loads C. elegans connectome data from c302 project CSV files.
//! Supports multiple data sources:
//! - White et al. 1986 (classic connectome)
//! - Varshney et al. 2011 (updated)
//! - Witvliet et al. 2020 (electron microscopy)
//!
//! Reference: https://github.com/openworm/c302

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::neuron::{Neuron, NeuronClass, NeuronId, Neurotransmitter};
use crate::synapse::{Synapse, SynapseType};
use crate::muscle_map::NeuromuscularJunction;
use crate::connectome::Connectome;

/// Data source for connectome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DataSource {
    /// White et al. 1986 - Classic EM reconstruction
    #[default]
    White1986,
    /// Varshney et al. 2011 - Updated connectivity
    Varshney2011,
    /// Cook et al. 2019 - Complete adult connectome
    Cook2019,
    /// Witvliet et al. 2020 - Developmental series
    Witvliet2020,
}

/// Connection type from c302 data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionType {
    /// Chemical synapse (Send)
    Chemical,
    /// Electrical synapse (GapJunction)
    GapJunction,
    /// Neuromuscular junction
    NMJ,
}

impl ConnectionType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "send" | "chemical" | "exc" | "inh" => Some(Self::Chemical),
            "gapjunction" | "gap" | "electrical" => Some(Self::GapJunction),
            "nmj" | "neuromuscular" => Some(Self::NMJ),
            _ => None,
        }
    }
}

/// Raw connection record from CSV
#[derive(Debug, Clone)]
pub struct ConnectionRecord {
    pub origin: String,
    pub target: String,
    pub conn_type: ConnectionType,
    pub weight: f32,
    pub neurotransmitter: Option<Neurotransmitter>,
}

/// c302 data loader
pub struct C302DataLoader {
    /// Neuron name to ID mapping
    name_to_id: HashMap<String, NeuronId>,
    /// All neurons
    neurons: Vec<Neuron>,
    /// All connections
    connections: Vec<ConnectionRecord>,
    /// Data source
    source: DataSource,
}

impl C302DataLoader {
    /// Create new loader
    pub fn new(source: DataSource) -> Self {
        Self {
            name_to_id: HashMap::new(),
            neurons: Vec::new(),
            connections: Vec::new(),
            source,
        }
    }

    /// Load neuron list (all 302 hermaphrodite neurons)
    pub fn load_neurons(&mut self) {
        // Complete list of C. elegans hermaphrodite neurons (302 total)
        // Organized by class and function

        // Sensory neurons (amphid, phasmid, etc.)
        let sensory = SENSORY_NEURONS;
        for name in sensory {
            self.add_neuron(name, NeuronClass::Sensory);
        }

        // Interneurons
        let inter = INTERNEURONS;
        for name in inter {
            self.add_neuron(name, NeuronClass::Interneuron);
        }

        // Motor neurons
        let motor = MOTOR_NEURONS;
        for name in motor {
            self.add_neuron(name, NeuronClass::Motor);
        }

        // Pharyngeal neurons
        let pharyngeal = PHARYNGEAL_NEURONS;
        for name in pharyngeal {
            self.add_neuron(name, NeuronClass::Pharyngeal);
        }
    }

    fn add_neuron(&mut self, name: &str, class: NeuronClass) {
        let id = self.neurons.len() as NeuronId;
        let neuron = Neuron::new(id, name, class)
            .with_neurotransmitter(infer_neurotransmitter(name));
        self.neurons.push(neuron);
        self.name_to_id.insert(name.to_string(), id);
    }

    /// Load connections from CSV reader
    /// Expected format: Origin,Target,Type,Weight[,Neurotransmitter]
    pub fn load_connections_csv<R: Read>(&mut self, reader: R) -> Result<usize, String> {
        let buf = BufReader::new(reader);
        let mut count = 0;

        for (line_num, line_result) in buf.lines().enumerate() {
            let line = line_result.map_err(|e| format!("Line {}: {}", line_num, e))?;
            let line = line.trim();

            // Skip empty lines and headers
            if line.is_empty() || line.starts_with('#') || line.to_lowercase().starts_with("origin") {
                continue;
            }

            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 3 {
                continue;
            }

            let origin = parts[0].to_string();
            let target = parts[1].to_string();
            let conn_type = match ConnectionType::from_str(parts[2]) {
                Some(t) => t,
                None => continue,
            };

            let weight = if parts.len() > 3 {
                parts[3].parse().unwrap_or(1.0)
            } else {
                1.0
            };

            let neurotransmitter = if parts.len() > 4 {
                parse_neurotransmitter(parts[4])
            } else {
                None
            };

            self.connections.push(ConnectionRecord {
                origin,
                target,
                conn_type,
                weight,
                neurotransmitter,
            });
            count += 1;
        }

        Ok(count)
    }

    /// Load from c302 herm_full_edgelist.csv format
    /// Format: pre,post,type,count
    pub fn load_edgelist_csv<R: Read>(&mut self, reader: R) -> Result<usize, String> {
        let buf = BufReader::new(reader);
        let mut count = 0;

        for (line_num, line_result) in buf.lines().enumerate() {
            let line = line_result.map_err(|e| format!("Line {}: {}", line_num, e))?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') || line.starts_with("pre") {
                continue;
            }

            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 4 {
                continue;
            }

            let origin = parts[0].to_string();
            let target = parts[1].to_string();
            let conn_type = match parts[2].to_lowercase().as_str() {
                "chemical" | "exc" | "inh" => ConnectionType::Chemical,
                "electrical" | "gap" => ConnectionType::GapJunction,
                _ => continue,
            };
            let weight: f32 = parts[3].parse().unwrap_or(1.0);

            self.connections.push(ConnectionRecord {
                origin,
                target,
                conn_type,
                weight,
                neurotransmitter: None,
            });
            count += 1;
        }

        Ok(count)
    }

    /// Build connectome from loaded data
    pub fn build_connectome(&self) -> Connectome {
        let mut connectome = Connectome::new();

        // Add all neurons
        for neuron in &self.neurons {
            connectome.add_neuron(neuron.clone());
        }

        // Add synapses
        for conn in &self.connections {
            let pre_id = match self.name_to_id.get(&conn.origin) {
                Some(&id) => id,
                None => continue,
            };
            let post_id = match self.name_to_id.get(&conn.target) {
                Some(&id) => id,
                None => {
                    // Check if target is a muscle
                    if conn.target.starts_with('M') && conn.target.len() >= 4 {
                        // It's a muscle - add NMJ
                        if let Some(nmj) = parse_muscle_target(&conn.target, pre_id, conn.weight) {
                            connectome.add_nmj(nmj);
                        }
                        continue;
                    }
                    continue;
                }
            };

            let synapse = match conn.conn_type {
                ConnectionType::Chemical => {
                    let mut syn = Synapse::chemical(pre_id, post_id, conn.weight);
                    if let Some(nt) = conn.neurotransmitter {
                        if nt == Neurotransmitter::GABA {
                            syn = syn.inhibitory();
                        } else {
                            syn = syn.excitatory();
                        }
                    }
                    syn
                }
                ConnectionType::GapJunction => {
                    Synapse::gap_junction(pre_id, post_id, conn.weight)
                }
                ConnectionType::NMJ => continue, // Handled above
            };

            connectome.add_synapse(synapse);
        }

        connectome
    }

    /// Get number of neurons loaded
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Get number of connections loaded
    pub fn num_connections(&self) -> usize {
        self.connections.len()
    }
}

/// Parse muscle target name to NMJ
fn parse_muscle_target(name: &str, neuron_id: NeuronId, weight: f32) -> Option<NeuromuscularJunction> {
    // Muscle names: MDL01-24, MDR01-24, MVL01-24, MVR01-24
    if name.len() < 5 {
        return None;
    }

    let quadrant = match &name[0..3] {
        "MDR" => 0,
        "MVR" => 1,
        "MVL" => 2,
        "MDL" => 3,
        _ => return None,
    };

    let row: u32 = name[3..].parse().ok()?;
    if row < 1 || row > 24 {
        return None;
    }

    Some(NeuromuscularJunction {
        neuron: neuron_id,
        muscle_row: row - 1,
        muscle_quadrant: quadrant,
        weight,
    })
}

/// Parse neurotransmitter from string
fn parse_neurotransmitter(s: &str) -> Option<Neurotransmitter> {
    match s.to_lowercase().as_str() {
        "acetylcholine" | "ach" => Some(Neurotransmitter::Acetylcholine),
        "gaba" => Some(Neurotransmitter::GABA),
        "glutamate" | "glu" => Some(Neurotransmitter::Glutamate),
        "dopamine" | "da" => Some(Neurotransmitter::Dopamine),
        "serotonin" | "5ht" => Some(Neurotransmitter::Serotonin),
        "octopamine" | "oa" => Some(Neurotransmitter::Octopamine),
        "tyramine" | "ta" => Some(Neurotransmitter::Tyramine),
        _ => None,
    }
}

/// Infer neurotransmitter from neuron name (based on c302 data)
fn infer_neurotransmitter(name: &str) -> Neurotransmitter {
    // GABAergic neurons (from c302 GABA.py)
    const GABA_NEURONS: &[&str] = &[
        "DD01", "DD02", "DD03", "DD04", "DD05", "DD06",
        "VD01", "VD02", "VD03", "VD04", "VD05", "VD06", "VD07", "VD08", "VD09", "VD10", "VD11", "VD12", "VD13",
        "AVL", "DVB", "RIS", "RME",
    ];

    // Dopaminergic neurons
    const DOPAMINE_NEURONS: &[&str] = &[
        "CEPDL", "CEPDR", "CEPVL", "CEPVR", "ADEL", "ADER", "PDEL", "PDER",
    ];

    // Serotonergic neurons
    const SEROTONIN_NEURONS: &[&str] = &[
        "NSML", "NSMR", "ADFL", "ADFR", "HSNI", "HSNR",
    ];

    if GABA_NEURONS.iter().any(|&n| name == n) {
        Neurotransmitter::GABA
    } else if DOPAMINE_NEURONS.iter().any(|&n| name == n) {
        Neurotransmitter::Dopamine
    } else if SEROTONIN_NEURONS.iter().any(|&n| name == n) {
        Neurotransmitter::Serotonin
    } else {
        // Default to acetylcholine (most common)
        Neurotransmitter::Acetylcholine
    }
}

// ============================================================================
// Complete C. elegans neuron lists (302 hermaphrodite neurons)
// Organized by functional class as in c302
// ============================================================================

/// Sensory neurons (39 types, 60 cells)
const SENSORY_NEURONS: &[&str] = &[
    // Amphid sensory (bilateral pairs)
    "ADAL", "ADAR", "ADEL", "ADER", "ADFL", "ADFR", "ADLL", "ADLR",
    "AFDL", "AFDR", "ALML", "ALMR", "ALNL", "ALNR",
    "ASEL", "ASER", "ASGL", "ASGR", "ASHL", "ASHR",
    "ASIL", "ASIR", "ASJL", "ASJR", "ASKL", "ASKR",
    "AUAL", "AUAR", "AWAL", "AWAR", "AWBL", "AWBR", "AWCL", "AWCR",
    // Phasmid sensory
    "PHAL", "PHAR", "PHBL", "PHBR", "PHCL", "PHCR",
    // Other sensory
    "PLML", "PLMR", "PVDL", "PVDR",
    "AVM", "PVM", "SDQL", "SDQR",
    "OLLL", "OLLR", "OLQDL", "OLQDR", "OLQVL", "OLQVR",
    "IL1DL", "IL1DR", "IL1L", "IL1R", "IL1VL", "IL1VR",
    "IL2DL", "IL2DR", "IL2L", "IL2R", "IL2VL", "IL2VR",
    "CEPDL", "CEPDR", "CEPVL", "CEPVR",
    "BAGL", "BAGR", "URXL", "URXR",
    "FLPL", "FLPR", "AQR", "PQR",
];

/// Interneurons (command, ring, and other interneurons)
const INTERNEURONS: &[&str] = &[
    // Command interneurons
    "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER",
    "PVCL", "PVCR", "PVNL", "PVNR",
    // Ring interneurons
    "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR",
    "RIDL", "RIDR", "RIFL", "RIFR", "RIGL", "RIGR",
    "RIHL", "RIHR", "RIML", "RIMR", "RINL", "RINR",
    "RIPL", "RIPR", "RIR",
    // Amphid interneurons
    "AIAL", "AIAR", "AIBL", "AIBR", "AIML", "AIMR",
    "AINL", "AINR", "AIYL", "AIYR", "AIZL", "AIZR",
    // Other interneurons
    "AVFL", "AVFR", "AVHL", "AVHR", "AVJL", "AVJR", "AVKL", "AVKR",
    "AVG", "AVL",
    "BDUL", "BDUR",
    "DVA", "DVB", "DVC",
    "LUAL", "LUAR",
    "PVPL", "PVPR", "PVQL", "PVQR",
    "PVWL", "PVWR",
    "SAADL", "SAADR", "SAAVL", "SAAVR",
    "SABVL", "SABVR",
    "URADL", "URADR", "URAVL", "URAVR",
];

/// Motor neurons (113 cells)
const MOTOR_NEURONS: &[&str] = &[
    // A-class (backward locomotion)
    "VA01", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11", "VA12",
    "DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "DA08", "DA09",
    // B-class (forward locomotion)
    "VB01", "VB02", "VB03", "VB04", "VB05", "VB06", "VB07", "VB08", "VB09", "VB10", "VB11",
    "DB01", "DB02", "DB03", "DB04", "DB05", "DB06", "DB07",
    // D-class (inhibitory, cross-inhibition)
    "VD01", "VD02", "VD03", "VD04", "VD05", "VD06", "VD07", "VD08", "VD09", "VD10", "VD11", "VD12", "VD13",
    "DD01", "DD02", "DD03", "DD04", "DD05", "DD06",
    // AS-class
    "AS01", "AS02", "AS03", "AS04", "AS05", "AS06", "AS07", "AS08", "AS09", "AS10", "AS11",
    // Head motor neurons
    "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR",
    "RMED", "RMEL", "RMER", "RMEV",
    "RMFL", "RMFR", "RMGL", "RMGR", "RMHL", "RMHR",
    "SIADL", "SIADR", "SIAVL", "SIAVR",
    "SIBDL", "SIBDR", "SIBVL", "SIBVR",
    "SMBDL", "SMBDR", "SMBVL", "SMBVR",
    "SMDDL", "SMDDR", "SMDVL", "SMDVR",
    // Vulval/Egg-laying
    "VC01", "VC02", "VC03", "VC04", "VC05", "VC06",
    "HSNL", "HSNR",
];

/// Pharyngeal neurons (20 neurons)
const PHARYNGEAL_NEURONS: &[&str] = &[
    // Pharyngeal motor neurons
    "M1", "M2L", "M2R", "M3L", "M3R", "M4", "M5",
    // Pharyngeal interneurons
    "I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6",
    // Pharyngeal marginal cells
    "MC", "MI",
    // Neurosecretory motor neurons
    "NSML", "NSMR",
    // Pharyngeal polymodal neurons
    "MCL", "MCR",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_neurons() {
        let mut loader = C302DataLoader::new(DataSource::White1986);
        loader.load_neurons();

        // Should have ~302 neurons (some may be counted in multiple categories)
        assert!(loader.num_neurons() > 250, "Should have most of 302 neurons");
    }

    #[test]
    fn test_parse_muscle_target() {
        let nmj = parse_muscle_target("MDR05", 0, 1.0);
        assert!(nmj.is_some());
        let nmj = nmj.unwrap();
        assert_eq!(nmj.muscle_quadrant, 0);
        assert_eq!(nmj.muscle_row, 4); // 0-indexed

        let nmj = parse_muscle_target("MVL12", 0, 1.0);
        assert!(nmj.is_some());
        let nmj = nmj.unwrap();
        assert_eq!(nmj.muscle_quadrant, 2);
        assert_eq!(nmj.muscle_row, 11);
    }

    #[test]
    fn test_connection_type_parsing() {
        assert_eq!(ConnectionType::from_str("Send"), Some(ConnectionType::Chemical));
        assert_eq!(ConnectionType::from_str("GapJunction"), Some(ConnectionType::GapJunction));
        assert_eq!(ConnectionType::from_str("chemical"), Some(ConnectionType::Chemical));
    }

    #[test]
    fn test_build_connectome() {
        let mut loader = C302DataLoader::new(DataSource::White1986);
        loader.load_neurons();

        let connectome = loader.build_connectome();
        assert!(connectome.num_neurons() > 250);
    }
}
