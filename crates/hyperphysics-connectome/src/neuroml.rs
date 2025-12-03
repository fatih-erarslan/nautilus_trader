//! NeuroML 2 Export
//!
//! Exports connectome and network models to NeuroML 2 format for interoperability
//! with other neuroscience tools (NEURON, Brian, NetPyNE, etc.).
//!
//! ## NeuroML 2 Format
//!
//! NeuroML 2 is an XML-based format for describing:
//! - Cell morphologies and biophysics
//! - Ion channels and synapses
//! - Network connectivity
//! - Simulation parameters
//!
//! ## Reference
//!
//! - NeuroML 2: https://neuroml.org/
//! - libNeuroML: https://github.com/NeuralEnsemble/libNeuroML
//! - c302 export: https://github.com/openworm/c302

use std::io::Write;
use std::fmt::Write as FmtWrite;

use crate::connectome::Connectome;
use crate::neuron::{Neuron, NeuronClass, Neurotransmitter};
use crate::synapse::{Synapse, SynapseType};
use crate::models::{ModelLevel, ModelParams};
use crate::muscle_map::NeuromuscularJunction;

/// NeuroML 2 namespace
const NEUROML_NAMESPACE: &str = "http://www.neuroml.org/schema/neuroml2";
/// NeuroML 2 schema location
const NEUROML_SCHEMA: &str = "https://raw.githubusercontent.com/NeuroML/NeuroML2/master/Schemas/NeuroML2/NeuroML_v2.3.xsd";

/// NeuroML exporter for connectome data
pub struct NeuroMLExporter {
    /// Model level (affects cell type)
    model_level: ModelLevel,
    /// Include cell morphology
    include_morphology: bool,
    /// Include biophysical properties
    include_biophysics: bool,
    /// Network ID
    network_id: String,
    /// Notes/description
    notes: String,
}

impl Default for NeuroMLExporter {
    fn default() -> Self {
        Self {
            model_level: ModelLevel::B,
            include_morphology: false,
            include_biophysics: true,
            network_id: "c302_network".to_string(),
            notes: "C. elegans connectome exported from HyperPhysics".to_string(),
        }
    }
}

impl NeuroMLExporter {
    /// Create a new exporter with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set model level (affects cell type definitions)
    pub fn with_model_level(mut self, level: ModelLevel) -> Self {
        self.model_level = level;
        self
    }

    /// Set network ID
    pub fn with_network_id(mut self, id: &str) -> Self {
        self.network_id = id.to_string();
        self
    }

    /// Set notes/description
    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = notes.to_string();
        self
    }

    /// Include morphology data
    pub fn with_morphology(mut self, include: bool) -> Self {
        self.include_morphology = include;
        self
    }

    /// Export connectome to NeuroML 2 XML string
    pub fn export(&self, connectome: &Connectome) -> String {
        let mut xml = String::with_capacity(1024 * 100);

        // XML header
        writeln!(xml, r#"<?xml version="1.0" encoding="UTF-8"?>"#).unwrap();

        // NeuroML root element
        writeln!(xml, r#"<neuroml xmlns="{}"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="{} {}"
    id="{}">"#,
            NEUROML_NAMESPACE,
            NEUROML_NAMESPACE,
            NEUROML_SCHEMA,
            self.network_id
        ).unwrap();

        // Notes
        writeln!(xml, "    <notes>{}</notes>", escape_xml(&self.notes)).unwrap();
        writeln!(xml).unwrap();

        // Include files for cell types
        self.write_includes(&mut xml);

        // Cell type definitions
        self.write_cell_types(&mut xml);

        // Synapse types
        self.write_synapse_types(&mut xml);

        // Network definition
        self.write_network(&mut xml, connectome);

        // Close root
        writeln!(xml, "</neuroml>").unwrap();

        xml
    }

    /// Export to a writer
    pub fn export_to<W: Write>(&self, connectome: &Connectome, writer: &mut W) -> std::io::Result<()> {
        let xml = self.export(connectome);
        writer.write_all(xml.as_bytes())
    }

    /// Write include statements for external cell/channel definitions
    fn write_includes(&self, xml: &mut String) {
        writeln!(xml, "    <!-- Cell and channel definitions -->").unwrap();

        match self.model_level {
            ModelLevel::A => {
                writeln!(xml, r#"    <include href="LeakyIntegrateAndFire.nml"/>"#).unwrap();
            }
            ModelLevel::B => {
                writeln!(xml, r#"    <include href="IzhikevichCell.nml"/>"#).unwrap();
            }
            ModelLevel::C | ModelLevel::D => {
                writeln!(xml, r#"    <include href="HodgkinHuxleyCell.nml"/>"#).unwrap();
                writeln!(xml, r#"    <include href="IonChannels.nml"/>"#).unwrap();
            }
            ModelLevel::C1 | ModelLevel::D1 => {
                writeln!(xml, r#"    <include href="HodgkinHuxleyCell.nml"/>"#).unwrap();
                writeln!(xml, r#"    <include href="IonChannels.nml"/>"#).unwrap();
                writeln!(xml, r#"    <include href="GradedSynapses.nml"/>"#).unwrap();
            }
        }

        writeln!(xml, r#"    <include href="Synapses.nml"/>"#).unwrap();
        writeln!(xml).unwrap();
    }

    /// Write cell type definitions
    fn write_cell_types(&self, xml: &mut String) {
        writeln!(xml, "    <!-- Cell type definitions -->").unwrap();

        match self.model_level {
            ModelLevel::A => {
                // Leaky Integrate-and-Fire
                writeln!(xml, r#"    <iafCell id="generic_iaf_cell"
        leakReversal="-65mV"
        thresh="-50mV"
        reset="-65mV"
        C="1nF"
        leakConductance="0.05uS"/>"#).unwrap();
            }
            ModelLevel::B => {
                // Izhikevich neuron
                writeln!(xml, r#"    <izhikevich2007Cell id="generic_izhikevich_cell"
        v0="-65mV"
        C="100pF"
        k="0.7nS_per_mV"
        vr="-60mV"
        vt="-40mV"
        vpeak="35mV"
        a="0.03per_ms"
        b="-2nS"
        c="-50mV"
        d="100pA"/>"#).unwrap();
            }
            ModelLevel::C | ModelLevel::D => {
                // Hodgkin-Huxley style
                writeln!(xml, r#"    <cell id="generic_hh_cell">
        <morphology id="morphology_generic">
            <segment id="0" name="soma">
                <proximal x="0" y="0" z="0" diameter="10"/>
                <distal x="10" y="0" z="0" diameter="10"/>
            </segment>
        </morphology>
        <biophysicalProperties id="biophys_generic">
            <membraneProperties>
                <channelDensity id="leak" ionChannel="leak" condDensity="0.3mS_per_cm2" erev="-65mV" ion="non_specific"/>
                <channelDensity id="naChans" ionChannel="NaChan" condDensity="120mS_per_cm2" erev="50mV" ion="na"/>
                <channelDensity id="kChans" ionChannel="KChan" condDensity="36mS_per_cm2" erev="-77mV" ion="k"/>
                <spikeThresh value="-20mV"/>
                <specificCapacitance value="1uF_per_cm2"/>
                <initMembPotential value="-65mV"/>
            </membraneProperties>
            <intracellularProperties>
                <resistivity value="0.1kohm_cm"/>
            </intracellularProperties>
        </biophysicalProperties>
    </cell>"#).unwrap();
            }
            ModelLevel::C1 | ModelLevel::D1 => {
                // Hodgkin-Huxley with graded synapse support
                writeln!(xml, r#"    <cell id="generic_hh_graded_cell">
        <morphology id="morphology_generic">
            <segment id="0" name="soma">
                <proximal x="0" y="0" z="0" diameter="10"/>
                <distal x="10" y="0" z="0" diameter="10"/>
            </segment>
        </morphology>
        <biophysicalProperties id="biophys_graded">
            <membraneProperties>
                <channelDensity id="leak" ionChannel="leak" condDensity="0.3mS_per_cm2" erev="-55mV" ion="non_specific"/>
                <channelDensity id="kFast" ionChannel="k_fast" condDensity="36mS_per_cm2" erev="-77mV" ion="k"/>
                <channelDensity id="kSlow" ionChannel="k_slow" condDensity="1.8mS_per_cm2" erev="-80mV" ion="k"/>
                <channelDensity id="ca" ionChannel="ca_boyle" condDensity="4mS_per_cm2" erev="60mV" ion="ca"/>
                <spikeThresh value="-20mV"/>
                <specificCapacitance value="1uF_per_cm2"/>
                <initMembPotential value="-55mV"/>
            </membraneProperties>
            <intracellularProperties>
                <resistivity value="0.1kohm_cm"/>
            </intracellularProperties>
        </biophysicalProperties>
    </cell>"#).unwrap();
            }
        }

        writeln!(xml).unwrap();
    }

    /// Write synapse type definitions
    fn write_synapse_types(&self, xml: &mut String) {
        writeln!(xml, "    <!-- Synapse definitions -->").unwrap();

        // Excitatory chemical synapse (ACh, Glutamate)
        writeln!(xml, r#"    <expTwoSynapse id="exc_syn"
        gbase="1nS"
        erev="0mV"
        tauRise="0.5ms"
        tauDecay="5ms"/>"#).unwrap();

        // Inhibitory chemical synapse (GABA)
        writeln!(xml, r#"    <expTwoSynapse id="inh_syn"
        gbase="1nS"
        erev="-80mV"
        tauRise="0.5ms"
        tauDecay="10ms"/>"#).unwrap();

        // Gap junction (electrical synapse)
        writeln!(xml, r#"    <gapJunction id="gap_junction"
        conductance="0.5nS"/>"#).unwrap();

        // Neuromuscular junction
        writeln!(xml, r#"    <expTwoSynapse id="nmj_syn"
        gbase="5nS"
        erev="0mV"
        tauRise="0.2ms"
        tauDecay="3ms"/>"#).unwrap();

        writeln!(xml).unwrap();
    }

    /// Write network definition
    fn write_network(&self, xml: &mut String, connectome: &Connectome) {
        writeln!(xml, r#"    <network id="{}" type="networkWithTemperature" temperature="20degC">"#,
            self.network_id).unwrap();

        // Populations (groups of neurons by class)
        self.write_populations(xml, connectome);

        // Projections (synaptic connections)
        self.write_projections(xml, connectome);

        // Electrical connections (gap junctions)
        self.write_electrical_connections(xml, connectome);

        // Input sources
        self.write_input_sources(xml, connectome);

        writeln!(xml, "    </network>").unwrap();
    }

    /// Write neuron populations
    fn write_populations(&self, xml: &mut String, connectome: &Connectome) {
        writeln!(xml, "        <!-- Neuron populations -->").unwrap();

        let cell_type = match self.model_level {
            ModelLevel::A => "generic_iaf_cell",
            ModelLevel::B => "generic_izhikevich_cell",
            ModelLevel::C | ModelLevel::D => "generic_hh_cell",
            ModelLevel::C1 | ModelLevel::D1 => "generic_hh_graded_cell",
        };

        // Group by neuron class
        for class in &[NeuronClass::Sensory, NeuronClass::Interneuron, NeuronClass::Motor, NeuronClass::Pharyngeal] {
            let neurons: Vec<&Neuron> = connectome.neurons()
                .iter()
                .filter(|n| n.class == *class)
                .collect();

            if neurons.is_empty() {
                continue;
            }

            let class_name = match class {
                NeuronClass::Sensory => "sensory",
                NeuronClass::Interneuron => "interneuron",
                NeuronClass::Motor => "motor",
                NeuronClass::Pharyngeal => "pharyngeal",
            };

            writeln!(xml, r#"        <population id="{}_neurons" component="{}" size="{}" type="populationList">"#,
                class_name, cell_type, neurons.len()).unwrap();

            for neuron in neurons {
                let (x, y, z) = (neuron.position[0], neuron.position[1], neuron.position[2]);
                writeln!(xml, r#"            <instance id="{}">
                <location x="{}" y="{}" z="{}"/>
            </instance>"#,
                    neuron.id, x, y, z).unwrap();
            }

            writeln!(xml, "        </population>").unwrap();
        }

        writeln!(xml).unwrap();
    }

    /// Write synaptic projections
    fn write_projections(&self, xml: &mut String, connectome: &Connectome) {
        writeln!(xml, "        <!-- Chemical synaptic projections -->").unwrap();

        // Excitatory projections
        let exc_synapses: Vec<&Synapse> = connectome.synapses()
            .iter()
            .filter(|s| s.synapse_type == SynapseType::Chemical && s.is_excitatory())
            .collect();

        if !exc_synapses.is_empty() {
            writeln!(xml, r#"        <projection id="exc_connections" presynapticPopulation="all_neurons" postsynapticPopulation="all_neurons" synapse="exc_syn">"#).unwrap();

            for (i, syn) in exc_synapses.iter().enumerate() {
                writeln!(xml, r#"            <connection id="{}" preCellId="../{}" postCellId="../{}"/>"#,
                    i, syn.pre, syn.post).unwrap();
            }

            writeln!(xml, "        </projection>").unwrap();
        }

        // Inhibitory projections
        let inh_synapses: Vec<&Synapse> = connectome.synapses()
            .iter()
            .filter(|s| s.synapse_type == SynapseType::Chemical && s.is_inhibitory())
            .collect();

        if !inh_synapses.is_empty() {
            writeln!(xml, r#"        <projection id="inh_connections" presynapticPopulation="all_neurons" postsynapticPopulation="all_neurons" synapse="inh_syn">"#).unwrap();

            for (i, syn) in inh_synapses.iter().enumerate() {
                writeln!(xml, r#"            <connection id="{}" preCellId="../{}" postCellId="../{}"/>"#,
                    i, syn.pre, syn.post).unwrap();
            }

            writeln!(xml, "        </projection>").unwrap();
        }

        writeln!(xml).unwrap();
    }

    /// Write electrical connections (gap junctions)
    fn write_electrical_connections(&self, xml: &mut String, connectome: &Connectome) {
        let gap_junctions: Vec<&Synapse> = connectome.synapses()
            .iter()
            .filter(|s| s.synapse_type == SynapseType::GapJunction)
            .collect();

        if gap_junctions.is_empty() {
            return;
        }

        writeln!(xml, "        <!-- Electrical connections (gap junctions) -->").unwrap();
        writeln!(xml, r#"        <electricalProjection id="gap_junctions" presynapticPopulation="all_neurons" postsynapticPopulation="all_neurons">"#).unwrap();

        for (i, syn) in gap_junctions.iter().enumerate() {
            writeln!(xml, r#"            <electricalConnection id="{}" preCell="{}" postCell="{}" synapse="gap_junction"/>"#,
                i, syn.pre, syn.post).unwrap();
        }

        writeln!(xml, "        </electricalProjection>").unwrap();
        writeln!(xml).unwrap();
    }

    /// Write input sources
    fn write_input_sources(&self, xml: &mut String, connectome: &Connectome) {
        // Add pulse generators for sensory neuron stimulation
        let sensory_neurons: Vec<&Neuron> = connectome.neurons()
            .iter()
            .filter(|n| n.class == NeuronClass::Sensory)
            .collect();

        if sensory_neurons.is_empty() {
            return;
        }

        writeln!(xml, "        <!-- Input sources for sensory neurons -->").unwrap();
        writeln!(xml, r#"        <pulseGenerator id="sensory_input" delay="100ms" duration="500ms" amplitude="10pA"/>"#).unwrap();

        writeln!(xml, r#"        <inputList id="sensory_inputs" population="sensory_neurons" component="sensory_input">"#).unwrap();

        for (i, neuron) in sensory_neurons.iter().take(10).enumerate() {
            writeln!(xml, r#"            <input id="{}" target="../sensory_neurons/{}" destination="synapses"/>"#,
                i, neuron.id).unwrap();
        }

        writeln!(xml, "        </inputList>").unwrap();
    }
}

/// Export connectome to LEMS simulation file
pub struct LEMSExporter {
    /// Simulation duration (ms)
    duration: f64,
    /// Time step (ms)
    dt: f64,
    /// Output file prefix
    output_prefix: String,
}

impl Default for LEMSExporter {
    fn default() -> Self {
        Self {
            duration: 1000.0,
            dt: 0.025,
            output_prefix: "c302_simulation".to_string(),
        }
    }
}

impl LEMSExporter {
    /// Create new LEMS exporter
    pub fn new() -> Self {
        Self::default()
    }

    /// Set simulation duration (ms)
    pub fn with_duration(mut self, duration: f64) -> Self {
        self.duration = duration;
        self
    }

    /// Set time step (ms)
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Export LEMS simulation file
    pub fn export(&self, neuroml_file: &str) -> String {
        let mut xml = String::with_capacity(4096);

        writeln!(xml, r#"<?xml version="1.0" encoding="UTF-8"?>"#).unwrap();
        writeln!(xml, r#"<Lems xmlns="http://www.neuroml.org/lems/0.7.6"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://www.neuroml.org/lems/0.7.6 https://raw.githubusercontent.com/LEMS/LEMS/master/Schemas/LEMS/LEMS_v0.7.6.xsd">"#).unwrap();

        // Include NeuroML file
        writeln!(xml, r#"    <Include file="{}"/>"#, neuroml_file).unwrap();

        // Simulation target
        writeln!(xml, r#"
    <Target component="sim1"/>

    <Simulation id="sim1" length="{}ms" step="{}ms" target="c302_network">

        <OutputFile id="output_volts" fileName="{}_voltages.dat">
            <OutputColumn id="v0" quantity="sensory_neurons[0]/v"/>
            <OutputColumn id="v1" quantity="motor_neurons[0]/v"/>
        </OutputFile>

        <OutputFile id="output_spikes" fileName="{}_spikes.dat" format="EVENTS_FORMAT">
            <EventSelection id="spikes" select="all_neurons[*]" eventPort="spike"/>
        </OutputFile>

    </Simulation>

</Lems>"#,
            self.duration,
            self.dt,
            self.output_prefix,
            self.output_prefix
        ).unwrap();

        xml
    }
}

/// Export to c302-compatible Python script
pub struct C302PythonExporter {
    /// Model level
    model_level: ModelLevel,
    /// Network name
    network_name: String,
}

impl Default for C302PythonExporter {
    fn default() -> Self {
        Self {
            model_level: ModelLevel::B,
            network_name: "c302_network".to_string(),
        }
    }
}

impl C302PythonExporter {
    /// Create new Python exporter
    pub fn new() -> Self {
        Self::default()
    }

    /// Set model level
    pub fn with_model_level(mut self, level: ModelLevel) -> Self {
        self.model_level = level;
        self
    }

    /// Export Python script for c302
    pub fn export(&self, connectome: &Connectome) -> String {
        let mut py = String::with_capacity(8192);

        // Header
        writeln!(py, r#"#!/usr/bin/env python3
"""
C. elegans connectome network generated by HyperPhysics
Compatible with c302 framework

Usage:
    python {} [simulator]

    simulator: jNeuroML, jNeuroML_NEURON, jNeuroML_NetPyNE
"""

import c302
import sys

def generate():
    """Generate the network model."""

    params = c302.C302Parameters()"#, self.network_name).unwrap();

        // Set model level
        let level_name = match self.model_level {
            ModelLevel::A => "A",
            ModelLevel::B => "B",
            ModelLevel::C => "C",
            ModelLevel::C1 => "C1",
            ModelLevel::D => "D",
            ModelLevel::D1 => "D1",
        };

        writeln!(py, r#"    params.set_model_level("{}")"#, level_name).unwrap();

        // Add neurons
        writeln!(py, r#"
    # Neurons"#).unwrap();

        for neuron in connectome.neurons() {
            writeln!(py, r#"    params.add_neuron("{}")"#, neuron.name).unwrap();
        }

        // Add synapses
        writeln!(py, r#"
    # Synaptic connections"#).unwrap();

        for syn in connectome.synapses() {
            let pre = connectome.get_neuron(syn.pre).map(|n| n.name.as_str()).unwrap_or("?");
            let post = connectome.get_neuron(syn.post).map(|n| n.name.as_str()).unwrap_or("?");

            match syn.synapse_type {
                SynapseType::Chemical => {
                    let syn_type = if syn.is_inhibitory() { "GABA" } else { "ACh" };
                    writeln!(py, r#"    params.add_synapse("{}", "{}", "{}", weight={})"#,
                        pre, post, syn_type, syn.weight).unwrap();
                }
                SynapseType::GapJunction => {
                    writeln!(py, r#"    params.add_gap_junction("{}", "{}", conductance={})"#,
                        pre, post, syn.weight).unwrap();
                }
            }
        }

        // Generate network
        writeln!(py, r#"
    return c302.generate_network(params, "{}")

if __name__ == "__main__":
    simulator = sys.argv[1] if len(sys.argv) > 1 else "jNeuroML"
    nml_doc = generate()
    c302.run_simulation(nml_doc, simulator)
"#, self.network_name).unwrap();

        py
    }
}

/// Escape special XML characters
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuroml_export() {
        let connectome = Connectome::celegans();
        let exporter = NeuroMLExporter::new()
            .with_model_level(ModelLevel::B)
            .with_network_id("test_network");

        let xml = exporter.export(&connectome);

        assert!(xml.contains("<?xml version"));
        assert!(xml.contains("<neuroml"));
        assert!(xml.contains("izhikevich2007Cell"));
        assert!(xml.contains("</neuroml>"));
    }

    #[test]
    fn test_lems_export() {
        let exporter = LEMSExporter::new()
            .with_duration(500.0)
            .with_dt(0.01);

        let xml = exporter.export("test_network.nml");

        assert!(xml.contains("<Lems"));
        assert!(xml.contains("500ms"));
        assert!(xml.contains("0.01ms"));
    }

    #[test]
    fn test_python_export() {
        let connectome = Connectome::celegans();
        let exporter = C302PythonExporter::new()
            .with_model_level(ModelLevel::B);

        let py = exporter.export(&connectome);

        assert!(py.contains("import c302"));
        assert!(py.contains("set_model_level"));
    }
}
