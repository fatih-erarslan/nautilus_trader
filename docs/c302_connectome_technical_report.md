# C302 C. elegans Connectome Implementation - Technical Analysis Report

## Executive Summary

The c302 project provides a comprehensive NeuroML2-compliant implementation of the C. elegans nervous system connectome with multiple levels of biophysical detail. This report analyzes the core architecture, neuron models, synaptic dynamics, and data sources.

**Key Metrics:**
- **302 neurons** (complete C. elegans hermaphrodite nervous system)
- **95 body wall muscles** (organized in 4 quadrants: MDL, MDR, MVL, MVR)
- **~7,000 neural connections** (chemical synapses + gap junctions)
- **~1,500 neuromuscular junctions**
- **6 model complexity levels** (A, B, C, C1, D, D1)
- **1,841 lines** in main orchestration module (`__init__.py`)

**Critical Features:**
1. **Flexible Data Sources**: Supports multiple connectome datasets (White 1986, Varshney 2011, Cook 2019, Witvliet 2020+)
2. **Modular Architecture**: Separate parameter files for each biophysical complexity level
3. **NeuroML2 Compliance**: Full integration with NeuroML2 ecosystem for standardization and reproducibility
4. **Owmeta Integration**: Biological metadata from curated OpenWorm database with offline caching
5. **Advanced Connection Control**: Regex-based parameter overrides, connection scaling, polarity modification
6. **Locomotion Support**: Sinusoidal stimulus with anatomically-informed phase offsets for motor pattern generation

**Model Progression:**
- **Level A**: Simple integrate-and-fire (IAF) neurons, event-based synapses
- **Level B**: IAF with activity variable, real gap junctions
- **Level C**: Conductance-based neurons with Hodgkin-Huxley channels, calcium dynamics
- **Level C1**: Graded (analog) synapses, voltage-dependent transmission
- **Level D**: Multi-compartmental neurons with realistic morphology
- **Level D1**: Most realistic - graded synapses with rise/decay kinetics, tuned conductances

---

## 1. SYSTEM ARCHITECTURE

### 1.1 Core Framework (`__init__.py` - 1841 lines)

**Primary Data Reader:**
- Default: `cect.SpreadsheetDataReader`
- Flexible data source system supporting multiple connectome datasets
- FW (Forward) mode: `cect.UpdatedSpreadsheetDataReader2`
- Cook2019 reader: `cect.Cook2019HermReader` (updated connectome data)

**Key Components:**
- **NeuroML2 Integration**: Full support for NeuroML2 cell models, synapses, and network structures
- **Network Generation**: Automated network construction from connectome data
- **Multiple Model Levels**: Parameters A, B, C, C0, C1, C2, D, D1 for different complexity levels
- **Owmeta Integration**: Optional integration with OpenWorm data repository (v6 bundle)

**Network Generation Pipeline:**
1. **Cell Population Creation** (lines 883-1031):
   - Loads morphology from `NeuroML2/*.cell.nml` files
   - Creates `Population` objects with single `Instance` at specified 3D location
   - Adds cell properties: color, type (sensory/interneuron/motor), neurotransmitter, receptor
   - For Level D: creates multi-compartmental neuron cells with actual morphology

2. **Muscle Population Creation** (lines 1046-1151):
   - Creates 95 body wall muscle populations
   - Positions muscles using formula: `(x, y, z) = (±80, -300+30*idx, ±80)`
   - Muscle color: `(0, 0.6, 0)` (green)

3. **Neural Connection Processing** (lines 1155-1427):
   - Reads connections from data reader
   - Determines synapse type based on neurotransmitter:
     - GABA → inhibitory (`inh`)
     - _GJ suffix → electrical (`elec`)
     - Default → excitatory (`exc`)
   - Creates `Projection` (chemical), `ElectricalProjection` (gap junctions), or `ContinuousProjection` (graded/analog)
   - Supports connection number override, scaling, and polarity override via regex patterns

4. **Neuromuscular Junction Processing** (lines 1428-1699):
   - Connects motor neurons to muscles
   - Similar synapse determination logic
   - Supports muscle-to-muscle connections (rare)

**Neuron Count**: 302 neurons (as per biological C. elegans connectome)

**Connection Polarity Determination Logic** (lines 1172-1174, 1456-1458):
```python
if "GABA" in conn.synclass:
    conn_pol = "inh"  # Inhibitory
elif "_GJ" in conn.synclass:
    conn_pol = "elec"  # Electrical (gap junction)
else:
    conn_pol = "exc"  # Excitatory (default)
```

**Neurotransmitter Classification** (via owmeta data, lines 605-610):
- GABA → inhibitory synapse (negative polarity marker)
- Others (Acetylcholine, Glutamate, etc.) → excitatory
- Unknown neurotransmitter → "?" marker

**Advanced Features:**
- **Connection number scaling**: `number_syns = pow(conn.number, scale)` with `global_connectivity_power_scaling` parameter
- **Regex-based parameter overrides**: Match connection patterns like `"AVB.*-DB\d+_elec_syn_gbase"` to set specific conductances
- **Mirrored electrical connections**: Automatically create symmetric gap junction parameters
- **Validation**: NeuroML2 schema validation (except for experimental levels B, C0, C2, D1)

**Muscle System**:
- 95 body wall muscles organized in 4 quadrants:
  - MDR (Muscle Dorsal Right): 24 muscles
  - MVR (Muscle Ventral Right): 24 muscles
  - MVL (Muscle Ventral Left): 24 muscles
  - MDL (Muscle Dorsal Left): 23 muscles
- Special muscles: MANAL, MVULVA

---

## 2. MODEL LEVELS (PARAMETER SETS)

### 2.1 Level A - Simple Integrate & Fire

**File**: `parameters_A.py`

**Neuron Model:**
- Type: `IafCell` (Integrate-and-Fire)
- Capacitance: 3 pF
- Leak reversal: -50 mV
- Threshold: -30 mV
- Reset: -50 mV
- Conductance: 0.1 nS

**Chemical Synapses:**
- Type: `ExpTwoSynapse` (double exponential)
- Excitatory:
  - gbase: 0.01 nS
  - Erev: 0 mV
  - Rise: 3 ms
  - Decay: 10 ms
- Inhibitory (GABA):
  - gbase: 0.01 nS
  - Erev: -80 mV
  - Rise: 3 ms
  - Decay: 10 ms

**Gap Junctions:**
- Implemented as event-based synapses (NOT real gap junctions)
- Default conductance: 0 nS (essentially disabled)

**Assessment**: Not biologically realistic; tends to over-excite; difficult to tune

---

### 2.2 Level B - IAF with Activity Variable

**File**: `parameters_B.py`

**Neuron Model:**
- Type: `IafActivityCell` (custom component)
- Same electrical parameters as Level A
- Added: tau1 = 50 ms (activity decay time constant)
- Activity variable tracks recent spiking

**Gap Junctions:**
- Type: `GapJunction` (real electrical connections)
- Neuron-to-neuron: 0.01 nS
- Neuron-to-muscle: 0.01 nS
- Current depends linearly on voltage difference

**Assessment**: Better than A, but still limited by IAF dynamics

---

### 2.3 Level C - Conductance-Based Cells

**File**: `parameters_C.py`

**Neuron Model:**
- Type: `Cell` (multi-compartmental capability)
- Diameter: 5 μm
- Initial membrane potential: -45 mV
- Specific capacitance: 1 μF/cm²

**Ion Channels (Hodgkin-Huxley style):**

1. **Leak Channel**:
   - Neurons: 0.005 mS/cm²
   - Muscles: 5e-7 S/cm²
   - Erev: -50 mV

2. **K_slow (Potassium - slow)**:
   - Neurons: 3 mS/cm²
   - Muscles: 0.0006 S/cm²
   - Erev: -60 mV

3. **K_fast (Potassium - fast)**:
   - Neurons: 0.0711 mS/cm²
   - Muscles: 0.0001 S/cm²
   - Erev: -60 mV

4. **Ca_boyle (Calcium)**:
   - Neurons: 3 mS/cm²
   - Muscles: 0.0007 S/cm²
   - Erev: 40 mV

**Calcium Dynamics:**
- Concentration model: `FixedFactorConcentrationModel`
- Decay time: 11.5943 ms
- Rho: 0.000238919 mol/(m·A·s)

**Chemical Synapses:**
- Type: `ExpTwoSynapse`
- Excitatory:
  - gbase: 0.1 nS
  - Erev: 0 mV
  - Rise: 1 ms
  - Decay: 5 ms
- Inhibitory:
  - gbase: 0.1 nS
  - Erev: -60 mV
  - Rise: 2 ms
  - Decay: 40 ms

**Gap Junctions:**
- Conductance: 0.0005 nS

**Assessment**: More realistic; can generate oscillatory behavior; requires spiking for event-based synapses

---

### 2.4 Level C1 - Graded Synapses

**File**: `parameters_C1.py`

**Inherits**: All cell parameters from Level C

**Chemical Synapses:**
- Type: `GradedSynapse` (voltage-dependent continuous transmission)
- Excitatory:
  - Conductance: 0.09 nS
  - Delta: 5 mV
  - Vth: 0 mV
  - Erev: 0 mV
  - k: 0.025 per ms
- Inhibitory:
  - Conductance: 0.09 nS
  - Delta: 5 mV
  - Vth: 0 mV
  - Erev: -70 mV
  - k: 0.025 per ms

**Gap Junctions:**
- Conductance: 0.00052 nS

**Assessment**: Good prospect; removes need for spiking; analog communication

---

### 2.5 Level D - Multicompartmental

**File**: `parameters_D.py`

**Neuron Model:**
- Type: `Cell` with realistic morphology
- Uses actual neuron morphology from NeuroML files
- Resistivity: 12 kΩ·cm
- Spike threshold: -26 mV

**Ion Channels** (similar to C but different densities):
- Leak: 0.02 mS/cm²
- K_slow: 2 mS/cm²
- K_fast: 0.2 mS/cm²
- Ca_boyle: 2 mS/cm²

**Chemical Synapses:**
- Type: `ExpTwoSynapse`
- Excitatory: 0.01 nS
- Inhibitory: 3 nS

**Gap Junctions:**
- Conductance: 0.0005 nS

**Muscle Model:**
- Length: 20 μm
- Diameter: 5 μm
- Similar ion channel composition

**Note**: Each neuron gets custom cell file based on its morphology (cells/{neuron}_D.cell.nml)

**Issues**:
- Either low resistance → rapid potential propagation but high input resistance
- Or high resistance → localized changes around soma
- See GitHub issue #71

---

### 2.6 Level D1 - Multicompartmental with Graded Synapses

**File**: `parameters_D1.py`

**Inherits**: Cell model from Level D

**Chemical Synapses:**
- Type: `GradedSynapse2` (custom component with rise/decay kinetics)
- Excitatory:
  - Conductance: 2 nS
  - ar (rise): 0.5 per s
  - ad (decay): 20 per s
  - beta: 0.25 per mV
  - Vth: -35 mV
  - Erev: 0 mV
- Inhibitory:
  - Conductance: 26 nS (neuron), 0.25 nS (muscle)
  - ar: 0.005 per s
  - ad: 10 per s
  - beta: 0.5 per mV
  - Vth: -55 mV
  - Erev: -80 mV

**Gap Junctions:**
- Neuron-to-neuron: 0.005 nS
- Neuron-to-muscle: 0.0001 nS

**Global Scaling:**
- `global_connectivity_power_scaling`: 0 (disabled by default)
- Can scale connection numbers: n_eff = n^scale

**Resistivity**: 3 kΩ·cm (lower than D for better propagation)

**Assessment**: Medium-term target for full-scale model; most biologically realistic

---

## 3. NEURON CLASSIFICATION

### 3.1 Complete Neuron List (302 neurons)

From `ConnectomeReader.py`:

**Sensory Neurons** (examples):
- ADAL, ADAR (amphid)
- AFDL, AFDR (amphid)
- ALML, ALMR (mechanosensory)
- ASEL, ASER (amphid chemosensory)
- AWA, AWB, AWC (chemosensory)
- BAG (O2 sensor)
- OLQDL/R, OLQVL/R (chemosensory)
- PLML, PLMR (mechanosensory)

**Interneurons** (examples):
- AIAL, AIAR
- AIBL, AIBR (command interneurons)
- AIML, AIMR
- AINL, AINR
- AIYL, AIYR
- AIZL, AIZR
- ALA (sleep neuron)
- AVA, AVB, AVD, AVE (command interneurons)
- RIA, RIB, RIC (interneurons)

**Motor Neurons**:
- VA class (VA1-VA12): Ventral A-type motor neurons - 12 neurons
- VB class (VB1-VB11): Ventral B-type motor neurons - 11 neurons
- DA class (DA1-DA9): Dorsal A-type motor neurons - 9 neurons
- DB class (DB1-DB7): Dorsal B-type motor neurons - 7 neurons
- VD class (VD1-VD13): Ventral D-type GABAergic motor neurons - 13 neurons
- DD class (DD1-DD6): Dorsal D-type GABAergic motor neurons - 6 neurons
- AS class (AS1-AS11): Ventral cord motor neurons - 11 neurons
- VC class (VC1-VC6): Hermaphrodite-specific motor neurons - 6 neurons

**Special Neurons**:
- DVA, DVB, DVC: Tail neurons
- PDA, PDB: Motor neurons
- PVT: Interneuron
- RIS: Sleep/quiescence neuron

### 3.2 Neurotransmitter Classification

**Cholinergic (Acetylcholine)** - Default excitatory:
- Most motor neurons (VA, VB, DA, DB, AS)
- Many interneurons

**GABAergic (GABA)** - Inhibitory:
- VD neurons (VD1-VD13)
- DD neurons (DD1-DD6)
- Automatically classified by name prefix in code

**Determination Logic** (`__init__.py`):
```python
def get_synclass(cell, syntype):
    if syntype == "GapJunction":
        return "Generic_GJ"
    elif cell.startswith("DD") or cell.startswith("VD"):
        return "GABA"
    else:
        return "Acetylcholine"
```

---

## 4. SYNAPSE TYPES AND DYNAMICS

### 4.1 Chemical Synapses

**Type 1: ExpTwoSynapse** (Levels A, B, C, D)
- Double exponential conductance waveform
- Formula: g(t) = gbase × (exp(-t/tau_decay) - exp(-t/tau_rise))
- Event-based (requires presynaptic spike)
- Current: I = g(t) × (V - Erev)

**Parameters vary by level:**
- Level A: Rise=3ms, Decay=10ms (exc), 10ms (inh)
- Level C: Rise=1ms, Decay=5ms (exc), Rise=2ms, Decay=40ms (inh)
- Level D: Similar to C with different gbase values

**Type 2: GradedSynapse** (Level C1)
- Continuous voltage-dependent transmission
- Sigmoid activation: s = 1 / (1 + exp(-(V - Vth)/delta))
- Conductance: g = conductance × s
- Rate constant k determines rise/fall speed
- NO spiking required - analog communication

**Type 3: GradedSynapse2** (Level D1)
- Custom implementation with explicit rise/decay kinetics
- Voltage-dependent activation: A = 1 / (1 + exp(-beta × (V - Vth)))
- State variable s with dynamics:
  - ds/dt = ar × A × (1 - s) - ad × s
  - ar = rise rate
  - ad = decay rate
- More biologically realistic than GradedSynapse
- Defined in `custom_synapses.xml`

### 4.2 Gap Junctions (Electrical Synapses)

**Implementation**:
- Type: `GapJunction` (NeuroML2 standard)
- Current: I = conductance × (V_pre - V_post)
- Bidirectional
- Linear voltage dependence

**Connection Pattern:**
- Neuron-to-neuron: High density in C. elegans
- Neuron-to-muscle: Present but less common
- Muscle-to-muscle: Rare but supported

**Conductance Values** (by level):
- Level A: 0 nS (disabled, uses pseudo-GJ with events)
- Level B: 0.01 nS
- Level C: 0.0005 nS
- Level C1: 0.00052 nS
- Level D: 0.0005 nS
- Level D1: 0.005 nS (neuron), 0.0001 nS (muscle)

### 4.3 Polarity Override System

Allows manual override of connection polarity:

```python
conn_polarity_override = {
    "AVAL-AVBR": "inh",  # Make this specific connection inhibitory
    "AVAR-.*": "exc"     # Regex: all AVAR connections excitatory
}
```

**Supported Polarities:**
- `"exc"` - Excitatory
- `"inh"` - Inhibitory
- `"elec"` - Electrical (gap junction)

---

## 5. MUSCLE MAPPING

### 5.1 Body Wall Muscle Organization

**Quadrants** (24 muscles each except MDL with 23):

```
Quadrant 0 (MDR): MDR01-MDR24  (Muscle Dorsal Right)
Quadrant 1 (MVR): MVR01-MVR24  (Muscle Ventral Right)
Quadrant 2 (MVL): MVL01-MVL24  (Muscle Ventral Left)
Quadrant 3 (MDL): MDL01-MDL24  (Muscle Dorsal Left)
```

**Naming Convention:**
- Old format: `BWM-VL12` (Body Wall Muscle - Ventral Left 12)
- New format: `MVL12` (preferred in c302)

**Conversion Function** (`ConnectomeReader.py`):
```python
def convert_to_preferred_muscle_name(muscle):
    if muscle.startswith("BWM-VL"): return "MVL%s" % muscle[6:]
    elif muscle.startswith("BWM-VR"): return "MVR%s" % muscle[6:]
    elif muscle.startswith("BWM-DL"): return "MDL%s" % muscle[6:]
    elif muscle.startswith("BWM-DR"): return "MDR%s" % muscle[6:]
```

### 5.2 Muscle Position Calculation

**3D Positioning** (`__init__.py`):
```python
def get_muscle_position(muscle, data_reader):
    # Pattern: M(V|D)(L|R)(01-24)
    # V=Ventral(z=80), D=Dorsal(z=-80)
    # L=Left(x=80), R=Right(x=-80)
    # Index: y = -300 + 30*index

    # Example: MVL12
    # x = 80 (Left)
    # y = -300 + 30*12 = 60
    # z = 80 (Ventral)
```

**Spatial Layout:**
- Anterior-posterior: y-axis (-300 to +420 μm)
- Left-right: x-axis (±80 μm)
- Dorsal-ventral: z-axis (±80 μm)
- Spacing: 30 μm between consecutive muscles

### 5.3 Neuromuscular Junctions

**Motor Neurons → Muscles:**

| Motor Neuron Class | Target Muscles | Synapse Type |
|--------------------|----------------|--------------|
| VA (VA1-VA12) | Ventral muscles (MVL/MVR) | Acetylcholine (exc) |
| VB (VB1-VB11) | Ventral muscles | Acetylcholine (exc) |
| DA (DA1-DA9) | Dorsal muscles (MDL/MDR) | Acetylcholine (exc) |
| DB (DB1-DB7) | Dorsal muscles | Acetylcholine (exc) |
| VD (VD1-VD13) | Ventral muscles | GABA (inh) |
| DD (DD1-DD6) | Dorsal muscles | GABA (inh) |
| AS (AS1-AS11) | Mixed | Acetylcholine |
| VC (VC1-VC6) | Vulval muscles | Acetylcholine |

**Connection Reading:**
```python
mneurons, all_muscles, muscle_conns = data_reader.read_muscle_data()
# Returns:
# - mneurons: Motor neurons with muscle connections
# - all_muscles: List of muscle cells
# - muscle_conns: ConnectionInfo objects for neuron→muscle
```

---

## 6. CONNECTOME DATA SOURCES

### 6.1 Cook 2019 Dataset

**File**: `Cook2019DataReader.py`
**Source**: "SI 5 Connectome adjacency matrices.xlsx"

**Data Sheets:**
1. **"hermaphrodite chemical"**:
   - Pre-synaptic: 300 cells (rows 4-304)
   - Post-synaptic: 453 cells (columns 4-457)
   - Connection type: Chemical synapses
   - Matrix format: Integer synapse counts

2. **"herm gap jn symmetric"**:
   - Pre-synaptic: 468 cells (rows 4-472)
   - Post-synaptic: 468 cells (columns 4-472)
   - Connection type: Gap junctions
   - Matrix format: Integer connection counts
   - Symmetrized data

**Key Features:**
- Reads Excel file using `openpyxl`
- Constructs connection matrices as NumPy arrays
- Filters body wall muscles from neuron-to-neuron connections
- Returns `ConnectionInfo` objects with:
  - pre_cell: Presynaptic neuron name
  - post_cell: Postsynaptic neuron name
  - number: Number of synapses/connections
  - syntype: "Send" (chemical) or "GapJunction"
  - synclass: "Acetylcholine", "GABA", or "Generic_GJ"

**Leading Zero Removal:**
```python
# VB01 → VB1, DA03 → DA3
def remove_leading_index_zero(cell):
    if is_neuron(cell) and cell[-2:].startswith("0"):
        return "%s%s" % (cell[:-2], cell[-1:])
    return cell
```

### 6.2 Alternative Data Readers

**Available Readers** (from imports):
- `cect.White_whole` - Original White et al. dataset
- `cect.Cook2019HermReader` - Cook 2019 hermaphrodite
- `cect.SpreadsheetDataReader` - Generic spreadsheet reader
- `cect.UpdatedSpreadsheetDataReader2` - Updated data
- `VarshneyDataReader` - Varshney et al. dataset
- `WitvlietDataReader1/2` - Witvliet et al. datasets
- `WormNeuroAtlasReader` - WormNeuroAtlas integration

**Reader Interface:**
```python
reader = load_data_reader(data_reader_name)
cells, conns = reader.read_data(include_nonconnected_cells=True)
neurons, muscles, muscle_conns = reader.read_muscle_data()
```

### 6.3 Connection Statistics (Example from Cook2019)

From analysis output:

**Neuron-to-Neuron:**
- Total connections: ~7,000 (typical)
- Neurotransmitter distribution:
  - Acetylcholine: ~5,000 connections
  - GABA: ~500 connections
  - Generic_GJ (gap junctions): ~2,000 connections
- Average synapses per connection: 1-5

**Neuron-to-Muscle:**
- Motor neurons: ~80-100 neurons with muscle connections
- Muscle connections: ~1,000-1,500 neuromuscular junctions
- Predominantly cholinergic (excitatory)
- Some GABAergic (inhibitory) from VD/DD classes

---

## 6A. OWMETA INTEGRATION & DATA CACHING

### 6A.1 OpenWorm Data Repository Integration (lines 43-56, 540-621)

**Purpose**: Provides biological metadata for neurons and muscles from curated OpenWorm database.

**Owmeta Dependencies:**
```python
from owmeta_core import __version__ as owc_version
from owmeta_core.bundle import Bundle
from owmeta_core.context import Context
from owmeta import __version__ as owmeta_version
from owmeta.cell import Cell
from owmeta.neuron import Neuron
from owmeta.muscle import Muscle
```

**Data Bundle:**
- Bundle: `"openworm/owmeta-data"`
- Version: 6
- Source: OpenWorm project's curated biological database

### 6A.2 Cell Information Retrieval (lines 540-621)

**Function**: `_get_cell_info(bnd, cells)`

**Retrieved Metadata:**
```python
neuron_info = {
    "cell_name": (
        cell_object,           # Owmeta Cell/Neuron object
        neuron_types,          # ("sensory", "interneuron", "motor")
        receptor,              # ("AMPA", "GABA-A", "NMDA", ...)
        neurotransmitter,      # ("Acetylcholine", "GABA", "Glutamate", ...)
        short_label,           # "Se) ASEL" or "Mo) VA1"
        color                  # "1 0.2 1" (RGB for visualization)
    )
}
```

**Neuron Type Classification:**
- **Sensory**: Color `(1, 0.2, 1)` (magenta), prefix "Se)"
- **Interneuron**: Color `(1, 0, 0.4)` (red-pink), prefix "In)"
- **Motor**: Color `(0.5, 0.4, 1)` (blue-purple), prefix "Mo)"
- **Muscle**: Color `(0, 0.6, 0)` (green), prefix "Mu)"
- **Unknown**: Color `(0.5, 0, 0)` (dark red)

**Polarity Indicator:**
- **GABA transmitter**: "-" prefix (inhibitory)
- **Other transmitter**: "+" prefix (excitatory)
- **Unknown transmitter**: "?" prefix

### 6A.3 Data Caching System (lines 537-556, 1791-1840)

**Motivation**: Avoid repeated owmeta bundle access (network I/O overhead).

**Cache File Location:**
```python
OWMETA_CACHED_DATA_FILE = "/path/to/c302/data/owmeta_cache.json"
```

**Cache Structure:**
```json
{
    "comment": "Information exported from owmeta v0.13.0 (owmeta core v0.14.0)",
    "neuron_info": {
        "AVAL": [
            "Neuron object repr",
            ["interneuron"],
            ["Acetylcholine receptor", "Glutamate receptor"],
            ["Acetylcholine", "Glutamate"],
            "+ In) AVAL",
            "1 0 0.4"
        ],
        "PLML": [
            "Neuron object repr",
            ["sensory"],
            ["Glutamate receptor"],
            ["Glutamate"],
            "+ Se) PLML",
            "1 0.2 1"
        ]
        // ... 302 neurons
    },
    "muscle_info": {
        "MDL01": [
            "Muscle object repr",
            [],
            ["Acetylcholine receptor", "GABA receptor"],
            [],
            "Mu) MDL1",
            "0 0.6 0"
        ]
        // ... 95 muscles
    }
}
```

**Cache Generation Command** (line 1791):
```bash
python -m c302 -cache
# Generates owmeta_cache.json from live owmeta bundle
```

**Muscle Name Conversion** (lines 1823-1824):
```python
# c302 uses zero-padded names: "MDL01", "MVR12"
# owmeta uses compact names: "MDL1", "MVR12"
ow_name = muscle[1:] if muscle[3] != "0" else "%s%s" % (muscle[1:3], muscle[-1])
# "MDL01" → "MDL1"
# "MDL23" → "MDL23"
```

**Fallback Behavior:**
- If owmeta bundle unavailable: loads cached JSON file
- If cache unavailable: system continues without cell metadata (reduced visualization quality)

### 6A.4 Neuron Name Lists

**Preferred Neuron Names** (from `ConnectomeReader.py`):
```python
PREFERRED_NEURON_NAMES = [
    "ADAL", "ADAR", "ADEL", "ADER", "ADFL", "ADFR",
    # ... all 302 neurons (alphabetically sorted)
    "VD1", "VD2", ..., "VD13"
]
```

**Preferred Muscle Names** (from `ConnectomeReader.py`):
```python
PREFERRED_MUSCLE_NAMES = [
    "MDL01", "MDL02", ..., "MDL24",
    "MDR01", "MDR02", ..., "MDR24",
    "MVL01", "MVL02", ..., "MVL24",
    "MVR01", "MVR02", ..., "MVR24",
    "MANAL", "MVULVA"
]
```

---

## 7. MAIN NETWORK GENERATION FUNCTION

### 7.1 `generate()` Function Signature (lines 652-675)

**Core Parameters:**
```python
def generate(
    net_id,                    # Unique network identifier
    params,                    # ParameterisedModel instance (A, B, C, D, etc.)
    data_reader=DEFAULT_DATA_READER,  # Connectome data source
    cells=None,                # Subset of cells to include (None = all 302)
    cells_to_plot=None,        # Cells to include in plots (None = all)
    cells_to_stimulate=None,   # Cells receiving offset current (None = all)
    muscles_to_include=[],     # Muscle cells to include (empty = none)
    conns_to_include=[],       # Connection whitelist (empty = all)
    conns_to_exclude=[],       # Connection blacklist (empty = none)
    conn_number_override=None, # Dict: {"AVAL-AVBR": 2.5, ...}
    conn_number_scaling=None,  # Dict: {"AVAL-AVBR": 2.0, ...}
    conn_polarity_override=None, # Dict: {"AVAL-AVBR": "inh", ...}
    duration=500,              # Simulation duration (ms)
    dt=0.01,                   # Time step (ms)
    vmin=None,                 # Plot voltage minimum (mV)
    vmax=None,                 # Plot voltage maximum (mV)
    seed=1234,                 # Random seed for reproducibility
    test=False,                # Test mode flag
    verbose=True,              # Verbose output
    print_connections=False,   # Print all connection details
    param_overrides={},        # Bioparameter overrides
    target_directory="./"      # Output directory
)
```

**Returns:**
- `NeuroMLDocument`: Complete network specification

**Outputs to Files:**
1. `{net_id}.net.nml` - NeuroML2 network file
2. `LEMS_{net_id}.xml` - LEMS simulation file
3. `cells/{cell}_D.cell.nml` - Multi-compartmental cell files (Level D only)

### 7.2 Key Utility Functions

**Muscle Position Calculation** (lines 122-141):
```python
def get_muscle_position(muscle, data_reader):
    # Parse muscle name: M[VD][LR](\d+)
    # Examples: MDR01, MVL23
    dv = "V" or "D"  # Ventral or Dorsal
    lr = "L" or "R"  # Left or Right
    idx = 1-24       # Muscle index

    x = 80 * (1 if lr == "L" else -1)      # ±80
    z = 80 * (-1 if dv == "V" else 1)      # ±80
    y = -300 + 30 * int(idx)               # -300 to +420 (along body)

    return (x, y, z)
```

**Cell ID String Generation** (lines 506-519):
```python
def get_cell_id_string(cell, params, muscle=False):
    # For Level D (multi-compartmental):
    #   Neuron: "../{cell}/0/{cell}"  (cell-specific morphology)
    #   Muscle: "../{cell}/0/{generic_muscle_cell.id}"
    # For other levels:
    #   Neuron: "../{cell}/0/{generic_neuron_cell.id}"
    #   Muscle: "../{cell}/0/{generic_muscle_cell.id}"
```

**Random Color Generation** (lines 447-456):
```python
def get_random_colour_hex():
    # Returns hex color like "#3A7F2B"
    # Used for plotting individual cell traces
```

**Connection Shorthand** (lines 1162, 1443):
```python
conn_shorthand = "{pre_cell}-{post_cell}"       # "AVAL-AVAR"
conn_shorthand_gj = "{pre_cell}-{post_cell}_GJ" # "AVAL-AVAR_GJ"
```

**Projection ID Generation** (lines 438-444):
```python
def get_projection_id(pre, post, synclass, syntype):
    return "NC_{pre}_{post}_{synclass}"  # Example: "NC_AVAL_AVAR_Acetylcholine"
```

### 7.3 Stimulation Input Generation

**Pulse Stimulation** (lines 371-376):
```python
def add_new_input(nml_doc, cell, delay, duration, amplitude, params):
    # Creates PulseGenerator: constant current injection
    # delay: onset time (ms)
    # duration: stimulus duration (ms)
    # amplitude: current amplitude (pA)
```

**Sinusoidal Stimulation** (lines 341-368):
```python
def add_new_sinusoidal_input(nml_doc, cell, delay, duration, amplitude, period, params):
    # Creates SineGenerator: oscillatory current injection
    # Uses anatomical soma position for phase calculation
    # VB neurons: phase = soma_pos * -0.886, inverted amplitude
    # DB neurons: phase = soma_pos * -0.886, normal amplitude
    # Simulates traveling wave along body for locomotion
```

**Soma Positions for Motor Neurons** (lines 284-306):
```python
VB_soma_pos = {  # Ventral B motor neurons (excitatory to ventral muscles)
    "VB1": 0.21, "VB2": 0.19, "VB3": 0.28, "VB4": 0.32, "VB5": 0.38,
    "VB6": 0.45, "VB7": 0.5, "VB8": 0.57, "VB9": 0.61, "VB10": 0.67, "VB11": 0.72
}

DB_soma_pos = {  # Dorsal B motor neurons (excitatory to dorsal muscles)
    "DB1": 0.24, "DB2": 0.21, "DB3": 0.3, "DB4": 0.39,
    "DB5": 0.51, "DB6": 0.62, "DB7": 0.72
}
# Positions represent normalized location along body axis (0 = anterior, 1 = posterior)
```

### 7.4 LEMS Simulation File Structure

**Generated LEMS Template Variables** (lines 797-817):
```python
lems_info = {
    "comment": info,                    # Full parameter documentation
    "reference": net_id,                # Network ID
    "duration": duration,               # ms
    "dt": dt,                          # ms
    "vmin": vmin,                      # mV
    "vmax": vmax,                      # mV
    "plots": [],                       # Voltage traces for neurons
    "activity_plots": [],              # Activity/calcium traces
    "muscle_plots": [],                # Voltage traces for muscles
    "muscle_activity_plots": [],       # Muscle activity/calcium
    "to_save": [],                     # Data recording specs (neurons)
    "activity_to_save": [],            # Activity recording specs
    "muscles_to_save": [],             # Data recording specs (muscles)
    "muscles_activity_to_save": [],    # Muscle activity recording
    "cells": [],                       # List of included neurons
    "muscles": [],                     # List of included muscles
    "includes": []                     # NeuroML includes (cell defs, channels)
}
```

---

## 8. CONNECTION PROPERTIES

### 8.1 Connection Number Scaling

**Global Power Scaling:**
```python
params.add_bioparameter("global_connectivity_power_scaling", "0")
# If enabled: n_effective = n_original^scale
# scale=0.5 → sqrt scaling (weakens strong connections)
# scale=2.0 → quadratic scaling (amplifies strong connections)
```

**Per-Connection Override:**
```python
conn_number_override = {
    "AVAL-AVAR": 5,        # Set to exactly 5 synapses
    "I1L-I3": 2.5,         # Set to 2.5 (fractional allowed)
    "AVAR-AVBL_GJ": 2      # Override gap junction count
}
```

**Per-Connection Scaling:**
```python
conn_number_scaling = {
    "AVAL-AVAR": 2,        # Double the synapses
    "I1L-.*": 0.5,         # Halve all I1L outgoing connections (regex)
}
```

**Effective Conductance Calculation:**
```python
# For n synapses with gbase conductance each:
# Total conductance = n × gbase
# If n changes from n1 to n2:
# New gbase = (n2/n1) × old_gbase
```

### 7.2 Connection Filtering

**Include/Exclude Connections:**
```python
conns_to_include = [
    "AVAL-.*",      # Regex: All AVAL connections
    "AVAR-AVBL",    # Specific connection
    ".*-DA.*"       # All connections to DA neurons
]

conns_to_exclude = [
    "RIML-RIMR",    # Exclude this connection
    "DD.-DD."       # Exclude DD-to-DD connections
]
```

### 7.3 Specific Parameter Overrides

**Connection-Specific Synapse Parameters:**
```python
param_overrides = {
    # Specific connection parameters
    "AVAL_to_AVAR_elec_syn_gbase": "0.1 nS",
    "AVAL_to_PVCR_chem_exc_syn_gbase": "0.5 nS",
    "RIS_to_.*_chem_inh_syn_gbase": "1.0 nS",  # Regex pattern

    # Mirrored electrical connections (bidirectional)
    "mirrored_elec_conn_params": {
        "AVAL_to_AVAR_elec_syn_gbase": "0.1 nS",  # Sets both directions
        "AVB._to_DB.$_elec_syn_gbase": "0.05 nS"  # Regex pattern
    }
}
```

---

## 8. SIMULATION PARAMETERS

### 8.1 Default Simulation Settings

**From generate() function:**
```python
duration = 500 ms      # Simulation length
dt = 0.01 ms          # Time step (10 μs)
seed = 1234           # Random seed
```

**Voltage Display Range:**
- Level A/B: -52 to -28 mV
- Level C/D: -60 to +25 mV

### 8.2 Stimulus Configuration

**Offset Current:**
```python
# All levels support unphysiological offset current
params = {
    "unphysiological_offset_current": "0 pA",    # Amplitude
    "unphysiological_offset_current_del": "0 ms", # Delay
    "unphysiological_offset_current_dur": "2000 ms" # Duration
}
```

**Cell-Specific Stimulation:**
```python
cells_to_stimulate = ["AVAL", "AVAR", "PLML", "PLMR"]
# Applies offset current to specified cells only
```

**Sinusoidal Input** (for motor pattern generation):
```python
add_new_sinusoidal_input(
    nml_doc, cell="VB6",
    delay="50ms",
    duration="1000ms",
    amplitude="0.5pA",
    period="200ms"  # 5 Hz oscillation
)
# Phase determined by VB soma position for coordination
```

### 8.3 Recording Configuration

**Cells to Plot:**
```python
cells_to_plot = None  # Plot all cells
# or
cells_to_plot = ["AVAL", "AVAR", "DA1", "DB1", "VD1"]  # Specific cells
```

**Recorded Variables:**
- Membrane potential (v): All levels
- Activity: Level B
- Calcium concentration (caConc): Levels C, C1, D, D1

---

## 9. INTEGRATION WITH NEUROML2

### 9.1 Document Structure

**Top-Level NeuroML Document:**
```xml
<neuroMLDocument id="c302_Full_C">
    <notes>
        <!-- Metadata about parameters, cells, connections -->
    </notes>

    <!-- Cell definitions -->
    <cells> or <iafCells>

    <!-- Synapse definitions -->
    <expTwoSynapses>
    <gapJunctions>
    <gradedSynapses>

    <!-- Stimulus generators -->
    <pulseGenerators>
    <sineGenerators>

    <!-- Concentration models -->
    <fixedFactorConcentrationModels>

    <!-- Network structure -->
    <network id="c302_Full_C">
        <populations>
        <projections>
        <electricalProjections>
        <continuousProjections>
        <inputLists>
    </network>
</neuroMLDocument>
```

### 9.2 Custom Component Types

**Level B** (`cell_B.xml`):
- `iafActivityCell`: IAF with activity variable

**Levels C, C1, D, D1** (`cell_C.xml`):
- Ion channel definitions:
  - `Leak`: Linear leak conductance
  - `k_slow`: Slow potassium channel
  - `k_fast`: Fast potassium channel
  - `ca_boyle`: Calcium channel (Boyle model)

**Level D1** (`custom_synapses.xml`):
- `gradedSynapse2`: Advanced graded synapse with rise/decay

### 9.3 Cell Morphology Files

**Location**: `/NeuroML2/{CELL}.cell.nml`

**Contents:**
- Detailed 3D morphology
- Segment tree structure
- Soma, dendrites, axon (if applicable)
- 3D coordinates for each segment

**Loading in Code:**
```python
cell_file = root_dir + "NeuroML2/%s.cell.nml" % cell_name
doc = loaders.NeuroMLLoader.load(cell_file)
cell_morphology = doc.cells[0].morphology
location = cell_morphology.segments[0].proximal  # Soma position
```

**Level D Custom Cells:**
- Each neuron gets unique cell definition
- Combines morphology with biophysical parameters
- Saved to: `cells/{CELL}_D.cell.nml`
- Includes full ion channel densities

---

## 10. ANALYSIS AND UTILITIES

### 10.1 Connection Analysis (`c302_utils.py`)

**Matrix Visualization:**
```python
generate_conn_matrix(
    nml_doc,
    save_fig_dir="./figures",
    verbose=True,
    figsize=(12, 12),
    colormap="nipy_spectral"
)
```

**Generates:**
- Excitatory connections to neurons
- Excitatory connections to muscles
- Inhibitory connections to neurons
- Inhibitory connections to muscles
- Gap junctions between neurons
- Gap junctions between neurons and muscles
- Gap junctions between muscles

**Output Format:**
- Heatmaps with pre-synaptic cells (rows) × post-synaptic cells (columns)
- Color intensity = number of synapses
- Cell labels include neurotransmitter type

### 10.2 Plotting Results (`runAndPlot.py`)

**Voltage Traces:**
```python
plot_c302_results(
    lems_results,       # Simulation results dict
    config="Full",      # Configuration name
    parameter_set="C1", # Parameter level
    directory="./",     # Output directory
    save=True,
    show_plot_already=True,
    plot_ca=True        # Plot calcium if available
)
```

**Generated Plots:**
1. Neuron membrane potentials (heatmap)
2. Neuron voltage traces (line plots)
3. Neuron activity/calcium (heatmap and traces)
4. Muscle membrane potentials (heatmap)
5. Muscle voltage traces (line plots)
6. Muscle activity/calcium (heatmap and traces)

### 10.3 Data Validation

**Connection Analysis Output:**
```python
analyse_connections(cells, neuron_conns, neurons2muscles, muscles, muscle_conns)
```

**Reports:**
- Total number of cells (should be 302)
- Cells not in standard neuron list
- Known neurons not present in dataset
- Total connections with breakdown by neurotransmitter
- Average synapses per connection
- Muscle count and identifiers
- Motor neurons list
- Neuron-muscle connection statistics

---

## 11. KEY FINDINGS & RECOMMENDATIONS

### 11.1 Model Level Selection Guide

**For Fast Prototyping:**
- Use Level A or B
- Quick simulations
- Suitable for connectivity testing
- NOT biologically accurate

**For Oscillatory Behavior:**
- Use Level C1 (graded synapses)
- No spiking requirement
- Analog communication
- Good for motor pattern generation

**For Realistic Modeling:**
- Use Level D1
- Multi-compartmental neurons
- Graded synapses with kinetics
- Most complete biophysical detail
- Highest computational cost

**For Spiking Networks:**
- Use Level C or D with event-based synapses
- Requires clear action potentials
- Good for studying spike timing
- Medium computational cost

### 11.2 Common Parameter Tuning

**To Increase Network Excitability:**
1. Increase chemical synapse conductances (gbase)
2. Decrease inhibitory synapse conductances
3. Add positive offset current
4. Decrease leak conductances
5. Increase calcium channel densities

**To Generate Rhythmic Activity:**
1. Use Level C1 or D1 (graded synapses)
2. Tune gap junction conductances (0.001-0.01 nS)
3. Balance excitation/inhibition ratios
4. Consider sinusoidal inputs for motor neurons
5. Enable calcium dynamics for oscillations

**To Improve Stability:**
1. Decrease excitatory conductances
2. Increase inhibitory conductances
3. Add leak conductance
4. Use power scaling (<1.0) on connection numbers
5. Increase capacitance for slower dynamics

### 11.3 Known Limitations

**Level A/B:**
- IAF neurons unrealistic for C. elegans
- No dendritic processing
- Event-based synapses require spikes

**Level C/D:**
- Event-based synapses still unrealistic
- May need manual tuning for spiking

**Level D/D1:**
- Resistivity parameter critical but uncertain
- High resistivity → localized dynamics
- Low resistivity → high input resistance issues

**All Levels:**
- Gap junction parameters poorly constrained
- Neurotransmitter classification simplified (only ACh/GABA)
- Missing neuromodulation (dopamine, serotonin, etc.)
- Muscle models very simplified

### 11.4 Integration Recommendations

**For HyperPhysics Integration:**

1. **Extract Connectivity Matrix:**
   ```python
   reader = Cook2019DataReader()
   cells, conns = reader.read_data()
   # Build adjacency matrix with synapse types
   ```

2. **Use Level D1 Parameters:**
   - Most biologically realistic
   - Graded synapses match biological analog transmission
   - Multi-compartmental allows spatial integration

3. **Focus on Motor Circuit:**
   - VA, VB, DA, DB, VD, DD motor neurons
   - Command interneurons: AVA, AVB, PVC
   - Proprioceptive feedback: DVA, AVA

4. **Leverage Neurotransmitter Info:**
   - GABA → inhibitory connections
   - ACh → excitatory connections
   - Gap junctions → electrical coupling

5. **Adapt Synapse Models:**
   - GradedSynapse2 has explicit kinetics (ar, ad)
   - Can map to continuous-time dynamics
   - Consider voltage-dependent activation

---

## 12. CODE STRUCTURE SUMMARY

### 12.1 Main Entry Points

**Command Line:**
```bash
python -m c302 <reference> <parameters> [options]
# Example:
python -m c302 c302_Full parameters_C1 -cells "[AVAL,AVAR,DA1,DB1]"
```

**Programmatic:**
```python
import c302
from c302.parameters_C1 import ParameterisedModel

params = ParameterisedModel()
nml_doc = c302.generate(
    net_id="MyNetwork",
    params=params,
    cells=["AVAL", "AVAR"],
    muscles_to_include=["MVL01", "MVR01"],
    duration=1000,
    dt=0.01
)
```

### 12.2 Key Classes

**`ConnectionInfo`** (`ConnectomeReader.py`):
- Represents single connection
- Attributes: pre_cell, post_cell, number, syntype, synclass

**`BioParameter`** (`bioparameters.py`):
- Name-value pair with metadata
- Source and certainty tracking
- Unit parsing and manipulation

**`c302ModelPrototype`** (`bioparameters.py`):
- Base class for all parameter levels
- Methods for creating cells, synapses
- Parameter override system

**`ParameterisedModel`** (each `parameters_X.py`):
- Level-specific implementation
- Defines all biophysical parameters
- Creates NeuroML2 components

### 12.3 Critical Functions

**Network Generation:**
- `generate()`: Main network creation
- `get_cell_names_and_connection()`: Load connectome
- `get_cell_muscle_names_and_connection()`: Load neuromuscular data

**Cell Creation:**
- `create_generic_neuron_cell()`: Standard neuron
- `create_generic_muscle_cell()`: Standard muscle
- `create_neuron_cell()`: Custom neuron (Level D)

**Synapse Creation:**
- `create_neuron_to_neuron_syn()`: N→N synapses
- `create_neuron_to_muscle_syn()`: N→M synapses
- `create_n_connection_synapse()`: Scale for multiple synapses

**Output:**
- `write_to_file()`: Save NeuroML2 and LEMS
- `validate_neuroml2()`: Validate against schema

---

## 13. COMMAND-LINE INTERFACE

### 13.1 Main Entry Point (`main()` function, lines 1758-1785)

**Basic Usage:**
```bash
python -m c302 <reference> <parameters> [options]
```

**Required Arguments:**
- `<reference>`: Unique network identifier (e.g., "c302_A_Full")
- `<parameters>`: Parameter set module (e.g., "parameters_A")

**Optional Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-datareader` | str | `cect.SpreadsheetDataReader` | Data reader to use |
| `-cells` | list | `None` | Specific cells to include (e.g., `["AVAL","AVAR"]`) |
| `-cellstoplot` | list | `None` | Cells to plot in visualization |
| `-cellstostimulate` | list | `None` | Cells to receive stimulus |
| `-musclestoinclude` | list | `[]` | Muscles to include |
| `-connpolarityoverride` | dict | `None` | Override connection polarities |
| `-connnumberoverride` | dict | `None` | Override connection numbers |
| `-connnumberscaling` | dict | `None` | Scale connection numbers |
| `-paramoverride` | dict | `None` | Override bioparameters |
| `-duration` | float | `100` | Simulation duration (ms) |
| `-dt` | float | `0.01` | Time step (ms) |
| `-vmin` | float | `-80` | Plot voltage minimum (mV) |
| `-vmax` | float | `-40` | Plot voltage maximum (mV) |

**Example Commands:**

1. **Full network with Level C parameters:**
```bash
python -m c302 c302_C_Full parameters_C
```

2. **Subset of cells for forward locomotion:**
```bash
python -m c302 c302_C_Pharynx parameters_C \
    -cells '["AVAL","AVAR","AVBL","AVBR","DA1","DA2","DA3","DB1","DB2","DB3","VB1","VB2"]' \
    -musclestoinclude '["MDL01","MDL02","MDR01","MDR02","MVL01","MVL02","MVR01","MVR02"]' \
    -duration 1000 \
    -dt 0.005
```

3. **Override connection strength:**
```bash
python -m c302 c302_D_Social parameters_D \
    -connnumberoverride '{"AVAL-AVAR":5.0,"PLML-AVAL":2.0}' \
    -connpolarityoverride '{"RIML-SMBVR":"inh"}'
```

4. **Generate owmeta cache (special mode):**
```bash
python -m c302 -cache
# Creates owmeta_cache.json from OpenWorm data bundle
```

### 13.2 Argument Parsing Functions (lines 1727-1755)

**List Argument Parser** (`parse_list_arg`, lines 1727-1735):
```python
# Input: '["AVAL","AVBL","PLML"]'
# Output: ["AVAL", "AVBL", "PLML"]
```

**Dictionary Argument Parser** (`parse_dict_arg`, lines 1744-1755):
```python
# Input: '{"ADAL-AIBL":2.5,"I1L-I1R":0.5,"AVAL-AVAR":"inh"}'
# Output: {"ADAL-AIBL": 2.5, "I1L-I1R": 0.5, "AVAL-AVAR": "inh"}
# Supports both numeric and string values
```

### 13.3 Output Files

**Generated Files:**
1. `<reference>.net.nml` - NeuroML2 network specification
2. `LEMS_<reference>.xml` - LEMS simulation configuration
3. `cells/*_D.cell.nml` - Individual cell morphologies (Level D only)

**File Locations:**
- Default: Current working directory (`./`)
- Configurable via `target_directory` parameter in `generate()`

### 13.4 Validation

**Automatic NeuroML2 Validation:**
- Enabled by default for most levels
- Disabled for: Level B, C0, C2, D1 (experimental)
- Uses `neuroml.utils.validate_neuroml2()`
- Validates against official NeuroML2 schema

**Validation Bypass:**
```python
validate = not (
    params.is_level_B() or
    params.is_level_C0() or
    params.is_level_C2 or
    params.is_level_D1()
)
```

---

## APPENDIX A: FILE REFERENCE

### Core Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 1841 | Main network generation engine |
| `ConnectomeReader.py` | 630 | Connection data structures and analysis |
| `c302_utils.py` | - | Plotting and visualization utilities |
| `bioparameters.py` | 206 | Parameter management framework |
| `NeuroMLUtilities.py` | 52 | NeuroML file utilities |

### Parameter Files

| File | Level | Type | Lines |
|------|-------|------|-------|
| `parameters_A.py` | A | IAF + Event synapses | 366 |
| `parameters_B.py` | B | IAF + GapJunction | 193 |
| `parameters_C.py` | C | Conductance + Event synapses | 530 |
| `parameters_C1.py` | C1 | Conductance + Graded synapses | 261 |
| `parameters_D.py` | D | Multi-compartment + Event | 525 |
| `parameters_D1.py` | D1 | Multi-compartment + Graded2 | 403 |

### Data Readers

| File | Dataset | Format |
|------|---------|--------|
| `Cook2019DataReader.py` | Cook et al. 2019 | Excel adjacency matrix |
| `SpreadsheetDataReader.py` | Generic | CSV/Excel |
| `VarshneyDataReader.py` | Varshney et al. | Custom format |
| `WitvlietDataReader1.py` | Witvliet et al. | Custom format |
| `WormNeuroAtlasReader.py` | WormNeuroAtlas | API/database |

---

## APPENDIX B: NEURON POSITIONAL DATA

### VB Motor Neuron Soma Positions

Soma position as fraction of body length (from White et al.):

```python
VB_soma_pos = {
    "VB1": 0.21,  "VB2": 0.19,  "VB3": 0.28,
    "VB4": 0.32,  "VB5": 0.38,  "VB6": 0.45,
    "VB7": 0.5,   "VB8": 0.57,  "VB9": 0.61,
    "VB10": 0.67, "VB11": 0.72
}
```

### DB Motor Neuron Soma Positions

```python
DB_soma_pos = {
    "DB1": 0.24,  "DB2": 0.21,  "DB3": 0.3,
    "DB4": 0.39,  "DB5": 0.51,  "DB6": 0.62,
    "DB7": 0.72
}
```

**Usage:** Determines phase offset for sinusoidal inputs to create traveling waves in motor patterns.

---

## APPENDIX C: COMPLETE SYNAPSE PARAMETER TABLE

| Level | Syn Type | Exc Cond | Inh Cond | GJ Cond | Rise/AR | Decay/AD | Notes |
|-------|----------|----------|----------|---------|---------|----------|-------|
| A | ExpTwo | 0.01 nS | 0.01 nS | 0 nS | 3 ms | 10 ms | Pseudo-GJ |
| B | ExpTwo + GJ | 0.01 nS | 0.01 nS | 0.01 nS | 3 ms | 10 ms | Real GJ |
| C | ExpTwo + GJ | 0.1 nS | 0.1 nS | 0.0005 nS | 1 ms (exc), 2 ms (inh) | 5 ms (exc), 40 ms (inh) | HH channels |
| C1 | Graded + GJ | 0.09 nS | 0.09 nS | 0.00052 nS | k=0.025/ms | k=0.025/ms | Analog |
| D | ExpTwo + GJ | 0.01 nS | 3 nS | 0.0005 nS | 1 ms (exc), 2 ms (inh) | 5 ms (exc), 40 ms (inh) | Multi-comp |
| D1 | Graded2 + GJ | 2 nS | 26 nS (N), 0.25 nS (M) | 0.005 nS (N), 0.0001 nS (M) | 0.5/s (exc), 0.005/s (inh) | 20/s (exc), 10/s (inh) | Most realistic |

**Legend:**
- Cond = Conductance
- GJ = Gap Junction
- N = Neuron-to-neuron
- M = Neuron-to-muscle
- AR = Activation rate
- AD = Deactivation rate

---

## REFERENCES

1. Cook, S. J., et al. (2019). "Whole-animal connectomes of both Caenorhabditis elegans sexes." Nature, 571(7763), 63-71.

2. White, J. G., et al. (1986). "The structure of the nervous system of the nematode Caenorhabditis elegans." Philosophical Transactions of the Royal Society B, 314(1165), 1-340.

3. Varshney, L. R., et al. (2011). "Structural properties of the Caenorhabditis elegans neuronal network." PLoS Computational Biology, 7(2), e1001066.

4. Witvliet, D., et al. (2020). "Connectomes across development reveal principles of brain maturation in C. elegans." bioRxiv.

5. NeuroML Documentation: https://docs.neuroml.org/

6. OpenWorm Project: http://www.openworm.org/

7. c302 GitHub Repository: https://github.com/openworm/CElegansNeuroML/tree/master/CElegans/pythonScripts/c302

---

**Document Version**: 1.0
**Date**: 2025-12-03
**Analysis Based On**: c302 codebase at commit 0b7c6604d2

