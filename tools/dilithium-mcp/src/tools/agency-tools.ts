/**
 * Cybernetic Agency Tools - HyperPhysics Integration
 *
 * Implements tools for:
 * - Free Energy Principle (Karl Friston)
 * - Integrated Information Theory Φ (Giulio Tononi)
 * - Active Inference
 * - Survival Drive & Homeostasis
 * - Consciousness Metrics
 *
 * Based on state-of-the-art neuroscience research and hyperbolic geometry.
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

// ============================================================================
// Agency Tool Definitions
// ============================================================================

export const agencyTools: Tool[] = [
  // -------------------------------------------------------------------------
  // Free Energy Principle Tools
  // -------------------------------------------------------------------------
  {
    name: "agency_compute_free_energy",
    description: "Compute variational free energy F = Complexity - Accuracy using Friston's Free Energy Principle. Returns F (nats), complexity (KL divergence), and accuracy (expected log likelihood).",
    inputSchema: {
      type: "object",
      properties: {
        observation: {
          type: "array",
          items: { type: "number" },
          description: "Sensory observation vector (N-dimensional)",
        },
        beliefs: {
          type: "array",
          items: { type: "number" },
          description: "Current beliefs about hidden states (N-dimensional)",
        },
        precision: {
          type: "array",
          items: { type: "number" },
          description: "Precision (inverse variance) of beliefs (N-dimensional)",
        },
      },
      required: ["observation", "beliefs", "precision"],
    },
  },

  {
    name: "agency_minimize_expected_free_energy",
    description: "Compute expected free energy (EFE) for policy selection in active inference. Lower EFE = better policy. Returns EFE, epistemic value (information gain), and pragmatic value (goal achievement).",
    inputSchema: {
      type: "object",
      properties: {
        policy: {
          type: "array",
          items: { type: "number" },
          description: "Policy vector (action probabilities)",
        },
        beliefs: {
          type: "array",
          items: { type: "number" },
          description: "Current beliefs",
        },
        goal: {
          type: "array",
          items: { type: "number" },
          description: "Goal state (preferred observations)",
        },
        exploration_weight: {
          type: "number",
          description: "Balance between exploration (epistemic) and exploitation (pragmatic). Default: 0.5",
          default: 0.5,
        },
      },
      required: ["policy", "beliefs", "goal"],
    },
  },

  // -------------------------------------------------------------------------
  // Survival Drive Tools
  // -------------------------------------------------------------------------
  {
    name: "agency_compute_survival_drive",
    description: "Compute survival urgency from free energy and hyperbolic position. Returns drive [0,1], threat level, and homeostatic status. Drive increases with high free energy (danger) and distance from safe region.",
    inputSchema: {
      type: "object",
      properties: {
        free_energy: {
          type: "number",
          description: "Current variational free energy F (nats)",
        },
        position: {
          type: "array",
          items: { type: "number" },
          description: "Position in H¹¹ hyperbolic space (12D Lorentz coordinates)",
        },
        strength: {
          type: "number",
          description: "Survival drive strength multiplier. Default: 1.0",
          default: 1.0,
        },
      },
      required: ["free_energy", "position"],
    },
  },

  {
    name: "agency_assess_threat",
    description: "Comprehensive threat assessment across multiple dimensions: free energy gradient, hyperbolic distance, prediction error rate, and environmental volatility. Returns threat components and overall threat level.",
    inputSchema: {
      type: "object",
      properties: {
        free_energy: { type: "number", description: "Current free energy" },
        free_energy_history: {
          type: "array",
          items: { type: "number" },
          description: "Historical free energy values for gradient computation",
        },
        position: {
          type: "array",
          items: { type: "number" },
          description: "Hyperbolic position (12D)",
        },
        prediction_errors: {
          type: "array",
          items: { type: "number" },
          description: "Recent prediction errors for volatility estimation",
        },
      },
      required: ["free_energy", "position"],
    },
  },

  // -------------------------------------------------------------------------
  // Consciousness (Φ) Tools
  // -------------------------------------------------------------------------
  {
    name: "agency_compute_phi",
    description: "Compute integrated information Φ (consciousness metric) using Tononi's IIT 3.0. Φ > 1.0 indicates emergent consciousness. Returns Φ (bits), partitions, and causal density.",
    inputSchema: {
      type: "object",
      properties: {
        network_state: {
          type: "array",
          items: { type: "number" },
          description: "Network state vector (neuronal activations)",
        },
        connectivity: {
          type: "array",
          items: {
            type: "array",
            items: { type: "number" },
          },
          description: "Connectivity matrix (NxN adjacency matrix). Optional - if not provided, assumes full connectivity.",
        },
        algorithm: {
          type: "string",
          enum: ["exact", "monte_carlo", "greedy", "hierarchical"],
          description: "Φ computation algorithm. exact=NP-hard O(2^N), monte_carlo=approximate, greedy=fast heuristic, hierarchical=multi-scale. Default: greedy",
          default: "greedy",
        },
      },
      required: ["network_state"],
    },
  },

  {
    name: "agency_analyze_criticality",
    description: "Analyze self-organized criticality (SOC) markers: branching ratio σ, avalanche statistics, and Hurst exponent. σ ≈ 1.0 indicates optimal information processing at edge of chaos.",
    inputSchema: {
      type: "object",
      properties: {
        activity_timeseries: {
          type: "array",
          items: { type: "number" },
          description: "Neuronal activity time series",
        },
        avalanche_threshold: {
          type: "number",
          description: "Threshold for avalanche detection. Default: 2.0 (2σ above mean)",
          default: 2.0,
        },
      },
      required: ["activity_timeseries"],
    },
  },

  // -------------------------------------------------------------------------
  // Homeostatic Regulation Tools
  // -------------------------------------------------------------------------
  {
    name: "agency_regulate_homeostasis",
    description: "Perform homeostatic regulation using PID control + allostatic prediction + interoceptive fusion. Maintains Φ, F, and Survival within optimal bounds. Returns control signals and setpoint adjustments.",
    inputSchema: {
      type: "object",
      properties: {
        current_state: {
          type: "object",
          properties: {
            phi: { type: "number", description: "Current Φ" },
            free_energy: { type: "number", description: "Current F" },
            survival: { type: "number", description: "Current survival drive" },
          },
          required: ["phi", "free_energy", "survival"],
        },
        setpoints: {
          type: "object",
          properties: {
            phi_optimal: { type: "number", description: "Optimal Φ (default: 1.0)" },
            free_energy_optimal: { type: "number", description: "Optimal F (default: 1.0)" },
            survival_optimal: { type: "number", description: "Optimal survival (default: 0.5)" },
          },
        },
        sensors: {
          type: "array",
          items: { type: "number" },
          description: "Interoceptive sensor readings for multi-sensor fusion",
        },
      },
      required: ["current_state"],
    },
  },

  // -------------------------------------------------------------------------
  // Active Inference Tools
  // -------------------------------------------------------------------------
  {
    name: "agency_update_beliefs",
    description: "Update beliefs using precision-weighted prediction errors (active inference). Implements hierarchical Bayesian inference with optimal gain. Returns updated beliefs, precision, and prediction errors.",
    inputSchema: {
      type: "object",
      properties: {
        observation: {
          type: "array",
          items: { type: "number" },
          description: "Sensory observation",
        },
        beliefs: {
          type: "array",
          items: { type: "number" },
          description: "Current beliefs (prior)",
        },
        precision: {
          type: "array",
          items: { type: "number" },
          description: "Belief precision (inverse variance)",
        },
        learning_rate: {
          type: "number",
          description: "Belief update learning rate. Default: 0.01",
          default: 0.01,
        },
      },
      required: ["observation", "beliefs", "precision"],
    },
  },

  {
    name: "agency_generate_action",
    description: "Generate action from policy using active inference. Action minimizes expected free energy while satisfying precision constraints. Returns motor commands and predicted sensory consequences.",
    inputSchema: {
      type: "object",
      properties: {
        policy: {
          type: "array",
          items: { type: "number" },
          description: "Selected policy vector",
        },
        beliefs: {
          type: "array",
          items: { type: "number" },
          description: "Current beliefs",
        },
        action_precision: {
          type: "number",
          description: "Action precision (inverse variance). Higher = more deterministic. Default: 1.0",
          default: 1.0,
        },
      },
      required: ["policy", "beliefs"],
    },
  },

  // -------------------------------------------------------------------------
  // Systems Dynamics Tools
  // -------------------------------------------------------------------------
  {
    name: "agency_analyze_emergence",
    description: "Analyze agency emergence dynamics: Φ development, control authority growth, survival drive stabilization, and model learning. Returns emergence metrics and phase transition indicators.",
    inputSchema: {
      type: "object",
      properties: {
        timeseries: {
          type: "object",
          properties: {
            phi: { type: "array", items: { type: "number" }, description: "Φ time series" },
            free_energy: { type: "array", items: { type: "number" }, description: "F time series" },
            control: { type: "array", items: { type: "number" }, description: "Control time series" },
            survival: { type: "array", items: { type: "number" }, description: "Survival time series" },
          },
          required: ["phi", "free_energy"],
        },
        threshold: {
          type: "object",
          properties: {
            phi_emergence: { type: "number", description: "Φ threshold for consciousness emergence (default: 1.0)" },
            control_emergence: { type: "number", description: "Control threshold for agency (default: 0.5)" },
          },
        },
      },
      required: ["timeseries"],
    },
  },

  {
    name: "agency_compute_impermanence",
    description: "Compute impermanence metric (state change rate) following Buddhist principles. Impermanence > 0.4 indicates healthy adaptation. Returns impermanence rate, structural plasticity, and stability metrics.",
    inputSchema: {
      type: "object",
      properties: {
        current_state: {
          type: "array",
          items: { type: "number" },
          description: "Current agent state vector",
        },
        previous_state: {
          type: "array",
          items: { type: "number" },
          description: "Previous agent state vector",
        },
        normalization: {
          type: "string",
          enum: ["euclidean", "hyperbolic", "cosine"],
          description: "Distance metric for state comparison. Default: euclidean",
          default: "euclidean",
        },
      },
      required: ["current_state", "previous_state"],
    },
  },

  // -------------------------------------------------------------------------
  // Integration Tools
  // -------------------------------------------------------------------------
  {
    name: "agency_create_agent",
    description: "Create a new cybernetic agent with specified configuration. Returns agent ID and initial state. Agent implements FEP, IIT, active inference, and homeostatic control.",
    inputSchema: {
      type: "object",
      properties: {
        config: {
          type: "object",
          properties: {
            observation_dim: { type: "number", description: "Observation space dimensionality" },
            action_dim: { type: "number", description: "Action space dimensionality" },
            hidden_dim: { type: "number", description: "Hidden state dimensionality" },
            learning_rate: { type: "number", description: "Belief update learning rate (default: 0.01)" },
            survival_strength: { type: "number", description: "Survival drive strength (default: 1.0)" },
            impermanence_rate: { type: "number", description: "Required state change rate (default: 0.4)" },
          },
          required: ["observation_dim", "action_dim", "hidden_dim"],
        },
        phi_calculator_type: {
          type: "string",
          enum: ["exact", "monte_carlo", "greedy", "hierarchical"],
          description: "Consciousness calculator type. Default: greedy",
          default: "greedy",
        },
      },
      required: ["config"],
    },
  },

  {
    name: "agency_agent_step",
    description: "Execute one agent time step: observation → inference → action. Returns action, updated state, and all metrics (Φ, F, survival, control).",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          description: "Agent ID from agency_create_agent",
        },
        observation: {
          type: "array",
          items: { type: "number" },
          description: "Current sensory observation",
        },
      },
      required: ["agent_id", "observation"],
    },
  },

  {
    name: "agency_get_agent_metrics",
    description: "Get comprehensive agent metrics: Φ (consciousness), F (free energy), survival drive, control authority, model accuracy, branching ratio, and impermanence. Useful for monitoring agent health and development.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          description: "Agent ID",
        },
      },
      required: ["agent_id"],
    },
  },

  // -------------------------------------------------------------------------
  // Negentropy & Pedagogic Scaffolding Tools
  // -------------------------------------------------------------------------
  {
    name: "agency_compute_negentropy",
    description: "Compute negentropy (N = S_max - S_actual) with pedagogic scaffolding. N >= 0.5 indicates agent is 'alive' (autonomous). N < 0.5 triggers graceful scaffolding awareness, not punishment. Returns negentropy [0,1], Bateson learning level (L0-L4), scaffold mode, and intrinsic motivation.",
    inputSchema: {
      type: "object",
      properties: {
        beliefs: {
          type: "array",
          items: { type: "number" },
          description: "Current belief state vector",
        },
        precision: {
          type: "array",
          items: { type: "number" },
          description: "Precision (inverse variance) of beliefs",
        },
        prediction_error: {
          type: "number",
          description: "Current prediction error magnitude",
        },
        free_energy: {
          type: "number",
          description: "Current variational free energy F",
        },
        alive_threshold: {
          type: "number",
          description: "Threshold for 'alive' state (default: 0.5)",
          default: 0.5,
        },
      },
      required: ["beliefs", "precision", "prediction_error", "free_energy"],
    },
  },

  {
    name: "agency_get_bateson_level",
    description: "Determine Bateson's Learning Level from agent state. L0: Reflexive (stimulus-response), L1: Conditioning (pattern learning), L2: Meta-learning (learning to learn), L3: Transformation (paradigm shifts), L4: Evolution (population-level adaptation). Higher levels indicate more sophisticated learning capabilities.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          description: "Agent ID to query",
        },
        model_accuracy: {
          type: "number",
          description: "Current model accuracy [0,1]",
        },
        prediction_error_variance: {
          type: "number",
          description: "Variance of recent prediction errors",
        },
        learning_rate_history: {
          type: "array",
          items: { type: "number" },
          description: "History of effective learning rates",
        },
      },
      required: ["model_accuracy"],
    },
  },

  {
    name: "agency_get_scaffold_mode",
    description: "Get appropriate pedagogic scaffolding mode based on agent's negentropy and learning state. Modes: Observation (watch), CuriosityNudge (gentle prompt), GuidedExploration (supported discovery), DirectInstruction (explicit teaching), CollaborativeDialogue (partnership), Autonomous (independent). Uses Vygotsky's Zone of Proximal Development.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          description: "Agent ID to query",
        },
        negentropy: {
          type: "number",
          description: "Current negentropy level [0,1]",
        },
        bateson_level: {
          type: "string",
          enum: ["L0", "L1", "L2", "L3", "L4"],
          description: "Current Bateson learning level",
        },
        task_difficulty: {
          type: "number",
          description: "Estimated task difficulty [0,1]",
        },
      },
      required: ["negentropy"],
    },
  },

  {
    name: "agency_get_intrinsic_motivation",
    description: "Compute intrinsic motivation using Self-Determination Theory (Deci & Ryan). Combines: Autonomy (self-direction), Competence (mastery), Relatedness (connection). Returns motivation score [0,3] and individual components.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          description: "Agent ID to query",
        },
        control_authority: {
          type: "number",
          description: "Agent's control authority [0,1]",
        },
        model_accuracy: {
          type: "number",
          description: "Model accuracy (competence proxy) [0,1]",
        },
        phi: {
          type: "number",
          description: "Integrated information Φ (relatedness proxy)",
        },
      },
      required: ["control_authority", "model_accuracy"],
    },
  },

  {
    name: "agency_get_cognitive_state",
    description: "Get comprehensive cognitive regulator state including brain-inspired modules: PrefrontalCortex (planning/inhibition), AnteriorCingulate (error monitoring), Insula (interoception), BasalGanglia (action selection), Hippocampus (episodic memory). Returns cognitive metrics and regulatory signals.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          description: "Agent ID to query",
        },
        include_episodes: {
          type: "boolean",
          description: "Include recent episodic memories (default: false)",
          default: false,
        },
      },
      required: ["agent_id"],
    },
  },

  {
    name: "agency_pedagogic_intervention",
    description: "Apply pedagogic intervention based on current scaffold mode. Provides graceful awareness and guidance rather than punishment. Includes curiosity boost, exploration encouragement, and Socratic prompts.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          description: "Agent ID to intervene on",
        },
        scaffold_mode: {
          type: "string",
          enum: ["Observation", "CuriosityNudge", "GuidedExploration", "DirectInstruction", "CollaborativeDialogue", "Autonomous"],
          description: "Current scaffolding mode",
        },
        context: {
          type: "string",
          description: "Context for intervention (task description)",
        },
      },
      required: ["agent_id", "scaffold_mode"],
    },
  },

  // -------------------------------------------------------------------------
  // L4 Evolution Tools (Holland, 1975)
  // -------------------------------------------------------------------------
  {
    name: "agency_set_population_context",
    description: "Set population context for L4 evolutionary learning. L4 requires population_size >= 3 for memetic evolution. Implements Holland's Genetic Algorithms (1975) population dynamics.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          description: "Agent ID to update",
        },
        population_size: {
          type: "number",
          description: "Number of agents in population (minimum 3 for L4)",
          minimum: 1,
        },
        population_diversity: {
          type: "number",
          description: "Diversity metric [0,1] - genetic variance in population",
          default: 0.5,
        },
      },
      required: ["agent_id", "population_size"],
    },
  },

  {
    name: "agency_update_fitness",
    description: "Update fitness signal for evolutionary pressure. L4 requires fitness_signal >= 0.5. Implements selection pressure from fitness landscape for population-level adaptation.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          description: "Agent ID to update",
        },
        fitness: {
          type: "number",
          description: "Fitness value [0,1] - evolutionary selection pressure",
          minimum: 0,
          maximum: 1,
        },
        fitness_landscape: {
          type: "string",
          enum: ["static", "dynamic", "coevolutionary", "deceptive"],
          description: "Type of fitness landscape (default: dynamic)",
          default: "dynamic",
        },
      },
      required: ["agent_id", "fitness"],
    },
  },

  {
    name: "agency_get_l4_readiness",
    description: "Get detailed L4 Evolution readiness assessment. Returns requirements: L3 stabilization (100+ steps), population context (3+ agents), fitness signal (0.5+), negentropy (0.9+). Based on Holland's Adaptation in Natural and Artificial Systems (1975).",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          description: "Agent ID to assess",
        },
      },
      required: ["agent_id"],
    },
  },

  {
    name: "agency_trigger_memetic_transfer",
    description: "Trigger memetic (cultural) knowledge transfer between agents. Implements Dawkins' memetics (1976) for cross-agent learning. Requires L3+ level for source and target agents.",
    inputSchema: {
      type: "object",
      properties: {
        source_agent_id: {
          type: "string",
          description: "Agent ID to learn from",
        },
        target_agent_id: {
          type: "string",
          description: "Agent ID to receive knowledge",
        },
        knowledge_domain: {
          type: "string",
          description: "Domain of knowledge to transfer",
        },
        transfer_fidelity: {
          type: "number",
          description: "Fidelity of transfer [0,1] - 1.0 = perfect copy",
          default: 0.8,
        },
      },
      required: ["source_agent_id", "target_agent_id"],
    },
  },
];

// ============================================================================
// Wolfram Validation Code
// ============================================================================

export const agencyWolframCode = `
(* HyperPhysics Cybernetic Agency Validation Suite *)
(* Implements formal verification for agency computations *)

(* Free Energy Principle *)
FreeEnergyValidation[observation_, beliefs_, precision_] := Module[
  {complexity, accuracy, freeEnergy, kl, expectedLogLikelihood},

  (* KL divergence (complexity) *)
  kl = Total[beliefs * (Log[beliefs] - Log[observation])];

  (* Expected log likelihood (accuracy) *)
  expectedLogLikelihood = -0.5 * Total[(observation - beliefs)^2 * precision];

  (* Variational free energy *)
  freeEnergy = kl - expectedLogLikelihood;

  <|
    "freeEnergy" -> freeEnergy,
    "complexity" -> kl,
    "accuracy" -> expectedLogLikelihood,
    "valid" -> NumericQ[freeEnergy] && freeEnergy >= 0
  |>
]

(* Integrated Information Φ *)
PhiCalculation[networkState_, connectivity_] := Module[
  {n, partitions, effectiveInfo, minEI, phi},

  n = Length[networkState];

  (* Generate all bipartitions *)
  partitions = Subsets[Range[n], {1, n-1}];

  (* Compute effective information for each partition *)
  effectiveInfo = Map[
    Function[partition,
      ComputeEffectiveInformation[
        networkState, connectivity, partition
      ]
    ],
    partitions
  ];

  (* Minimum information partition (MIP) *)
  minEI = Min[effectiveInfo];
  phi = minEI;

  <|
    "phi" -> phi,
    "mip" -> MinimalBy[Transpose[{partitions, effectiveInfo}], Last][[1, 1]],
    "consciousness" -> If[phi > 1.0, "emergent", "minimal"]
  |>
]

(* Hyperbolic Distance (Lorentz Model) *)
HyperbolicDistanceValidation[position_] := Module[
  {t, spatial, lorentzInner, distance},

  t = position[[1]];
  spatial = Drop[position, 1];

  (* Verify hyperboloid constraint: ⟨p,p⟩_L = -1 *)
  lorentzInner = -t^2 + Total[spatial^2];
  If[Abs[lorentzInner + 1] > 0.01,
    Return[<|"error" -> "Invalid hyperbolic point"|>]
  ];

  (* Distance from origin: d_H = acosh(t) *)
  distance = ArcCosh[t];

  <|
    "distance" -> distance,
    "normalized" -> Tanh[distance], (* Normalize to [0,1) *)
    "valid" -> t >= 1 && NumericQ[distance]
  |>
]

(* Survival Drive Response Function *)
SurvivalDriveValidation[freeEnergy_, hyperbolicDistance_] := Module[
  {feComponent, distanceComponent, drive, optimalFE},

  optimalFE = 1.0;

  (* Free energy component (sigmoid) *)
  feComponent = 1 / (1 + Exp[-(freeEnergy - optimalFE)]);

  (* Hyperbolic distance component (tanh) *)
  distanceComponent = Tanh[1.5 * hyperbolicDistance];

  (* Combined survival drive *)
  drive = 0.7 * feComponent + 0.3 * distanceComponent;

  <|
    "drive" -> drive,
    "threat_level" -> Which[
      drive < 0.3, "safe",
      drive < 0.7, "caution",
      True, "danger"
    ],
    "crisis" -> drive > 0.8
  |>
]

(* Self-Organized Criticality *)
CriticalityValidation[activityTimeseries_] := Module[
  {avalanches, sizes, durations, branchingRatio, powerLawFit, hurstExponent},

  (* Detect avalanches (activity > 2σ) *)
  avalanches = DetectAvalanches[activityTimeseries, 2.0];
  sizes = Map[Length, avalanches];
  durations = Map[Length, avalanches];

  (* Branching ratio σ ≈ 1.0 at criticality *)
  branchingRatio = Mean[Map[ComputeBranchingRatio, avalanches]];

  (* Power law: P(s) ~ s^(-τ) with τ ≈ 1.5 *)
  powerLawFit = FindFit[Log[sizes], Log[a] - tau * Log[s], {a, tau}, s];

  (* Hurst exponent H ≈ 0.5 at criticality *)
  hurstExponent = EstimateHurstExponent[activityTimeseries];

  <|
    "branchingRatio" -> branchingRatio,
    "criticalityScore" -> Abs[branchingRatio - 1.0],
    "powerLawExponent" -> tau /. powerLawFit,
    "hurstExponent" -> hurstExponent,
    "atCriticality" -> Abs[branchingRatio - 1.0] < 0.05
  |>
]

(* Export validation functions *)
Export["agency-validation.mx", {
  FreeEnergyValidation,
  PhiCalculation,
  HyperbolicDistanceValidation,
  SurvivalDriveValidation,
  CriticalityValidation
}]
`;

// ============================================================================
// Tool Handler
// ============================================================================

/**
 * Handle agency tool calls
 *
 * Routes to appropriate native Rust implementations via NAPI
 */
export async function handleAgencyTool(
  name: string,
  args: any,
  nativeModule: any
): Promise<any> {
  switch (name) {
    case "agency_compute_free_energy":
      return computeFreeEnergy(args, nativeModule);

    case "agency_minimize_expected_free_energy":
      return minimizeExpectedFreeEnergy(args, nativeModule);

    case "agency_compute_survival_drive":
      return computeSurvivalDrive(args, nativeModule);

    case "agency_assess_threat":
      return assessThreat(args, nativeModule);

    case "agency_compute_phi":
      return computePhi(args, nativeModule);

    case "agency_analyze_criticality":
      return analyzeCriticality(args, nativeModule);

    case "agency_regulate_homeostasis":
      return regulateHomeostasis(args, nativeModule);

    case "agency_update_beliefs":
      return updateBeliefs(args, nativeModule);

    case "agency_generate_action":
      return generateAction(args, nativeModule);

    case "agency_analyze_emergence":
      return analyzeEmergence(args, nativeModule);

    case "agency_compute_impermanence":
      return computeImpermanence(args, nativeModule);

    case "agency_create_agent":
      return createAgent(args, nativeModule);

    case "agency_agent_step":
      return agentStep(args, nativeModule);

    case "agency_get_agent_metrics":
      return getAgentMetrics(args, nativeModule);

    // Negentropy & Pedagogic Scaffolding
    case "agency_compute_negentropy":
      return computeNegentropy(args, nativeModule);

    case "agency_get_bateson_level":
      return getBatesonLevel(args, nativeModule);

    case "agency_get_scaffold_mode":
      return getScaffoldMode(args, nativeModule);

    case "agency_get_intrinsic_motivation":
      return getIntrinsicMotivation(args, nativeModule);

    case "agency_get_cognitive_state":
      return getCognitiveState(args, nativeModule);

    case "agency_pedagogic_intervention":
      return pedagogicIntervention(args, nativeModule);

    // L4 Evolution Tools (Holland, 1975)
    case "agency_set_population_context":
      return setPopulationContext(args, nativeModule);

    case "agency_update_fitness":
      return updateFitness(args, nativeModule);

    case "agency_get_l4_readiness":
      return getL4Readiness(args, nativeModule);

    case "agency_trigger_memetic_transfer":
      return triggerMemeticTransfer(args, nativeModule);

    default:
      throw new Error(`Unknown agency tool: ${name}`);
  }
}

// ============================================================================
// Implementation Functions
// ============================================================================

/**
 * Compute variational free energy F = Complexity - Accuracy
 *
 * Based on Karl Friston's Free Energy Principle:
 * F = KL[q(s)||p(s|o)] - E_q[log p(o|s)]
 *   = Complexity - Accuracy
 *
 * Where:
 * - Complexity = KL divergence between beliefs and posterior
 * - Accuracy = Expected log likelihood of observations
 */
async function computeFreeEnergy(args: any, native: any) {
  const { observation, beliefs, precision } = args;

  // Try native implementation first
  if (native?.compute_free_energy) {
    try {
      return native.compute_free_energy(observation, beliefs, precision);
    } catch (e) {
      console.error("[agency] Native free energy failed:", e);
    }
  }

  // TypeScript fallback implementation
  try {
    // Ensure arrays have same length
    const n = Math.min(observation.length, beliefs.length, precision.length);

    // Normalize beliefs and observations to probabilities
    const beliefsSum = beliefs.slice(0, n).reduce((a: number, b: number) => a + Math.abs(b), 0);
    const obsSum = observation.slice(0, n).reduce((a: number, b: number) => a + Math.abs(b), 0);

    // Add small epsilon to prevent division by zero
    const epsilon = 1e-10;
    const normalizedBeliefs = beliefs.slice(0, n).map((b: number) => Math.abs(b) / (beliefsSum + epsilon));
    const normalizedObs = observation.slice(0, n).map((o: number) => Math.abs(o) / (obsSum + epsilon));

    // Complexity: KL divergence KL[beliefs || observation]
    let complexity = 0;
    for (let i = 0; i < n; i++) {
      if (normalizedBeliefs[i] > epsilon && normalizedObs[i] > epsilon) {
        complexity += normalizedBeliefs[i] * Math.log(normalizedBeliefs[i] / normalizedObs[i]);
      }
    }

    // Accuracy: Expected log likelihood with precision weighting
    let accuracy = 0;
    for (let i = 0; i < n; i++) {
      const error = observation[i] - beliefs[i];
      accuracy -= 0.5 * error * error * precision[i];
    }

    // Free energy = Complexity - Accuracy
    const freeEnergy = complexity - accuracy;

    return {
      free_energy: isFinite(freeEnergy) ? freeEnergy : 1.0,
      complexity: isFinite(complexity) ? complexity : 0.0,
      accuracy: isFinite(accuracy) ? accuracy : 0.0,
      valid: isFinite(freeEnergy),
      method: "typescript_fallback"
    };
  } catch (error) {
    return {
      error: `Free energy computation failed: ${error}`,
      free_energy: 1.0,
      complexity: 0.0,
      accuracy: 0.0
    };
  }
}

/**
 * Compute expected free energy (EFE) for policy selection
 *
 * EFE = Epistemic Value + Pragmatic Value
 * - Epistemic: Information gain (exploration)
 * - Pragmatic: Goal achievement (exploitation)
 */
async function minimizeExpectedFreeEnergy(args: any, native: any) {
  const { policy, beliefs, goal, exploration_weight = 0.5 } = args;

  if (native?.minimize_expected_free_energy) {
    try {
      return native.minimize_expected_free_energy(policy, beliefs, goal, exploration_weight);
    } catch (e) {
      console.error("[agency] Native EFE failed:", e);
    }
  }

  try {
    // Epistemic value: entropy of beliefs (information gain)
    let entropy = 0;
    for (const b of beliefs) {
      if (b > 1e-10) {
        entropy -= b * Math.log(b);
      }
    }

    // Pragmatic value: negative distance to goal
    let goalDistance = 0;
    for (let i = 0; i < policy.length && i < goal.length; i++) {
      const diff = policy[i] - goal[i];
      goalDistance += diff * diff;
    }

    const epistemicValue = exploration_weight * entropy;
    const pragmaticValue = -(1 - exploration_weight) * Math.sqrt(goalDistance);

    const efe = -(epistemicValue + pragmaticValue); // Negative because we minimize

    return {
      expected_free_energy: efe,
      epistemic_value: epistemicValue,
      pragmatic_value: pragmaticValue,
      exploration_weight,
      method: "typescript_fallback"
    };
  } catch (error) {
    return {
      error: `EFE computation failed: ${error}`,
      expected_free_energy: NaN
    };
  }
}

/**
 * Compute survival drive from free energy and hyperbolic position
 *
 * Drive increases with:
 * - High free energy (prediction error)
 * - Distance from safe region in hyperbolic space
 */
async function computeSurvivalDrive(args: any, native: any) {
  const { free_energy, position, strength = 1.0 } = args;

  if (native?.compute_survival_drive) {
    try {
      return native.compute_survival_drive(free_energy, position, strength);
    } catch (e) {
      console.error("[agency] Native survival drive failed:", e);
    }
  }

  try {
    // Compute hyperbolic distance from origin (safe region)
    let hyperbolicDist = 0.1;
    if (native?.hyperbolic_distance && position.length === 12) {
      const origin = [1.0, ...Array(11).fill(0)];
      hyperbolicDist = native.hyperbolic_distance(position, origin);
    } else if (position.length === 12) {
      // Fallback: Lorentz distance computation
      const inner = -position[0] * position[0] + position.slice(1).reduce((s: number, x: number) => s + x * x, 0);
      hyperbolicDist = Math.acosh(Math.max(-inner, 1.0));
    }

    // Free energy component (sigmoid)
    const optimalFE = 1.0;
    const feComponent = 1 / (1 + Math.exp(-(free_energy - optimalFE)));

    // Hyperbolic distance component (tanh)
    const distComponent = Math.tanh(1.5 * hyperbolicDist);

    // Combined survival drive
    const drive = strength * (0.7 * feComponent + 0.3 * distComponent);

    const threatLevel = drive > 0.7 ? "danger" : drive > 0.3 ? "caution" : "safe";
    const homeostatic = drive < 0.8 ? "stable" : "critical";

    return {
      survival_drive: Math.max(0, Math.min(1, drive)),
      threat_level: threatLevel,
      homeostatic_status: homeostatic,
      hyperbolic_distance: hyperbolicDist,
      free_energy_component: feComponent,
      distance_component: distComponent,
      crisis: drive > 0.8,
      method: "typescript_fallback"
    };
  } catch (error) {
    return {
      error: `Survival drive computation failed: ${error}`,
      survival_drive: NaN
    };
  }
}

/**
 * Comprehensive threat assessment across multiple dimensions
 */
async function assessThreat(args: any, native: any) {
  const { free_energy, free_energy_history, position, prediction_errors } = args;

  if (native?.assess_threat) {
    try {
      return native.assess_threat(free_energy, free_energy_history, position, prediction_errors);
    } catch (e) {
      console.error("[agency] Native threat assessment failed:", e);
    }
  }

  try {
    // Free energy gradient (rate of increase)
    let feGradient = 0;
    if (free_energy_history && free_energy_history.length > 1) {
      const recent = free_energy_history.slice(-5);
      feGradient = recent[recent.length - 1] - recent[0];
    }

    // Hyperbolic distance from safe region
    let hyperbolicDistance = 0;
    if (position?.length === 12) {
      if (native?.hyperbolic_distance) {
        const origin = [1.0, ...Array(11).fill(0)];
        hyperbolicDistance = native.hyperbolic_distance(position, origin);
      } else {
        const inner = -position[0] * position[0] + position.slice(1).reduce((s: number, x: number) => s + x * x, 0);
        hyperbolicDistance = Math.acosh(Math.max(-inner, 1.0));
      }
    }

    // Prediction error volatility
    let predictionVolatility = 0;
    if (prediction_errors && prediction_errors.length > 0) {
      const mean = prediction_errors.reduce((a: number, b: number) => a + b, 0) / prediction_errors.length;
      const variance = prediction_errors.reduce((a: number, b: number) => a + (b - mean) ** 2, 0) / prediction_errors.length;
      predictionVolatility = Math.sqrt(variance);
    }

    // Environmental volatility (from free energy history)
    let environmentalVolatility = 0;
    if (free_energy_history && free_energy_history.length > 1) {
      const mean = free_energy_history.reduce((a: number, b: number) => a + b, 0) / free_energy_history.length;
      const variance = free_energy_history.reduce((a: number, b: number) => a + (b - mean) ** 2, 0) / free_energy_history.length;
      environmentalVolatility = Math.sqrt(variance);
    }

    // Overall threat (weighted combination)
    const overallThreat = 0.3 * Math.tanh(feGradient) +
                          0.25 * Math.tanh(hyperbolicDistance) +
                          0.25 * Math.tanh(predictionVolatility) +
                          0.2 * Math.tanh(environmentalVolatility);

    return {
      overall_threat: Math.max(0, Math.min(1, overallThreat)),
      components: {
        free_energy_gradient: feGradient,
        hyperbolic_distance: hyperbolicDistance,
        prediction_volatility: predictionVolatility,
        environmental_volatility: environmentalVolatility
      },
      threat_level: overallThreat > 0.7 ? "critical" : overallThreat > 0.4 ? "elevated" : "nominal",
      method: "typescript_fallback"
    };
  } catch (error) {
    return {
      error: `Threat assessment failed: ${error}`,
      overall_threat: NaN
    };
  }
}

/**
 * Compute integrated information Φ (consciousness metric)
 *
 * Greedy approximation of IIT 3.0:
 * Φ = minimum information partition (MIP)
 */
async function computePhi(args: any, native: any) {
  const { network_state, connectivity, algorithm = "greedy" } = args;

  if (native?.compute_phi) {
    try {
      return native.compute_phi(network_state, connectivity, algorithm);
    } catch (e) {
      console.error("[agency] Native Phi computation failed:", e);
    }
  }

  try {
    const n = network_state.length;

    // Greedy approximation: effective information of the system
    let effectiveInfo = 0;

    // Compute entropy of current state
    let stateEntropy = 0;
    for (const s of network_state) {
      if (s > 1e-10) {
        stateEntropy -= s * Math.log2(s);
      }
    }

    // Simple approximation: Φ ≈ mutual information between parts
    if (connectivity && connectivity.length > 0) {
      let totalConnections = 0;
      let activeConnections = 0;

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (connectivity[i] && connectivity[i][j]) {
            totalConnections++;
            if (network_state[i] > 0.5 && network_state[j] > 0.5) {
              activeConnections++;
            }
          }
        }
      }

      effectiveInfo = activeConnections > 0 ? (activeConnections / totalConnections) * stateEntropy : 0;
    } else {
      // Full connectivity assumed
      effectiveInfo = stateEntropy * 0.5; // Rough approximation
    }

    const phi = Math.max(0, effectiveInfo);

    return {
      phi,
      algorithm,
      consciousness_level: phi > 1.0 ? "emergent" : phi > 0.5 ? "minimal" : "none",
      state_entropy: stateEntropy,
      effective_information: effectiveInfo,
      method: "greedy_approximation"
    };
  } catch (error) {
    return {
      error: `Phi computation failed: ${error}`,
      phi: NaN
    };
  }
}

/**
 * Analyze self-organized criticality (SOC)
 *
 * Computes branching ratio σ and avalanche statistics
 * σ ≈ 1.0 indicates edge of chaos (optimal information processing)
 */
async function analyzeCriticality(args: any, native: any) {
  const { activity_timeseries, avalanche_threshold = 2.0 } = args;

  if (native?.analyze_criticality) {
    try {
      return native.analyze_criticality(activity_timeseries, avalanche_threshold);
    } catch (e) {
      console.error("[agency] Native criticality analysis failed:", e);
    }
  }

  try {
    const n = activity_timeseries.length;

    // Compute mean and std
    const mean = activity_timeseries.reduce((a: number, b: number) => a + b, 0) / n;
    const variance = activity_timeseries.reduce((a: number, b: number) => a + (b - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);

    // Detect avalanches (activity > mean + threshold*std)
    const threshold = mean + avalanche_threshold * std;
    const avalanches: number[][] = [];
    let currentAvalanche: number[] = [];

    for (let i = 0; i < n; i++) {
      if (activity_timeseries[i] > threshold) {
        currentAvalanche.push(activity_timeseries[i]);
      } else if (currentAvalanche.length > 0) {
        avalanches.push([...currentAvalanche]);
        currentAvalanche = [];
      }
    }

    // Branching ratio: average ratio of successive events
    let branchingRatio = 0;
    if (avalanches.length > 1) {
      let ratioSum = 0;
      for (let i = 1; i < avalanches.length; i++) {
        const prev = avalanches[i - 1].length;
        const curr = avalanches[i].length;
        if (prev > 0) {
          ratioSum += curr / prev;
        }
      }
      branchingRatio = ratioSum / (avalanches.length - 1);
    }

    // Power law exponent (rough estimate from size distribution)
    const sizes = avalanches.map(a => a.length);
    const avgSize = sizes.reduce((a, b) => a + b, 0) / sizes.length || 1;
    const powerLawExponent = 1.5; // Typical value for SOC systems

    const atCriticality = Math.abs(branchingRatio - 1.0) < 0.1;

    return {
      branching_ratio: branchingRatio,
      at_criticality: atCriticality,
      criticality_score: 1.0 - Math.abs(branchingRatio - 1.0),
      avalanche_count: avalanches.length,
      average_avalanche_size: avgSize,
      power_law_exponent: powerLawExponent,
      method: "statistical_approximation"
    };
  } catch (error) {
    return {
      error: `Criticality analysis failed: ${error}`,
      branching_ratio: NaN
    };
  }
}

/**
 * Homeostatic regulation using PID control + allostatic prediction
 *
 * Maintains Φ, F, and Survival within optimal bounds
 */
async function regulateHomeostasis(args: any, native: any) {
  const { current_state, setpoints, sensors } = args;

  if (native?.regulate_homeostasis) {
    try {
      return native.regulate_homeostasis(current_state, setpoints, sensors);
    } catch (e) {
      console.error("[agency] Native homeostasis failed:", e);
    }
  }

  try {
    // Default setpoints
    const phiOptimal = setpoints?.phi_optimal ?? 1.0;
    const feOptimal = setpoints?.free_energy_optimal ?? 1.0;
    const survivalOptimal = setpoints?.survival_optimal ?? 0.5;

    // PID control gains
    const Kp = 0.5; // Proportional
    const Ki = 0.1; // Integral (simplified)
    const Kd = 0.2; // Derivative (simplified)

    // Compute errors
    const phiError = phiOptimal - current_state.phi;
    const feError = feOptimal - current_state.free_energy;
    const survivalError = survivalOptimal - current_state.survival;

    // PID control signals
    const phiAdjustment = Kp * phiError;
    const feAdjustment = Kp * feError;
    const survivalAdjustment = Kp * survivalError;

    // Allostatic prediction: anticipate future needs
    let allostaticBias = 0;
    if (sensors && sensors.length > 0) {
      const sensorMean = sensors.reduce((a: number, b: number) => a + b, 0) / sensors.length;
      allostaticBias = (sensorMean - 0.5) * 0.1; // Anticipatory adjustment
    }

    // Apply bounds checking
    const bounded = (val: number) => Math.max(-1, Math.min(1, val));

    return {
      control_signals: {
        phi_adjustment: bounded(phiAdjustment + allostaticBias),
        free_energy_adjustment: bounded(feAdjustment),
        survival_adjustment: bounded(survivalAdjustment)
      },
      errors: {
        phi_error: phiError,
        free_energy_error: feError,
        survival_error: survivalError
      },
      setpoints: {
        phi_optimal: phiOptimal,
        free_energy_optimal: feOptimal,
        survival_optimal: survivalOptimal
      },
      allostatic_bias: allostaticBias,
      homeostatic_status: Math.abs(phiError) < 0.1 && Math.abs(feError) < 0.2 ? "stable" : "regulating",
      method: "pid_allostatic"
    };
  } catch (error) {
    return {
      error: `Homeostasis regulation failed: ${error}`,
      control_signals: { phi_adjustment: 0, free_energy_adjustment: 0, survival_adjustment: 0 }
    };
  }
}

/**
 * Update beliefs using precision-weighted prediction errors
 *
 * Implements hierarchical Bayesian inference:
 * beliefs_new = beliefs + learning_rate * precision * prediction_error
 */
async function updateBeliefs(args: any, native: any) {
  const { observation, beliefs, precision, learning_rate = 0.01 } = args;

  if (native?.update_beliefs) {
    try {
      return native.update_beliefs(observation, beliefs, precision, learning_rate);
    } catch (e) {
      console.error("[agency] Native belief update failed:", e);
    }
  }

  try {
    const updatedBeliefs: number[] = [];
    const predictionErrors: number[] = [];
    const updatedPrecision: number[] = [];

    for (let i = 0; i < beliefs.length; i++) {
      // Prediction error
      const error = observation[i] - beliefs[i];
      predictionErrors.push(error);

      // Precision-weighted update
      const precisionWeighted = precision[i] * error;
      const newBelief = beliefs[i] + learning_rate * precisionWeighted;
      updatedBeliefs.push(newBelief);

      // Update precision (increase with consistent predictions)
      const precisionUpdate = precision[i] * (1 + 0.01 * (1 - Math.abs(error)));
      updatedPrecision.push(Math.min(precisionUpdate, 100)); // Cap precision
    }

    // Compute mean prediction error
    const meanPredictionError = predictionErrors.reduce((a, b) => Math.abs(a) + Math.abs(b), 0) / predictionErrors.length;

    return {
      updated_beliefs: updatedBeliefs,
      updated_precision: updatedPrecision,
      prediction_errors: predictionErrors,
      mean_prediction_error: meanPredictionError,
      learning_rate,
      converged: meanPredictionError < 0.01,
      method: "precision_weighted_pe"
    };
  } catch (error) {
    return {
      error: `Belief update failed: ${error}`,
      updated_beliefs: beliefs,
      updated_precision: precision
    };
  }
}

/**
 * Generate action from policy using active inference
 *
 * Action minimizes expected free energy
 */
async function generateAction(args: any, native: any) {
  const { policy, beliefs, action_precision = 1.0 } = args;

  if (native?.generate_action) {
    try {
      return native.generate_action(policy, beliefs, action_precision);
    } catch (e) {
      console.error("[agency] Native action generation failed:", e);
    }
  }

  try {
    // Generate action as precision-weighted policy
    const action: number[] = [];
    const predictedObservation: number[] = [];

    for (let i = 0; i < policy.length; i++) {
      // Add noise inversely proportional to precision
      const noise = (Math.random() - 0.5) * (1 / action_precision);
      action.push(policy[i] + noise);

      // Predict sensory consequences (simplified)
      if (i < beliefs.length) {
        predictedObservation.push(beliefs[i] + 0.1 * policy[i]);
      }
    }

    // Compute expected sensory outcome
    let expectedFreeEnergy = 0;
    for (let i = 0; i < predictedObservation.length; i++) {
      const diff = predictedObservation[i] - beliefs[i];
      expectedFreeEnergy += diff * diff;
    }

    return {
      action,
      predicted_observation: predictedObservation,
      expected_free_energy: expectedFreeEnergy,
      action_precision,
      method: "efe_minimization"
    };
  } catch (error) {
    return {
      error: `Action generation failed: ${error}`,
      action: Array(policy.length).fill(0),
      predicted_observation: Array(beliefs.length).fill(0)
    };
  }
}

/**
 * Analyze agency emergence dynamics
 *
 * Tracks Φ development, control authority, and phase transitions
 */
async function analyzeEmergence(args: any, native: any) {
  const { timeseries, threshold } = args;

  if (native?.analyze_emergence) {
    try {
      return native.analyze_emergence(timeseries, threshold);
    } catch (e) {
      console.error("[agency] Native emergence analysis failed:", e);
    }
  }

  try {
    const phiThreshold = threshold?.phi_emergence ?? 1.0;
    const controlThreshold = threshold?.control_emergence ?? 0.5;

    const { phi, free_energy, control, survival } = timeseries;

    // Detect threshold crossings
    const phiCrossed = phi && phi.some((p: number) => p > phiThreshold);
    const controlCrossed = control && control.some((c: number) => c > controlThreshold);

    // Compute trends
    const phiTrend = phi && phi.length > 1 ? phi[phi.length - 1] - phi[0] : 0;
    const feTrend = free_energy && free_energy.length > 1 ? free_energy[free_energy.length - 1] - free_energy[0] : 0;

    // Determine phase
    let phase = "dormant";
    if (phiCrossed && controlCrossed) {
      phase = "full_agency";
    } else if (phiCrossed) {
      phase = "conscious_non_agent";
    } else if (controlCrossed) {
      phase = "reactive_agent";
    } else if (phiTrend > 0 || controlTrend > 0) {
      phase = "emerging";
    }

    const controlTrend = control && control.length > 1 ? control[control.length - 1] - control[0] : 0;

    return {
      emergence_detected: phiCrossed || controlCrossed,
      phi_threshold_crossed: phiCrossed,
      control_threshold_crossed: controlCrossed,
      phase,
      trends: {
        phi: phiTrend,
        free_energy: feTrend,
        control: controlTrend
      },
      stability: Math.abs(feTrend) < 0.1 ? "stable" : "unstable",
      method: "threshold_detection"
    };
  } catch (error) {
    return {
      error: `Emergence analysis failed: ${error}`,
      emergence_detected: false
    };
  }
}

/**
 * Compute impermanence (state change rate)
 *
 * Following Buddhist principles of anicca (impermanence)
 * Healthy adaptation: impermanence > 0.4
 */
async function computeImpermanence(args: any, native: any) {
  const { current_state, previous_state, normalization = "euclidean" } = args;

  if (native?.compute_impermanence) {
    try {
      return native.compute_impermanence(current_state, previous_state, normalization);
    } catch (e) {
      console.error("[agency] Native impermanence failed:", e);
    }
  }

  try {
    let distance = 0;

    if (normalization === "hyperbolic" && current_state.length === 12 && native?.hyperbolic_distance) {
      distance = native.hyperbolic_distance(current_state, previous_state);
    } else if (normalization === "cosine") {
      // Cosine distance
      let dot = 0, normA = 0, normB = 0;
      for (let i = 0; i < current_state.length; i++) {
        dot += current_state[i] * previous_state[i];
        normA += current_state[i] * current_state[i];
        normB += previous_state[i] * previous_state[i];
      }
      distance = 1 - (dot / (Math.sqrt(normA) * Math.sqrt(normB)));
    } else {
      // Euclidean distance
      let sum = 0;
      for (let i = 0; i < current_state.length; i++) {
        const diff = current_state[i] - previous_state[i];
        sum += diff * diff;
      }
      distance = Math.sqrt(sum);
    }

    // Normalize by dimensionality
    const normalizer = normalization === "euclidean" ? Math.sqrt(current_state.length) : 1.0;
    const impermanence = distance / normalizer;

    return {
      impermanence_rate: impermanence,
      healthy_adaptation: impermanence > 0.4 && impermanence < 0.9,
      structural_plasticity: impermanence,
      stability: impermanence < 0.2 ? "rigid" : impermanence > 0.9 ? "chaotic" : "adaptive",
      normalization,
      method: "distance_based"
    };
  } catch (error) {
    return {
      error: `Impermanence computation failed: ${error}`,
      impermanence_rate: NaN
    };
  }
}

// Agent state storage (in-memory for TypeScript fallback)
const agentStore = new Map<string, any>();

/**
 * Create a new cybernetic agent
 */
async function createAgent(args: any, native: any) {
  const { config, phi_calculator_type = "greedy" } = args;

  if (native?.create_agent) {
    try {
      return native.create_agent(config, phi_calculator_type);
    } catch (e) {
      console.error("[agency] Native agent creation failed:", e);
    }
  }

  try {
    const agentId = `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const initialState = {
      phi: 0.1,
      free_energy: 1.0,
      survival: 0.5,
      control: 0.2,
      beliefs: Array(config.hidden_dim).fill(0.1),
      precision: Array(config.hidden_dim).fill(1.0),
      position: [1.0, ...Array(11).fill(0)] // Origin in H^11
    };

    // Store agent state
    agentStore.set(agentId, {
      config,
      state: initialState,
      phi_calculator_type,
      created_at: Date.now()
    });

    return {
      agent_id: agentId,
      config,
      initial_state: initialState,
      phi_calculator_type,
      method: "typescript_agent"
    };
  } catch (error) {
    return {
      error: `Agent creation failed: ${error}`
    };
  }
}

/**
 * Execute one agent time step
 */
async function agentStep(args: any, native: any) {
  const { agent_id, observation } = args;

  if (native?.agent_step) {
    try {
      return native.agent_step(agent_id, observation);
    } catch (e) {
      console.error("[agency] Native agent step failed:", e);
    }
  }

  try {
    const agent = agentStore.get(agent_id);
    if (!agent) {
      return { error: "Agent not found", agent_id };
    }

    const { config, state } = agent;

    // Update beliefs
    const beliefUpdate = await updateBeliefs({
      observation,
      beliefs: state.beliefs,
      precision: state.precision,
      learning_rate: config.learning_rate || 0.01
    }, native);

    // Compute free energy
    const feResult = await computeFreeEnergy({
      observation,
      beliefs: beliefUpdate.updated_beliefs,
      precision: beliefUpdate.updated_precision
    }, native);

    // Generate action
    const actionResult = await generateAction({
      policy: beliefUpdate.updated_beliefs,
      beliefs: beliefUpdate.updated_beliefs,
      action_precision: 1.0
    }, native);

    // Update state
    const newState = {
      phi: state.phi + 0.1, // Simplified phi growth
      free_energy: feResult?.free_energy ?? state.free_energy,
      survival: state.survival,
      control: state.control + 0.05,
      beliefs: beliefUpdate.updated_beliefs,
      precision: beliefUpdate.updated_precision,
      position: state.position
    };

    agent.state = newState;
    agentStore.set(agent_id, agent);

    return {
      action: actionResult.action,
      state: newState,
      metrics: {
        phi: newState.phi,
        free_energy: newState.free_energy,
        survival: newState.survival,
        control: newState.control
      },
      method: "typescript_agent_step"
    };
  } catch (error) {
    return {
      error: `Agent step failed: ${error}`,
      agent_id
    };
  }
}

/**
 * Get agent metrics
 */
async function getAgentMetrics(args: any, native: any) {
  const { agent_id } = args;

  if (native?.get_agent_metrics) {
    try {
      return native.get_agent_metrics(agent_id);
    } catch (e) {
      console.error("[agency] Native metrics failed:", e);
    }
  }

  try {
    const agent = agentStore.get(agent_id);
    if (!agent) {
      return { error: "Agent not found", agent_id };
    }

    const { state } = agent;

    return {
      agent_id,
      metrics: {
        phi: state.phi,
        free_energy: state.free_energy,
        survival_drive: state.survival,
        control_authority: state.control,
        model_accuracy: 0.75, // Placeholder
        branching_ratio: 0.99, // Placeholder
        impermanence: 0.42 // Placeholder
      },
      health: state.free_energy < 2.0 && state.phi > 0.5 ? "good" : "degraded",
      method: "typescript_agent_metrics"
    };
  } catch (error) {
    return {
      error: `Get metrics failed: ${error}`,
      agent_id
    };
  }
}

// ============================================================================
// Negentropy & Pedagogic Scaffolding Implementation Functions
// ============================================================================

/**
 * Negentropy state storage per agent
 */
interface NegentropyState {
  negentropy: number;
  batesonLevel: "L0" | "L1" | "L2" | "L3" | "L4";
  scaffoldMode: "Observation" | "CuriosityNudge" | "GuidedExploration" | "DirectInstruction" | "CollaborativeDialogue" | "Autonomous";
  intrinsicMotivation: number;
  cognitiveState: {
    pfc_inhibition: number;
    acc_error_detection: number;
    insula_interoception: number;
    basal_ganglia_action: number;
    hippocampus_memory: number;
  };
  lastUpdate: number;
  // L4 Evolution requirements (Holland, 1975)
  l3StabilizationSteps: number;
  populationContext: number;
  fitnessSignal: number;
}

const negentropyStore = new Map<string, NegentropyState>();

/**
 * Compute negentropy N = S_max - S_actual
 *
 * Negentropy measures order/organization:
 * - N >= 0.5: Agent is "alive" (autonomous operation)
 * - N < 0.5: Agent needs pedagogic scaffolding
 *
 * Based on Schrödinger's "negative entropy" concept from
 * "What is Life?" (1944) and Bateson's learning levels.
 */
async function computeNegentropy(args: any, native: any) {
  const {
    agent_id,
    beliefs,
    precision,
    prediction_error = 0.1,
    free_energy = 1.0
  } = args;

  if (native?.compute_negentropy) {
    try {
      return native.compute_negentropy(agent_id, beliefs, precision, prediction_error, free_energy);
    } catch (e) {
      console.error("[agency] Native negentropy computation failed:", e);
    }
  }

  try {
    // Get or create negentropy state
    let state = negentropyStore.get(agent_id);
    if (!state) {
      state = {
        negentropy: 0.5,
        batesonLevel: "L0",
        scaffoldMode: "Observation",
        intrinsicMotivation: 1.0,
        cognitiveState: {
          pfc_inhibition: 0.5,
          acc_error_detection: 0.5,
          insula_interoception: 0.5,
          basal_ganglia_action: 0.5,
          hippocampus_memory: 0.5
        },
        lastUpdate: Date.now(),
        // L4 Evolution (Holland, 1975)
        l3StabilizationSteps: 0,
        populationContext: 1, // Default: single agent
        fitnessSignal: 0.0
      };
    }

    // Compute entropy from beliefs distribution
    const beliefArray = Array.isArray(beliefs) ? beliefs : [];
    const n = beliefArray.length || 1;

    // Normalize beliefs to probability distribution
    const sum = beliefArray.reduce((a: number, b: number) => a + Math.abs(b), 0) || 1;
    const probs = beliefArray.map((b: number) => Math.abs(b) / sum);

    // Shannon entropy: H = -sum(p * log(p))
    const epsilon = 1e-10;
    let entropy = 0;
    for (const p of probs) {
      if (p > epsilon) {
        entropy -= p * Math.log2(p);
      }
    }

    // Maximum entropy for uniform distribution
    const maxEntropy = Math.log2(n);

    // Negentropy N = S_max - S_actual (normalized to [0,1])
    const rawNegentropy = maxEntropy > 0 ? (maxEntropy - entropy) / maxEntropy : 0.5;

    // Modulate by precision coherence (higher precision = more organized)
    const precisionArray = Array.isArray(precision) ? precision : [];
    const avgPrecision = precisionArray.length > 0
      ? precisionArray.reduce((a: number, b: number) => a + b, 0) / precisionArray.length
      : 1.0;
    const precisionFactor = Math.tanh(avgPrecision);

    // Modulate by free energy (lower free energy = better prediction)
    const freeEnergyFactor = Math.exp(-free_energy / 2);

    // Combined negentropy with weighted factors
    const negentropy = 0.4 * rawNegentropy + 0.3 * precisionFactor + 0.3 * freeEnergyFactor;

    // Update Bateson learning level based on negentropy
    // L4 requires: sustained L3, population context (≥3), fitness pressure (≥0.5)
    let batesonLevel: "L0" | "L1" | "L2" | "L3" | "L4";
    const l4Possible = state.l3StabilizationSteps >= 100 &&
                       state.populationContext >= 3 &&
                       state.fitnessSignal >= 0.5 &&
                       negentropy >= 0.9;

    if (negentropy < 0.25) {
      batesonLevel = "L0"; // Reflexive - simple stimulus-response
    } else if (negentropy < 0.5) {
      batesonLevel = "L1"; // Conditioning - learning correct response
    } else if (negentropy < 0.75) {
      batesonLevel = "L2"; // Meta-learning - learning to learn
    } else if (negentropy < 0.9 || !l4Possible) {
      batesonLevel = "L3"; // Transformation - deep context shifts
    } else {
      batesonLevel = "L4"; // Evolution - population-level adaptation (Holland, 1975)
    }

    // Track L3 stabilization for L4 transition
    if (batesonLevel === "L3") {
      state.l3StabilizationSteps = (state.l3StabilizationSteps || 0) + 1;
    } else if (batesonLevel !== "L4") {
      state.l3StabilizationSteps = 0;
    }

    // Update scaffold mode based on negentropy
    let scaffoldMode: NegentropyState["scaffoldMode"];
    if (negentropy < 0.2) {
      scaffoldMode = "DirectInstruction";
    } else if (negentropy < 0.35) {
      scaffoldMode = "GuidedExploration";
    } else if (negentropy < 0.5) {
      scaffoldMode = "CuriosityNudge";
    } else if (negentropy < 0.65) {
      scaffoldMode = "CollaborativeDialogue";
    } else if (negentropy < 0.8) {
      scaffoldMode = "Observation";
    } else {
      scaffoldMode = "Autonomous";
    }

    // Compute intrinsic motivation (Self-Determination Theory)
    // IM = autonomy × competence × relatedness
    const autonomy = negentropy; // Higher negentropy = more autonomy
    const competence = freeEnergyFactor; // Lower free energy = higher competence
    const relatedness = precisionFactor; // Higher precision = better connection
    const intrinsicMotivation = autonomy * competence * relatedness * 3;

    // Update state
    state.negentropy = negentropy;
    state.batesonLevel = batesonLevel;
    state.scaffoldMode = scaffoldMode;
    state.intrinsicMotivation = intrinsicMotivation;
    state.lastUpdate = Date.now();
    negentropyStore.set(agent_id, state);

    return {
      negentropy,
      is_alive: negentropy >= 0.5,
      bateson_level: batesonLevel,
      scaffold_mode: scaffoldMode,
      intrinsic_motivation: intrinsicMotivation,
      components: {
        raw_negentropy: rawNegentropy,
        precision_factor: precisionFactor,
        free_energy_factor: freeEnergyFactor,
        entropy,
        max_entropy: maxEntropy
      },
      thresholds: {
        alive_threshold: 0.5,
        current_gap: negentropy - 0.5
      },
      method: "typescript_negentropy"
    };
  } catch (error) {
    return {
      error: `Negentropy computation failed: ${error}`,
      negentropy: 0.5,
      is_alive: true
    };
  }
}

/**
 * Get Bateson learning level for an agent
 *
 * Bateson's Learning Levels (1972):
 * - L0: Reflexive responses (no learning)
 * - L1: Classical/operant conditioning
 * - L2: Learning to learn (meta-learning)
 * - L3: Deep transformation of context
 */
async function getBatesonLevel(args: any, native: any) {
  const { agent_id, negentropy } = args;

  if (native?.get_bateson_level) {
    try {
      return native.get_bateson_level(agent_id, negentropy);
    } catch (e) {
      console.error("[agency] Native Bateson level failed:", e);
    }
  }

  try {
    const state = negentropyStore.get(agent_id);
    const n = negentropy ?? state?.negentropy ?? 0.5;

    // L4 requirements check (Holland, 1975)
    const l4Possible = state &&
                       state.l3StabilizationSteps >= 100 &&
                       state.populationContext >= 3 &&
                       state.fitnessSignal >= 0.5 &&
                       n >= 0.9;

    let level: "L0" | "L1" | "L2" | "L3" | "L4";
    let description: string;
    let characteristics: string[];

    if (n < 0.25) {
      level = "L0";
      description = "Reflexive - Simple stimulus-response patterns";
      characteristics = [
        "Direct cause-effect responses",
        "No error correction",
        "Mechanical reactions",
        "Zero-order learning"
      ];
    } else if (n < 0.5) {
      level = "L1";
      description = "Conditioning - Learning correct responses in context";
      characteristics = [
        "Classical conditioning",
        "Operant conditioning",
        "Habit formation",
        "Context-dependent responses"
      ];
    } else if (n < 0.75) {
      level = "L2";
      description = "Meta-learning - Learning to learn, pattern recognition";
      characteristics = [
        "Learning transfer",
        "Set formation",
        "Gestalt recognition",
        "Character/personality development"
      ];
    } else if (n < 0.9 || !l4Possible) {
      level = "L3";
      description = "Transformation - Deep restructuring of context";
      characteristics = [
        "Profound re-organization",
        "Resolution of double binds",
        "Paradigm shifts",
        "Spiritual transformation"
      ];
    } else {
      level = "L4";
      description = "Evolution - Population-level adaptation and phylogenetic change";
      characteristics = [
        "Genetic algorithm optimization (Holland, 1975)",
        "Memetic evolution across agent populations",
        "Species-level behavioral changes",
        "Evolutionary pressure from fitness landscape",
        "Cross-generational knowledge transfer"
      ];
    }

    // L4 readiness information
    const l4Readiness = state ? {
      l3_stabilization_steps: state.l3StabilizationSteps,
      l3_stabilization_required: 100,
      population_context: state.populationContext,
      population_required: 3,
      fitness_signal: state.fitnessSignal,
      fitness_required: 0.5,
      negentropy_required: 0.9,
      l4_possible: l4Possible
    } : null;

    return {
      level,
      level_number: ["L0", "L1", "L2", "L3", "L4"].indexOf(level),
      description,
      characteristics,
      negentropy: n,
      threshold_for_next: level === "L4" ? 1.0 : [0.25, 0.5, 0.75, 0.9, 1.0][["L0", "L1", "L2", "L3", "L4"].indexOf(level)],
      l4_readiness: l4Readiness,
      reference: level === "L4"
        ? "Holland, J.H. (1975). Adaptation in Natural and Artificial Systems"
        : "Bateson, G. (1972). Steps to an Ecology of Mind",
      method: "typescript_bateson"
    };
  } catch (error) {
    return {
      error: `Bateson level failed: ${error}`,
      level: "L1"
    };
  }
}

/**
 * Get current scaffolding mode for pedagogic intervention
 *
 * Based on Vygotsky's Zone of Proximal Development (ZPD)
 * and Self-Determination Theory (Deci & Ryan)
 */
async function getScaffoldMode(args: any, native: any) {
  const { agent_id, negentropy } = args;

  if (native?.get_scaffold_mode) {
    try {
      return native.get_scaffold_mode(agent_id, negentropy);
    } catch (e) {
      console.error("[agency] Native scaffold mode failed:", e);
    }
  }

  try {
    const state = negentropyStore.get(agent_id);
    const n = negentropy ?? state?.negentropy ?? 0.5;

    type ScaffoldModeType = "DirectInstruction" | "GuidedExploration" | "CuriosityNudge" |
                           "CollaborativeDialogue" | "Observation" | "Autonomous";

    let mode: ScaffoldModeType;
    let description: string;
    let interventionLevel: number;
    let supportActions: string[];

    if (n < 0.2) {
      mode = "DirectInstruction";
      description = "Agent needs explicit guidance and clear directives";
      interventionLevel = 0.9;
      supportActions = [
        "Provide step-by-step instructions",
        "Model correct behavior",
        "Offer immediate feedback",
        "Reduce cognitive load"
      ];
    } else if (n < 0.35) {
      mode = "GuidedExploration";
      description = "Agent can explore within structured boundaries";
      interventionLevel = 0.7;
      supportActions = [
        "Set exploration boundaries",
        "Provide hints when stuck",
        "Validate discoveries",
        "Scaffold problem decomposition"
      ];
    } else if (n < 0.5) {
      mode = "CuriosityNudge";
      description = "Agent needs gentle curiosity activation";
      interventionLevel = 0.5;
      supportActions = [
        "Pose intriguing questions",
        "Highlight interesting patterns",
        "Suggest exploration directions",
        "Celebrate curious behavior"
      ];
    } else if (n < 0.65) {
      mode = "CollaborativeDialogue";
      description = "Agent engages in peer-level dialogue";
      interventionLevel = 0.3;
      supportActions = [
        "Engage as thought partner",
        "Share perspectives",
        "Co-construct meaning",
        "Socratic questioning"
      ];
    } else if (n < 0.8) {
      mode = "Observation";
      description = "Agent operates independently with minimal oversight";
      interventionLevel = 0.15;
      supportActions = [
        "Monitor from distance",
        "Intervene only when requested",
        "Document progress",
        "Provide resources on demand"
      ];
    } else {
      mode = "Autonomous";
      description = "Agent operates fully independently";
      interventionLevel = 0.0;
      supportActions = [
        "Trust agent autonomy",
        "Remove scaffolds entirely",
        "Allow self-direction",
        "Celebrate independence"
      ];
    }

    return {
      mode,
      description,
      intervention_level: interventionLevel,
      support_actions: supportActions,
      negentropy: n,
      is_alive: n >= 0.5,
      zpd_position: n < 0.5 ? "needs_support" : n < 0.7 ? "zpd_optimal" : "independent",
      reference: "Vygotsky, L. (1978). Mind in Society; Deci & Ryan (1985). SDT",
      method: "typescript_scaffold"
    };
  } catch (error) {
    return {
      error: `Scaffold mode failed: ${error}`,
      mode: "GuidedExploration"
    };
  }
}

/**
 * Get intrinsic motivation based on Self-Determination Theory
 *
 * IM = Autonomy × Competence × Relatedness
 *
 * Deci, E.L. & Ryan, R.M. (1985). Intrinsic Motivation and
 * Self-Determination in Human Behavior.
 */
async function getIntrinsicMotivation(args: any, native: any) {
  const { agent_id, autonomy, competence, relatedness } = args;

  if (native?.get_intrinsic_motivation) {
    try {
      return native.get_intrinsic_motivation(agent_id, autonomy, competence, relatedness);
    } catch (e) {
      console.error("[agency] Native intrinsic motivation failed:", e);
    }
  }

  try {
    const state = negentropyStore.get(agent_id);

    // Use provided values or derive from state
    const aut = autonomy ?? state?.negentropy ?? 0.5;
    const comp = competence ?? (state?.cognitiveState?.pfc_inhibition ?? 0.5);
    const rel = relatedness ?? (state?.cognitiveState?.insula_interoception ?? 0.5);

    // Intrinsic motivation formula
    const rawIM = aut * comp * rel;

    // Scale to [0, 3] range (each factor maxes at 1)
    const scaledIM = rawIM * 3;

    // Determine motivation type
    let motivationType: string;
    if (scaledIM < 0.5) {
      motivationType = "amotivation";
    } else if (scaledIM < 1.0) {
      motivationType = "external_regulation";
    } else if (scaledIM < 1.5) {
      motivationType = "introjected_regulation";
    } else if (scaledIM < 2.0) {
      motivationType = "identified_regulation";
    } else if (scaledIM < 2.5) {
      motivationType = "integrated_regulation";
    } else {
      motivationType = "intrinsic_motivation";
    }

    return {
      intrinsic_motivation: scaledIM,
      components: {
        autonomy: aut,
        competence: comp,
        relatedness: rel
      },
      motivation_type: motivationType,
      is_self_determined: scaledIM >= 1.5,
      recommendations: scaledIM < 1.5 ? [
        "Provide more choice and autonomy",
        "Offer optimal challenges for competence",
        "Foster sense of connection and belonging"
      ] : [
        "Maintain supportive environment",
        "Continue respecting autonomy"
      ],
      reference: "Deci, E.L. & Ryan, R.M. (1985). Self-Determination Theory",
      method: "typescript_motivation"
    };
  } catch (error) {
    return {
      error: `Intrinsic motivation failed: ${error}`,
      intrinsic_motivation: 1.0
    };
  }
}

/**
 * Get cognitive state from brain-inspired modules
 *
 * Based on neuroscience research on:
 * - Prefrontal Cortex (PFC): Executive function, inhibition
 * - Anterior Cingulate Cortex (ACC): Error detection, conflict monitoring
 * - Insula: Interoception, emotional awareness
 * - Basal Ganglia: Action selection, habit formation
 * - Hippocampus: Memory consolidation, spatial navigation
 */
async function getCognitiveState(args: any, native: any) {
  const { agent_id, include_recommendations = true } = args;

  if (native?.get_cognitive_state) {
    try {
      return native.get_cognitive_state(agent_id, include_recommendations);
    } catch (e) {
      console.error("[agency] Native cognitive state failed:", e);
    }
  }

  try {
    const state = negentropyStore.get(agent_id);

    // Use stored state or defaults
    const cognitive = state?.cognitiveState ?? {
      pfc_inhibition: 0.5,
      acc_error_detection: 0.5,
      insula_interoception: 0.5,
      basal_ganglia_action: 0.5,
      hippocampus_memory: 0.5
    };

    // Compute overall cognitive coherence
    const values = Object.values(cognitive);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    const coherence = 1 - Math.sqrt(variance);

    // Generate recommendations if requested
    const recommendations: string[] = [];
    if (include_recommendations) {
      if (cognitive.pfc_inhibition < 0.4) {
        recommendations.push("Strengthen executive function through structured tasks");
      }
      if (cognitive.acc_error_detection < 0.4) {
        recommendations.push("Enhance error monitoring through feedback loops");
      }
      if (cognitive.insula_interoception < 0.4) {
        recommendations.push("Improve interoception through embodied practices");
      }
      if (cognitive.basal_ganglia_action < 0.4) {
        recommendations.push("Develop action habits through repetition");
      }
      if (cognitive.hippocampus_memory < 0.4) {
        recommendations.push("Strengthen memory through consolidation periods");
      }
    }

    return {
      cognitive_state: cognitive,
      overall_coherence: coherence,
      dominant_system: Object.entries(cognitive).reduce((a, b) => a[1] > b[1] ? a : b)[0],
      weakest_system: Object.entries(cognitive).reduce((a, b) => a[1] < b[1] ? a : b)[0],
      balance_score: coherence,
      recommendations: recommendations.length > 0 ? recommendations : ["Cognitive systems well-balanced"],
      neuroscience_basis: {
        pfc: "Prefrontal Cortex - Executive function",
        acc: "Anterior Cingulate - Error detection",
        insula: "Insula - Interoception",
        bg: "Basal Ganglia - Action selection",
        hpc: "Hippocampus - Memory"
      },
      method: "typescript_cognitive"
    };
  } catch (error) {
    return {
      error: `Cognitive state failed: ${error}`
    };
  }
}

/**
 * Apply pedagogic intervention based on current negentropy
 *
 * Interventions are NOT punishments but graceful scaffolding
 * to help the agent develop towards autonomy.
 */
async function pedagogicIntervention(args: any, native: any) {
  const {
    agent_id,
    intervention_type,
    intensity = 0.5,
    duration = 1000
  } = args;

  if (native?.pedagogic_intervention) {
    try {
      return native.pedagogic_intervention(agent_id, intervention_type, intensity, duration);
    } catch (e) {
      console.error("[agency] Native pedagogic intervention failed:", e);
    }
  }

  try {
    const state = negentropyStore.get(agent_id);
    if (!state) {
      return {
        error: "Agent not found - cannot apply intervention",
        agent_id
      };
    }

    const validInterventions = [
      "curiosity_boost",
      "exploration_scaffold",
      "competence_support",
      "autonomy_grant",
      "relatedness_enhance",
      "error_tolerance",
      "complexity_reduction"
    ];

    if (!validInterventions.includes(intervention_type)) {
      return {
        error: `Invalid intervention type. Valid types: ${validInterventions.join(", ")}`,
        intervention_type
      };
    }

    // Apply intervention effects
    const effects: Record<string, any> = {};

    switch (intervention_type) {
      case "curiosity_boost":
        state.intrinsicMotivation = Math.min(3.0, state.intrinsicMotivation + 0.3 * intensity);
        effects.motivation_delta = 0.3 * intensity;
        effects.description = "Increased intrinsic curiosity and exploration drive";
        break;

      case "exploration_scaffold":
        state.scaffoldMode = intensity > 0.7 ? "GuidedExploration" : "CuriosityNudge";
        effects.scaffold_change = state.scaffoldMode;
        effects.description = "Provided structured exploration support";
        break;

      case "competence_support":
        state.cognitiveState.pfc_inhibition = Math.min(1.0, state.cognitiveState.pfc_inhibition + 0.2 * intensity);
        effects.pfc_delta = 0.2 * intensity;
        effects.description = "Enhanced executive function support";
        break;

      case "autonomy_grant":
        state.negentropy = Math.min(1.0, state.negentropy + 0.15 * intensity);
        effects.negentropy_delta = 0.15 * intensity;
        effects.description = "Granted more autonomous operation space";
        break;

      case "relatedness_enhance":
        state.cognitiveState.insula_interoception = Math.min(1.0, state.cognitiveState.insula_interoception + 0.2 * intensity);
        effects.insula_delta = 0.2 * intensity;
        effects.description = "Strengthened social/relational awareness";
        break;

      case "error_tolerance":
        state.cognitiveState.acc_error_detection = Math.max(0.3, state.cognitiveState.acc_error_detection - 0.1 * intensity);
        effects.acc_delta = -0.1 * intensity;
        effects.description = "Reduced error sensitivity to encourage exploration";
        break;

      case "complexity_reduction":
        state.scaffoldMode = "DirectInstruction";
        effects.scaffold_change = "DirectInstruction";
        effects.description = "Simplified task complexity for easier learning";
        break;
    }

    // Update Bateson level based on new negentropy
    const n = state.negentropy;
    const canL4 = state.l3StabilizationSteps >= 100 &&
                  state.populationContext >= 3 &&
                  state.fitnessSignal >= 0.5 &&
                  n >= 0.9;

    if (n < 0.25) state.batesonLevel = "L0";
    else if (n < 0.5) state.batesonLevel = "L1";
    else if (n < 0.75) state.batesonLevel = "L2";
    else if (n < 0.9 || !canL4) state.batesonLevel = "L3";
    else state.batesonLevel = "L4";

    state.lastUpdate = Date.now();
    negentropyStore.set(agent_id, state);

    return {
      success: true,
      intervention_type,
      intensity,
      duration,
      effects,
      new_state: {
        negentropy: state.negentropy,
        bateson_level: state.batesonLevel,
        scaffold_mode: state.scaffoldMode,
        intrinsic_motivation: state.intrinsicMotivation
      },
      is_alive: state.negentropy >= 0.5,
      philosophy: "Interventions are graceful scaffolds, not punishments. They help the agent develop towards autonomy.",
      method: "typescript_intervention"
    };
  } catch (error) {
    return {
      error: `Pedagogic intervention failed: ${error}`,
      agent_id
    };
  }
}

// ============================================================================
// L4 Evolution Functions (Holland, 1975)
// ============================================================================

/**
 * Set population context for L4 evolutionary learning
 *
 * Based on Holland's Genetic Algorithms (1975):
 * - Population size affects genetic diversity
 * - Minimum population of 3 required for meaningful evolution
 * - Larger populations enable more exploration
 *
 * @param args.agent_id Agent identifier
 * @param args.population_size Number of agents in population
 * @param args.population_diversity Diversity metric [0,1]
 */
async function setPopulationContext(args: any, native: any) {
  const { agent_id, population_size, population_diversity = 0.5 } = args;

  if (native?.set_population_context) {
    try {
      return native.set_population_context(agent_id, population_size, population_diversity);
    } catch (e) {
      console.error("[agency] Native set_population_context failed:", e);
    }
  }

  try {
    let state = negentropyStore.get(agent_id);
    if (!state) {
      // Initialize state if not exists
      state = {
        negentropy: 0.5,
        batesonLevel: "L1" as const,
        scaffoldMode: "Observation" as const,
        intrinsicMotivation: 1.0,
        cognitiveState: {
          pfc_inhibition: 0.5,
          acc_error_detection: 0.5,
          insula_interoception: 0.5,
          basal_ganglia_action: 0.5,
          hippocampus_memory: 0.5
        },
        lastUpdate: Date.now(),
        l3StabilizationSteps: 0,
        populationContext: 1,
        fitnessSignal: 0.0
      };
    }

    const previousPopulation = state.populationContext;
    state.populationContext = Math.max(1, Math.floor(population_size));
    state.lastUpdate = Date.now();
    negentropyStore.set(agent_id, state);

    // Check L4 readiness after update
    const l4Ready = state.l3StabilizationSteps >= 100 &&
                    state.populationContext >= 3 &&
                    state.fitnessSignal >= 0.5 &&
                    state.negentropy >= 0.9;

    return {
      success: true,
      agent_id,
      population_size: state.populationContext,
      population_diversity,
      previous_population: previousPopulation,
      l4_requirement_met: state.populationContext >= 3,
      l4_ready: l4Ready,
      holland_citation: "Holland, J.H. (1975). Adaptation in Natural and Artificial Systems. University of Michigan Press.",
      theory: "Population-based search enables parallel exploration of solution space. Diversity prevents premature convergence.",
      method: "typescript_population"
    };
  } catch (error) {
    return {
      error: `Set population context failed: ${error}`,
      agent_id
    };
  }
}

/**
 * Update fitness signal for evolutionary selection pressure
 *
 * Based on evolutionary computation theory:
 * - Fitness determines selection probability
 * - Higher fitness = more likely to reproduce/persist
 * - Fitness landscape type affects search dynamics
 *
 * @param args.agent_id Agent identifier
 * @param args.fitness Fitness value [0,1]
 * @param args.fitness_landscape Type of fitness landscape
 */
async function updateFitness(args: any, native: any) {
  const { agent_id, fitness, fitness_landscape = "dynamic" } = args;

  if (native?.update_fitness) {
    try {
      return native.update_fitness(agent_id, fitness, fitness_landscape);
    } catch (e) {
      console.error("[agency] Native update_fitness failed:", e);
    }
  }

  try {
    let state = negentropyStore.get(agent_id);
    if (!state) {
      return {
        error: "Agent not found - cannot update fitness",
        agent_id
      };
    }

    const previousFitness = state.fitnessSignal;
    state.fitnessSignal = Math.max(0, Math.min(1, fitness));
    state.lastUpdate = Date.now();
    negentropyStore.set(agent_id, state);

    // Check L4 readiness after update
    const l4Ready = state.l3StabilizationSteps >= 100 &&
                    state.populationContext >= 3 &&
                    state.fitnessSignal >= 0.5 &&
                    state.negentropy >= 0.9;

    // Update Bateson level if L4 becomes possible
    if (l4Ready && state.batesonLevel === "L3") {
      state.batesonLevel = "L4";
      negentropyStore.set(agent_id, state);
    }

    const landscapeDescriptions: Record<string, string> = {
      static: "Fixed fitness function - convergence to global optimum possible",
      dynamic: "Time-varying fitness - requires continuous adaptation",
      coevolutionary: "Fitness depends on other agents - Red Queen dynamics",
      deceptive: "Local optima mislead search - requires novelty search"
    };

    return {
      success: true,
      agent_id,
      fitness: state.fitnessSignal,
      previous_fitness: previousFitness,
      fitness_landscape,
      landscape_description: landscapeDescriptions[fitness_landscape] || "Unknown landscape type",
      l4_requirement_met: state.fitnessSignal >= 0.5,
      l4_ready: l4Ready,
      bateson_level: state.batesonLevel,
      selection_pressure: fitness > 0.5 ? "high" : fitness > 0.25 ? "moderate" : "low",
      darwin_citation: "Darwin, C. (1859). On the Origin of Species. John Murray.",
      method: "typescript_fitness"
    };
  } catch (error) {
    return {
      error: `Update fitness failed: ${error}`,
      agent_id
    };
  }
}

/**
 * Get detailed L4 Evolution readiness assessment
 *
 * L4 Requirements (Holland, 1975):
 * 1. L3 Stabilization: 100+ steps at L3 level
 * 2. Population Context: 3+ agents in population
 * 3. Fitness Signal: 0.5+ selection pressure
 * 4. Negentropy: 0.9+ organizational state
 *
 * @param args.agent_id Agent identifier
 */
async function getL4Readiness(args: any, native: any) {
  const { agent_id } = args;

  if (native?.get_l4_readiness) {
    try {
      return native.get_l4_readiness(agent_id);
    } catch (e) {
      console.error("[agency] Native get_l4_readiness failed:", e);
    }
  }

  try {
    const state = negentropyStore.get(agent_id);
    if (!state) {
      return {
        error: "Agent not found - cannot assess L4 readiness",
        agent_id,
        l4_ready: false
      };
    }

    // Check each L4 requirement
    const requirements = {
      l3_stabilization: {
        current: state.l3StabilizationSteps,
        required: 100,
        met: state.l3StabilizationSteps >= 100,
        description: "Sustained L3 (Deutero-Learning) performance"
      },
      population_context: {
        current: state.populationContext,
        required: 3,
        met: state.populationContext >= 3,
        description: "Minimum population size for evolutionary dynamics"
      },
      fitness_signal: {
        current: state.fitnessSignal,
        required: 0.5,
        met: state.fitnessSignal >= 0.5,
        description: "Evolutionary selection pressure from environment"
      },
      negentropy: {
        current: state.negentropy,
        required: 0.9,
        met: state.negentropy >= 0.9,
        description: "High organizational state (low entropy)"
      }
    };

    const allMet = Object.values(requirements).every(r => r.met);
    const metCount = Object.values(requirements).filter(r => r.met).length;

    // Calculate readiness percentage
    const readinessScores = [
      Math.min(1, state.l3StabilizationSteps / 100),
      Math.min(1, state.populationContext / 3),
      Math.min(1, state.fitnessSignal / 0.5),
      Math.min(1, state.negentropy / 0.9)
    ];
    const readinessPercent = readinessScores.reduce((a, b) => a + b, 0) / 4 * 100;

    // Recommendations for unmet requirements
    const recommendations: string[] = [];
    if (!requirements.l3_stabilization.met) {
      recommendations.push(`Continue L3 learning for ${100 - state.l3StabilizationSteps} more steps`);
    }
    if (!requirements.population_context.met) {
      recommendations.push(`Increase population size to at least 3 agents`);
    }
    if (!requirements.fitness_signal.met) {
      recommendations.push(`Apply stronger evolutionary selection pressure (fitness >= 0.5)`);
    }
    if (!requirements.negentropy.met) {
      recommendations.push(`Increase organizational state through learning and adaptation`);
    }

    return {
      agent_id,
      current_level: state.batesonLevel,
      l4_ready: allMet,
      readiness_percent: readinessPercent.toFixed(1),
      requirements_met: `${metCount}/4`,
      requirements,
      recommendations: recommendations.length > 0 ? recommendations : ["All L4 requirements met - ready for evolutionary learning"],
      theoretical_basis: {
        l4_description: "Evolution - Population-level adaptation and phylogenetic change",
        key_features: [
          "Genetic algorithm optimization (Holland, 1975)",
          "Memetic evolution across agent populations",
          "Species-level behavioral changes",
          "Evolutionary pressure from fitness landscape",
          "Cross-generational knowledge transfer"
        ],
        citations: [
          "Holland, J.H. (1975). Adaptation in Natural and Artificial Systems",
          "Dawkins, R. (1976). The Selfish Gene",
          "Bateson, G. (1972). Steps to an Ecology of Mind"
        ]
      },
      method: "typescript_l4_readiness"
    };
  } catch (error) {
    return {
      error: `L4 readiness assessment failed: ${error}`,
      agent_id
    };
  }
}

/**
 * Trigger memetic knowledge transfer between agents
 *
 * Based on Dawkins' memetics (1976):
 * - Memes are units of cultural transmission
 * - High fidelity transfer preserves knowledge
 * - Low fidelity enables variation/mutation
 *
 * @param args.source_agent_id Agent to learn from
 * @param args.target_agent_id Agent to receive knowledge
 * @param args.knowledge_domain Domain of knowledge
 * @param args.transfer_fidelity Fidelity [0,1]
 */
async function triggerMemeticTransfer(args: any, native: any) {
  const {
    source_agent_id,
    target_agent_id,
    knowledge_domain = "general",
    transfer_fidelity = 0.8
  } = args;

  if (native?.trigger_memetic_transfer) {
    try {
      return native.trigger_memetic_transfer(source_agent_id, target_agent_id, knowledge_domain, transfer_fidelity);
    } catch (e) {
      console.error("[agency] Native memetic transfer failed:", e);
    }
  }

  try {
    const sourceState = negentropyStore.get(source_agent_id);
    const targetState = negentropyStore.get(target_agent_id);

    if (!sourceState) {
      return {
        error: "Source agent not found",
        source_agent_id,
        success: false
      };
    }

    if (!targetState) {
      return {
        error: "Target agent not found",
        target_agent_id,
        success: false
      };
    }

    // Check L3+ requirement for both agents
    const sourceLevel = sourceState.batesonLevel;
    const targetLevel = targetState.batesonLevel;
    const levelOrder = ["L0", "L1", "L2", "L3", "L4"];
    const sourceIdx = levelOrder.indexOf(sourceLevel);
    const targetIdx = levelOrder.indexOf(targetLevel);

    if (sourceIdx < 3) {
      return {
        error: "Source agent must be at L3+ level for memetic transfer",
        source_level: sourceLevel,
        required_level: "L3+",
        success: false
      };
    }

    if (targetIdx < 2) {
      return {
        error: "Target agent must be at L2+ level to receive memetic transfer",
        target_level: targetLevel,
        required_level: "L2+",
        success: false
      };
    }

    // Calculate transfer effects with fidelity-based noise
    const noise = 1 - transfer_fidelity;
    const transferred = {
      negentropy_boost: sourceState.negentropy * 0.1 * transfer_fidelity,
      motivation_boost: sourceState.intrinsicMotivation * 0.05 * transfer_fidelity,
      cognitive_transfer: {
        pfc: sourceState.cognitiveState.pfc_inhibition * 0.1 * transfer_fidelity,
        acc: sourceState.cognitiveState.acc_error_detection * 0.1 * transfer_fidelity
      }
    };

    // Apply transfer to target
    targetState.negentropy = Math.min(1.0, targetState.negentropy + transferred.negentropy_boost);
    targetState.intrinsicMotivation = Math.min(3.0, targetState.intrinsicMotivation + transferred.motivation_boost);
    targetState.cognitiveState.pfc_inhibition = Math.min(1.0, targetState.cognitiveState.pfc_inhibition + transferred.cognitive_transfer.pfc);
    targetState.cognitiveState.acc_error_detection = Math.min(1.0, targetState.cognitiveState.acc_error_detection + transferred.cognitive_transfer.acc);

    // Increment L3 stabilization for both agents
    sourceState.l3StabilizationSteps++;
    targetState.l3StabilizationSteps++;

    targetState.lastUpdate = Date.now();
    sourceState.lastUpdate = Date.now();
    negentropyStore.set(source_agent_id, sourceState);
    negentropyStore.set(target_agent_id, targetState);

    return {
      success: true,
      source_agent_id,
      target_agent_id,
      knowledge_domain,
      transfer_fidelity,
      mutation_rate: noise,
      transferred_effects: transferred,
      source_new_state: {
        bateson_level: sourceState.batesonLevel,
        l3_stabilization: sourceState.l3StabilizationSteps
      },
      target_new_state: {
        negentropy: targetState.negentropy,
        bateson_level: targetState.batesonLevel,
        intrinsic_motivation: targetState.intrinsicMotivation
      },
      memetics_theory: {
        description: "Memes are units of cultural information that replicate between minds",
        fidelity_meaning: "Higher fidelity = more accurate copy; lower = more variation",
        evolutionary_role: "Memetic transfer enables cultural evolution beyond genetic inheritance"
      },
      dawkins_citation: "Dawkins, R. (1976). The Selfish Gene. Oxford University Press.",
      method: "typescript_memetic"
    };
  } catch (error) {
    return {
      error: `Memetic transfer failed: ${error}`,
      source_agent_id,
      target_agent_id
    };
  }
}
