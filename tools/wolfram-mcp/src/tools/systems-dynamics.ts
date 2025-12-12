/**
 * Systems Dynamics & Analysis Tools
 * 
 * Wolfram-powered systems thinking capabilities:
 * - System modeling and simulation
 * - Equilibrium analysis
 * - Control theory
 * - Feedback loop analysis
 * - Causal loop diagrams
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";

export const systemsDynamicsTools: Tool[] = [
  // ============================================================================
  // System Modeling
  // ============================================================================
  {
    name: "systems_model_create",
    description: "Create a system dynamics model with stocks, flows, and feedback loops.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Model name" },
        stocks: { 
          type: "array", 
          items: { 
            type: "object",
            properties: {
              name: { type: "string" },
              initial: { type: "number" },
              unit: { type: "string" }
            }
          },
          description: "Stock variables (accumulators)"
        },
        flows: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              from: { type: "string" },
              to: { type: "string" },
              rate: { type: "string", description: "Rate expression" }
            }
          },
          description: "Flow variables"
        },
        parameters: {
          type: "object",
          description: "Model parameters"
        },
      },
      required: ["name", "stocks"],
    },
  },
  {
    name: "systems_model_simulate",
    description: "Simulate a system model over time and return trajectories.",
    inputSchema: {
      type: "object",
      properties: {
        equations: { type: "array", items: { type: "string" }, description: "Differential equations" },
        initialConditions: { type: "object", description: "Initial values for each variable" },
        parameters: { type: "object", description: "Parameter values" },
        timeSpan: { type: "array", items: { type: "number" }, description: "[t_start, t_end]" },
        outputVariables: { type: "array", items: { type: "string" } },
      },
      required: ["equations", "initialConditions", "timeSpan"],
    },
  },

  // ============================================================================
  // Equilibrium Analysis
  // ============================================================================
  {
    name: "systems_equilibrium_find",
    description: "Find equilibrium points (fixed points, steady states) of a dynamical system.",
    inputSchema: {
      type: "object",
      properties: {
        equations: { type: "array", items: { type: "string" }, description: "System equations (set to 0 for equilibrium)" },
        variables: { type: "array", items: { type: "string" }, description: "State variables" },
        constraints: { type: "object", description: "Variable constraints (bounds)" },
      },
      required: ["equations", "variables"],
    },
  },
  {
    name: "systems_equilibrium_stability",
    description: "Analyze stability of equilibrium points using eigenvalue analysis.",
    inputSchema: {
      type: "object",
      properties: {
        jacobian: { type: "array", items: { type: "array" }, description: "Jacobian matrix at equilibrium" },
        equilibriumPoint: { type: "object", description: "The equilibrium point to analyze" },
      },
      required: ["jacobian"],
    },
  },
  {
    name: "systems_equilibrium_bifurcation",
    description: "Analyze bifurcation behavior as parameters change.",
    inputSchema: {
      type: "object",
      properties: {
        equations: { type: "array", items: { type: "string" } },
        variables: { type: "array", items: { type: "string" } },
        bifurcationParameter: { type: "string", description: "Parameter to vary" },
        parameterRange: { type: "array", items: { type: "number" }, description: "[min, max]" },
      },
      required: ["equations", "variables", "bifurcationParameter", "parameterRange"],
    },
  },

  // ============================================================================
  // Control Theory
  // ============================================================================
  {
    name: "systems_control_design",
    description: "Design a controller for a system (PID, state feedback, optimal control).",
    inputSchema: {
      type: "object",
      properties: {
        systemModel: { type: "object", description: "State-space or transfer function model" },
        controllerType: { type: "string", enum: ["pid", "state_feedback", "lqr", "mpc"], description: "Controller type" },
        specifications: { type: "object", description: "Control specifications (settling time, overshoot, etc.)" },
      },
      required: ["systemModel", "controllerType"],
    },
  },
  {
    name: "systems_control_analyze",
    description: "Analyze controllability, observability, and stability of a control system.",
    inputSchema: {
      type: "object",
      properties: {
        A: { type: "array", items: { type: "array" }, description: "State matrix" },
        B: { type: "array", items: { type: "array" }, description: "Input matrix" },
        C: { type: "array", items: { type: "array" }, description: "Output matrix" },
        D: { type: "array", items: { type: "array" }, description: "Feedthrough matrix" },
      },
      required: ["A", "B"],
    },
  },

  // ============================================================================
  // Feedback Loop Analysis
  // ============================================================================
  {
    name: "systems_feedback_causal_loop",
    description: "Analyze causal loop diagrams and identify feedback loops.",
    inputSchema: {
      type: "object",
      properties: {
        variables: { type: "array", items: { type: "string" } },
        connections: {
          type: "array",
          items: {
            type: "object",
            properties: {
              from: { type: "string" },
              to: { type: "string" },
              polarity: { type: "string", enum: ["+", "-"], description: "Positive or negative influence" }
            }
          }
        },
      },
      required: ["variables", "connections"],
    },
  },
  {
    name: "systems_feedback_loop_gain",
    description: "Calculate loop gain and phase margin for stability analysis.",
    inputSchema: {
      type: "object",
      properties: {
        transferFunction: { type: "string", description: "Open-loop transfer function" },
        frequency: { type: "number", description: "Frequency of interest (rad/s)" },
      },
      required: ["transferFunction"],
    },
  },

  // ============================================================================
  // Network Analysis
  // ============================================================================
  {
    name: "systems_network_analyze",
    description: "Analyze system as a network - centrality, clustering, flow.",
    inputSchema: {
      type: "object",
      properties: {
        nodes: { type: "array", items: { type: "string" } },
        edges: { 
          type: "array", 
          items: { 
            type: "object",
            properties: {
              from: { type: "string" },
              to: { type: "string" },
              weight: { type: "number" }
            }
          }
        },
        analysisType: { 
          type: "string", 
          enum: ["centrality", "clustering", "flow", "communities", "all"],
          description: "Type of network analysis"
        },
      },
      required: ["nodes", "edges"],
    },
  },
  {
    name: "systems_network_optimize",
    description: "Optimize network flow or structure.",
    inputSchema: {
      type: "object",
      properties: {
        network: { type: "object", description: "Network specification" },
        objective: { type: "string", enum: ["max_flow", "min_cost", "shortest_path", "min_spanning_tree"] },
        constraints: { type: "object" },
      },
      required: ["network", "objective"],
    },
  },

  // ============================================================================
  // Sensitivity Analysis
  // ============================================================================
  {
    name: "systems_sensitivity_analyze",
    description: "Analyze parameter sensitivity - how outputs change with inputs.",
    inputSchema: {
      type: "object",
      properties: {
        model: { type: "string", description: "Model expression or function" },
        parameters: { type: "array", items: { type: "string" } },
        nominalValues: { type: "object" },
        perturbation: { type: "number", description: "Perturbation fraction (default: 0.01)" },
      },
      required: ["model", "parameters", "nominalValues"],
    },
  },
  {
    name: "systems_monte_carlo",
    description: "Run Monte Carlo simulation for uncertainty quantification.",
    inputSchema: {
      type: "object",
      properties: {
        model: { type: "string" },
        parameterDistributions: { 
          type: "object", 
          description: "Parameter distributions {param: {type: 'normal', mean: x, std: y}}" 
        },
        iterations: { type: "number", description: "Number of Monte Carlo iterations" },
        outputMetrics: { type: "array", items: { type: "string" } },
      },
      required: ["model", "parameterDistributions"],
    },
  },
];

export const systemsDynamicsWolframCode: Record<string, (args: any) => string> = {
  "systems_equilibrium_find": (args) => {
    const eqs = args.equations?.map((e: string) => `${e} == 0`).join(", ") || "";
    const vars = args.variables?.join(", ") || "x";
    return `Solve[{${eqs}}, {${vars}}] // ToString`;
  },
  
  "systems_equilibrium_stability": (args) => {
    const jacobian = JSON.stringify(args.jacobian || [[0]]);
    return `Module[{J = ${jacobian}, eigs},
      eigs = Eigenvalues[J];
      <|
        "eigenvalues" -> eigs,
        "stable" -> AllTrue[Re[eigs], # < 0 &],
        "type" -> Which[
          AllTrue[Re[eigs], # < 0 &], "Stable node/focus",
          AllTrue[Re[eigs], # > 0 &], "Unstable node/focus",
          True, "Saddle point"
        ]
      |>
    ] // ToString`;
  },
  
  "systems_model_simulate": (args) => {
    const eqs = args.equations?.join(", ") || "";
    const initial = Object.entries(args.initialConditions || {})
      .map(([k, v]) => `${k}[0] == ${v}`)
      .join(", ");
    const tSpan = args.timeSpan || [0, 10];
    const vars = args.outputVariables?.join(", ") || "x";
    return `NDSolve[{${eqs}, ${initial}}, {${vars}}, {t, ${tSpan[0]}, ${tSpan[1]}}] // ToString`;
  },
  
  "systems_control_analyze": (args) => {
    const A = JSON.stringify(args.A || [[0]]);
    const B = JSON.stringify(args.B || [[1]]);
    return `Module[{sys = StateSpaceModel[{${A}, ${B}}]},
      <|
        "controllable" -> ControllableModelQ[sys],
        "controllabilityMatrix" -> ControllabilityMatrix[sys],
        "poles" -> SystemsModelExtract[sys, "Poles"]
      |>
    ] // ToString`;
  },
  
  "systems_feedback_causal_loop": (args) => {
    const edges = (args.connections || [])
      .map((c: any) => `DirectedEdge["${c.from}", "${c.to}"]`)
      .join(", ");
    return `Module[{g = Graph[{${edges}}], cycles},
      cycles = FindCycle[g, Infinity, All];
      <|
        "loopCount" -> Length[cycles],
        "loops" -> cycles,
        "reinforcingLoops" -> Select[cycles, EvenQ[Count[#, _?(MemberQ[{"+"}, #] &)]] &],
        "balancingLoops" -> Select[cycles, OddQ[Count[#, _?(MemberQ[{"-"}, #] &)]] &]
      |>
    ] // ToString`;
  },
  
  "systems_network_analyze": (args) => {
    const edges = (args.edges || [])
      .map((e: any) => `"${e.from}" -> "${e.to}"`)
      .join(", ");
    return `Module[{g = Graph[{${edges}}]},
      <|
        "vertexCount" -> VertexCount[g],
        "edgeCount" -> EdgeCount[g],
        "centrality" -> BetweennessCentrality[g],
        "clustering" -> GlobalClusteringCoefficient[g],
        "communities" -> FindGraphCommunities[g],
        "diameter" -> GraphDiameter[g]
      |>
    ] // ToString`;
  },
  
  "systems_sensitivity_analyze": (args) => {
    const model = args.model || "x";
    const params = args.parameters?.join(", ") || "a";
    return `Module[{f = ${model}, sensitivities},
      sensitivities = Table[
        D[f, p],
        {p, {${params}}}
      ];
      <|
        "gradients" -> sensitivities,
        "elasticity" -> sensitivities * {${params}} / f
      |>
    ] // ToString`;
  },
};
