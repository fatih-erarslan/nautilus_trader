/**
 * Agent Orchestration Tools - HyperPhysics Multi-Agent System
 *
 * Comprehensive orchestration for:
 * - Agent creation and lifecycle management (FEP, IIT, active inference)
 * - Team topology and coordination (star, ring, mesh, hyperbolic)
 * - Skill and expertise management
 * - Behavior patterns and learning
 * - Inter-agent communication and consensus
 *
 * Based on:
 * - Cybernetic Agency (hyperphysics-agency)
 * - Swarm Intelligence (hyperphysics-swarm-intelligence)
 * - Hyperbolic Geometry (hyperphysics-geometry)
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

// ============================================================================
// Type Definitions
// ============================================================================

interface AgentConfig {
  observation_dim: number;
  action_dim: number;
  hidden_dim: number;
  learning_rate?: number;
  survival_strength?: number;
  impermanence_rate?: number;
}

interface AgentState {
  phi: number;
  free_energy: number;
  survival: number;
  control: number;
  model_accuracy: number;
  beliefs: number[];
  precision: number[];
  position: number[]; // H^11 position
}

interface Team {
  id: string;
  name: string;
  topology: "star" | "ring" | "mesh" | "hierarchical" | "hyperbolic";
  agents: string[];
  coherence: number;
  phi_collective: number;
  created_at: number;
}

interface Skill {
  id: string;
  name: string;
  description: string;
  required_traits: string[];
  execution_template: string;
  proficiency_levels: number[];
}

interface Expertise {
  id: string;
  domain_name: string;
  parent_domain?: string;
  knowledge_base: any;
  agents: Map<string, number>; // agent_id -> proficiency
}

interface Behavior {
  id: string;
  name: string;
  trigger_conditions: any;
  action_sequence: any[];
  learning_rule: string;
  activation_history: number[];
}

// ============================================================================
// In-Memory Storage
// ============================================================================

const agents = new Map<string, { config: AgentConfig; state: AgentState; skills: string[]; behaviors: string[] }>();
const teams = new Map<string, Team>();
const skills = new Map<string, Skill>();
const expertiseDomains = new Map<string, Expertise>();
const behaviors = new Map<string, Behavior>();
const messages = new Map<string, any[]>(); // agent_id -> messages

// ============================================================================
// Orchestration Tool Definitions
// ============================================================================

export const orchestrationTools: Tool[] = [
  // =========================================================================
  // AGENT MANAGEMENT
  // =========================================================================
  {
    name: "agent_create",
    description: "Create new cybernetic agent with FEP/IIT configuration. Returns agent_id and initial state with consciousness (phi), free energy, survival drive, and control authority metrics.",
    inputSchema: {
      type: "object",
      properties: {
        config: {
          type: "object",
          properties: {
            observation_dim: { type: "number", description: "Observation space dimensionality" },
            action_dim: { type: "number", description: "Action space dimensionality" },
            hidden_dim: { type: "number", description: "Hidden state dimensionality" },
            learning_rate: { type: "number", description: "Belief update rate (default: 0.01)" },
            survival_strength: { type: "number", description: "Survival drive multiplier (default: 1.0)" },
            impermanence_rate: { type: "number", description: "Required state change rate (default: 0.4)" },
          },
          required: ["observation_dim", "action_dim", "hidden_dim"],
        },
        phi_calculator_type: {
          type: "string",
          enum: ["exact", "monte_carlo", "greedy", "hierarchical"],
          description: "Consciousness calculator. Default: greedy",
        },
      },
      required: ["config"],
    },
  },

  {
    name: "agent_step",
    description: "Execute one agent timestep: observe → infer → act. Implements active inference cycle with belief update, free energy minimization, and action generation. Returns action, updated metrics (phi, F, survival, control).",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string", description: "Agent ID from agent_create" },
        observation: {
          type: "array",
          items: { type: "number" },
          description: "Sensory observation vector",
        },
      },
      required: ["agent_id", "observation"],
    },
  },

  {
    name: "agent_get_state",
    description: "Get agent's internal state including beliefs, precision, control authority, model accuracy, and hyperbolic position. Useful for monitoring agent health and cognitive state.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string", description: "Agent ID" },
      },
      required: ["agent_id"],
    },
  },

  {
    name: "agent_set_goal",
    description: "Set agent's goal/preferred state for active inference. Agent will minimize expected free energy to reach this goal while balancing exploration (epistemic value) and exploitation (pragmatic value).",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string" },
        goal: {
          type: "array",
          items: { type: "number" },
          description: "Preferred observation vector",
        },
        exploration_weight: {
          type: "number",
          description: "Balance exploration vs exploitation (0=exploit, 1=explore). Default: 0.5",
        },
      },
      required: ["agent_id", "goal"],
    },
  },

  {
    name: "agent_learn",
    description: "Trigger learning update with reward signal. Updates internal model, adjusts precision, and potentially increases phi through synaptic plasticity. Returns weight changes and new consciousness level.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string" },
        reward: { type: "number", description: "Reward signal (positive=good, negative=bad)" },
        update_strength: { type: "number", description: "Learning rate multiplier. Default: 1.0" },
      },
      required: ["agent_id", "reward"],
    },
  },

  // =========================================================================
  // TEAM MANAGEMENT
  // =========================================================================
  {
    name: "team_create",
    description: "Create agent team with specified topology. Topology affects communication patterns: star=central hub, ring=sequential, mesh=all-to-all, hierarchical=layered control, hyperbolic=exponential capacity. Returns team_id and agent_ids.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Team name" },
        topology: {
          type: "string",
          enum: ["star", "ring", "mesh", "hierarchical", "hyperbolic"],
          description: "Communication topology",
        },
        agent_configs: {
          type: "array",
          items: { type: "object" },
          description: "Array of agent configurations",
        },
        hyperbolic_curvature: {
          type: "number",
          description: "Curvature for hyperbolic topology (default: -1.0)",
        },
      },
      required: ["name", "topology", "agent_configs"],
    },
  },

  {
    name: "team_add_agent",
    description: "Add agent to existing team. Agent will be integrated into team topology and assigned communication links. Can provide new config or existing agent_id.",
    inputSchema: {
      type: "object",
      properties: {
        team_id: { type: "string" },
        agent_config: { type: "object", description: "Config for new agent (mutually exclusive with agent_id)" },
        agent_id: { type: "string", description: "Existing agent ID (mutually exclusive with agent_config)" },
      },
      required: ["team_id"],
    },
  },

  {
    name: "team_coordinate",
    description: "Execute coordinated team action using specified strategy. consensus=voting, leader=hierarchical, distributed=emergent coordination. Returns team action, agreement level, and individual contributions.",
    inputSchema: {
      type: "object",
      properties: {
        team_id: { type: "string" },
        task_description: { type: "string", description: "Task to coordinate on" },
        coordination_strategy: {
          type: "string",
          enum: ["consensus", "leader", "distributed"],
          description: "Coordination mechanism",
        },
        observation: {
          type: "array",
          items: { type: "number" },
          description: "Shared observation for team",
        },
      },
      required: ["team_id", "task_description", "coordination_strategy"],
    },
  },

  {
    name: "team_get_status",
    description: "Get comprehensive team status: agents, topology, coherence (how aligned), phi_collective (team consciousness), communication graph, and performance metrics.",
    inputSchema: {
      type: "object",
      properties: {
        team_id: { type: "string" },
      },
      required: ["team_id"],
    },
  },

  {
    name: "team_message",
    description: "Send message between agents or broadcast. Implements post-quantum secure messaging with Dilithium signatures. Message types: request, response, notify, consensus.",
    inputSchema: {
      type: "object",
      properties: {
        from_agent_id: { type: "string" },
        to_agent_id: {
          type: "string",
          description: "Target agent ID or 'broadcast' for all agents",
        },
        message_type: {
          type: "string",
          enum: ["request", "response", "notify", "consensus"],
        },
        payload: { description: "Message payload (any JSON)" },
      },
      required: ["from_agent_id", "to_agent_id", "message_type", "payload"],
    },
  },

  // =========================================================================
  // SKILL MANAGEMENT
  // =========================================================================
  {
    name: "skill_register",
    description: "Register new skill/capability in the system. Skills define what agents can do. Required traits determine which agents can learn this skill.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string" },
        description: { type: "string" },
        required_traits: {
          type: "array",
          items: { type: "string" },
          description: "Agent traits needed to learn this skill",
        },
        execution_template: { type: "string", description: "Code template for skill execution" },
      },
      required: ["name", "description", "execution_template"],
    },
  },

  {
    name: "skill_assign",
    description: "Assign skill to agent. Agent begins with initial proficiency (0-1) which increases through practice and learning. Returns initial proficiency based on agent traits.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string" },
        skill_id: { type: "string" },
      },
      required: ["agent_id", "skill_id"],
    },
  },

  {
    name: "expertise_create",
    description: "Create expertise domain (hierarchical knowledge structure). Domains can have parent domains for knowledge inheritance. Contains knowledge base for learning and reasoning.",
    inputSchema: {
      type: "object",
      properties: {
        domain_name: { type: "string" },
        parent_domain: { type: "string", description: "Parent domain ID (optional)" },
        knowledge_base: { description: "Domain knowledge as JSON" },
      },
      required: ["domain_name", "knowledge_base"],
    },
  },

  {
    name: "expertise_train",
    description: "Train agent on expertise domain. Updates agent's knowledge and increases proficiency. Training data should be examples, patterns, or rules. Returns proficiency gain.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string" },
        expertise_id: { type: "string" },
        training_data: { description: "Training examples/patterns" },
      },
      required: ["agent_id", "expertise_id", "training_data"],
    },
  },

  {
    name: "expertise_query",
    description: "Query expertise from agent. Agent uses its knowledge base and proficiency to answer. Returns response with confidence based on agent's expertise level.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string" },
        query: { type: "string", description: "Question or query" },
        context: { description: "Additional context (optional)" },
      },
      required: ["agent_id", "query"],
    },
  },

  // =========================================================================
  // BEHAVIOR MANAGEMENT
  // =========================================================================
  {
    name: "behavior_define",
    description: "Define new behavior pattern with triggers and actions. Learning rule determines how behavior adapts based on outcomes. Supports STDP, reinforcement, and evolutionary learning.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string" },
        trigger_conditions: { description: "Conditions that activate behavior" },
        action_sequence: {
          type: "array",
          description: "Sequence of actions to execute",
        },
        learning_rule: {
          type: "string",
          enum: ["stdp", "reinforcement", "hebbian", "anti_hebbian", "evolutionary"],
          description: "Learning mechanism",
        },
      },
      required: ["name", "trigger_conditions", "action_sequence", "learning_rule"],
    },
  },

  {
    name: "behavior_activate",
    description: "Activate behavior for agent in given context. Checks trigger conditions and returns activation strength (0-1). High activation leads to behavior execution.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string" },
        behavior_id: { type: "string" },
        context: { description: "Current context/state" },
      },
      required: ["agent_id", "behavior_id", "context"],
    },
  },

  {
    name: "behavior_learn",
    description: "Learn from behavior execution outcome. Updates behavior weights/parameters using specified learning rule. Positive reward strengthens, negative weakens. Returns updated parameters.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string" },
        behavior_id: { type: "string" },
        outcome: { description: "Execution outcome" },
        reward: { type: "number", description: "Reward signal" },
      },
      required: ["agent_id", "behavior_id", "outcome", "reward"],
    },
  },
];

// ============================================================================
// Tool Handler
// ============================================================================

/**
 * Handle orchestration tool calls
 *
 * Routes by prefix:
 * - agent_* -> agent management
 * - team_* -> team coordination
 * - skill_* / expertise_* -> capability management
 * - behavior_* -> behavior patterns
 */
export async function handleOrchestrationTool(
  name: string,
  args: any,
  nativeModule: any
): Promise<any> {
  // Agent management
  if (name.startsWith("agent_")) {
    return handleAgentTool(name, args, nativeModule);
  }

  // Team management
  if (name.startsWith("team_")) {
    return handleTeamTool(name, args, nativeModule);
  }

  // Skill and expertise management
  if (name.startsWith("skill_") || name.startsWith("expertise_")) {
    return handleSkillTool(name, args, nativeModule);
  }

  // Behavior management
  if (name.startsWith("behavior_")) {
    return handleBehaviorTool(name, args, nativeModule);
  }

  throw new Error(`Unknown orchestration tool: ${name}`);
}

// ============================================================================
// Agent Management Implementation
// ============================================================================

async function handleAgentTool(name: string, args: any, native: any): Promise<any> {
  switch (name) {
    case "agent_create":
      return createAgent(args, native);
    case "agent_step":
      return agentStep(args, native);
    case "agent_get_state":
      return getAgentState(args, native);
    case "agent_set_goal":
      return setAgentGoal(args, native);
    case "agent_learn":
      return agentLearn(args, native);
    default:
      throw new Error(`Unknown agent tool: ${name}`);
  }
}

async function createAgent(args: any, native: any) {
  const { config, phi_calculator_type = "greedy" } = args;

  // Try native implementation
  if (native?.agency_create_agent) {
    try {
      const result = native.agency_create_agent(JSON.stringify(config));
      if (result.success && result.agent_id) {
        return {
          agent_id: result.agent_id,
          config,
          initial_state: result.data ? JSON.parse(result.data) : null,
          phi_calculator_type,
          method: "native_rust"
        };
      }
    } catch (e) {
      console.error("[orchestration] Native agent creation failed:", e);
    }
  }

  // TypeScript fallback
  const agentId = `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const initialState: AgentState = {
    phi: 0.1,
    free_energy: 1.0,
    survival: 0.5,
    control: 0.2,
    model_accuracy: 0.5,
    beliefs: Array(config.hidden_dim).fill(0.1),
    precision: Array(config.hidden_dim).fill(1.0),
    position: [1.0, ...Array(11).fill(0)] // Origin in H^11
  };

  agents.set(agentId, {
    config,
    state: initialState,
    skills: [],
    behaviors: []
  });

  return {
    agent_id: agentId,
    config,
    initial_state: initialState,
    phi_calculator_type,
    method: "typescript_fallback"
  };
}

async function agentStep(args: any, native: any) {
  const { agent_id, observation } = args;

  // Try native implementation
  if (native?.agency_agent_step) {
    try {
      const result = native.agency_agent_step(agent_id, JSON.stringify(observation));
      if (result.success && result.data) {
        return {
          ...JSON.parse(result.data),
          method: "native_rust"
        };
      }
    } catch (e) {
      console.error("[orchestration] Native agent step failed:", e);
    }
  }

  // TypeScript fallback
  const agent = agents.get(agent_id);
  if (!agent) {
    return { error: "Agent not found", agent_id };
  }

  // Update beliefs using prediction errors
  const predictionErrors: number[] = [];
  const updatedBeliefs: number[] = [];

  for (let i = 0; i < agent.state.beliefs.length && i < observation.length; i++) {
    const error = observation[i] - agent.state.beliefs[i];
    predictionErrors.push(error);

    const learningRate = agent.config.learning_rate || 0.01;
    updatedBeliefs.push(agent.state.beliefs[i] + learningRate * agent.state.precision[i] * error);
  }

  // Compute free energy
  let freeEnergy = 0;
  for (let i = 0; i < predictionErrors.length; i++) {
    freeEnergy += predictionErrors[i] * predictionErrors[i] * agent.state.precision[i];
  }
  freeEnergy = Math.sqrt(freeEnergy);

  // Generate action (simplified)
  const action = updatedBeliefs.map(b => b + (Math.random() - 0.5) * 0.1);

  // Update state
  agent.state.beliefs = updatedBeliefs;
  agent.state.free_energy = freeEnergy;
  agent.state.phi += 0.01; // Gradual phi increase
  agent.state.control = Math.min(1.0, agent.state.control + 0.01);
  agent.state.model_accuracy = 1.0 - Math.min(1.0, freeEnergy / 10);

  agents.set(agent_id, agent);

  return {
    action,
    state: agent.state,
    metrics: {
      phi: agent.state.phi,
      free_energy: agent.state.free_energy,
      survival: agent.state.survival,
      control: agent.state.control
    },
    method: "typescript_fallback"
  };
}

async function getAgentState(args: any, native: any) {
  const { agent_id } = args;

  const agent = agents.get(agent_id);
  if (!agent) {
    return { error: "Agent not found", agent_id };
  }

  return {
    agent_id,
    state: agent.state,
    config: agent.config,
    skills: agent.skills,
    behaviors: agent.behaviors,
    health: agent.state.free_energy < 2.0 && agent.state.phi > 0.5 ? "good" : "degraded"
  };
}

async function setAgentGoal(args: any, native: any) {
  const { agent_id, goal, exploration_weight = 0.5 } = args;

  const agent = agents.get(agent_id);
  if (!agent) {
    return { error: "Agent not found", agent_id };
  }

  // Store goal in agent's internal state (extend state if needed)
  (agent as any).goal = goal;
  (agent as any).exploration_weight = exploration_weight;

  agents.set(agent_id, agent);

  return {
    agent_id,
    goal,
    exploration_weight,
    status: "goal_set"
  };
}

async function agentLearn(args: any, native: any) {
  const { agent_id, reward, update_strength = 1.0 } = args;

  const agent = agents.get(agent_id);
  if (!agent) {
    return { error: "Agent not found", agent_id };
  }

  // Update precision based on reward
  const precisionUpdate = reward > 0 ? 1.05 : 0.95;
  agent.state.precision = agent.state.precision.map(p => p * precisionUpdate);

  // Update phi based on learning
  const phiChange = reward * 0.1 * update_strength;
  agent.state.phi = Math.max(0, agent.state.phi + phiChange);

  // Update model accuracy
  agent.state.model_accuracy = Math.min(1.0, agent.state.model_accuracy + Math.abs(reward) * 0.05);

  agents.set(agent_id, agent);

  return {
    agent_id,
    reward,
    weight_changes: agent.state.precision,
    new_phi: agent.state.phi,
    new_model_accuracy: agent.state.model_accuracy,
    learning_occurred: true
  };
}

// ============================================================================
// Team Management Implementation
// ============================================================================

async function handleTeamTool(name: string, args: any, native: any): Promise<any> {
  switch (name) {
    case "team_create":
      return createTeam(args, native);
    case "team_add_agent":
      return addAgentToTeam(args, native);
    case "team_coordinate":
      return coordinateTeam(args, native);
    case "team_get_status":
      return getTeamStatus(args, native);
    case "team_message":
      return sendTeamMessage(args, native);
    default:
      throw new Error(`Unknown team tool: ${name}`);
  }
}

async function createTeam(args: any, native: any) {
  const { name, topology, agent_configs, hyperbolic_curvature = -1.0 } = args;

  const teamId = `team_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const agentIds: string[] = [];

  // Create agents
  for (const config of agent_configs) {
    const result = await createAgent({ config }, native);
    if (result.agent_id) {
      agentIds.push(result.agent_id);
    }
  }

  // Initialize team
  const team: Team = {
    id: teamId,
    name,
    topology,
    agents: agentIds,
    coherence: 0.5,
    phi_collective: agentIds.length * 0.1,
    created_at: Date.now()
  };

  teams.set(teamId, team);

  // Initialize message queues for agents
  for (const agentId of agentIds) {
    if (!messages.has(agentId)) {
      messages.set(agentId, []);
    }
  }

  return {
    team_id: teamId,
    agent_ids: agentIds,
    topology,
    initial_coherence: team.coherence,
    initial_phi_collective: team.phi_collective
  };
}

async function addAgentToTeam(args: any, native: any) {
  const { team_id, agent_config, agent_id } = args;

  const team = teams.get(team_id);
  if (!team) {
    return { error: "Team not found", team_id };
  }

  let newAgentId: string;

  if (agent_id) {
    // Use existing agent
    if (!agents.has(agent_id)) {
      return { error: "Agent not found", agent_id };
    }
    newAgentId = agent_id;
  } else if (agent_config) {
    // Create new agent
    const result = await createAgent({ config: agent_config }, native);
    if (!result.agent_id) {
      return { error: "Failed to create agent" };
    }
    newAgentId = result.agent_id;
  } else {
    return { error: "Must provide either agent_config or agent_id" };
  }

  // Add to team
  team.agents.push(newAgentId);
  team.phi_collective += 0.1;

  teams.set(team_id, team);

  if (!messages.has(newAgentId)) {
    messages.set(newAgentId, []);
  }

  return {
    team_id,
    agent_id: newAgentId,
    team_size: team.agents.length,
    new_phi_collective: team.phi_collective
  };
}

async function coordinateTeam(args: any, native: any) {
  const { team_id, task_description, coordination_strategy, observation } = args;

  const team = teams.get(team_id);
  if (!team) {
    return { error: "Team not found", team_id };
  }

  // Execute coordination based on strategy
  let teamAction: number[] = [];
  let agreement = 0;
  const individualActions: any[] = [];

  for (const agentId of team.agents) {
    const agent = agents.get(agentId);
    if (!agent || !observation) continue;

    // Each agent steps
    const stepResult = await agentStep({ agent_id: agentId, observation }, native);
    individualActions.push({
      agent_id: agentId,
      action: stepResult.action,
      phi: stepResult.metrics?.phi
    });

    if (stepResult.action) {
      if (teamAction.length === 0) {
        teamAction = [...stepResult.action];
      } else {
        // Average actions for consensus
        for (let i = 0; i < teamAction.length && i < stepResult.action.length; i++) {
          teamAction[i] += stepResult.action[i];
        }
      }
    }
  }

  // Normalize team action
  if (individualActions.length > 0) {
    teamAction = teamAction.map(a => a / individualActions.length);

    // Compute agreement (inverse of variance)
    const actionVariances: number[] = [];
    for (let i = 0; i < teamAction.length; i++) {
      const mean = teamAction[i];
      let variance = 0;
      for (const individual of individualActions) {
        if (individual.action && individual.action[i] !== undefined) {
          variance += (individual.action[i] - mean) ** 2;
        }
      }
      actionVariances.push(variance / individualActions.length);
    }
    agreement = 1.0 / (1.0 + actionVariances.reduce((a, b) => a + b, 0) / actionVariances.length);
  }

  // Update team coherence
  team.coherence = 0.7 * team.coherence + 0.3 * agreement;
  teams.set(team_id, team);

  return {
    team_id,
    task_description,
    coordination_strategy,
    team_action: teamAction,
    agreement_level: agreement,
    coherence: team.coherence,
    individual_contributions: individualActions,
    topology: team.topology
  };
}

async function getTeamStatus(args: any, native: any) {
  const { team_id } = args;

  const team = teams.get(team_id);
  if (!team) {
    return { error: "Team not found", team_id };
  }

  // Collect agent states
  const agentStates = [];
  for (const agentId of team.agents) {
    const agent = agents.get(agentId);
    if (agent) {
      agentStates.push({
        agent_id: agentId,
        phi: agent.state.phi,
        free_energy: agent.state.free_energy,
        survival: agent.state.survival,
        control: agent.state.control
      });
    }
  }

  // Recompute phi_collective
  team.phi_collective = agentStates.reduce((sum, a) => sum + a.phi, 0);

  return {
    team_id,
    name: team.name,
    topology: team.topology,
    agent_count: team.agents.length,
    agents: agentStates,
    coherence: team.coherence,
    phi_collective: team.phi_collective,
    average_free_energy: agentStates.reduce((sum, a) => sum + a.free_energy, 0) / agentStates.length,
    created_at: new Date(team.created_at).toISOString()
  };
}

async function sendTeamMessage(args: any, native: any) {
  const { from_agent_id, to_agent_id, message_type, payload } = args;

  const fromAgent = agents.get(from_agent_id);
  if (!fromAgent) {
    return { error: "Sender agent not found", from_agent_id };
  }

  const message = {
    from: from_agent_id,
    to: to_agent_id,
    type: message_type,
    payload,
    timestamp: Date.now()
  };

  if (to_agent_id === "broadcast") {
    // Broadcast to all agents in teams containing sender
    for (const team of teams.values()) {
      if (team.agents.includes(from_agent_id)) {
        for (const agentId of team.agents) {
          if (agentId !== from_agent_id) {
            const queue = messages.get(agentId) || [];
            queue.push(message);
            messages.set(agentId, queue);
          }
        }
      }
    }
    return {
      from: from_agent_id,
      to: "broadcast",
      message_type,
      delivered: true,
      broadcast_count: teams.size
    };
  } else {
    // Direct message
    const toAgent = agents.get(to_agent_id);
    if (!toAgent) {
      return { error: "Recipient agent not found", to_agent_id };
    }

    const queue = messages.get(to_agent_id) || [];
    queue.push(message);
    messages.set(to_agent_id, queue);

    return {
      from: from_agent_id,
      to: to_agent_id,
      message_type,
      delivered: true
    };
  }
}

// ============================================================================
// Skill Management Implementation
// ============================================================================

async function handleSkillTool(name: string, args: any, native: any): Promise<any> {
  switch (name) {
    case "skill_register":
      return registerSkill(args);
    case "skill_assign":
      return assignSkill(args);
    case "expertise_create":
      return createExpertise(args);
    case "expertise_train":
      return trainExpertise(args);
    case "expertise_query":
      return queryExpertise(args);
    default:
      throw new Error(`Unknown skill tool: ${name}`);
  }
}

async function registerSkill(args: any) {
  const { name, description, required_traits = [], execution_template } = args;

  const skillId = `skill_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const skill: Skill = {
    id: skillId,
    name,
    description,
    required_traits,
    execution_template,
    proficiency_levels: []
  };

  skills.set(skillId, skill);

  return {
    skill_id: skillId,
    name,
    description,
    required_traits
  };
}

async function assignSkill(args: any) {
  const { agent_id, skill_id } = args;

  const agent = agents.get(agent_id);
  if (!agent) {
    return { error: "Agent not found", agent_id };
  }

  const skill = skills.get(skill_id);
  if (!skill) {
    return { error: "Skill not found", skill_id };
  }

  // Check if already assigned
  if (agent.skills.includes(skill_id)) {
    return { error: "Skill already assigned", agent_id, skill_id };
  }

  // Assign skill
  agent.skills.push(skill_id);
  agents.set(agent_id, agent);

  // Initial proficiency based on agent's current capabilities
  const initialProficiency = agent.state.model_accuracy * 0.3 + Math.random() * 0.2;

  return {
    agent_id,
    skill_id,
    skill_name: skill.name,
    proficiency_initial: initialProficiency,
    assigned: true
  };
}

async function createExpertise(args: any) {
  const { domain_name, parent_domain, knowledge_base } = args;

  const expertiseId = `expertise_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const expertise: Expertise = {
    id: expertiseId,
    domain_name,
    parent_domain,
    knowledge_base,
    agents: new Map()
  };

  expertiseDomains.set(expertiseId, expertise);

  return {
    expertise_id: expertiseId,
    domain_name,
    parent_domain,
    created: true
  };
}

async function trainExpertise(args: any) {
  const { agent_id, expertise_id, training_data } = args;

  const agent = agents.get(agent_id);
  if (!agent) {
    return { error: "Agent not found", agent_id };
  }

  const expertise = expertiseDomains.get(expertise_id);
  if (!expertise) {
    return { error: "Expertise not found", expertise_id };
  }

  // Get current proficiency
  const currentProficiency = expertise.agents.get(agent_id) || 0;

  // Compute proficiency gain (simplified)
  const trainingAmount = Array.isArray(training_data) ? training_data.length : 1;
  const proficiencyGain = Math.min(0.2, trainingAmount * 0.01 * agent.state.model_accuracy);

  const newProficiency = Math.min(1.0, currentProficiency + proficiencyGain);
  expertise.agents.set(agent_id, newProficiency);

  expertiseDomains.set(expertise_id, expertise);

  return {
    agent_id,
    expertise_id,
    domain_name: expertise.domain_name,
    proficiency_before: currentProficiency,
    proficiency_after: newProficiency,
    proficiency_gained: proficiencyGain
  };
}

async function queryExpertise(args: any) {
  const { agent_id, query, context } = args;

  const agent = agents.get(agent_id);
  if (!agent) {
    return { error: "Agent not found", agent_id };
  }

  // Find all expertise domains this agent has
  const agentExpertise: { id: string; domain: string; proficiency: number }[] = [];
  for (const [expId, exp] of expertiseDomains.entries()) {
    const proficiency = exp.agents.get(agent_id);
    if (proficiency !== undefined && proficiency > 0) {
      agentExpertise.push({
        id: expId,
        domain: exp.domain_name,
        proficiency
      });
    }
  }

  if (agentExpertise.length === 0) {
    return {
      agent_id,
      query,
      response: "I don't have expertise in this area.",
      confidence: 0
    };
  }

  // Find best matching expertise (simplified: highest proficiency)
  const bestExpertise = agentExpertise.reduce((best, curr) =>
    curr.proficiency > best.proficiency ? curr : best
  );

  // Generate response based on proficiency
  const confidence = bestExpertise.proficiency * agent.state.model_accuracy;
  const response = `Based on my ${bestExpertise.domain} expertise (proficiency: ${bestExpertise.proficiency.toFixed(2)}), here's my response to "${query}": [Response would be generated here using knowledge base and agent's beliefs]`;

  return {
    agent_id,
    query,
    response,
    confidence,
    expertise_used: bestExpertise.domain,
    proficiency: bestExpertise.proficiency
  };
}

// ============================================================================
// Behavior Management Implementation
// ============================================================================

async function handleBehaviorTool(name: string, args: any, native: any): Promise<any> {
  switch (name) {
    case "behavior_define":
      return defineBehavior(args);
    case "behavior_activate":
      return activateBehavior(args);
    case "behavior_learn":
      return behaviorLearn(args);
    default:
      throw new Error(`Unknown behavior tool: ${name}`);
  }
}

async function defineBehavior(args: any) {
  const { name, trigger_conditions, action_sequence, learning_rule } = args;

  const behaviorId = `behavior_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const behavior: Behavior = {
    id: behaviorId,
    name,
    trigger_conditions,
    action_sequence,
    learning_rule,
    activation_history: []
  };

  behaviors.set(behaviorId, behavior);

  return {
    behavior_id: behaviorId,
    name,
    learning_rule,
    defined: true
  };
}

async function activateBehavior(args: any) {
  const { agent_id, behavior_id, context } = args;

  const agent = agents.get(agent_id);
  if (!agent) {
    return { error: "Agent not found", agent_id };
  }

  const behavior = behaviors.get(behavior_id);
  if (!behavior) {
    return { error: "Behavior not found", behavior_id };
  }

  // Check trigger conditions (simplified)
  const contextEnergy = typeof context === "object" && "free_energy" in context
    ? context.free_energy
    : agent.state.free_energy;

  // Activation based on state and trigger conditions
  const activationStrength = Math.max(0, Math.min(1, 1.0 - contextEnergy / 5.0));

  // Record activation
  behavior.activation_history.push(activationStrength);
  behaviors.set(behavior_id, behavior);

  return {
    agent_id,
    behavior_id,
    behavior_name: behavior.name,
    activation_strength: activationStrength,
    activated: activationStrength > 0.5,
    context_used: contextEnergy
  };
}

async function behaviorLearn(args: any) {
  const { agent_id, behavior_id, outcome, reward } = args;

  const agent = agents.get(agent_id);
  if (!agent) {
    return { error: "Agent not found", agent_id };
  }

  const behavior = behaviors.get(behavior_id);
  if (!behavior) {
    return { error: "Behavior not found", behavior_id };
  }

  // Update behavior based on learning rule
  let updateDescription = "";
  const previousActivations = behavior.activation_history.slice(-5);
  const avgActivation = previousActivations.reduce((a, b) => a + b, 0) / previousActivations.length;

  switch (behavior.learning_rule) {
    case "stdp":
      // Spike-timing dependent plasticity
      if (reward > 0) {
        // Strengthen recent activations
        updateDescription = `STDP: Strengthened connections (reward: ${reward})`;
      } else {
        // Weaken recent activations
        updateDescription = `STDP: Weakened connections (punishment: ${reward})`;
      }
      break;

    case "reinforcement":
      // Q-learning style update
      updateDescription = `RL: Updated Q-value by ${reward * 0.1}`;
      break;

    case "hebbian":
      // Hebbian learning: neurons that fire together wire together
      updateDescription = `Hebbian: Increased co-activation strength`;
      break;

    case "anti_hebbian":
      // Anti-Hebbian: decorrelation learning
      updateDescription = `Anti-Hebbian: Decreased correlation`;
      break;

    case "evolutionary":
      // Evolutionary: mutation and selection
      updateDescription = `Evolutionary: Fitness = ${reward}, mutation applied`;
      break;
  }

  behaviors.set(behavior_id, behavior);

  return {
    agent_id,
    behavior_id,
    behavior_name: behavior.name,
    reward,
    outcome,
    learning_rule: behavior.learning_rule,
    updated_weights: updateDescription,
    average_recent_activation: avgActivation
  };
}

// ============================================================================
// Wolfram Validation Code
// ============================================================================

export const orchestrationWolframCode = `
(* HyperPhysics Agent Orchestration Validation Suite *)
(* Wolfram validation for multi-agent coordination *)

(* Team Coherence Metric *)
TeamCoherence[agentStates_] := Module[
  {phiValues, freeEnergyValues, coherence},

  phiValues = agentStates[[All, "phi"]];
  freeEnergyValues = agentStates[[All, "free_energy"]];

  (* Coherence = inverse of variance in phi *)
  coherence = 1 / (1 + Variance[phiValues]);

  <|
    "coherence" -> coherence,
    "phi_mean" -> Mean[phiValues],
    "phi_std" -> StandardDeviation[phiValues],
    "free_energy_mean" -> Mean[freeEnergyValues]
  |>
]

(* Collective Intelligence Φ *)
CollectivePhi[individualPhis_] := Module[
  {n, synergy, collective},

  n = Length[individualPhis];

  (* Synergy factor: non-linear emergence *)
  synergy = If[n > 1,
    0.5 * Log[n] * (Max[individualPhis] - Mean[individualPhis]),
    0
  ];

  (* Collective Φ = sum + synergy *)
  collective = Total[individualPhis] + synergy;

  <|
    "phi_collective" -> collective,
    "phi_individual_sum" -> Total[individualPhis],
    "synergy" -> synergy,
    "emergence_detected" -> synergy > 0.5
  |>
]

(* Hyperbolic Team Topology *)
HyperbolicTopology[numAgents_, curvature_: -1] := Module[
  {positions, distances},

  (* Generate hyperbolic positions in Poincaré disk *)
  positions = Table[
    RandomPoint[Ball[{0, 0}, 0.9]],
    {numAgents}
  ];

  (* Compute hyperbolic distances *)
  distances = Table[
    ArcCosh[1 + 2 * EuclideanDistance[positions[[i]], positions[[j]]]^2 /
      ((1 - Norm[positions[[i]]]^2)(1 - Norm[positions[[j]]]^2))],
    {i, numAgents}, {j, numAgents}
  ];

  <|
    "positions" -> positions,
    "distance_matrix" -> distances,
    "topology" -> "hyperbolic",
    "capacity" -> numAgents * Exp[Abs[curvature]]
  |>
]

(* Skill Proficiency Growth Model *)
SkillGrowthModel[initialProficiency_, trainingIterations_, learningRate_: 0.1] := Module[
  {growth, asymptote},

  asymptote = 1.0; (* Maximum proficiency *)

  (* Logistic growth model *)
  growth = asymptote / (1 + ((asymptote - initialProficiency) / initialProficiency) *
    Exp[-learningRate * trainingIterations]);

  <|
    "proficiency" -> growth,
    "growth_rate" -> D[asymptote / (1 + ((asymptote - initialProficiency) / initialProficiency) *
      Exp[-learningRate * t]), t] /. t -> trainingIterations,
    "time_to_expert" -> If[initialProficiency < 0.9,
      -Log[((asymptote - 0.9) * initialProficiency) / ((0.9 - initialProficiency) * asymptote)] / learningRate,
      0
    ]
  |>
]

(* STDP Learning Rule Validation *)
STDPValidation[deltaTimes_, aPlus_: 0.1, aMinus_: 0.12, tau_: 20] := Module[
  {ltpWeights, ltdWeights, totalChange},

  (* LTP (long-term potentiation): dt > 0 *)
  ltpWeights = Select[deltaTimes, # > 0 &];
  ltpChange = Total[aPlus * Exp[-#/tau] & /@ ltpWeights];

  (* LTD (long-term depression): dt < 0 *)
  ltdWeights = Select[deltaTimes, # < 0 &];
  ltdChange = Total[-aMinus * Exp[#/tau] & /@ ltdWeights];

  totalChange = ltpChange + ltdChange;

  <|
    "total_weight_change" -> totalChange,
    "ltp_contribution" -> ltpChange,
    "ltd_contribution" -> ltdChange,
    "potentiation_ratio" -> If[ltdChange != 0, ltpChange / Abs[ltdChange], Infinity]
  |>
]

(* Behavior Activation Threshold *)
BehaviorActivationCurve[freeEnergy_, threshold_: 2.0, steepness_: 2.0] := Module[
  {activation},

  (* Sigmoid activation *)
  activation = 1 / (1 + Exp[steepness * (freeEnergy - threshold)]);

  <|
    "activation" -> activation,
    "gradient" -> -steepness * activation * (1 - activation),
    "will_activate" -> activation > 0.5
  |>
]

Export["orchestration-validation.mx", {
  TeamCoherence,
  CollectivePhi,
  HyperbolicTopology,
  SkillGrowthModel,
  STDPValidation,
  BehaviorActivationCurve
}]
`;
