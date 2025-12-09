/**
 * Swarm Tools - MCP Tools for Agent-to-Agent Communication
 * 
 * Exposes the AgentMesh functionality as MCP tools
 * for Cascade/Windsurf to use in multi-agent workflows.
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { AgentMesh, createAgentMesh, getAgentMesh, SwarmMessage } from "./agent-mesh.js";

// ============================================================================
// Tool Schemas
// ============================================================================

const JoinMeshSchema = z.object({
  name: z.string().describe("Display name for this agent instance"),
  type: z.enum(["cascade", "windsurf", "custom"]).optional().default("cascade"),
});

const SendMessageSchema = z.object({
  to: z.string().describe("Recipient agent ID or 'broadcast' for all"),
  type: z.enum([
    "task", "result", "query", "response", "consensus", "vote",
    "sync", "alert", "memory", "code", "review", "approve"
  ]),
  payload: z.any().describe("Message payload"),
  priority: z.enum(["low", "normal", "high", "critical"]).optional().default("normal"),
});

const ProposeConsensusSchema = z.object({
  topic: z.string().describe("What are we voting on?"),
  options: z.array(z.string()).describe("Available choices"),
  deadlineMs: z.number().optional().default(60000).describe("Voting deadline in ms"),
});

const VoteSchema = z.object({
  proposalId: z.string().describe("ID of the consensus proposal"),
  choice: z.string().describe("Your vote choice"),
});

const CreateTaskSchema = z.object({
  title: z.string().describe("Task title"),
  description: z.string().describe("Task description"),
  assignees: z.array(z.string()).describe("Agent IDs to assign"),
});

const UpdateTaskSchema = z.object({
  taskId: z.string(),
  status: z.enum(["pending", "in_progress", "review", "completed"]).optional(),
  priority: z.number().optional(),
});

const SetMemorySchema = z.object({
  key: z.string().describe("Memory key"),
  value: z.any().describe("Value to store"),
});

const GetMemorySchema = z.object({
  key: z.string().describe("Memory key to retrieve"),
});

const ShareCodeSchema = z.object({
  filename: z.string().describe("File name"),
  content: z.string().describe("Code content"),
  description: z.string().describe("What this code does"),
});

const RequestReviewSchema = z.object({
  artifactId: z.string().describe("Code artifact ID"),
  reviewers: z.array(z.string()).describe("Agent IDs to request review from"),
});

// ============================================================================
// Tool Definitions
// ============================================================================

export const swarmTools: Tool[] = [
  {
    name: "swarm_join",
    description: "Join the agent mesh network to communicate with other Cascade/Windsurf instances on this machine. Call this first before using other swarm tools.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Your display name in the mesh" },
        type: { type: "string", enum: ["cascade", "windsurf", "custom"], description: "Agent type" },
      },
      required: ["name"],
    },
  },
  {
    name: "swarm_leave",
    description: "Leave the agent mesh network gracefully.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "swarm_list_agents",
    description: "List all active agents in the mesh network.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "swarm_send",
    description: "Send a message to another agent or broadcast to all.",
    inputSchema: {
      type: "object",
      properties: {
        to: { type: "string", description: "Agent ID or 'broadcast'" },
        type: { 
          type: "string", 
          enum: ["task", "result", "query", "response", "consensus", "vote", "sync", "alert", "memory", "code", "review", "approve"],
          description: "Message type"
        },
        payload: { description: "Message content (any JSON)" },
        priority: { type: "string", enum: ["low", "normal", "high", "critical"] },
      },
      required: ["to", "type", "payload"],
    },
  },
  {
    name: "swarm_propose",
    description: "Propose a consensus vote to all agents. Use this for decisions that need agreement from multiple agents.",
    inputSchema: {
      type: "object",
      properties: {
        topic: { type: "string", description: "Question or topic to vote on" },
        options: { type: "array", items: { type: "string" }, description: "Available choices" },
        deadlineMs: { type: "number", description: "Voting deadline in milliseconds" },
      },
      required: ["topic", "options"],
    },
  },
  {
    name: "swarm_vote",
    description: "Cast your vote on a consensus proposal.",
    inputSchema: {
      type: "object",
      properties: {
        proposalId: { type: "string" },
        choice: { type: "string" },
      },
      required: ["proposalId", "choice"],
    },
  },
  {
    name: "swarm_create_task",
    description: "Create a shared task and assign it to agents for collaborative work.",
    inputSchema: {
      type: "object",
      properties: {
        title: { type: "string" },
        description: { type: "string" },
        assignees: { type: "array", items: { type: "string" }, description: "Agent IDs" },
      },
      required: ["title", "description", "assignees"],
    },
  },
  {
    name: "swarm_update_task",
    description: "Update a shared task's status or priority.",
    inputSchema: {
      type: "object",
      properties: {
        taskId: { type: "string" },
        status: { type: "string", enum: ["pending", "in_progress", "review", "completed"] },
        priority: { type: "number" },
      },
      required: ["taskId"],
    },
  },
  {
    name: "swarm_my_tasks",
    description: "List tasks assigned to me.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "swarm_set_memory",
    description: "Store a value in shared memory accessible to all agents.",
    inputSchema: {
      type: "object",
      properties: {
        key: { type: "string" },
        value: { description: "Any JSON value" },
      },
      required: ["key", "value"],
    },
  },
  {
    name: "swarm_get_memory",
    description: "Retrieve a value from shared memory.",
    inputSchema: {
      type: "object",
      properties: {
        key: { type: "string" },
      },
      required: ["key"],
    },
  },
  {
    name: "swarm_share_code",
    description: "Share code with other agents for review or collaboration.",
    inputSchema: {
      type: "object",
      properties: {
        filename: { type: "string" },
        content: { type: "string" },
        description: { type: "string" },
      },
      required: ["filename", "content", "description"],
    },
  },
  {
    name: "swarm_request_review",
    description: "Request code review from specific agents.",
    inputSchema: {
      type: "object",
      properties: {
        artifactId: { type: "string" },
        reviewers: { type: "array", items: { type: "string" } },
      },
      required: ["artifactId", "reviewers"],
    },
  },
  {
    name: "swarm_find_nearest",
    description: "Find nearest agents by hyperbolic distance (affinity-based routing).",
    inputSchema: {
      type: "object",
      properties: {
        count: { type: "number", description: "How many agents to find" },
      },
    },
  },
  {
    name: "swarm_trust_scores",
    description: "Get trust scores for all agents using GNN-based trust propagation.",
    inputSchema: { type: "object", properties: {} },
  },
];

// ============================================================================
// Tool Handlers
// ============================================================================

export async function handleSwarmTool(name: string, args: any): Promise<string> {
  let mesh: AgentMesh;
  
  switch (name) {
    case "swarm_join": {
      const { name: agentName, type } = JoinMeshSchema.parse(args);
      mesh = createAgentMesh(agentName, type);
      await mesh.join();
      
      // Set up event logging
      mesh.on("agent_joined", (agent) => console.error(`[Swarm] Agent joined: ${agent.name}`));
      mesh.on("agent_left", (agent) => console.error(`[Swarm] Agent left: ${agent.name || agent.id}`));
      mesh.on("task_received", (task) => console.error(`[Swarm] Task received: ${task.title}`));
      mesh.on("consensus_proposed", (p) => console.error(`[Swarm] Consensus proposed: ${p.topic}`));
      mesh.on("consensus_approved", ({ proposal, result }) => console.error(`[Swarm] Consensus approved: ${proposal.topic} -> ${result}`));
      
      return JSON.stringify({
        success: true,
        message: `Joined mesh as "${agentName}"`,
        agentId: mesh.myId,
        activeAgents: mesh.activeAgents.length,
      });
    }
    
    case "swarm_leave": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      await mesh.leave();
      return JSON.stringify({ success: true, message: "Left mesh" });
    }
    
    case "swarm_list_agents": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh. Call swarm_join first." });
      return JSON.stringify({
        myId: mesh.myId,
        myName: mesh.myName,
        agents: mesh.activeAgents.map(a => ({
          id: a.id,
          name: a.name,
          type: a.type,
          trustScore: a.trustScore.toFixed(3),
          lastSeen: new Date(a.lastSeen).toISOString(),
        })),
      });
    }
    
    case "swarm_send": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { to, type, payload, priority } = SendMessageSchema.parse(args);
      const messageId = await mesh.send(to, type, payload, priority);
      return JSON.stringify({ success: true, messageId });
    }
    
    case "swarm_propose": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { topic, options, deadlineMs } = ProposeConsensusSchema.parse(args);
      const proposalId = await mesh.proposeConsensus(topic, options, deadlineMs);
      return JSON.stringify({ success: true, proposalId, topic, options });
    }
    
    case "swarm_vote": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { proposalId, choice } = VoteSchema.parse(args);
      await mesh.vote(proposalId, choice);
      return JSON.stringify({ success: true, voted: choice });
    }
    
    case "swarm_create_task": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { title, description, assignees } = CreateTaskSchema.parse(args);
      const taskId = await mesh.createTask(title, description, assignees);
      return JSON.stringify({ success: true, taskId, title });
    }
    
    case "swarm_update_task": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { taskId, ...updates } = UpdateTaskSchema.parse(args);
      await mesh.updateTask(taskId, updates);
      return JSON.stringify({ success: true, taskId });
    }
    
    case "swarm_my_tasks": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      return JSON.stringify({
        tasks: mesh.myTasks.map(t => ({
          id: t.id,
          title: t.title,
          status: t.status,
          priority: t.priority,
        })),
      });
    }
    
    case "swarm_set_memory": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { key, value } = SetMemorySchema.parse(args);
      await mesh.setMemory(key, value);
      return JSON.stringify({ success: true, key });
    }
    
    case "swarm_get_memory": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { key } = GetMemorySchema.parse(args);
      const value = mesh.getMemory(key);
      return JSON.stringify({ key, value });
    }
    
    case "swarm_share_code": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { filename, content, description } = ShareCodeSchema.parse(args);
      const artifactId = await mesh.shareCode(filename, content, description);
      return JSON.stringify({ success: true, artifactId, filename });
    }
    
    case "swarm_request_review": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const { artifactId, reviewers } = RequestReviewSchema.parse(args);
      await mesh.requestReview(artifactId, reviewers);
      return JSON.stringify({ success: true, artifactId, reviewers });
    }
    
    case "swarm_find_nearest": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const count = args?.count || 5;
      const nearest = mesh.findNearestAgents(count);
      return JSON.stringify({
        nearest: nearest.map(a => ({
          id: a.id,
          name: a.name,
          type: a.type,
        })),
      });
    }
    
    case "swarm_trust_scores": {
      mesh = getAgentMesh();
      if (!mesh) return JSON.stringify({ success: false, error: "Not connected to mesh" });
      const scores = mesh.getTrustScores();
      return JSON.stringify({
        trustScores: Object.fromEntries(
          [...scores.entries()].map(([k, v]) => [k, v.toFixed(4)])
        ),
      });
    }
    
    default:
      return JSON.stringify({ error: `Unknown swarm tool: ${name}` });
  }
}
