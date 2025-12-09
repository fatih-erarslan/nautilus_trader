/**
 * Agent Swarm Module
 * 
 * Multi-agent communication for Cascade/Windsurf instances
 * using HyperPhysics-inspired algorithms.
 * 
 * Re-exports from original wolfram-mcp swarm implementation.
 */

export * from "./agent-mesh.js";
export * from "./swarm-tools.js";

// Re-export local types for backwards compatibility
import { Tool } from "@modelcontextprotocol/sdk/types.js";

// ============================================================================
// Agent Types
// ============================================================================

export interface SwarmAgent {
  id: string;
  publicKey: string;  // Dilithium public key
  capabilities: string[];
  status: "active" | "idle" | "busy" | "offline";
  lastSeen: string;
  metadata: Record<string, unknown>;
}

export interface SwarmMessage {
  id: string;
  from: string;
  to: string | "broadcast";
  type: "request" | "response" | "notify" | "consensus";
  payload: unknown;
  signature: string;
  timestamp: string;
}

export interface ConsensusProposal {
  id: string;
  proposer: string;
  topic: string;
  options: string[];
  votes: Map<string, number>;
  deadline: string;
  status: "open" | "closed" | "accepted" | "rejected";
}

export interface SharedMemory {
  key: string;
  value: unknown;
  version: number;
  lastUpdatedBy: string;
  timestamp: string;
}

// ============================================================================
// Swarm State
// ============================================================================

class SwarmCoordinator {
  private agents: Map<string, SwarmAgent> = new Map();
  private messages: SwarmMessage[] = [];
  private proposals: Map<string, ConsensusProposal> = new Map();
  private sharedMemory: Map<string, SharedMemory> = new Map();
  
  // === Agent Management ===
  
  registerAgent(agent: Omit<SwarmAgent, "status" | "lastSeen">): SwarmAgent {
    const fullAgent: SwarmAgent = {
      ...agent,
      status: "active",
      lastSeen: new Date().toISOString(),
    };
    this.agents.set(agent.id, fullAgent);
    return fullAgent;
  }
  
  updateAgentStatus(agentId: string, status: SwarmAgent["status"]): boolean {
    const agent = this.agents.get(agentId);
    if (agent) {
      agent.status = status;
      agent.lastSeen = new Date().toISOString();
      return true;
    }
    return false;
  }
  
  getAgent(agentId: string): SwarmAgent | undefined {
    return this.agents.get(agentId);
  }
  
  listAgents(filter?: { status?: SwarmAgent["status"]; capability?: string }): SwarmAgent[] {
    let result = Array.from(this.agents.values());
    
    if (filter?.status) {
      result = result.filter(a => a.status === filter.status);
    }
    if (filter?.capability) {
      result = result.filter(a => a.capabilities.includes(filter.capability!));
    }
    
    return result;
  }
  
  // === Messaging ===
  
  sendMessage(message: Omit<SwarmMessage, "id" | "timestamp">): SwarmMessage {
    const fullMessage: SwarmMessage = {
      ...message,
      id: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
    };
    this.messages.push(fullMessage);
    
    // Keep messages bounded
    if (this.messages.length > 10000) {
      this.messages = this.messages.slice(-5000);
    }
    
    return fullMessage;
  }
  
  getMessages(agentId: string, since?: string): SwarmMessage[] {
    let result = this.messages.filter(m => 
      m.to === agentId || m.to === "broadcast"
    );
    
    if (since) {
      result = result.filter(m => m.timestamp > since);
    }
    
    return result;
  }
  
  // === Consensus ===
  
  createProposal(
    proposer: string,
    topic: string,
    options: string[],
    durationMs: number = 60000
  ): ConsensusProposal {
    const proposal: ConsensusProposal = {
      id: crypto.randomUUID(),
      proposer,
      topic,
      options,
      votes: new Map(),
      deadline: new Date(Date.now() + durationMs).toISOString(),
      status: "open",
    };
    this.proposals.set(proposal.id, proposal);
    return proposal;
  }
  
  vote(proposalId: string, agentId: string, optionIndex: number): boolean {
    const proposal = this.proposals.get(proposalId);
    if (!proposal || proposal.status !== "open") {
      return false;
    }
    if (new Date() > new Date(proposal.deadline)) {
      this.closeProposal(proposalId);
      return false;
    }
    if (optionIndex < 0 || optionIndex >= proposal.options.length) {
      return false;
    }
    
    proposal.votes.set(agentId, optionIndex);
    return true;
  }
  
  closeProposal(proposalId: string): ConsensusProposal | undefined {
    const proposal = this.proposals.get(proposalId);
    if (!proposal) return undefined;
    
    // Count votes
    const voteCounts = new Map<number, number>();
    for (const [_, option] of proposal.votes) {
      voteCounts.set(option, (voteCounts.get(option) || 0) + 1);
    }
    
    // Find winner (simple majority)
    let maxVotes = 0;
    let winner = -1;
    for (const [option, count] of voteCounts) {
      if (count > maxVotes) {
        maxVotes = count;
        winner = option;
      }
    }
    
    proposal.status = winner >= 0 ? "accepted" : "rejected";
    return proposal;
  }
  
  // === Shared Memory ===
  
  setSharedMemory(key: string, value: unknown, updatedBy: string): SharedMemory {
    const existing = this.sharedMemory.get(key);
    const entry: SharedMemory = {
      key,
      value,
      version: (existing?.version || 0) + 1,
      lastUpdatedBy: updatedBy,
      timestamp: new Date().toISOString(),
    };
    this.sharedMemory.set(key, entry);
    return entry;
  }
  
  getSharedMemory(key: string): SharedMemory | undefined {
    return this.sharedMemory.get(key);
  }
  
  listSharedMemory(): SharedMemory[] {
    return Array.from(this.sharedMemory.values());
  }
}

// Global coordinator instance
const coordinator = new SwarmCoordinator();

// ============================================================================
// Tool Definitions
// ============================================================================

export const swarmTools: Tool[] = [
  {
    name: "swarm_register_agent",
    description: "Register an agent with the swarm coordinator",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Unique agent identifier" },
        public_key: { type: "string", description: "Dilithium public key" },
        capabilities: { 
          type: "array", 
          items: { type: "string" },
          description: "Agent capabilities" 
        },
        metadata: { type: "object", description: "Additional metadata" },
      },
      required: ["id", "public_key"],
    },
  },
  {
    name: "swarm_list_agents",
    description: "List registered agents with optional filters",
    inputSchema: {
      type: "object",
      properties: {
        status: { type: "string", enum: ["active", "idle", "busy", "offline"] },
        capability: { type: "string", description: "Filter by capability" },
      },
    },
  },
  {
    name: "swarm_send_message",
    description: "Send a message to another agent or broadcast",
    inputSchema: {
      type: "object",
      properties: {
        from: { type: "string", description: "Sender agent ID" },
        to: { type: "string", description: "Recipient ID or 'broadcast'" },
        type: { type: "string", enum: ["request", "response", "notify", "consensus"] },
        payload: { description: "Message payload" },
        signature: { type: "string", description: "Dilithium signature" },
      },
      required: ["from", "to", "type", "payload"],
    },
  },
  {
    name: "swarm_get_messages",
    description: "Get messages for an agent",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: { type: "string" },
        since: { type: "string", description: "ISO timestamp to filter from" },
      },
      required: ["agent_id"],
    },
  },
  {
    name: "swarm_create_proposal",
    description: "Create a consensus proposal for swarm voting",
    inputSchema: {
      type: "object",
      properties: {
        proposer: { type: "string", description: "Proposing agent ID" },
        topic: { type: "string", description: "Proposal topic" },
        options: { type: "array", items: { type: "string" }, description: "Voting options" },
        duration_ms: { type: "number", description: "Voting duration in ms" },
      },
      required: ["proposer", "topic", "options"],
    },
  },
  {
    name: "swarm_vote",
    description: "Vote on a consensus proposal",
    inputSchema: {
      type: "object",
      properties: {
        proposal_id: { type: "string" },
        agent_id: { type: "string" },
        option_index: { type: "number", description: "Index of chosen option" },
      },
      required: ["proposal_id", "agent_id", "option_index"],
    },
  },
  {
    name: "swarm_set_memory",
    description: "Set a value in shared swarm memory",
    inputSchema: {
      type: "object",
      properties: {
        key: { type: "string" },
        value: { description: "Value to store" },
        updated_by: { type: "string", description: "Agent ID making update" },
      },
      required: ["key", "value", "updated_by"],
    },
  },
  {
    name: "swarm_get_memory",
    description: "Get a value from shared swarm memory",
    inputSchema: {
      type: "object",
      properties: {
        key: { type: "string" },
      },
      required: ["key"],
    },
  },
];

// ============================================================================
// Tool Handler
// ============================================================================

export function handleSwarmTool(name: string, args: Record<string, unknown>): string {
  switch (name) {
    case "swarm_register_agent":
      const agent = coordinator.registerAgent({
        id: args.id as string,
        publicKey: args.public_key as string,
        capabilities: (args.capabilities as string[]) || [],
        metadata: (args.metadata as Record<string, unknown>) || {},
      });
      return JSON.stringify(agent);
      
    case "swarm_list_agents":
      const agents = coordinator.listAgents({
        status: args.status as SwarmAgent["status"],
        capability: args.capability as string,
      });
      return JSON.stringify(agents);
      
    case "swarm_send_message":
      const message = coordinator.sendMessage({
        from: args.from as string,
        to: args.to as string,
        type: args.type as SwarmMessage["type"],
        payload: args.payload,
        signature: (args.signature as string) || "",
      });
      return JSON.stringify(message);
      
    case "swarm_get_messages":
      const messages = coordinator.getMessages(
        args.agent_id as string,
        args.since as string
      );
      return JSON.stringify(messages);
      
    case "swarm_create_proposal":
      const proposal = coordinator.createProposal(
        args.proposer as string,
        args.topic as string,
        args.options as string[],
        (args.duration_ms as number) || 60000
      );
      return JSON.stringify(proposal);
      
    case "swarm_vote":
      const voteResult = coordinator.vote(
        args.proposal_id as string,
        args.agent_id as string,
        args.option_index as number
      );
      return JSON.stringify({ success: voteResult });
      
    case "swarm_set_memory":
      const mem = coordinator.setSharedMemory(
        args.key as string,
        args.value,
        args.updated_by as string
      );
      return JSON.stringify(mem);
      
    case "swarm_get_memory":
      const entry = coordinator.getSharedMemory(args.key as string);
      return JSON.stringify(entry || { error: "Key not found" });
      
    default:
      return JSON.stringify({ error: `Unknown swarm tool: ${name}` });
  }
}
