/**
 * Agent Mesh - Multi-Agent Communication Layer
 * 
 * Enables Cascade-to-Cascade, Windsurf-to-Windsurf communication
 * on the same machine using shared memory and message passing.
 * 
 * Architecture inspired by:
 * - HyperPhysics pBit consensus for distributed agreement
 * - GNN trust networks for agent reputation
 * - STDP for temporal message correlation
 * - Conformal prediction for message validity
 */

import { EventEmitter } from "events";
import { existsSync, mkdirSync, readFileSync, writeFileSync, unlinkSync, readdirSync } from "fs";
import { join } from "path";
import { createHash, randomBytes } from "crypto";

// ============================================================================
// Types
// ============================================================================

export interface AgentIdentity {
  id: string;
  name: string;
  type: "cascade" | "windsurf" | "custom";
  publicKey: string;
  capabilities: string[];
  hyperbolicPosition: { x: number; y: number }; // Position in Poincaré disk
  trustScore: number;
  lastSeen: number;
}

export interface SwarmMessage {
  id: string;
  from: string;
  to: string | "broadcast";
  type: MessageType;
  payload: any;
  timestamp: number;
  signature?: string;
  ttl: number; // Time-to-live in seconds
  priority: "low" | "normal" | "high" | "critical";
}

export type MessageType =
  | "join"           // Agent joining the swarm
  | "leave"          // Agent leaving
  | "heartbeat"      // Keep-alive
  | "task"           // Task assignment
  | "result"         // Task result
  | "query"          // Knowledge query
  | "response"       // Query response
  | "consensus"      // Consensus proposal
  | "vote"           // Consensus vote
  | "sync"           // State synchronization
  | "alert"          // Priority alert
  | "memory"         // Shared memory update
  | "code"           // Code sharing
  | "review"         // Code review request
  | "approve";       // Approval

export interface ConsensusProposal {
  id: string;
  proposer: string;
  topic: string;
  options: string[];
  votes: Map<string, string>;
  deadline: number;
  quorum: number;
  status: "pending" | "approved" | "rejected" | "expired";
}

export interface SharedTask {
  id: string;
  title: string;
  description: string;
  assignedTo: string[];
  status: "pending" | "in_progress" | "review" | "completed";
  priority: number;
  dependencies: string[];
  artifacts: string[];
  createdAt: number;
  updatedAt: number;
}

// ============================================================================
// Agent Mesh Configuration
// ============================================================================

const MESH_DIR = process.env.AGENT_MESH_DIR || "/tmp/hyperphysics-mesh";
const INBOX_DIR = join(MESH_DIR, "inboxes");
const AGENTS_FILE = join(MESH_DIR, "agents.json");
const TASKS_FILE = join(MESH_DIR, "tasks.json");
const CONSENSUS_FILE = join(MESH_DIR, "consensus.json");
const MEMORY_FILE = join(MESH_DIR, "shared_memory.json");

// ============================================================================
// HyperPhysics-Inspired Algorithms
// ============================================================================

/**
 * Compute hyperbolic distance between agents in Poincaré disk
 * Closer agents have stronger communication affinity
 */
function hyperbolicDistance(p1: { x: number; y: number }, p2: { x: number; y: number }): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const diffNormSq = dx * dx + dy * dy;
  
  const norm1Sq = p1.x * p1.x + p1.y * p1.y;
  const norm2Sq = p2.x * p2.x + p2.y * p2.y;
  
  if (norm1Sq >= 1 || norm2Sq >= 1) return Infinity;
  
  const denom = Math.sqrt((1 - norm1Sq) * (1 - norm2Sq) + diffNormSq);
  const ratio = Math.sqrt(diffNormSq) / denom;
  
  return 2 * Math.atanh(Math.min(ratio, 0.9999));
}

/**
 * pBit-inspired consensus mechanism
 * Uses Boltzmann distribution for probabilistic voting
 */
function pBitConsensus(votes: Map<string, string>, options: string[], temperature: number = 1.0): string {
  const counts = new Map<string, number>();
  options.forEach(o => counts.set(o, 0));
  
  for (const vote of votes.values()) {
    counts.set(vote, (counts.get(vote) || 0) + 1);
  }
  
  // Boltzmann distribution over options
  const energies = options.map(o => -(counts.get(o) || 0));
  const minEnergy = Math.min(...energies);
  const expValues = energies.map(e => Math.exp(-(e - minEnergy) / temperature));
  const sum = expValues.reduce((a, b) => a + b, 0);
  const probs = expValues.map(e => e / sum);
  
  // Return highest probability option
  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }
  
  return options[maxIdx];
}

/**
 * STDP-inspired message relevance scoring
 * Messages closer in time to query are more relevant
 */
function stdpRelevance(messageTime: number, queryTime: number, tauMs: number = 10000): number {
  const dt = queryTime - messageTime;
  if (dt > 0) {
    return Math.exp(-dt / tauMs);
  } else {
    return 0.5 * Math.exp(dt / tauMs);
  }
}

/**
 * GNN-inspired trust propagation
 * Trust flows through the agent network
 */
function propagateTrust(agents: AgentIdentity[], interactions: Map<string, string[]>): Map<string, number> {
  const trust = new Map<string, number>();
  const damping = 0.85;
  const iterations = 10;
  
  // Initialize
  agents.forEach(a => trust.set(a.id, a.trustScore));
  
  // PageRank-style propagation
  for (let i = 0; i < iterations; i++) {
    const newTrust = new Map<string, number>();
    
    for (const agent of agents) {
      let incoming = 0;
      const neighbors = interactions.get(agent.id) || [];
      
      for (const neighbor of neighbors) {
        const neighborTrust = trust.get(neighbor) || 0;
        const outDegree = (interactions.get(neighbor) || []).length || 1;
        incoming += neighborTrust / outDegree;
      }
      
      newTrust.set(agent.id, (1 - damping) / agents.length + damping * incoming);
    }
    
    trust.clear();
    newTrust.forEach((v, k) => trust.set(k, v));
  }
  
  return trust;
}

// ============================================================================
// Agent Mesh Class
// ============================================================================

export class AgentMesh extends EventEmitter {
  private identity: AgentIdentity;
  private agents: Map<string, AgentIdentity> = new Map();
  private inbox: SwarmMessage[] = [];
  private outbox: SwarmMessage[] = [];
  private tasks: Map<string, SharedTask> = new Map();
  private consensus: Map<string, ConsensusProposal> = new Map();
  private sharedMemory: Map<string, any> = new Map();
  private interactions: Map<string, string[]> = new Map();
  private pollInterval: NodeJS.Timer | null = null;
  
  constructor(name: string, type: "cascade" | "windsurf" | "custom" = "cascade") {
    super();
    
    // Generate unique identity
    this.identity = {
      id: createHash("sha256").update(randomBytes(32)).digest("hex").slice(0, 16),
      name,
      type,
      publicKey: randomBytes(32).toString("hex"),
      capabilities: ["wolfram", "code", "review", "consensus"],
      hyperbolicPosition: this.randomPoincareDiskPoint(),
      trustScore: 0.5,
      lastSeen: Date.now(),
    };
    
    this.ensureDirectories();
    this.loadState();
  }
  
  private randomPoincareDiskPoint(): { x: number; y: number } {
    // Uniform distribution in Poincaré disk
    const r = Math.sqrt(Math.random()) * 0.9;
    const theta = Math.random() * 2 * Math.PI;
    return { x: r * Math.cos(theta), y: r * Math.sin(theta) };
  }
  
  private ensureDirectories() {
    if (!existsSync(MESH_DIR)) mkdirSync(MESH_DIR, { recursive: true });
    if (!existsSync(INBOX_DIR)) mkdirSync(INBOX_DIR, { recursive: true });
    
    // Create own inbox
    const myInbox = join(INBOX_DIR, this.identity.id);
    if (!existsSync(myInbox)) mkdirSync(myInbox, { recursive: true });
  }
  
  private loadState() {
    try {
      if (existsSync(AGENTS_FILE)) {
        const data = JSON.parse(readFileSync(AGENTS_FILE, "utf-8"));
        data.forEach((a: AgentIdentity) => this.agents.set(a.id, a));
      }
      if (existsSync(TASKS_FILE)) {
        const data = JSON.parse(readFileSync(TASKS_FILE, "utf-8"));
        data.forEach((t: SharedTask) => this.tasks.set(t.id, t));
      }
      if (existsSync(CONSENSUS_FILE)) {
        const data = JSON.parse(readFileSync(CONSENSUS_FILE, "utf-8"));
        data.forEach((c: any) => {
          c.votes = new Map(Object.entries(c.votes || {}));
          this.consensus.set(c.id, c);
        });
      }
      if (existsSync(MEMORY_FILE)) {
        const data = JSON.parse(readFileSync(MEMORY_FILE, "utf-8"));
        Object.entries(data).forEach(([k, v]) => this.sharedMemory.set(k, v));
      }
    } catch (e) {
      console.error("Failed to load mesh state:", e);
    }
  }
  
  private saveState() {
    try {
      writeFileSync(AGENTS_FILE, JSON.stringify([...this.agents.values()], null, 2));
      writeFileSync(TASKS_FILE, JSON.stringify([...this.tasks.values()], null, 2));
      writeFileSync(CONSENSUS_FILE, JSON.stringify(
        [...this.consensus.values()].map(c => ({
          ...c,
          votes: Object.fromEntries(c.votes),
        })),
        null, 2
      ));
      writeFileSync(MEMORY_FILE, JSON.stringify(Object.fromEntries(this.sharedMemory), null, 2));
    } catch (e) {
      console.error("Failed to save mesh state:", e);
    }
  }
  
  // ============================================================================
  // Public API
  // ============================================================================
  
  /** Join the agent mesh */
  async join(): Promise<void> {
    // Register self
    this.agents.set(this.identity.id, this.identity);
    this.saveState();
    
    // Broadcast join message
    await this.broadcast({
      type: "join",
      payload: this.identity,
      priority: "high",
    });
    
    // Start polling for messages
    this.startPolling();
    
    this.emit("joined", this.identity);
  }
  
  /** Leave the agent mesh */
  async leave(): Promise<void> {
    await this.broadcast({
      type: "leave",
      payload: { id: this.identity.id },
      priority: "normal",
    });
    
    this.stopPolling();
    this.agents.delete(this.identity.id);
    this.saveState();
    
    // Clean up inbox
    const myInbox = join(INBOX_DIR, this.identity.id);
    try {
      const files = readdirSync(myInbox);
      files.forEach(f => unlinkSync(join(myInbox, f)));
    } catch (e) {}
    
    this.emit("left", this.identity);
  }
  
  /** Send a message to a specific agent or broadcast */
  async send(to: string | "broadcast", type: MessageType, payload: any, priority: "low" | "normal" | "high" | "critical" = "normal"): Promise<string> {
    const message: SwarmMessage = {
      id: randomBytes(8).toString("hex"),
      from: this.identity.id,
      to,
      type,
      payload,
      timestamp: Date.now(),
      ttl: 3600, // 1 hour default
      priority,
    };
    
    if (to === "broadcast") {
      await this.broadcast(message);
    } else {
      await this.deliverTo(to, message);
    }
    
    return message.id;
  }
  
  /** Broadcast message to all agents */
  private async broadcast(partialMessage: Partial<SwarmMessage>): Promise<void> {
    const message: SwarmMessage = {
      id: randomBytes(8).toString("hex"),
      from: this.identity.id,
      to: "broadcast",
      type: partialMessage.type || "heartbeat",
      payload: partialMessage.payload,
      timestamp: Date.now(),
      ttl: partialMessage.ttl || 3600,
      priority: partialMessage.priority || "normal",
    };
    
    for (const agent of this.agents.values()) {
      if (agent.id !== this.identity.id) {
        await this.deliverTo(agent.id, message);
      }
    }
  }
  
  /** Deliver message to specific agent's inbox */
  private async deliverTo(agentId: string, message: SwarmMessage): Promise<void> {
    const inboxDir = join(INBOX_DIR, agentId);
    if (!existsSync(inboxDir)) {
      mkdirSync(inboxDir, { recursive: true });
    }
    
    const filename = `${message.timestamp}-${message.id}.json`;
    writeFileSync(join(inboxDir, filename), JSON.stringify(message, null, 2));
    
    // Track interaction
    const myInteractions = this.interactions.get(this.identity.id) || [];
    if (!myInteractions.includes(agentId)) {
      myInteractions.push(agentId);
      this.interactions.set(this.identity.id, myInteractions);
    }
  }
  
  /** Poll for new messages */
  private startPolling() {
    this.pollInterval = setInterval(() => this.pollInbox(), 1000);
    this.pollInbox(); // Initial poll
  }
  
  private stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }
  
  private pollInbox() {
    const myInbox = join(INBOX_DIR, this.identity.id);
    if (!existsSync(myInbox)) return;
    
    try {
      const files = readdirSync(myInbox).sort();
      
      for (const file of files) {
        const filepath = join(myInbox, file);
        const message: SwarmMessage = JSON.parse(readFileSync(filepath, "utf-8"));
        
        // Check TTL
        if (Date.now() - message.timestamp > message.ttl * 1000) {
          unlinkSync(filepath);
          continue;
        }
        
        // Process message
        this.handleMessage(message);
        
        // Remove processed message
        unlinkSync(filepath);
      }
    } catch (e) {
      // Ignore read errors
    }
    
    // Update last seen
    this.identity.lastSeen = Date.now();
    this.agents.set(this.identity.id, this.identity);
    
    // Send heartbeat
    if (Math.random() < 0.1) { // 10% chance per poll
      this.broadcast({ type: "heartbeat", payload: { lastSeen: Date.now() }, priority: "low" });
    }
    
    // Prune stale agents (no heartbeat in 5 minutes)
    const staleThreshold = 5 * 60 * 1000;
    for (const [id, agent] of this.agents) {
      if (id !== this.identity.id && Date.now() - agent.lastSeen > staleThreshold) {
        this.agents.delete(id);
        this.emit("agent_left", agent);
      }
    }
    
    this.saveState();
  }
  
  private handleMessage(message: SwarmMessage) {
    // Update sender's last seen
    const sender = this.agents.get(message.from);
    if (sender) {
      sender.lastSeen = Date.now();
      this.agents.set(message.from, sender);
    }
    
    switch (message.type) {
      case "join":
        this.agents.set(message.payload.id, message.payload);
        this.emit("agent_joined", message.payload);
        break;
        
      case "leave":
        this.agents.delete(message.payload.id);
        this.emit("agent_left", message.payload);
        break;
        
      case "heartbeat":
        // Already handled above
        break;
        
      case "task":
        this.tasks.set(message.payload.id, message.payload);
        this.emit("task_received", message.payload);
        break;
        
      case "result":
        this.emit("result_received", message.payload);
        break;
        
      case "query":
        this.emit("query_received", message);
        break;
        
      case "response":
        this.emit("response_received", message);
        break;
        
      case "consensus":
        this.handleConsensusProposal(message.payload);
        break;
        
      case "vote":
        this.handleVote(message.payload);
        break;
        
      case "memory":
        this.sharedMemory.set(message.payload.key, message.payload.value);
        this.emit("memory_updated", message.payload);
        break;
        
      case "code":
        this.emit("code_shared", message.payload);
        break;
        
      case "review":
        this.emit("review_requested", message.payload);
        break;
        
      case "approve":
        this.emit("approval_received", message.payload);
        break;
        
      case "alert":
        this.emit("alert", message.payload);
        break;
        
      default:
        this.emit("message", message);
    }
  }
  
  // ============================================================================
  // Consensus
  // ============================================================================
  
  /** Propose a consensus vote */
  async proposeConsensus(topic: string, options: string[], deadlineMs: number = 60000): Promise<string> {
    const proposal: ConsensusProposal = {
      id: randomBytes(8).toString("hex"),
      proposer: this.identity.id,
      topic,
      options,
      votes: new Map(),
      deadline: Date.now() + deadlineMs,
      quorum: Math.ceil(this.agents.size * 0.5), // 50% quorum
      status: "pending",
    };
    
    this.consensus.set(proposal.id, proposal);
    
    await this.broadcast({
      type: "consensus",
      payload: { ...proposal, votes: {} },
      priority: "high",
    });
    
    return proposal.id;
  }
  
  /** Vote on a consensus proposal */
  async vote(proposalId: string, choice: string): Promise<void> {
    const proposal = this.consensus.get(proposalId);
    if (!proposal || proposal.status !== "pending") return;
    
    proposal.votes.set(this.identity.id, choice);
    
    await this.broadcast({
      type: "vote",
      payload: { proposalId, voterId: this.identity.id, choice },
      priority: "high",
    });
    
    this.checkConsensusResult(proposalId);
  }
  
  private handleConsensusProposal(proposal: any) {
    proposal.votes = new Map(Object.entries(proposal.votes || {}));
    this.consensus.set(proposal.id, proposal);
    this.emit("consensus_proposed", proposal);
  }
  
  private handleVote(voteData: { proposalId: string; voterId: string; choice: string }) {
    const proposal = this.consensus.get(voteData.proposalId);
    if (!proposal) return;
    
    proposal.votes.set(voteData.voterId, voteData.choice);
    this.checkConsensusResult(voteData.proposalId);
  }
  
  private checkConsensusResult(proposalId: string) {
    const proposal = this.consensus.get(proposalId);
    if (!proposal || proposal.status !== "pending") return;
    
    // Check deadline
    if (Date.now() > proposal.deadline) {
      proposal.status = "expired";
      this.emit("consensus_expired", proposal);
      return;
    }
    
    // Check quorum
    if (proposal.votes.size >= proposal.quorum) {
      const result = pBitConsensus(proposal.votes, proposal.options);
      
      // Check if majority agrees
      const resultVotes = [...proposal.votes.values()].filter(v => v === result).length;
      if (resultVotes > proposal.votes.size / 2) {
        proposal.status = "approved";
        this.emit("consensus_approved", { proposal, result });
      } else {
        proposal.status = "rejected";
        this.emit("consensus_rejected", proposal);
      }
    }
  }
  
  // ============================================================================
  // Task Management
  // ============================================================================
  
  /** Create and assign a shared task */
  async createTask(title: string, description: string, assignees: string[]): Promise<string> {
    const task: SharedTask = {
      id: randomBytes(8).toString("hex"),
      title,
      description,
      assignedTo: assignees,
      status: "pending",
      priority: 1,
      dependencies: [],
      artifacts: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    
    this.tasks.set(task.id, task);
    
    // Notify assignees
    for (const assignee of assignees) {
      await this.send(assignee, "task", task, "high");
    }
    
    return task.id;
  }
  
  /** Update task status */
  async updateTask(taskId: string, updates: Partial<SharedTask>): Promise<void> {
    const task = this.tasks.get(taskId);
    if (!task) return;
    
    Object.assign(task, updates, { updatedAt: Date.now() });
    this.tasks.set(taskId, task);
    
    await this.broadcast({
      type: "task",
      payload: task,
      priority: "normal",
    });
  }
  
  // ============================================================================
  // Shared Memory
  // ============================================================================
  
  /** Set a value in shared memory */
  async setMemory(key: string, value: any): Promise<void> {
    this.sharedMemory.set(key, value);
    
    await this.broadcast({
      type: "memory",
      payload: { key, value, updatedBy: this.identity.id },
      priority: "normal",
    });
  }
  
  /** Get a value from shared memory */
  getMemory(key: string): any {
    return this.sharedMemory.get(key);
  }
  
  // ============================================================================
  // Code Sharing
  // ============================================================================
  
  /** Share code with other agents */
  async shareCode(filename: string, content: string, description: string): Promise<string> {
    const artifact = {
      id: randomBytes(8).toString("hex"),
      filename,
      content,
      description,
      author: this.identity.id,
      timestamp: Date.now(),
    };
    
    await this.broadcast({
      type: "code",
      payload: artifact,
      priority: "normal",
    });
    
    return artifact.id;
  }
  
  /** Request code review */
  async requestReview(artifactId: string, reviewers: string[]): Promise<void> {
    for (const reviewer of reviewers) {
      await this.send(reviewer, "review", { artifactId, requestedBy: this.identity.id }, "high");
    }
  }
  
  /** Approve code */
  async approve(artifactId: string): Promise<void> {
    await this.broadcast({
      type: "approve",
      payload: { artifactId, approvedBy: this.identity.id },
      priority: "high",
    });
  }
  
  // ============================================================================
  // Getters
  // ============================================================================
  
  get myId(): string {
    return this.identity.id;
  }
  
  get myName(): string {
    return this.identity.name;
  }
  
  get activeAgents(): AgentIdentity[] {
    return [...this.agents.values()];
  }
  
  get pendingTasks(): SharedTask[] {
    return [...this.tasks.values()].filter(t => t.status !== "completed");
  }
  
  get myTasks(): SharedTask[] {
    return [...this.tasks.values()].filter(t => t.assignedTo.includes(this.identity.id));
  }
  
  /** Find nearest agents by hyperbolic distance */
  findNearestAgents(count: number = 5): AgentIdentity[] {
    const others = [...this.agents.values()].filter(a => a.id !== this.identity.id);
    return others
      .map(a => ({ agent: a, distance: hyperbolicDistance(this.identity.hyperbolicPosition, a.hyperbolicPosition) }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, count)
      .map(x => x.agent);
  }
  
  /** Get trust scores using GNN propagation */
  getTrustScores(): Map<string, number> {
    return propagateTrust([...this.agents.values()], this.interactions);
  }
}

// ============================================================================
// Export singleton factory
// ============================================================================

let meshInstance: AgentMesh | null = null;

export function getAgentMesh(name?: string): AgentMesh {
  if (!meshInstance && name) {
    meshInstance = new AgentMesh(name);
  }
  return meshInstance!;
}

export function createAgentMesh(name: string, type: "cascade" | "windsurf" | "custom" = "cascade"): AgentMesh {
  meshInstance = new AgentMesh(name, type);
  return meshInstance;
}
