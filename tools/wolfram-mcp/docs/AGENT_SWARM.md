# Agent Swarm - Multi-Agent Communication

Enable **Cascade-to-Cascade** and **Windsurf-to-Windsurf** communication for collaborative multi-agent development.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent Mesh Network                            │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Cascade #1  │◄──►│  Cascade #2  │◄──►│  Windsurf    │          │
│  │  (Agent A)   │    │  (Agent B)   │    │  (Agent C)   │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                   │
│         └───────────────────┼───────────────────┘                   │
│                             │                                       │
│                    ┌────────▼────────┐                              │
│                    │  Shared State   │                              │
│                    │  /tmp/hyperphys │                              │
│                    │  -ics-mesh/     │                              │
│                    └─────────────────┘                              │
│                                                                      │
│  HyperPhysics-Inspired Algorithms:                                  │
│  • Hyperbolic distance for agent affinity                           │
│  • pBit consensus for distributed voting                            │
│  • STDP for temporal message relevance                              │
│  • GNN for trust propagation                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Join the Mesh (Required First)

```
swarm_join(name: "Cascade-Main", type: "cascade")
```

### 2. List Active Agents

```
swarm_list_agents()
```

### 3. Send Messages

```
// To a specific agent
swarm_send(to: "abc123", type: "query", payload: { question: "How do we implement X?" })

// Broadcast to all
swarm_send(to: "broadcast", type: "alert", payload: { message: "Build complete!" })
```

### 4. Propose Consensus

```
swarm_propose(
  topic: "Should we refactor the database layer?",
  options: ["yes", "no", "defer"],
  deadlineMs: 120000
)
```

### 5. Vote

```
swarm_vote(proposalId: "abc123", choice: "yes")
```

### 6. Create Shared Tasks

```
swarm_create_task(
  title: "Implement authentication",
  description: "Add OAuth2 support",
  assignees: ["agent1", "agent2"]
)
```

### 7. Share Memory

```
swarm_set_memory(key: "api_spec", value: { version: "2.0", endpoints: [...] })
swarm_get_memory(key: "api_spec")
```

### 8. Share Code

```
swarm_share_code(
  filename: "auth.rs",
  content: "pub fn authenticate(...) {...}",
  description: "OAuth2 implementation"
)
```

## Tool Reference

### Mesh Management
| Tool | Description |
|------|-------------|
| `swarm_join` | Join the mesh network |
| `swarm_leave` | Leave gracefully |
| `swarm_list_agents` | List active agents |

### Communication
| Tool | Description |
|------|-------------|
| `swarm_send` | Send message to agent or broadcast |
| `swarm_find_nearest` | Find agents by hyperbolic distance |
| `swarm_trust_scores` | Get GNN-computed trust scores |

### Collaboration
| Tool | Description |
|------|-------------|
| `swarm_propose` | Start a consensus vote |
| `swarm_vote` | Cast your vote |
| `swarm_create_task` | Create shared task |
| `swarm_update_task` | Update task status |
| `swarm_my_tasks` | List my assigned tasks |

### Shared State
| Tool | Description |
|------|-------------|
| `swarm_set_memory` | Store in shared memory |
| `swarm_get_memory` | Retrieve from shared memory |
| `swarm_share_code` | Share code artifact |
| `swarm_request_review` | Request code review |

## Message Types

| Type | Purpose |
|------|---------|
| `task` | Task assignment |
| `result` | Task completion result |
| `query` | Ask a question |
| `response` | Answer to query |
| `consensus` | Consensus proposal |
| `vote` | Consensus vote |
| `sync` | State synchronization |
| `alert` | Priority notification |
| `memory` | Shared memory update |
| `code` | Code sharing |
| `review` | Review request |
| `approve` | Code approval |

## HyperPhysics Integration

### Hyperbolic Distance (Agent Affinity)

Agents are positioned in the Poincaré disk. Closer agents have higher communication affinity:

```typescript
distance = 2 * arctanh(|z1 - z2| / √((1-|z1|²)(1-|z2|²) + |z1-z2|²))
```

Use `swarm_find_nearest()` to find agents with highest affinity.

### pBit Consensus

Votes are weighted using Boltzmann distribution:

```typescript
P(option) = exp(-E(option) / T) / Σ exp(-E(i) / T)
```

This allows probabilistic consensus that handles uncertainty.

### STDP Message Relevance

Messages are scored by temporal proximity to query:

```typescript
relevance = exp(-Δt / τ)  // for recent messages
```

Recent messages are more relevant than older ones.

### GNN Trust Propagation

Trust flows through agent interaction graph using PageRank-style algorithm:

```typescript
trust(i) = (1-d)/N + d * Σ trust(j) / outDegree(j)
```

Use `swarm_trust_scores()` to see computed trust levels.

## Example Workflows

### Multi-Agent Code Review

```
// Agent A shares code
swarm_share_code(filename: "feature.rs", content: "...", description: "New feature")
// Returns: { artifactId: "xyz789" }

// Agent A requests review
swarm_request_review(artifactId: "xyz789", reviewers: ["agent_b", "agent_c"])

// Agent B receives review request and approves
// (after reviewing)
swarm_send(to: "agent_a", type: "approve", payload: { artifactId: "xyz789", approved: true })
```

### Collaborative Decision Making

```
// Agent A proposes
swarm_propose(topic: "Use GraphQL or REST?", options: ["graphql", "rest", "both"])

// Agent B votes
swarm_vote(proposalId: "...", choice: "graphql")

// Agent C votes  
swarm_vote(proposalId: "...", choice: "graphql")

// Consensus reached: "graphql" wins
```

### Distributed Task Assignment

```
// Project lead creates task
swarm_create_task(
  title: "Database migration",
  description: "Migrate from SQLite to PostgreSQL",
  assignees: ["agent_b", "agent_c"]
)

// Agent B starts work
swarm_update_task(taskId: "...", status: "in_progress")

// Agent B completes
swarm_update_task(taskId: "...", status: "completed")
swarm_send(to: "broadcast", type: "result", payload: { task: "Database migration", success: true })
```

## File Locations

```
/tmp/hyperphysics-mesh/
├── agents.json        # Active agents registry
├── tasks.json         # Shared tasks
├── consensus.json     # Pending votes
├── shared_memory.json # Key-value store
└── inboxes/
    ├── {agent_id_1}/  # Agent 1's inbox
    ├── {agent_id_2}/  # Agent 2's inbox
    └── ...
```

## Configuration

Set `AGENT_MESH_DIR` to customize the mesh directory:

```bash
export AGENT_MESH_DIR=/custom/path/mesh
```

## Security Considerations

1. **Local only**: Communication is file-based, limited to same machine
2. **No encryption**: Messages are stored in plaintext JSON
3. **Trust initialization**: All agents start with 0.5 trust score

For production use, consider:
- Adding Dilithium signatures (see `HYPERPHYSICS_FROM_DILITHIUM_RECIPE.md`)
- Encrypting shared memory
- Rate limiting message sends
