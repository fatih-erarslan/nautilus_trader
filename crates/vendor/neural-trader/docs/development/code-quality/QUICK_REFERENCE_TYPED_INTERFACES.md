# Quick Reference: TypeScript Typed Interfaces

## New Interfaces Cheat Sheet

### RiskMetrics
```typescript
interface RiskMetrics {
  volatility: number        // Price volatility
  correlation: number       // Asset correlation
  var95: number            // Value at Risk (95%)
  cvar95: number           // Conditional VaR (95%)
  sharpeRatio: number      // Risk-adjusted return
  maxDrawdown: number      // Maximum drawdown
  beta: number             // Market beta
  concentrationRisk: number // Concentration risk
}
```

### StakeSizing
```typescript
interface StakeSizing {
  conservative: number     // Conservative stake
  moderate: number         // Moderate stake
  aggressive: number       // Aggressive stake
  kellyOptimal: number     // Kelly criterion optimal
  recommendation: string   // Recommended approach
}
```

### AgentParameters
```typescript
interface AgentParameters {
  maxPositionSize?: number  // Max position size
  stopLoss?: number        // Stop loss threshold
  takeProfit?: number      // Take profit target
  timeframe?: string       // Trading timeframe
  indicators?: string[]    // Technical indicators
  riskPerTrade?: number    // Risk per trade
  [key: string]: any       // Custom parameters
}
```

### AuditDetails
```typescript
interface AuditDetails {
  previousValue?: string          // Previous value
  newValue?: string              // New value
  reason?: string                // Change reason
  metadata?: Record<string, any> // Additional metadata
}
```

### MemberStatisticsUpdate
```typescript
interface MemberStatisticsUpdate {
  betsProposed?: number        // Bets proposed
  betsWon?: number            // Bets won
  betsLost?: number           // Bets lost
  totalProfit?: string        // Total profit
  totalStaked?: string        // Total staked
  roi?: number                // ROI percentage
  winRate?: number            // Win rate
  accuracy?: number           // Accuracy score
  profitContribution?: string // Profit contribution
  votesCast?: number          // Votes cast
  strategyContributions?: number // Strategy contributions
}
```

---

## Common Patterns

### Pattern 1: Type-Safe Allocation
```typescript
import { AllocationResult, RiskMetrics, StakeSizing } from '@neural-trader/backend';

const result: AllocationResult = {
  amount: "1000",
  percentageOfBankroll: 5,
  reasoning: "Good risk/reward",
  riskMetrics: {  // ← Typed object
    volatility: 0.25,
    correlation: 0.7,
    var95: 0.05,
    cvar95: 0.08,
    sharpeRatio: 1.5,
    maxDrawdown: 0.15,
    beta: 1.2,
    concentrationRisk: 0.3
  },
  approvalRequired: false,
  warnings: [],
  recommendedStakeSizing: {  // ← Typed object
    conservative: 100,
    moderate: 150,
    aggressive: 250,
    kellyOptimal: 180,
    recommendation: "moderate"
  }
};
```

### Pattern 2: Swarm Initialization
```typescript
import { SwarmConfig, SwarmTopology, DistributionStrategy } from '@neural-trader/backend';

const config: SwarmConfig = {
  topology: SwarmTopology.Mesh,
  maxAgents: 5,
  distributionStrategy: DistributionStrategy.Adaptive,
  enableGpu: true,
  autoScaling: true
};

const swarm = await initE2bSwarm("mesh", config);
```

### Pattern 3: Agent Deployment
```typescript
import { AgentParameters } from '@neural-trader/backend';

const params: AgentParameters = {
  maxPositionSize: 10000,
  stopLoss: 0.02,
  indicators: ["SMA", "RSI"],
  customParam: "value"  // ← Index signature allows this
};

await deployTradingAgent("sandbox-1", "momentum", ["AAPL"], params);
```

### Pattern 4: Audit Logging
```typescript
import { AuditDetails } from '@neural-trader/backend';

const details: AuditDetails = {
  previousValue: "pending",
  newValue: "executed",
  reason: "Market conditions favorable",
  metadata: { price: 150.25 }
};

logAuditEvent("info", "trading", "execute", "success",
              "user-1", "john", "192.168.1.1", "trade-1", details);
```

### Pattern 5: Type Guards (Backward Compat)
```typescript
function processAllocation(result: AllocationResult) {
  // Handle both typed object and JSON string
  const riskMetrics: RiskMetrics =
    typeof result.riskMetrics === 'string'
      ? JSON.parse(result.riskMetrics)  // ← Parse string
      : result.riskMetrics;              // ← Use object

  console.log(riskMetrics.volatility);  // ← Type-safe access
}
```

---

## Quick Lookup Table

| Use Case | Old Way | New Way |
|----------|---------|---------|
| **Risk Metrics** | `riskMetrics: string` | `riskMetrics: RiskMetrics \| string` |
| **Stake Sizing** | `stakeSizing: string` | `stakeSizing: StakeSizing \| string` |
| **Agent Config** | `params?: string` | `params?: AgentParameters \| string` |
| **Swarm Init** | `config: string` | `config: SwarmConfig \| string` |
| **Audit Details** | `details?: string` | `details?: AuditDetails \| string` |
| **Member Stats** | `statistics: string` | `statistics: MemberStatisticsUpdate \| string` |

---

## Function Signature Quick Reference

```typescript
// Swarm initialization
initE2bSwarm(topology: string, config: SwarmConfig | string): Promise<SwarmInit>

// Agent deployment
deployTradingAgent(
  sandboxId: string,
  agentType: string,
  symbols: string[],
  params?: AgentParameters | string
): Promise<AgentDeployment>

// Audit logging
logAuditEvent(
  level: string,
  category: string,
  action: string,
  outcome: string,
  userId?: string,
  username?: string,
  ipAddress?: string,
  resource?: string,
  details?: AuditDetails | string
): string

// Member statistics
memberManager.updateMemberStatistics(
  memberId: string,
  statistics: MemberStatisticsUpdate | string
): void
```

---

## Enum Quick Reference

### SwarmTopology
```typescript
enum SwarmTopology {
  Mesh = 0,         // Fully connected
  Hierarchical = 1, // Tree structure
  Ring = 2,         // Circular
  Star = 3          // Centralized hub
}
```

### DistributionStrategy
```typescript
enum DistributionStrategy {
  RoundRobin = 0,   // Round-robin
  LeastLoaded = 1,  // Least loaded
  Specialized = 2,  // Specialized
  Consensus = 3,    // Consensus-based
  Adaptive = 4      // Adaptive
}
```

### AllocationStrategy
```typescript
enum AllocationStrategy {
  KellyCriterion = 0,      // Kelly Criterion
  FixedPercentage = 1,     // Fixed percentage
  DynamicConfidence = 2,   // Dynamic confidence
  RiskParity = 3,          // Risk parity
  Martingale = 4,          // Martingale
  AntiMartingale = 5       // Anti-martingale
}
```

---

## Common Gotchas

### ✅ DO: Use type guards for mixed input
```typescript
const metrics: RiskMetrics =
  typeof result.riskMetrics === 'string'
    ? JSON.parse(result.riskMetrics)
    : result.riskMetrics;
```

### ❌ DON'T: Assume type without checking
```typescript
// BAD: Might fail if riskMetrics is a string
console.log(result.riskMetrics.volatility);

// GOOD: Type guard first
if (typeof result.riskMetrics === 'object') {
  console.log(result.riskMetrics.volatility);
}
```

### ✅ DO: Use index signatures for extensibility
```typescript
const params: AgentParameters = {
  stopLoss: 0.02,
  customParam: "works!"  // ← Allowed via [key: string]: any
};
```

### ✅ DO: Leverage partial updates
```typescript
// Only update specific fields
const update: MemberStatisticsUpdate = {
  betsWon: 7,  // ← Only this field
  roi: 15.5    // ← And this one
  // Other fields remain unchanged
};
```

---

## IDE Setup Tips

### VS Code
1. Install TypeScript extension (usually built-in)
2. Enable strict type checking in `tsconfig.json`:
   ```json
   {
     "compilerOptions": {
       "strict": true
     }
   }
   ```

### IntelliJ/WebStorm
- Type hints work automatically
- Cmd+Click (Mac) or Ctrl+Click (Windows) to jump to definitions

### Benefits You'll See
- ✅ Autocomplete for all fields
- ✅ Inline documentation on hover
- ✅ Compile-time error detection
- ✅ Refactoring support

---

**Quick Tips:**
1. **Always** use typed objects for new code
2. **Gradually** migrate legacy code during updates
3. **Use** type guards when handling mixed input
4. **Leverage** IDE autocomplete - it knows all the fields!

**More Info:** See `/workspaces/neural-trader/docs/TYPESCRIPT_INTERFACE_IMPROVEMENTS.md`
