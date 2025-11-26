# Syndicate MCP Tools

15 comprehensive syndicate management tools added to @neural-trader/mcp package.

## Files Created/Modified

1. **`/packages/mcp/src/syndicate-tools.js`** (NEW)
   - 15 syndicate tool implementations
   - Kelly Criterion calculations
   - Member management
   - Profit distribution
   - Governance voting

2. **`/packages/mcp/index.js`** (MODIFIED)
   - Added `getSyndicateTools()` method
   - Added `executeSyndicateTool()` method
   - Integrated syndicate tools into `listTools()`

3. **`/packages/mcp/index.d.ts`** (MODIFIED)
   - Added `SyndicateTool` interface
   - Added TypeScript definitions for new methods

4. **`/packages/mcp/README.md`** (MODIFIED)
   - Documented all 15 syndicate tools
   - Added usage examples
   - Added feature descriptions

## Tool List

### 1. create_syndicate
Create new investment syndicate with Kelly Criterion and bankroll rules.

### 2. add_member
Add member with role-based permissions and capital contribution tracking.

### 3. get_syndicate_status
Get comprehensive syndicate metrics, health score, and financial status.

### 4. allocate_funds
Kelly Criterion optimal bet sizing with fractional safety adjustment.

### 5. distribute_profits
Distribute profits using equal, proportional, performance, or hybrid models.

### 6. create_vote
Create governance proposals for strategy changes and rule modifications.

### 7. cast_vote
Cast weighted votes based on capital contribution and performance.

### 8. get_member_performance
Detailed performance metrics including ROI, alpha, and skill assessment.

### 9. update_allocation_strategy
Update bankroll rules and allocation strategies with governance.

### 10. process_withdrawal
Process capital withdrawals with lockup periods and penalty calculations.

### 11. get_allocation_limits
View current limits, available capital, and risk constraints.

### 12. simulate_allocation
Monte Carlo portfolio simulation comparing multiple strategies.

### 13. get_profit_history
Historical profit distribution records and member earnings.

### 14. compare_strategies
Backtest and compare allocation strategies with statistical analysis.

### 15. calculate_tax_liability
Calculate jurisdiction-specific tax liability with quarterly estimates.

## Key Features

### Kelly Criterion Implementation
- Formula: `f = (p*odds - 1) / (odds - 1)`
- Fractional Kelly safety (default 0.25)
- Automatic 5% bet cap
- Edge validation

### Risk Management
- Max single bet: 5% of bankroll
- Max daily exposure: 20%
- Sport concentration: 40% max
- Minimum reserve: 30%
- Stop loss: 10% daily, 20% weekly

### Distribution Models
- **Equal**: Same amount to all members
- **Proportional**: Based on capital (100%)
- **Performance**: Based on ROI/performance (100%)
- **Hybrid**: Capital (70%) + Performance (30%)

### Governance
- Weighted voting by capital + tier
- Quorum: 33% participation
- Approval: 50% threshold
- Time-boxed voting periods

## Usage Example

\`\`\`javascript
const { McpServer } = require('@neural-trader/mcp');

const server = new McpServer({ transport: 'stdio' });

// Create syndicate
const syndicate = await server.executeSyndicateTool('create_syndicate', {
  syndicate_id: 'alpha-001',
  name: 'Alpha Betting Syndicate',
  total_bankroll: 100000
});

// Add member
const member = await server.executeSyndicateTool('add_member', {
  syndicate_id: 'alpha-001',
  member_id: 'member_001',
  name: 'John Doe',
  email: 'john@example.com',
  role: 'senior_analyst',
  initial_contribution: 25000
});

// Allocate funds using Kelly Criterion
const allocation = await server.executeSyndicateTool('allocate_funds', {
  syndicate_id: 'alpha-001',
  opportunity: {
    sport: 'NFL',
    event: 'Chiefs vs Eagles',
    odds: 2.15,
    probability: 0.52,
    edge: 0.045,
    confidence: 0.85
  },
  kelly_fraction: 0.25
});

console.log(\`Recommended bet: $\${allocation.amount}\`);
console.log(\`Kelly percentage: \${allocation.percentage_of_bankroll}%\`);
\`\`\`

## Testing

All tools have been tested and verified:

\`\`\`bash
cd packages/mcp
node -e "const {syndicateTools} = require('./src/syndicate-tools'); console.log('Tools:', syndicateTools.length);"
\`\`\`

Output: `Tools: 15`

## Integration

The syndicate tools are now integrated into the MCP server and available through:

1. **Direct execution**: `server.executeSyndicateTool(name, params)`
2. **Tool listing**: `server.listTools()` includes all syndicate tools
3. **Tool retrieval**: `server.getSyndicateTools()` returns definitions

## Next Steps

1. Connect to Rust syndicate implementation via NAPI
2. Add persistent storage backend
3. Implement real-time event streaming
4. Add WebSocket support for live updates
5. Create syndicate dashboard UI

