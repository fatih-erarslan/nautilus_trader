# Syndicate CLI - Quick Start Guide

## Installation

```bash
cd /workspaces/neural-trader/neural-trader-rust/packages/syndicate
npm install
npm link  # Optional: for global 'syndicate' command
```

## 5-Minute Quick Start

### 1. Create Your First Syndicate (30 seconds)

```bash
node bin/syndicate.js create my-first-fund --bankroll 100000
```

Expected output:
```
✓ Syndicate 'my-first-fund' created successfully
ℹ Bankroll: $100,000.00
ℹ Created: [timestamp]
```

### 2. Add Team Members (1 minute)

```bash
node bin/syndicate.js member add "Alice" alice@example.com trader --capital 30000
node bin/syndicate.js member add "Bob" bob@example.com analyst --capital 25000
node bin/syndicate.js member add "Carol" carol@example.com manager --capital 20000
```

### 3. View Your Team (10 seconds)

```bash
node bin/syndicate.js member list
```

### 4. Allocate Funds (1 minute)

```bash
node bin/syndicate.js allocate examples/opportunity.json --strategy kelly
```

### 5. Distribute Profits (30 seconds)

Preview first:
```bash
node bin/syndicate.js distribute preview 5000 --model hybrid
```

Then distribute:
```bash
node bin/syndicate.js distribute 5000 --model hybrid
```

### 6. Check Statistics (10 seconds)

```bash
node bin/syndicate.js stats --syndicate my-first-fund
```

## Common Commands Cheat Sheet

| Task | Command |
|------|---------|
| Create syndicate | `syndicate create <id> --bankroll <amount>` |
| Add member | `syndicate member add <name> <email> <role> --capital <amount>` |
| List members | `syndicate member list` |
| Member stats | `syndicate member stats <member-id>` |
| Allocate funds | `syndicate allocate <file.json> --strategy <kelly\|fixed\|dynamic\|risk-parity>` |
| Distribute profit | `syndicate distribute <amount> --model <proportional\|performance\|tiered\|hybrid>` |
| Preview distribution | `syndicate distribute preview <amount> --model <model>` |
| Request withdrawal | `syndicate withdraw request <member-id> <amount>` |
| Create vote | `syndicate vote create "<proposal>" --options "<csv>"` |
| Cast vote | `syndicate vote cast <vote-id> <option> --member <member-id>` |
| View stats | `syndicate stats --syndicate <id>` |

## Allocation Strategies Explained

### Kelly Criterion (Recommended)
Mathematically optimal bet sizing based on probability and odds.
```bash
syndicate allocate opportunity.json --strategy kelly
```

### Fixed
Proportional to capital contribution.
```bash
syndicate allocate opportunity.json --strategy fixed
```

### Dynamic
Performance-based allocation (rewards better performers).
```bash
syndicate allocate opportunity.json --strategy dynamic
```

### Risk Parity
Risk-adjusted allocation based on opportunity risk level.
```bash
syndicate allocate opportunity.json --strategy risk-parity
```

## Distribution Models Explained

### Proportional
Based on capital contribution (fair for equal effort).
```bash
syndicate distribute 10000 --model proportional
```

### Performance
Based on historical performance (rewards success).
```bash
syndicate distribute 10000 --model performance
```

### Tiered
Based on capital tiers (incentivizes larger investments).
```bash
syndicate distribute 10000 --model tiered
```

### Hybrid
60% proportional + 40% performance (balanced approach).
```bash
syndicate distribute 10000 --model hybrid
```

## JSON Output for Automation

Add `--json` to any command:

```bash
syndicate member list --json > members.json
syndicate stats --syndicate my-fund --json > stats.json
```

## Real-World Workflow Example

```bash
# Morning: Create syndicate and add members
syndicate create sports-betting --bankroll 500000
syndicate member add "Lead Trader" lead@example.com trader --capital 150000
syndicate member add "Risk Manager" risk@example.com manager --capital 100000

# Midday: Review opportunity and allocate
cat opportunities/nba-game-123.json
syndicate allocate opportunities/nba-game-123.json --strategy kelly

# Evening: Distribute winnings
syndicate distribute 25000 --model hybrid
syndicate stats --syndicate sports-betting

# Weekly: Process withdrawals
syndicate withdraw list --pending
syndicate withdraw approve with-123456
syndicate withdraw process with-123456
```

## Troubleshooting

### Command not found
```bash
# Use full path
node /workspaces/neural-trader/neural-trader-rust/packages/syndicate/bin/syndicate.js

# Or link globally
npm link
```

### Member ID not found
```bash
# List all members to get IDs
syndicate member list
```

### Invalid opportunity file
```bash
# Check example format
cat examples/opportunity.json
```

### View detailed errors
```bash
# Add --verbose flag
syndicate <command> --verbose
```

## Next Steps

1. **Customize Rules**: Edit `examples/rules.json` and create with `--rules` flag
2. **Automate**: Create bash scripts for daily operations
3. **Integrate**: Connect with neural-trader MCP for AI recommendations
4. **Govern**: Use voting for major decisions
5. **Track**: Regular `stats` checks for performance monitoring

## Support

- Full documentation: `packages/syndicate/README.md`
- Example files: `packages/syndicate/examples/`
- Run demo: `cd examples && ./demo.sh`

---

**Pro Tip**: Use `--json` output with `jq` for powerful filtering:

```bash
syndicate member list --json | jq '.[] | select(.capital > 50000)'
syndicate stats --syndicate my-fund --json | jq '.statistics.roi'
```
