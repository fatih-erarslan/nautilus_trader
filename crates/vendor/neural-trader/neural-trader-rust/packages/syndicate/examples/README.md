# Syndicate CLI Examples

This directory contains example files and scripts for the Syndicate CLI.

## Files

### opportunity.json
Example betting/investment opportunity with all required fields:
- `name`: Opportunity description
- `totalAmount`: Total amount to allocate
- `probability`: Win probability (0-1)
- `odds`: Betting odds
- `riskLevel`: Risk classification (low/medium/high)

### rules.json
Example syndicate rules configuration:
- Allocation limits
- Withdrawal policies
- Voting rules
- Profit sharing terms
- Risk limits

### demo.sh
Interactive demo script that:
1. Creates a syndicate
2. Adds members
3. Allocates funds
4. Distributes profits
5. Creates governance votes
6. Shows statistics

## Running the Demo

```bash
cd examples
./demo.sh
```

## Quick Examples

### Create Syndicate with Rules
```bash
node ../bin/syndicate.js create my-fund --bankroll 500000 --rules rules.json
```

### Add Members
```bash
node ../bin/syndicate.js member add "John Doe" john@example.com trader --capital 50000
```

### Allocate with Kelly Criterion
```bash
node ../bin/syndicate.js allocate opportunity.json --strategy kelly
```

### Distribute Profits (Hybrid Model)
```bash
node ../bin/syndicate.js distribute 10000 --model hybrid
```

### Create Vote
```bash
node ../bin/syndicate.js vote create "Increase risk allocation?" --options "Yes,No"
```

## Custom Opportunity Files

Create your own opportunity files with this structure:

```json
{
  "name": "Your Opportunity Name",
  "totalAmount": 10000,
  "probability": 0.6,
  "odds": 2.0,
  "riskLevel": "medium",
  "description": "Optional description",
  "expectedReturn": 0.2
}
```

## Custom Rules Files

Customize syndicate rules:

```json
{
  "maxAllocationPercent": 25,
  "minCapitalRequirement": 10000,
  "withdrawalRules": {
    "minWithdrawalAmount": 1000,
    "maxWithdrawalPercent": 50
  },
  "votingRules": {
    "quorumPercent": 60,
    "majorityPercent": 51
  },
  "profitSharing": {
    "performanceFeePercent": 10,
    "managementFeePercent": 2
  },
  "riskLimits": {
    "maxSingleBetPercent": 10,
    "maxDailyLossPercent": 15
  }
}
```

## JSON Output

All commands support JSON output with the `--json` flag:

```bash
node ../bin/syndicate.js member list --json > members.json
node ../bin/syndicate.js stats --syndicate my-fund --json > stats.json
```

## Integration with Neural Trader MCP

Use with the neural-trader MCP server for AI-powered recommendations:

```bash
# Get allocation recommendations from neural trader
# Then allocate using the CLI
node ../bin/syndicate.js allocate neural-recommendation.json --strategy dynamic
```
