# @neural-trader/syndicate

[![npm version](https://badge.fury.io/js/%40neural-trader%2Fsyndicate.svg)](https://www.npmjs.com/package/@neural-trader/syndicate)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/neural-trader/neural-trader/workflows/CI/badge.svg)](https://github.com/neural-trader/neural-trader/actions)
[![Node Version](https://img.shields.io/badge/node-%3E%3D14.0.0-brightgreen.svg)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-Ready-blue.svg)](https://www.typescriptlang.org/)

> **Sophisticated investment syndicate management for collaborative sports betting and trading**

Enterprise-grade CLI and programmatic API for managing investment syndicates with Kelly Criterion allocation, multi-tier governance, bankroll management, and performance analytics.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [CLI Commands](#-cli-commands)
- [Allocation Strategies](#-allocation-strategies)
- [Distribution Models](#-distribution-models)
- [Governance & Voting](#-governance--voting)
- [API Reference](#-api-reference)
- [MCP Integration](#-mcp-integration)
- [Architecture](#-architecture)
- [Examples](#-examples)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

---

## ✨ Features

### Core Capabilities

#### 🎯 **4-Tier Membership System**
- **Trader** - Execute trades and allocations
- **Analyst** - Research and recommendations
- **Manager** - Approve withdrawals and governance
- **Admin** - Full system control

#### 💰 **Advanced Allocation Strategies**
- **Kelly Criterion** - Mathematically optimal bet sizing
- **Fixed Allocation** - Capital-proportional distribution
- **Dynamic Allocation** - Performance-based sizing
- **Risk Parity** - Risk-adjusted allocation

#### 📊 **Profit Distribution Models**
- **Proportional** - Capital-based distribution (fair for equal effort)
- **Performance** - Historical success-based rewards
- **Tiered** - Capital tier-based incentives
- **Hybrid** - 60% proportional + 40% performance

#### 🏛️ **18-Permission Governance System**
Fine-grained permission control:
- Member management (add, remove, update)
- Fund operations (allocate, distribute, withdraw)
- Voting rights (create, cast, view)
- Configuration access (view, modify, rules)

#### 📏 **9 Bankroll Management Rules**
Comprehensive risk management:
- Kelly fraction limits (25-100%)
- Max allocation per bet (1-25%)
- Min/max bet amounts
- Risk levels and member minimums

### Technical Features

- ✅ **24 CLI Commands** - Complete command-line interface
- ✅ **15 MCP Tools** - Model Context Protocol integration
- ✅ **Beautiful Output** - Colored tables, spinners, and progress indicators
- ✅ **JSON API** - Programmatic access for automation
- ✅ **Persistent Storage** - File-based data management
- ✅ **Real-time Analytics** - Member, syndicate, and performance tracking
- ✅ **Withdrawal Workflows** - Request → Approve → Process pipeline
- ✅ **Voting & Governance** - Democratic decision-making system

---

## 🚀 Quick Start

### 5-Minute Setup

```bash
# 1. Install dependencies
npm install @neural-trader/syndicate

# 2. Create your first syndicate
npx syndicate create my-fund --bankroll 100000

# 3. Add team members
npx syndicate member add "Alice" alice@example.com trader --capital 30000
npx syndicate member add "Bob" bob@example.com analyst --capital 25000

# 4. View your team
npx syndicate member list

# 5. Allocate funds with Kelly Criterion
npx syndicate allocate opportunities/bet.json --strategy kelly

# 6. Distribute profits
npx syndicate distribute 5000 --model hybrid

# 7. Check statistics
npx syndicate stats
```

**👉 See [QUICK_START.md](./QUICK_START.md) for detailed step-by-step guide**

---

## 📦 Installation

### NPM Package

```bash
npm install @neural-trader/syndicate
```

### Global CLI

```bash
npm install -g @neural-trader/syndicate
syndicate --help
```

### From Source

```bash
git clone https://github.com/ruvnet/neural-trader.git
cd neural-trader/neural-trader-rust/packages/syndicate
npm install
npm link  # Optional: for global 'syndicate' command
```

### Requirements

- **Node.js**: >= 14.0.0
- **NPM**: >= 6.0.0
- **OS**: Linux, macOS, Windows

---

## 🎮 CLI Commands

### Command Overview (24 Total)

| Category | Commands | Description |
|----------|----------|-------------|
| **Creation** | `create` | Create new syndicate |
| **Members** | `add`, `list`, `stats`, `update`, `remove` | Member management |
| **Allocation** | `allocate`, `allocate list`, `allocate history` | Fund allocation |
| **Distribution** | `distribute`, `distribute history`, `distribute preview` | Profit distribution |
| **Withdrawals** | `request`, `approve`, `process`, `list` | Withdrawal workflow |
| **Governance** | `vote create`, `vote cast`, `vote results`, `vote list` | Voting system |
| **Analytics** | `stats` | Statistics and reporting |
| **Config** | `config set`, `config get`, `config rules` | Configuration |

### Essential Commands

#### Create Syndicate

```bash
syndicate create <id> [options]

Options:
  --bankroll <amount>      Initial bankroll (required)
  --rules <file.json>      Custom rules configuration

Example:
  syndicate create sports-betting --bankroll 500000 --rules rules.json
```

#### Member Management

```bash
# Add member
syndicate member add <name> <email> <role> --capital <amount>

# List members
syndicate member list [--format table|json]

# Member statistics
syndicate member stats <member-id>

# Update member
syndicate member update <member-id> [options]

# Remove member
syndicate member remove <member-id> [--reason <text>]

Examples:
  syndicate member add "John Doe" john@example.com trader --capital 50000
  syndicate member list --format json
  syndicate member stats mem-123456
```

#### Fund Allocation

```bash
syndicate allocate <opportunity.json> [options]

Options:
  --strategy <type>        kelly|fixed|dynamic|risk-parity (default: kelly)

Examples:
  syndicate allocate bet.json --strategy kelly
  syndicate allocate list
  syndicate allocate history
```

**Opportunity File Format:**

```json
{
  "description": "NBA: Lakers vs Celtics",
  "amount": 5000,
  "probability": 0.55,
  "odds": 2.2,
  "risk_level": "medium"
}
```

#### Profit Distribution

```bash
syndicate distribute <amount> [options]

Options:
  --model <type>           proportional|performance|tiered|hybrid (default: hybrid)

Examples:
  syndicate distribute 10000 --model hybrid
  syndicate distribute preview 10000 --model proportional
  syndicate distribute history
```

#### Withdrawal Management

```bash
# Request withdrawal
syndicate withdraw request <member-id> <amount> [--emergency]

# Approve withdrawal (requires manager/admin)
syndicate withdraw approve <withdrawal-id>

# Process withdrawal
syndicate withdraw process <withdrawal-id>

# List withdrawals
syndicate withdraw list [--pending]

Examples:
  syndicate withdraw request mem-123456 5000
  syndicate withdraw approve with-789012
  syndicate withdraw list --pending
```

#### Voting & Governance

```bash
# Create vote
syndicate vote create "<proposal>" --options "<option1,option2,option3>"

# Cast vote
syndicate vote cast <vote-id> <option> --member <member-id>

# View results
syndicate vote results <vote-id>

# List votes
syndicate vote list [--active]

Examples:
  syndicate vote create "Increase Kelly fraction to 35%" --options "approve,reject,abstain"
  syndicate vote cast vote-123 approve --member mem-456
  syndicate vote results vote-123
```

#### Statistics & Analytics

```bash
syndicate stats [options]

Options:
  --syndicate <id>         Specific syndicate
  --member <id>            Specific member
  --performance            Performance overview
  --json                   JSON output

Examples:
  syndicate stats --syndicate my-fund
  syndicate stats --member mem-123456
  syndicate stats --performance --json
```

### Global Options

Available for all commands:

```bash
--json                    Output in JSON format
--verbose                 Enable verbose logging
--config <file>           Custom config file
--no-color                Disable colored output
```

---

## 💡 Allocation Strategies

### Kelly Criterion (Recommended)

**Mathematically optimal bet sizing** based on probability and odds.

```bash
syndicate allocate opportunity.json --strategy kelly
```

**Formula**: `f* = (bp - q) / b`
- `f*` = fraction of bankroll to bet
- `b` = odds - 1
- `p` = probability of winning
- `q` = probability of losing (1 - p)

**When to use**:
- ✅ High-confidence opportunities with known probabilities
- ✅ Long-term wealth maximization
- ✅ Asymmetric risk/reward scenarios

**Example**: With 55% win probability and 2.2 odds:
- Kelly fraction: ~17.3% of bankroll
- With $100,000 bankroll: $17,300 allocation

**👉 See [KELLY_CRITERION_GUIDE.md](./KELLY_CRITERION_GUIDE.md) for comprehensive details**

### Fixed Allocation

**Capital-proportional** allocation based on member contributions.

```bash
syndicate allocate opportunity.json --strategy fixed
```

**When to use**:
- ✅ Equal risk tolerance across members
- ✅ Simple, transparent allocation
- ✅ New syndicates without performance history

### Dynamic Allocation

**Performance-based** allocation rewarding historical success.

```bash
syndicate allocate opportunity.json --strategy dynamic
```

**When to use**:
- ✅ Established syndicates with track record
- ✅ Incentivizing consistent performers
- ✅ Competitive team environments

### Risk Parity

**Risk-adjusted** allocation based on opportunity risk levels.

```bash
syndicate allocate opportunity.json --strategy risk-parity
```

**When to use**:
- ✅ Diverse opportunity types
- ✅ Different member risk tolerances
- ✅ Portfolio balancing

---

## 📈 Distribution Models

### Proportional Model

**Capital-based** distribution (fair for equal effort).

```bash
syndicate distribute 10000 --model proportional
```

**Formula**: `share = (member_capital / total_capital) * profit`

**Best for**:
- Equal contribution from all members
- Simple, transparent splitting
- New partnerships

**Example**: $10,000 profit with $75,000 total capital
- Alice ($30,000): $4,000
- Bob ($25,000): $3,333
- Carol ($20,000): $2,667

### Performance Model

**Success-based** rewards for historical performance.

```bash
syndicate distribute 10000 --model performance
```

**Formula**: Based on historical ROI and win rates

**Best for**:
- Experienced syndicates
- Incentivizing consistent performance
- Merit-based rewards

### Tiered Model

**Capital tier-based** incentives for larger investments.

```bash
syndicate distribute 10000 --model tiered
```

**Tiers**:
- **Platinum** (>$100k): 1.3x multiplier
- **Gold** ($50k-$100k): 1.2x multiplier
- **Silver** ($25k-$50k): 1.1x multiplier
- **Bronze** (<$25k): 1.0x multiplier

**Best for**:
- Attracting large investors
- Scaling syndicate capital
- Incentivizing growth

### Hybrid Model (Recommended)

**Balanced approach**: 60% proportional + 40% performance.

```bash
syndicate distribute 10000 --model hybrid
```

**Best for**:
- Most syndicates
- Balancing fairness and merit
- Long-term partnerships

**Preview Before Distributing**:

```bash
syndicate distribute preview 10000 --model hybrid
```

---

## 🏛️ Governance & Voting

### Permission System (18 Permissions)

#### Member Permissions
- `member:add` - Add new members
- `member:remove` - Remove members
- `member:update` - Update member details
- `member:view` - View member information

#### Financial Permissions
- `funds:allocate` - Allocate funds to opportunities
- `funds:distribute` - Distribute profits
- `funds:withdraw_approve` - Approve withdrawal requests

#### Governance Permissions
- `vote:create` - Create new proposals
- `vote:cast` - Cast votes on proposals
- `vote:view` - View voting results

#### Configuration Permissions
- `config:view` - View configuration
- `config:modify` - Modify configuration
- `config:rules` - Manage syndicate rules

### Creating Votes

```bash
syndicate vote create "<proposal>" --options "<option1,option2,option3>"

Examples:
  # Simple approval
  syndicate vote create "Increase max bet to 20%" --options "approve,reject"

  # Multiple choices
  syndicate vote create "New allocation strategy?" --options "kelly,fixed,dynamic,risk-parity"

  # With abstain option
  syndicate vote create "Change to tiered distribution?" --options "yes,no,abstain"
```

### Voting Process

1. **Create** - Manager/Admin creates proposal
2. **Cast** - Members vote based on permissions
3. **Results** - View real-time results and participation
4. **Execute** - Implement approved changes

### Governance Tiers

| Tier | Voting Weight | Approval Required |
|------|---------------|-------------------|
| Admin | 3x | 66% (supermajority) |
| Manager | 2x | 60% |
| Trader | 1.5x | 50% (majority) |
| Analyst | 1x | 50% |

**👉 See [GOVERNANCE_GUIDE.md](./GOVERNANCE_GUIDE.md) for detailed governance framework**

---

## 🔧 API Reference

### Programmatic Usage

```javascript
const Syndicate = require('@neural-trader/syndicate');

// Initialize syndicate
const syndicate = new Syndicate({
  id: 'my-fund',
  bankroll: 100000,
  dataDir: '~/.syndicate'
});

// Add member
await syndicate.addMember({
  name: 'Alice',
  email: 'alice@example.com',
  role: 'trader',
  capital: 30000
});

// Allocate funds
const allocation = await syndicate.allocate({
  opportunity: {
    description: 'NBA: Lakers vs Celtics',
    amount: 5000,
    probability: 0.55,
    odds: 2.2,
    risk_level: 'medium'
  },
  strategy: 'kelly'
});

// Distribute profits
const distribution = await syndicate.distribute({
  amount: 10000,
  model: 'hybrid'
});

// Get statistics
const stats = await syndicate.getStats();
console.log(stats);
```

### TypeScript Support

Full TypeScript definitions included:

```typescript
import { Syndicate, Member, Allocation, Distribution } from '@neural-trader/syndicate';

interface SyndicateConfig {
  id: string;
  bankroll: number;
  dataDir?: string;
  rules?: Rules;
}

interface Member {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'manager' | 'trader' | 'analyst';
  capital: number;
  profit: number;
  status: 'active' | 'inactive';
}

interface Allocation {
  id: string;
  timestamp: number;
  opportunity: Opportunity;
  strategy: 'kelly' | 'fixed' | 'dynamic' | 'risk-parity';
  allocations: MemberAllocation[];
  total: number;
}
```

---

## 🔗 MCP Integration

### 15 MCP Tools Available

The Syndicate package includes **15 Model Context Protocol (MCP) tools** for AI integration:

#### Syndicate Management
- `mcp__syndicate__create` - Create new syndicate
- `mcp__syndicate__get_status` - Get syndicate status
- `mcp__syndicate__list` - List all syndicates

#### Member Operations
- `mcp__syndicate__add_member` - Add new member
- `mcp__syndicate__list_members` - List all members
- `mcp__syndicate__get_member_stats` - Get member statistics

#### Fund Operations
- `mcp__syndicate__allocate_funds` - Allocate funds
- `mcp__syndicate__distribute_profits` - Distribute profits
- `mcp__syndicate__preview_distribution` - Preview distribution

#### Withdrawal Management
- `mcp__syndicate__request_withdrawal` - Request withdrawal
- `mcp__syndicate__approve_withdrawal` - Approve withdrawal
- `mcp__syndicate__list_withdrawals` - List withdrawals

#### Governance
- `mcp__syndicate__create_vote` - Create vote proposal
- `mcp__syndicate__cast_vote` - Cast vote
- `mcp__syndicate__get_vote_results` - Get voting results

### MCP Server Setup

```bash
# Install MCP server
npm install -g @neural-trader/mcp-server

# Configure MCP
export MCP_SYNDICATE_DATA_DIR=~/.syndicate

# Start MCP server
neural-trader-mcp start
```

### Using MCP Tools

```javascript
// With Claude or other MCP-compatible AI
const result = await mcp.call('mcp__syndicate__allocate_funds', {
  syndicate_id: 'my-fund',
  opportunity: {
    description: 'NBA game',
    amount: 5000,
    probability: 0.55,
    odds: 2.2
  },
  strategy: 'kelly'
});
```

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────┐
│                  CLI Interface                       │
│              (24 Commands)                           │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────────┐
│              Business Logic Layer                    │
│  • Allocation Strategies (4)                        │
│  • Distribution Models (4)                          │
│  • Governance Engine (18 permissions)               │
│  • Bankroll Management (9 rules)                    │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────────┐
│              Storage Layer                           │
│  • JSON File Storage                                │
│  • ~/.syndicate/ directory                          │
│  • Per-syndicate data files                         │
│  • Global configuration                             │
└─────────────────────────────────────────────────────┘
```

### Data Structure

```
~/.syndicate/
├── config.json           # Global configuration
│   ├── default_strategy
│   ├── default_model
│   └── syndicates[]      # List of syndicate IDs
└── data/
    ├── syndicate-1.json  # Syndicate 1 data
    │   ├── id
    │   ├── bankroll
    │   ├── members[]
    │   ├── allocations[]
    │   ├── distributions[]
    │   ├── withdrawals[]
    │   └── votes[]
    └── syndicate-2.json  # Syndicate 2 data
```

### Technology Stack

- **Runtime**: Node.js >= 14.0.0
- **CLI Framework**: yargs 17.7.2
- **Styling**: chalk 4.1.2
- **UI Components**: ora 5.4.1, cli-table3 0.6.3
- **Storage**: File system (JSON)
- **API**: Native JavaScript/TypeScript

---

## 📚 Examples

### Complete Workflow

```bash
# 1. Create syndicate with rules
syndicate create sports-betting \
  --bankroll 500000 \
  --rules examples/rules.json

# 2. Add team
syndicate member add "Lead Trader" lead@example.com trader --capital 150000
syndicate member add "Risk Manager" risk@example.com manager --capital 100000
syndicate member add "Analyst" analyst@example.com analyst --capital 75000

# 3. Review opportunity
cat opportunities/nba-game-123.json
{
  "description": "NBA: Lakers vs Celtics",
  "amount": 25000,
  "probability": 0.58,
  "odds": 2.1,
  "risk_level": "medium"
}

# 4. Allocate with Kelly Criterion
syndicate allocate opportunities/nba-game-123.json --strategy kelly

# 5. After win: distribute profits
syndicate distribute preview 35000 --model hybrid
syndicate distribute 35000 --model hybrid

# 6. View statistics
syndicate stats --syndicate sports-betting --json
```

### Automation with JSON

```bash
# Export data for analysis
syndicate member list --json > members.json
syndicate stats --syndicate my-fund --json > stats.json
syndicate allocate history --json > allocations.json

# Process with jq
cat members.json | jq '.[] | select(.capital > 50000)'
cat stats.json | jq '.statistics.roi'
```

### Integration with Scripts

```bash
#!/bin/bash
# daily-operations.sh

SYNDICATE="sports-betting"

# Morning routine
echo "📊 Morning Report"
syndicate stats --syndicate $SYNDICATE

# Check pending withdrawals
echo "💰 Pending Withdrawals"
syndicate withdraw list --pending --json | jq -r '.[] | .id'

# Generate allocation report
echo "📈 Recent Allocations"
syndicate allocate history | tail -n 10
```

---

## 📖 Documentation

### Complete Documentation Set

| Document | Description | Size |
|----------|-------------|------|
| **[README.md](./README.md)** | Main documentation (this file) | Comprehensive |
| **[QUICK_START.md](./QUICK_START.md)** | 5-minute quick start guide | 213 lines |
| **[KELLY_CRITERION_GUIDE.md](./KELLY_CRITERION_GUIDE.md)** | Kelly Criterion deep dive | 733 lines |
| **[GOVERNANCE_GUIDE.md](./GOVERNANCE_GUIDE.md)** | Governance framework | 619 lines |
| **[FEATURES.md](./FEATURES.md)** | Complete feature list | 317 lines |
| **[SUMMARY.md](./SUMMARY.md)** | Project summary | 366 lines |
| **[examples/README.md](./examples/README.md)** | Example usage | - |

### Help System

```bash
# Main help
syndicate --help

# Command help
syndicate member --help
syndicate allocate --help

# Subcommand help
syndicate member add --help
syndicate distribute preview --help
```

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/ruvnet/neural-trader.git
cd neural-trader/neural-trader-rust/packages/syndicate

# Install dependencies
npm install

# Run tests
npm test

# Build
npm run build
```

### Running Tests

```bash
# Run test suite
npm test

# Manual testing
cd examples
./demo.sh
```

---

## 📄 License

MIT License - See [LICENSE](../../LICENSE) for details.

---

## 🆘 Support

- **Documentation**: Complete docs in this repository
- **Issues**: [GitHub Issues](https://github.com/ruvnet/neural-trader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ruvnet/neural-trader/discussions)
- **Examples**: See [examples/](./examples/) directory

---

## 🎯 Use Cases

### Investment Syndicates
- Pool capital from multiple investors
- Track performance and ROI
- Distribute profits fairly
- Democratic governance

### Sports Betting Groups
- Manage bankroll collectively
- Kelly Criterion optimal sizing
- Track wins and losses
- Process withdrawals

### Trading Partnerships
- Share trading capital
- Performance-based rewards
- Risk management
- Member analytics

### Venture Capital Pools
- Pool investment capital
- Allocate to opportunities
- Track returns
- Investor reporting

---

## 🚀 Quick Links

- 📝 [Quick Start Guide](./QUICK_START.md) - Get started in 5 minutes
- 📊 [Kelly Criterion Guide](./KELLY_CRITERION_GUIDE.md) - Mathematical allocation strategy
- 🏛️ [Governance Guide](./GOVERNANCE_GUIDE.md) - Complete governance framework
- ⚡ [Features](./FEATURES.md) - Complete feature list
- 📦 [Examples](./examples/) - Example files and demos

---

**Made with ❤️ by the Neural Trader team**

**Star us on GitHub** ⭐ if you find Syndicate useful!
