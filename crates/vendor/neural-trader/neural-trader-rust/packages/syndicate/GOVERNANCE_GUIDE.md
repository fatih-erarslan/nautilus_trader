# Governance Guide

Comprehensive guide to syndicate governance, voting, and decision-making processes.

## Table of Contents

- [Introduction](#introduction)
- [Governance Structure](#governance-structure)
- [Voting System](#voting-system)
- [Vote Types](#vote-types)
- [Permission System](#permission-system)
- [Proposal Process](#proposal-process)
- [Dispute Resolution](#dispute-resolution)
- [Best Practices](#best-practices)

## Introduction

Effective governance is critical for syndicate success. This guide covers the democratic decision-making processes that ensure fair and transparent syndicate management.

### Governance Principles

1. **Democracy**: Member-driven decision making
2. **Transparency**: All information shared openly
3. **Fairness**: Equal voice for all members (weighted by contribution)
4. **Accountability**: Leaders accountable to members
5. **Efficiency**: Timely decision-making processes

## Governance Structure

### Member Roles

```typescript
enum MemberRole {
  LeadInvestor = 'lead_investor',      // Executive authority
  SeniorAnalyst = 'senior_analyst',    // Strategic decisions
  Analyst = 'analyst',                 // Analysis and research
  Member = 'member',                   // Basic participation
  Observer = 'observer'                // View-only
}
```

### Role Hierarchy

```
┌─────────────────┐
│  Lead Investor  │  Full control, emergency powers
└────────┬────────┘
         │
┌────────▼────────┐
│ Senior Analyst  │  Strategy and analysis authority
└────────┬────────┘
         │
┌────────▼────────┐
│    Analyst      │  Research and recommendations
└────────┬────────┘
         │
┌────────▼────────┐
│     Member      │  Voting and basic participation
└────────┬────────┘
         │
┌────────▼────────┐
│    Observer     │  View-only access
└─────────────────┘
```

### Permission Matrix

| Permission | Lead | Senior | Analyst | Member | Observer |
|-----------|------|--------|---------|--------|----------|
| View Status | ✅ | ✅ | ✅ | ✅ | ✅ |
| View Financials | ✅ | ✅ | ✅ | ✅ | ❌ |
| Propose Vote | ✅ | ✅ | ✅ | ❌ | ❌ |
| Cast Vote | ✅ | ✅ | ✅ | ✅ | ❌ |
| Allocate Funds | ✅ | ✅ | ❌ | ❌ | ❌ |
| Add Members | ✅ | ❌ | ❌ | ❌ | ❌ |
| Remove Members | ✅ | ❌ | ❌ | ❌ | ❌ |
| Modify Strategy | ✅ | ✅ | ❌ | ❌ | ❌ |
| Emergency Action | ✅ | ❌ | ❌ | ❌ | ❌ |
| View Private Data | ✅ | ✅ | ❌ | ❌ | ❌ |

## Voting System

### Vote Weighting

#### 1. Capital-Weighted (Default)

```typescript
const voteWeight = member.contribution / syndicate.totalCapital;
```

Example:
```
Alice: $40,000 / $100,000 = 40% vote weight
Bob: $30,000 / $100,000 = 30% vote weight
Carol: $20,000 / $100,000 = 20% vote weight
David: $10,000 / $100,000 = 10% vote weight
```

#### 2. Tier-Weighted

```typescript
const tierWeights = {
  platinum: 4,
  gold: 3,
  silver: 2,
  bronze: 1
};

const totalWeight = members.reduce((sum, m) => sum + tierWeights[m.tier], 0);
const voteWeight = tierWeights[member.tier] / totalWeight;
```

Example:
```
Alice (Platinum): 4 / 10 = 40% vote weight
Bob (Gold): 3 / 10 = 30% vote weight
Carol (Silver): 2 / 10 = 20% vote weight
David (Bronze): 1 / 10 = 10% vote weight
```

#### 3. Equal-Weighted

```typescript
const voteWeight = 1 / syndicate.memberCount;
```

Example:
```
Alice: 1 / 4 = 25% vote weight
Bob: 1 / 4 = 25% vote weight
Carol: 1 / 4 = 25% vote weight
David: 1 / 4 = 25% vote weight
```

### Quorum Requirements

```typescript
interface QuorumConfig {
  default: number;           // 60% default
  strategy: number;          // 70% for strategy changes
  allocation: number;        // 50% for allocations
  member: number;           // 75% for member changes
  withdrawal: number;       // 60% for withdrawals
  amendment: number;        // 80% for rule amendments
  emergency: number;        // 40% for emergency votes
}
```

### Voting Duration

```typescript
interface DurationConfig {
  standard: number;          // 48 hours
  urgent: number;           // 24 hours
  emergency: number;        // 12 hours
  amendment: number;        // 96 hours (4 days)
}
```

## Vote Types

### 1. Strategy Votes

Change allocation strategies or betting approaches.

```typescript
// Example: Change to risk-adjusted allocation
const vote = await syndicate.createVote({
  type: VoteType.Strategy,
  proposal: 'Switch from Kelly to Risk-Adjusted allocation',
  description: `
    Current: Kelly Criterion (Quarter Kelly)
    Proposed: Risk-Adjusted with 5% max risk per bet

    Rationale:
    - Better drawdown protection
    - More consistent returns
    - Lower volatility

    Impact:
    - Average bet size: -15%
    - Expected volatility: -25%
    - Max drawdown: -30%
  `,
  options: ['approve', 'reject', 'modify'],
  duration: 48,
  quorum: 0.70
});
```

**Approval Requirements:**
- Quorum: 70%
- Approval: Simple majority (>50%)
- Duration: 48 hours

### 2. Allocation Votes

Approve specific large allocations.

```typescript
// Example: Large Super Bowl bet
const vote = await syndicate.createVote({
  type: VoteType.Allocation,
  proposal: 'Allocate $25,000 to Super Bowl Chiefs ML',
  description: `
    Opportunity: Kansas City Chiefs ML vs 49ers
    Odds: 2.15 (DraftKings)
    Our Probability: 58%
    Edge: 24.7%
    Recommended: $25,000 (5% of bankroll)

    Analysis:
    - Chiefs 4-0 in Super Bowls under Mahomes
    - 49ers key injuries on defense
    - Weather favors Chiefs offense
    - Line movement in our favor

    Risk Assessment:
    - Kelly: 12.4% of bankroll
    - Recommended: 5% (conservative)
    - Impact on reserves: Within limits
  `,
  options: ['approve', 'reject', 'reduce_to_$15000', 'reduce_to_$20000'],
  duration: 24,
  quorum: 0.60
});
```

**Approval Requirements:**
- Quorum: 60%
- Approval: Simple majority
- Duration: 24 hours (urgent)

### 3. Member Votes

Add, remove, or modify member roles.

```typescript
// Example: Promote member to analyst
const vote = await syndicate.createVote({
  type: VoteType.Member,
  proposal: 'Promote Bob Smith to Senior Analyst',
  description: `
    Current Role: Analyst
    Proposed Role: Senior Analyst

    Qualifications:
    - 2 years as Analyst
    - 68% win rate on recommendations
    - +$127,000 profit contribution
    - Strong strategic thinking

    New Permissions:
    - Allocate funds independently
    - Modify allocation strategies
    - Access to private financial data
  `,
  options: ['approve', 'reject', 'defer_6_months'],
  duration: 72,
  quorum: 0.75
});
```

**Approval Requirements:**
- Quorum: 75% (high for member changes)
- Approval: 60% supermajority
- Duration: 72 hours

### 4. Withdrawal Votes

Approve member capital withdrawals.

```typescript
// Example: Large withdrawal request
const vote = await syndicate.createVote({
  type: VoteType.Withdrawal,
  proposal: 'Approve Alice $50,000 emergency withdrawal',
  description: `
    Member: Alice Johnson
    Current Capital: $85,000
    Withdrawal Amount: $50,000
    Reason: Medical emergency

    Impact Analysis:
    - Syndicate capital: $250,000 → $200,000 (-20%)
    - Alice equity: $85,000 → $35,000
    - Reserve ratio: 12% → 10% (still above min)
    - Active positions: Sufficient liquidity

    Timeline:
    - If approved: Funds in 24 hours
    - No penalty for emergency withdrawal
  `,
  options: ['approve_full', 'approve_$30000', 'reject', 'defer_7_days'],
  duration: 12,  // Emergency fast-track
  quorum: 0.60
});
```

**Approval Requirements:**
- Quorum: 60%
- Approval: Simple majority
- Duration: 12 hours (emergency) or 72 hours (standard)

### 5. Amendment Votes

Modify syndicate rules and configuration.

```typescript
// Example: Change quorum requirement
const vote = await syndicate.createVote({
  type: VoteType.Amendment,
  proposal: 'Reduce voting quorum from 60% to 50%',
  description: `
    Current: 60% quorum required
    Proposed: 50% quorum required

    Rationale:
    - 60% too difficult to achieve
    - Average participation: 55%
    - Slowing decision-making
    - Many votes expiring without resolution

    Safeguards:
    - Keep supermajority for member changes (75%)
    - Keep high quorum for amendments (80%)
    - Only affects standard votes

    Impact:
    - Faster decision-making
    - More votes reaching resolution
    - Still requires majority approval
  `,
  options: ['approve', 'reject', 'reduce_to_55%'],
  duration: 96,  // 4 days for constitutional changes
  quorum: 0.80   // High bar for amendments
});
```

**Approval Requirements:**
- Quorum: 80% (very high)
- Approval: 66% supermajority
- Duration: 96 hours (4 days)

## Permission System

### Permission Types

```typescript
enum Permission {
  ViewStatus = 'view_status',
  ViewFinancials = 'view_financials',
  ProposeVote = 'propose_vote',
  CastVote = 'cast_vote',
  AllocateFunds = 'allocate_funds',
  AddMember = 'add_member',
  RemoveMember = 'remove_member',
  ModifyStrategy = 'modify_strategy',
  EmergencyAction = 'emergency_action',
  ViewPrivateData = 'view_private_data',
  ModifyPermissions = 'modify_permissions'
}
```

### Permission Checking

```typescript
class PermissionManager {
  hasPermission(member: Member, permission: Permission): boolean {
    // Check role-based permissions
    const rolePermissions = this.getRolePermissions(member.role);
    if (rolePermissions.includes(permission)) {
      return true;
    }

    // Check custom permissions
    if (member.customPermissions?.includes(permission)) {
      return true;
    }

    return false;
  }

  requirePermission(member: Member, permission: Permission): void {
    if (!this.hasPermission(member, permission)) {
      throw new Error(
        `Member ${member.name} lacks permission: ${permission}`
      );
    }
  }
}
```

### Custom Permissions

```typescript
// Grant custom permission to specific member
await syndicate.grantPermission({
  memberId: 'bob-001',
  permission: Permission.AllocateFunds,
  reason: 'Temporary delegation during Alice vacation',
  expiresAt: new Date('2024-02-01')
});

// Revoke permission
await syndicate.revokePermission({
  memberId: 'bob-001',
  permission: Permission.AllocateFunds
});
```

## Proposal Process

### Step-by-Step Process

#### 1. Proposal Creation

```typescript
const proposal = {
  type: VoteType.Strategy,
  title: 'Increase fractional Kelly to 0.30',
  description: 'Detailed rationale...',
  impact: 'Expected outcomes...',
  risks: 'Potential risks...',
  alternatives: 'Other options considered...',
  timeline: 'Implementation timeline...'
};
```

#### 2. Discussion Period

```typescript
// Enable discussion before voting
const discussion = await syndicate.createDiscussion({
  proposalId: proposal.id,
  duration: 24  // 24 hours before voting opens
});

// Members can comment
await discussion.addComment({
  memberId: 'bob-001',
  comment: 'I support this but suggest 0.28 instead of 0.30'
});
```

#### 3. Voting Period

```typescript
// Open vote after discussion
const vote = await syndicate.createVote({
  proposalId: proposal.id,
  options: ['approve', 'reject', 'modify'],
  duration: 48,
  quorum: 0.70
});

// Members cast votes
await syndicate.castVote(vote.id, 'alice-001', 'approve');
await syndicate.castVote(vote.id, 'bob-001', 'modify');
await syndicate.castVote(vote.id, 'carol-001', 'approve');
await syndicate.castVote(vote.id, 'david-001', 'approve');
```

#### 4. Resolution

```typescript
// Check vote outcome
const outcome = await vote.getOutcome();

if (outcome.approved) {
  // Execute proposal
  await proposal.execute();

  // Notify members
  await syndicate.notifyMembers({
    subject: 'Proposal Approved',
    message: `Proposal "${proposal.title}" has been approved and executed.`
  });
} else {
  // Proposal rejected
  await syndicate.notifyMembers({
    subject: 'Proposal Rejected',
    message: `Proposal "${proposal.title}" was not approved.`
  });
}
```

## Dispute Resolution

### Dispute Types

1. **Member Disputes**: Conflicts between members
2. **Allocation Disputes**: Disagreements on allocations
3. **Profit Disputes**: Distribution disagreements
4. **Governance Disputes**: Process or rule violations

### Resolution Process

#### Step 1: Internal Mediation

```typescript
const dispute = await syndicate.createDispute({
  type: DisputeType.MemberDispute,
  plaintiff: 'alice-001',
  defendant: 'bob-001',
  description: 'Disagreement over allocation decision',
  evidence: ['...']
});

// Assign mediator (Lead Investor)
await dispute.assignMediator('lead-001');

// Mediation attempt
const mediation = await dispute.mediate({
  duration: 7  // 7 days for resolution
});
```

#### Step 2: Member Vote

```typescript
// If mediation fails, member vote
if (!mediation.resolved) {
  const vote = await syndicate.createVote({
    type: VoteType.Dispute,
    proposal: dispute.summary,
    options: ['support_plaintiff', 'support_defendant', 'compromise'],
    duration: 48,
    quorum: 0.70
  });
}
```

#### Step 3: Binding Arbitration

```typescript
// If vote fails to resolve, binding arbitration
if (!vote.resolved) {
  const arbitration = await dispute.arbitrate({
    arbitrator: externalArbitrator,
    bindingDecision: true
  });

  // Execute arbitrator's decision
  await arbitration.decision.execute();
}
```

## Best Practices

### 1. Clear Communication

```typescript
// Always provide detailed proposals
const goodProposal = {
  title: 'Increase Kelly to 0.30',
  description: 'Detailed 3-paragraph explanation',
  rationale: 'Why this is beneficial',
  impact: 'Expected outcomes with numbers',
  risks: 'Potential downsides',
  alternatives: 'Other options considered',
  recommendation: 'Clear recommendation'
};
```

### 2. Appropriate Quorum

```typescript
// Set quorum based on importance
const quorumLevels = {
  routine: 0.50,      // Routine decisions
  important: 0.60,    // Important decisions
  critical: 0.70,     // Critical decisions
  constitutional: 0.80 // Rule changes
};
```

### 3. Timely Voting

```typescript
// Send reminders
await syndicate.sendVoteReminders({
  voteId: vote.id,
  reminderSchedule: [24, 12, 6, 1]  // Hours before closing
});
```

### 4. Document Everything

```typescript
// Keep audit trail
await syndicate.logGovernanceAction({
  type: 'vote_created',
  voteId: vote.id,
  creator: member.id,
  timestamp: new Date(),
  details: vote
});
```

### 5. Regular Reviews

```typescript
// Quarterly governance review
await syndicate.conductGovernanceReview({
  reviewPeriod: 'Q1-2024',
  metrics: {
    votesCreated: 24,
    votesCompleted: 22,
    averageParticipation: 0.73,
    disputesResolved: 3,
    amendmentsPassed: 1
  }
});
```

---

**Good governance is the foundation of syndicate success!**
