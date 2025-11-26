# Syndicate System Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Initialize a Syndicate

```python
from src.syndicate.member_management import SyndicateMemberManager, MemberRole
from src.syndicate.capital_management import FundAllocationEngine
from decimal import Decimal

# Create syndicate
syndicate = SyndicateMemberManager("my-syndicate-001")

# Add lead investor
lead = syndicate.add_member(
    name="Alice Johnson",
    email="alice@syndicate.com",
    role=MemberRole.LEAD_INVESTOR,
    initial_contribution=Decimal("50000")
)

# Add other members
analyst = syndicate.add_member(
    name="Bob Smith",
    email="bob@syndicate.com", 
    role=MemberRole.SENIOR_ANALYST,
    initial_contribution=Decimal("25000")
)

member = syndicate.add_member(
    name="Charlie Brown",
    email="charlie@syndicate.com",
    role=MemberRole.CONTRIBUTING_MEMBER,
    initial_contribution=Decimal("10000")
)

print(f"Total syndicate capital: ${syndicate.get_total_capital()}")
```

### 2. Set Up Fund Allocation

```python
# Initialize allocation engine
allocator = FundAllocationEngine(
    syndicate_id="my-syndicate-001",
    total_bankroll=syndicate.get_total_capital()
)

# Create a betting opportunity
from src.syndicate.capital_management import BettingOpportunity, AllocationStrategy
from datetime import timedelta

opportunity = BettingOpportunity(
    sport="NBA",
    event="Lakers vs Celtics",
    bet_type="moneyline",
    selection="Lakers ML",
    odds=2.15,
    probability=0.52,  # 52% win probability
    edge=0.068,        # 6.8% edge
    confidence=0.80,   # 80% model confidence
    model_agreement=0.85,
    time_until_event=timedelta(hours=3),
    liquidity=100000
)

# Get allocation recommendation
allocation = allocator.allocate_funds(
    opportunity,
    strategy=AllocationStrategy.KELLY_CRITERION
)

print(f"Recommended bet size: ${allocation.amount}")
print(f"Percentage of bankroll: {allocation.percentage_of_bankroll:.1%}")
print(f"Risk warnings: {allocation.warnings}")
```

### 3. Execute and Track Bets

```python
# Place the bet (integrate with your betting API)
bet_result = {
    "bet_id": "BET-20250108-001",
    "sport": "NBA",
    "amount": allocation.amount,
    "is_live": False
}

# Update exposure tracking
allocator.update_exposure(bet_result)

# Track performance (after bet settles)
syndicate.performance_tracker.track_bet_outcome(
    member_id=analyst.id,  # Who made the pick
    bet_details={
        "bet_id": bet_result["bet_id"],
        "sport": "NBA",
        "bet_type": "moneyline",
        "odds": 2.15,
        "stake": float(allocation.amount),
        "outcome": "won",  # or "lost"
        "profit": float(allocation.amount) * 1.15,  # If won
        "confidence": 0.80
    }
)
```

### 4. Distribute Profits

```python
from src.syndicate.capital_management import ProfitDistributionSystem, DistributionModel

# Weekly profit distribution
distributor = ProfitDistributionSystem("my-syndicate-001")

# Calculate distributions (e.g., $5,000 weekly profit)
distributions = distributor.calculate_distribution(
    total_profit=Decimal("5000"),
    members=list(syndicate.members.values()),
    model=DistributionModel.HYBRID  # 50% capital, 30% performance, 20% equal
)

# View distributions
for member_id, details in distributions.items():
    member = syndicate.members[member_id]
    print(f"{member.name}: ${details['net_amount']} (after {details['tax_withheld']} tax)")
```

### 5. Monitor Performance

```python
# Get member performance
for member_id in syndicate.members:
    report = syndicate.get_member_performance_report(member_id)
    
    print(f"\n{report['member_info']['name']} Performance:")
    print(f"  Role: {report['member_info']['role']}")
    print(f"  Capital: ${report['financial_summary']['capital_contribution']}")
    print(f"  ROI: {report['financial_summary']['roi']:.1%}")
    print(f"  Win Rate: {report['betting_performance']['win_rate']:.1%}")
    print(f"  Alpha: {report['alpha_analysis']['alpha']:.2%}")
```

## 🎯 Common Use Cases

### Creating a Vote

```python
# Propose strategy change
vote_id = syndicate.voting_system.create_vote(
    proposal_type="strategy_change",
    proposal_details={
        "change": "Increase NBA exposure limit",
        "from": "40%",
        "to": "50%",
        "reason": "Strong performance in NBA picks"
    },
    proposed_by=analyst.id,
    voting_period_hours=48
)

# Cast votes
syndicate.voting_system.cast_vote(vote_id, lead.id, "approve")
syndicate.voting_system.cast_vote(vote_id, member.id, "approve")

# Check results
results = syndicate.voting_system.get_vote_results(vote_id)
print(f"Approval: {results['approval_percentage']:.1%}")
```

### Processing Withdrawals

```python
from src.syndicate.capital_management import WithdrawalManager

# Initialize withdrawal manager
withdrawals = WithdrawalManager("my-syndicate-001")

# Member requests withdrawal
request = withdrawals.request_withdrawal(
    member_id=member.id,
    member_balance=Decimal("15000"),
    amount=Decimal("5000"),
    is_emergency=False  # 7-day wait period
)

print(f"Withdrawal scheduled for: {request['scheduled_for']}")
print(f"Net amount: ${request['net_amount']}")
```

### Risk Monitoring

```python
# Check current exposure
exposure = allocator.current_exposure

print(f"Daily exposure: ${exposure['daily']} ({float(exposure['daily']/allocator.total_bankroll)*100:.1%})")
print(f"Weekly exposure: ${exposure['weekly']}")
print(f"Live betting exposure: ${exposure['live_betting']}")

# Sport concentration
for sport, amount in exposure['by_sport'].items():
    percentage = float(amount/allocator.total_bankroll)*100
    print(f"{sport}: ${amount} ({percentage:.1%})")
```

## 📊 Key Metrics to Monitor

1. **Syndicate Health**
   - Total capital
   - Active members
   - Daily/weekly P&L
   - Cash reserve percentage

2. **Member Performance**
   - Individual ROI
   - Win rates by sport
   - Alpha generation
   - Contribution to profits

3. **Risk Metrics**
   - Current exposure
   - Concentration by sport
   - Drawdown tracking
   - Stop loss proximity

4. **Operational Metrics**
   - Pending withdrawals
   - Active votes
   - Tax obligations
   - Distribution history

## 🛡️ Best Practices

1. **Start Conservative**
   - Use 25% Kelly or less
   - Maintain 40%+ cash reserve
   - Set tight stop losses

2. **Regular Reviews**
   - Weekly performance analysis
   - Monthly strategy reviews
   - Quarterly member evaluations

3. **Clear Communication**
   - Document all decisions
   - Share performance reports
   - Explain strategy changes

4. **Risk First**
   - Never exceed position limits
   - Diversify across sports
   - Monitor correlation risk

## 🔗 Integration Examples

### With Trading APIs

```python
from src.trading_apis.orchestrator.execution_router import ExecutionRouter

# Route syndicate orders through trading APIs
router = ExecutionRouter(trading_apis)

# Execute syndicate allocation
result = await router.execute_order(
    order=allocation.to_order(),
    strategy='balanced'
)
```

### With AI Models

```python
from src.strategies.neural_strategy import NeuralStrategy

# Use AI for opportunity identification
neural = NeuralStrategy()
opportunities = neural.scan_markets()

# Allocate to best opportunities
for opp in opportunities[:5]:  # Top 5
    allocation = allocator.allocate_funds(opp)
    if allocation.amount > 0:
        execute_bet(allocation)
```

## 🚨 Troubleshooting

### Common Issues

1. **"Insufficient funds"**
   - Check cash reserve requirement (30%)
   - Review daily exposure limits
   - Verify member contributions

2. **"Approval required"**
   - Large bets need lead investor approval
   - Check permission settings
   - Review syndicate rules

3. **"Allocation too small"**
   - Edge might be too low
   - Confidence threshold not met
   - Kelly criterion returning minimal size

### Debug Commands

```python
# Check syndicate status
print(f"Total Capital: ${syndicate.get_total_capital()}")
print(f"Active Members: {len([m for m in syndicate.members.values() if m.is_active])}")
print(f"Current Exposure: {allocator.current_exposure}")

# Verify permissions
member = syndicate.members[member_id]
print(f"Can propose bets: {member.permissions.propose_bets}")
print(f"Can vote: {member.permissions.vote_on_major_decisions}")

# Check allocation logic
print(f"Bankroll rules: {allocator.rules}")
```

## 📚 Next Steps

1. Read the [full documentation](SYNDICATE_SYSTEM.md)
2. Review example implementations in `tests/`
3. Set up monitoring dashboards
4. Configure tax settings
5. Establish syndicate rules and bylaws

---

**Ready to start?** The syndicate system is production-ready for collaborative investing!