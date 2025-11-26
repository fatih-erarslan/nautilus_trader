# Sports Betting Risk Management Framework

A comprehensive risk management system specifically designed for sports betting syndicate operations. This framework provides sophisticated risk controls, portfolio optimization, consensus mechanisms, and performance monitoring for professional sports betting operations.

## Overview

The Sports Betting Risk Management Framework integrates five core components to provide comprehensive risk management for syndicate betting operations:

1. **Portfolio Risk Management** - Kelly criterion implementation and multi-sport optimization
2. **Betting Limits and Controls** - Exposure limits, drawdown controls, and circuit breakers
3. **Market Risk Analysis** - Odds movement tracking and counterparty risk assessment
4. **Syndicate Risk Controls** - Member limits, consensus requirements, and governance
5. **Performance Monitoring** - Real-time P&L tracking and risk-adjusted returns

## Key Features

### 🎯 Portfolio Risk Management
- **Kelly Criterion Implementation**: Optimal bet sizing based on edge and probability
- **Fractional Kelly**: Risk-adjusted position sizing (typically 25% of full Kelly)
- **Multi-Sport Portfolio Optimization**: Correlation-aware allocation across sports
- **Dynamic Risk Allocation**: Expertise-weighted risk distribution among members

### 🛡️ Betting Limits and Controls
- **Per-Bet Limits**: Maximum exposure per individual bet
- **Daily/Weekly Limits**: Time-based exposure controls
- **Sport-Specific Limits**: Category-based risk allocation
- **Circuit Breakers**: Automated trading halts on adverse conditions
- **Drawdown Controls**: Multi-tier drawdown protection with automatic actions

### 📊 Market Risk Analysis
- **Odds Movement Tracking**: Real-time monitoring of line movements
- **Steam Move Detection**: Identification of sharp money action
- **Liquidity Assessment**: Market depth and spread analysis
- **Counterparty Risk**: Bookmaker credit evaluation and exposure limits
- **Regulatory Compliance**: Jurisdiction-specific restriction monitoring

### 👥 Syndicate Risk Controls
- **Member-Level Limits**: Individual betting authority and expertise weighting
- **Consensus Requirements**: Tiered approval process for large bets
- **Expertise-Based Allocation**: Risk allocation based on domain knowledge
- **Emergency Procedures**: Automated shutdown protocols and governance

### 📈 Performance Monitoring
- **Real-Time P&L**: Continuous performance tracking
- **Risk-Adjusted Returns**: Sharpe, Sortino, and Calmar ratios
- **Drawdown Analysis**: Maximum and current drawdown monitoring
- **Alert System**: Automated notifications for risk breaches
- **Performance Attribution**: Analysis by sport, member, and strategy

## Installation

```bash
# Install the framework as part of the AI News Trading Platform
pip install -e .

# Or install dependencies separately
pip install numpy pandas dataclasses datetime logging
```

## Quick Start

```python
from sports_betting.risk_management import RiskFramework, BetOpportunity, SyndicateMember, MemberRole

# Initialize risk framework
risk_framework = RiskFramework(
    syndicate_name="Elite Sports Syndicate",
    initial_bankroll=1000000,  # $1M
    config={
        'max_kelly_fraction': 0.25,
        'max_portfolio_risk': 0.10,
        'max_drawdown_percentage': 0.20
    }
)

# Add syndicate members
admin = SyndicateMember(
    member_id="ADMIN001",
    name="John Smith",
    role=MemberRole.ADMIN,
    expertise_areas={'football': ExpertiseLevel.EXPERT},
    betting_limit=100000
)
risk_framework.syndicate_controller.add_member(admin)

# Create betting opportunity
bet_opportunity = BetOpportunity(
    bet_id="NFL_2024_W1_001",
    sport="football",
    event="Chiefs vs Lions",
    selection="Chiefs -3.5",
    odds=1.91,
    probability=0.55,  # 55% win probability
    confidence=0.85    # 85% confidence
)

# Evaluate bet
decision = risk_framework.evaluate_betting_opportunity(
    bet_opportunity=bet_opportunity,
    bookmaker="Pinnacle",
    jurisdiction="US",
    proposer_id="ADMIN001"
)

# Place bet if approved
if decision.approved:
    risk_framework.place_bet(bet_opportunity, decision, "Pinnacle")
```

## Core Components

### 1. Portfolio Risk Manager

Implements sophisticated position sizing and portfolio optimization:

```python
from sports_betting.risk_management import PortfolioRiskManager, BetOpportunity

portfolio_manager = PortfolioRiskManager(
    bankroll=1000000,
    max_kelly_fraction=0.25,
    max_portfolio_risk=0.10
)

# Calculate Kelly percentage for a bet
kelly_pct = portfolio_manager.calculate_kelly_percentage(bet_opportunity)

# Optimize portfolio across multiple bets
allocations = portfolio_manager.optimize_multi_sport_portfolio(bet_opportunities)
```

### 2. Betting Limits Controller

Manages exposure limits and risk controls:

```python
from sports_betting.risk_management import BettingLimitsController

limits_controller = BettingLimitsController(
    bankroll=1000000,
    max_bet_percentage=0.05,
    max_daily_loss_percentage=0.10
)

# Check if bet is within limits
is_allowed, violations = limits_controller.check_bet_limits(
    amount=50000,
    sport="football"
)

# Check circuit breakers
breakers_ok, triggered = limits_controller.check_circuit_breakers()
```

### 3. Market Risk Analyzer

Monitors market conditions and counterparty risk:

```python
from sports_betting.risk_management import MarketRiskAnalyzer

market_analyzer = MarketRiskAnalyzer(
    max_odds_volatility=0.10,
    min_liquidity_score=0.7
)

# Track odds movement
market_analyzer.track_odds_movement(
    market_id="NFL_W1_001",
    odds=1.91,
    bookmaker="Pinnacle"
)

# Assess comprehensive market risk
assessment = market_analyzer.perform_comprehensive_risk_assessment(
    market_id="NFL_W1_001",
    bookmaker="Pinnacle",
    jurisdiction="US",
    sport="football",
    proposed_stake=50000
)
```

### 4. Syndicate Risk Controller

Manages member governance and consensus:

```python
from sports_betting.risk_management import SyndicateRiskController

syndicate_controller = SyndicateRiskController(
    syndicate_name="Elite Syndicate",
    total_bankroll=1000000
)

# Create betting proposal for large stakes
proposal = syndicate_controller.create_betting_proposal(
    proposer_id="SENIOR001",
    sport="football",
    event="Chiefs vs Lions",
    selection="Chiefs -3.5",
    odds=1.91,
    proposed_stake=75000,
    rationale="Strong model edge with high confidence"
)

# Vote on proposal
syndicate_controller.vote_on_proposal(
    proposal.proposal_id,
    "ADMIN001",
    True,
    "Approved based on risk analysis"
)
```

### 5. Performance Monitor

Tracks performance and generates alerts:

```python
from sports_betting.risk_management import PerformanceMonitor, BettingTransaction

performance_monitor = PerformanceMonitor(
    initial_bankroll=1000000,
    risk_free_rate=0.02,
    target_return=0.20
)

# Record betting transaction
transaction = BettingTransaction(
    transaction_id="NFL_W1_001",
    timestamp=datetime.datetime.now(),
    sport="football",
    event="Chiefs vs Lions",
    selection="Chiefs -3.5",
    bet_type="spread",
    stake=50000,
    odds=1.91
)
performance_monitor.record_transaction(transaction)

# Get performance metrics
metrics = performance_monitor.get_performance_metrics()
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Current Drawdown: {metrics.current_drawdown:.2%}")
```

## Configuration

The framework supports extensive configuration options:

```python
config = {
    # Portfolio Risk Settings
    'max_kelly_fraction': 0.25,        # Use 25% of Kelly
    'max_portfolio_risk': 0.10,        # 10% max portfolio risk
    'correlation_threshold': 0.5,       # Correlation threshold
    
    # Betting Limits
    'max_bet_percentage': 0.05,         # 5% max per bet
    'max_daily_loss_percentage': 0.10,  # 10% daily loss limit
    'max_drawdown_percentage': 0.20,    # 20% max drawdown
    
    # Market Risk
    'max_odds_volatility': 0.10,        # 10% max odds volatility
    'min_liquidity_score': 0.7,         # Minimum liquidity score
    'max_bookmaker_exposure': 50000,    # Max exposure per bookmaker
    
    # Performance
    'risk_free_rate': 0.02,             # 2% risk-free rate
    'target_return': 0.20,              # 20% target return
}

risk_framework = RiskFramework(
    syndicate_name="My Syndicate",
    initial_bankroll=1000000,
    config=config
)
```

## Risk Management Workflows

### Standard Bet Evaluation
1. **Portfolio Analysis**: Calculate optimal position size using Kelly criterion
2. **Limit Checks**: Verify bet is within all applicable limits
3. **Market Risk**: Assess odds volatility, liquidity, and counterparty risk
4. **Syndicate Controls**: Check member limits and consensus requirements
5. **Decision**: Approve or reject with detailed reasoning

### Large Bet Consensus Process
1. **Proposal Creation**: Member creates formal betting proposal
2. **Notification**: Relevant members notified for voting
3. **Voting Period**: Timed voting with role-based requirements
4. **Consensus Check**: Algorithm determines if consensus reached
5. **Execution**: Approved bets placed with risk monitoring

### Emergency Procedures
1. **Real-time Monitoring**: Continuous tracking of drawdown and losses
2. **Alert Triggers**: Automated alerts at predefined thresholds
3. **Circuit Breakers**: Automatic trading halts on adverse conditions
4. **Emergency Shutdown**: Complete syndicate shutdown if needed
5. **Recovery Procedures**: Systematic recovery and restart protocols

## Performance Metrics

The framework calculates comprehensive performance metrics:

- **Return Metrics**: Total return, ROI, CAGR
- **Risk Metrics**: Volatility, VaR, maximum drawdown
- **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Operational Metrics**: Win rate, average win/loss, profit factor
- **Portfolio Metrics**: Diversification, correlation, concentration

## Alert System

Automated alerts for various conditions:

- **Drawdown Alerts**: Current and maximum drawdown breaches
- **Losing Streak Alerts**: Consecutive loss thresholds
- **Performance Alerts**: Underperformance vs. benchmarks
- **Risk Breach Alerts**: Limit violations and risk threshold breaches
- **Market Alerts**: Significant odds movements and liquidity issues

## Best Practices

### Position Sizing
- Use fractional Kelly (typically 25% of full Kelly) for risk management
- Consider correlation between bets when sizing positions
- Apply confidence adjustments to reduce size for uncertain bets
- Monitor portfolio heat and diversification metrics

### Risk Controls
- Set conservative drawdown limits (15-20% maximum)
- Implement multiple circuit breaker levels
- Regular review and adjustment of limits based on performance
- Maintain detailed records for regulatory compliance

### Syndicate Management
- Clear role definitions and betting authorities
- Formal consensus process for large bets
- Regular performance review and member evaluation
- Emergency procedures and communication protocols

## Integration with Trading Platform

This framework integrates seamlessly with the broader AI News Trading Platform:

```python
# Integration example
from src.sports_betting.risk_management import RiskFramework
from src.neural_forecast import NeuralForecaster

# Combine with neural forecasting
neural_forecaster = NeuralForecaster()
predicted_outcome = neural_forecaster.predict_game_outcome(
    team1="Chiefs", team2="Lions", features=game_features
)

# Use prediction in bet opportunity
bet_opportunity = BetOpportunity(
    bet_id="NFL_W1_001",
    sport="football",
    event="Chiefs vs Lions",
    selection="Chiefs -3.5",
    odds=1.91,
    probability=predicted_outcome['win_probability'],
    confidence=predicted_outcome['confidence']
)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all risk management tests
python -m pytest src/sports_betting/tests/ -v

# Run specific component tests
python -m pytest src/sports_betting/tests/test_portfolio_risk.py -v

# Run example usage
python src/sports_betting/example_usage.py
```

## Support and Documentation

- **API Documentation**: Generated from docstrings
- **Example Usage**: Comprehensive examples in `example_usage.py`
- **Test Suite**: Extensive testing for all components
- **Performance Benchmarks**: Included performance validation

## License

This framework is part of the AI News Trading Platform and follows the same licensing terms.

## Contributing

Contributions welcome! Please ensure:
- Comprehensive test coverage
- Detailed documentation
- Risk management best practices
- Regulatory compliance considerations