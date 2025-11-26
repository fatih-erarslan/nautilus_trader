"""
Example Usage of Sports Betting Risk Management Framework

Demonstrates how to use the comprehensive risk management system
for syndicate sports betting operations.
"""

import datetime
from typing import List

from sports_betting.risk_management import (
    RiskFramework,
    PortfolioRiskManager,
    BetOpportunity,
    SyndicateMember,
    MemberRole,
    ExpertiseLevel,
    RegulatoryAlert,
    RiskLevel
)


def setup_syndicate():
    """Set up a betting syndicate with risk management"""
    
    # Initialize risk framework
    risk_framework = RiskFramework(
        syndicate_name="Elite Sports Syndicate",
        initial_bankroll=1000000,  # $1M initial bankroll
        config={
            'max_kelly_fraction': 0.25,  # Use 25% of Kelly
            'max_portfolio_risk': 0.10,  # 10% max portfolio risk
            'max_bet_percentage': 0.05,  # 5% max per bet
            'max_daily_loss_percentage': 0.10,  # 10% daily loss limit
            'max_drawdown_percentage': 0.20,  # 20% max drawdown
        }
    )
    
    # Add syndicate members
    members = [
        SyndicateMember(
            member_id="ADMIN001",
            name="John Smith",
            role=MemberRole.ADMIN,
            expertise_areas={
                'football': ExpertiseLevel.EXPERT,
                'basketball': ExpertiseLevel.ADVANCED
            },
            betting_limit=100000,
            daily_limit=300000
        ),
        SyndicateMember(
            member_id="SENIOR001",
            name="Sarah Johnson",
            role=MemberRole.SENIOR_TRADER,
            expertise_areas={
                'soccer': ExpertiseLevel.EXPERT,
                'tennis': ExpertiseLevel.EXPERT
            },
            betting_limit=70000,
            daily_limit=200000
        ),
        SyndicateMember(
            member_id="TRADER001",
            name="Mike Chen",
            role=MemberRole.TRADER,
            expertise_areas={
                'basketball': ExpertiseLevel.ADVANCED,
                'baseball': ExpertiseLevel.INTERMEDIATE
            },
            betting_limit=50000,
            daily_limit=150000
        ),
        SyndicateMember(
            member_id="TRADER002",
            name="Emma Davis",
            role=MemberRole.TRADER,
            expertise_areas={
                'football': ExpertiseLevel.ADVANCED,
                'soccer': ExpertiseLevel.INTERMEDIATE
            },
            betting_limit=50000,
            daily_limit=150000
        ),
        SyndicateMember(
            member_id="ANALYST001",
            name="David Wilson",
            role=MemberRole.ANALYST,
            expertise_areas={
                'football': ExpertiseLevel.INTERMEDIATE,
                'basketball': ExpertiseLevel.BEGINNER
            },
            betting_limit=20000,
            daily_limit=60000
        )
    ]
    
    for member in members:
        risk_framework.syndicate_controller.add_member(member)
        
    print(f"Syndicate '{risk_framework.syndicate_name}' initialized with {len(members)} members")
    print(f"Total bankroll: ${risk_framework.initial_bankroll:,.2f}\n")
    
    return risk_framework


def example_single_bet_evaluation(risk_framework: RiskFramework):
    """Example of evaluating a single betting opportunity"""
    print("=== Single Bet Evaluation Example ===\n")
    
    # Create a betting opportunity
    bet_opportunity = BetOpportunity(
        bet_id="NFL_2024_W1_001",
        sport="football",
        event="Kansas City Chiefs vs Detroit Lions",
        selection="Kansas City Chiefs -3.5",
        odds=1.91,  # -110 in American odds
        probability=0.55,  # Our model gives 55% chance
        confidence=0.85  # 85% confidence in our model
    )
    
    print(f"Evaluating bet: {bet_opportunity.selection}")
    print(f"Odds: {bet_opportunity.odds} (implied prob: {1/bet_opportunity.odds:.2%})")
    print(f"Our probability: {bet_opportunity.probability:.2%}")
    print(f"Edge: {bet_opportunity.edge:.2%}")
    print(f"Confidence: {bet_opportunity.confidence:.2f}\n")
    
    # Track odds movement
    risk_framework.market_analyzer.track_odds_movement(
        market_id=bet_opportunity.bet_id,
        odds=1.91,
        bookmaker="Pinnacle",
        volume=50000
    )
    
    # Evaluate the opportunity
    decision = risk_framework.evaluate_betting_opportunity(
        bet_opportunity=bet_opportunity,
        bookmaker="Pinnacle",
        jurisdiction="US",
        proposer_id="SENIOR001",
        participating_members=["SENIOR001", "TRADER001", "TRADER002"]
    )
    
    print(f"Decision: {'APPROVED' if decision.approved else 'REJECTED'}")
    print(f"Risk Score: {decision.risk_score:.2f}")
    print(f"Allocated Amount: ${decision.allocated_amount:,.2f}")
    
    if decision.violations:
        print(f"\nViolations:")
        for violation in decision.violations:
            print(f"  - {violation}")
            
    if decision.warnings:
        print(f"\nWarnings:")
        for warning in decision.warnings:
            print(f"  - {warning}")
            
    if decision.recommendations:
        print(f"\nRecommendations:")
        for rec in decision.recommendations:
            print(f"  - {rec}")
            
    if decision.member_allocations:
        print(f"\nMember Allocations:")
        for member_id, amount in decision.member_allocations.items():
            print(f"  - {member_id}: ${amount:,.2f}")
            
    # Place bet if approved
    if decision.approved:
        success = risk_framework.place_bet(bet_opportunity, decision, "Pinnacle")
        print(f"\nBet placement: {'SUCCESS' if success else 'FAILED'}")
        
    print("\n" + "="*50 + "\n")
    
    return decision


def example_portfolio_optimization(risk_framework: RiskFramework):
    """Example of portfolio optimization across multiple bets"""
    print("=== Portfolio Optimization Example ===\n")
    
    # Create multiple betting opportunities
    opportunities = [
        BetOpportunity(
            bet_id="NFL_2024_W1_002",
            sport="football",
            event="Buffalo Bills vs Miami Dolphins",
            selection="Buffalo Bills ML",
            odds=1.75,
            probability=0.62,
            confidence=0.90,
            correlation_group="NFL_Week1"
        ),
        BetOpportunity(
            bet_id="NBA_2024_001",
            sport="basketball",
            event="Lakers vs Warriors",
            selection="Total Over 225.5",
            odds=1.90,
            probability=0.56,
            confidence=0.75,
            correlation_group="NBA_Opening"
        ),
        BetOpportunity(
            bet_id="EPL_2024_001",
            sport="soccer",
            event="Man City vs Arsenal",
            selection="Both Teams to Score",
            odds=1.85,
            probability=0.58,
            confidence=0.80,
            correlation_group="EPL_Matchday1"
        ),
        BetOpportunity(
            bet_id="NFL_2024_W1_003",
            sport="football",
            event="Cowboys vs Giants",
            selection="Cowboys -7",
            odds=1.95,
            probability=0.54,
            confidence=0.70,
            correlation_group="NFL_Week1"
        )
    ]
    
    # Optimize portfolio allocation
    allocations = risk_framework.portfolio_manager.optimize_multi_sport_portfolio(opportunities)
    
    print("Portfolio Optimization Results:")
    print(f"Number of opportunities: {len(opportunities)}")
    print(f"Number of selected bets: {len(allocations)}\n")
    
    total_allocation = 0
    for alloc in allocations:
        bet = next(o for o in opportunities if o.bet_id == alloc.bet_id)
        stake = risk_framework.portfolio_manager.get_bet_size(alloc.allocation_percentage)
        
        print(f"Bet: {bet.selection}")
        print(f"  - Kelly %: {alloc.kelly_percentage:.2%}")
        print(f"  - Allocated %: {alloc.allocation_percentage:.2%}")
        print(f"  - Stake: ${stake:,.2f}")
        print(f"  - Risk Contribution: {alloc.risk_contribution:.2%}")
        print(f"  - Expected Return: {alloc.expected_return:.2%}\n")
        
        total_allocation += alloc.allocation_percentage
        
    # Calculate portfolio metrics
    metrics = risk_framework.portfolio_manager.calculate_portfolio_metrics(allocations, opportunities)
    
    print(f"Portfolio Metrics:")
    print(f"  - Total Allocation: {metrics['total_allocation']:.2%}")
    print(f"  - Expected Return: {metrics['expected_return']:.2%}")
    print(f"  - Portfolio Std Dev: {metrics['portfolio_std']:.2%}")
    print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  - Diversification: {metrics['number_of_bets']} bets\n")
    
    print("="*50 + "\n")


def example_consensus_betting(risk_framework: RiskFramework):
    """Example of consensus-based betting for large stakes"""
    print("=== Consensus Betting Example ===\n")
    
    # Create a large betting opportunity requiring consensus
    large_bet = BetOpportunity(
        bet_id="SUPERBOWL_2024_001",
        sport="football",
        event="Super Bowl LVIII",
        selection="Kansas City Chiefs to Win",
        odds=2.10,
        probability=0.52,
        confidence=0.95
    )
    
    print(f"Large bet proposal: {large_bet.selection}")
    print(f"Proposed stake calculation based on edge and confidence...")
    
    # This will trigger consensus requirement due to size
    decision = risk_framework.evaluate_betting_opportunity(
        bet_opportunity=large_bet,
        bookmaker="Bet365",
        jurisdiction="US",
        proposer_id="SENIOR001",
        participating_members=["ADMIN001", "SENIOR001", "TRADER001", "TRADER002"]
    )
    
    if decision.proposal_id:
        print(f"\nProposal {decision.proposal_id} created - requires consensus")
        
        # Simulate voting
        votes = [
            ("ADMIN001", True, "Strong value with our model confidence"),
            ("TRADER001", True, "Agree with the edge assessment"),
            ("TRADER002", False, "Concerned about public money influence")
        ]
        
        for member_id, vote, comment in votes:
            success = risk_framework.syndicate_controller.vote_on_proposal(
                decision.proposal_id,
                member_id,
                vote,
                comment
            )
            print(f"  - {member_id} voted: {'YES' if vote else 'NO'} - {comment}")
            
        # Check proposal status
        proposal = risk_framework.syndicate_controller.active_proposals.get(decision.proposal_id)
        if proposal:
            print(f"\nProposal Status: {proposal.status}")
            print(f"Votes: {sum(1 for v in proposal.votes.values() if v)}/{len(proposal.votes)}")
            
    print("\n" + "="*50 + "\n")


def example_risk_monitoring(risk_framework: RiskFramework):
    """Example of risk monitoring and alerts"""
    print("=== Risk Monitoring Example ===\n")
    
    # Simulate some bet results to trigger monitoring
    bet_results = [
        ("NFL_2024_W1_001", "win", 45000),    # Win $45k
        ("NFL_2024_W1_002", "loss", 0),       # Loss
        ("NBA_2024_001", "loss", 0),          # Loss
        ("EPL_2024_001", "win", 20000),       # Win $20k
        ("NFL_2024_W1_003", "loss", 0),       # Loss
    ]
    
    print("Updating bet results...")
    for bet_id, result, settlement in bet_results:
        if bet_id in risk_framework.active_bets:
            risk_framework.update_bet_result(bet_id, result, settlement)
            print(f"  - {bet_id}: {result}")
            
    # Perform health check
    health_check = risk_framework.perform_health_check()
    
    print(f"\nSystem Health Check:")
    print(f"Overall Status: {health_check.overall_status.upper()}")
    print(f"\nComponent Status:")
    for component, status in health_check.components.items():
        print(f"  - {component}: {status}")
        
    print(f"\nKey Metrics:")
    for metric, value in health_check.metrics.items():
        if isinstance(value, float):
            print(f"  - {metric}: {value:.2f}")
        else:
            print(f"  - {metric}: {value}")
            
    if health_check.issues:
        print(f"\nIssues Detected:")
        for issue in health_check.issues:
            print(f"  - {issue}")
            
    if health_check.recommendations:
        print(f"\nRecommendations:")
        for rec in health_check.recommendations:
            print(f"  - {rec}")
            
    print("\n" + "="*50 + "\n")


def example_emergency_procedures(risk_framework: RiskFramework):
    """Example of emergency procedures and circuit breakers"""
    print("=== Emergency Procedures Example ===\n")
    
    # Simulate rapid losses to trigger circuit breakers
    print("Simulating rapid loss scenario...")
    
    # Add a regulatory alert
    regulatory_alert = RegulatoryAlert(
        jurisdiction="NY",
        alert_type="restriction",
        severity=RiskLevel.HIGH,
        description="New restrictions on college sports betting",
        effective_date=datetime.datetime.now(),
        sports_affected=["college_football", "college_basketball"]
    )
    
    risk_framework.market_analyzer.add_regulatory_alert(regulatory_alert)
    print(f"Regulatory Alert Added: {regulatory_alert.description}")
    
    # Check emergency conditions
    emergency_status, triggered = risk_framework.syndicate_controller.check_emergency_conditions()
    
    print(f"\nEmergency Status: {emergency_status.value.upper()}")
    if triggered:
        print("Triggered Protocols:")
        for protocol in triggered:
            print(f"  - {protocol}")
            
    # Get risk dashboard
    dashboard = risk_framework.get_risk_dashboard()
    
    print(f"\nRisk Dashboard Summary:")
    print(f"  - Active Bets: {dashboard['framework']['active_bets']}")
    print(f"  - Total Exposure: ${dashboard['framework']['total_exposure']:,.2f}")
    print(f"  - Approval Rate: {dashboard['framework']['approval_rate']}")
    print(f"  - Circuit Breakers: {len(dashboard['limits']['circuit_breakers'])} configured")
    print(f"  - Emergency Status: {dashboard['syndicate']['emergency_status']}")
    
    print("\n" + "="*50 + "\n")


def main():
    """Run all examples"""
    print("="*60)
    print("SPORTS BETTING RISK MANAGEMENT FRAMEWORK DEMONSTRATION")
    print("="*60 + "\n")
    
    # Set up syndicate
    risk_framework = setup_syndicate()
    
    # Run examples
    example_single_bet_evaluation(risk_framework)
    example_portfolio_optimization(risk_framework)
    example_consensus_betting(risk_framework)
    example_risk_monitoring(risk_framework)
    example_emergency_procedures(risk_framework)
    
    # Final summary
    print("=== Final Summary ===\n")
    performance_summary = risk_framework.performance_monitor.get_performance_summary()
    
    print(f"Current Bankroll: ${performance_summary['current_bankroll']:,.2f}")
    print(f"Total P&L: ${performance_summary['total_pnl']:,.2f}")
    print(f"ROI: {performance_summary['roi']}")
    print(f"Win Rate: {performance_summary['win_rate']}")
    print(f"Current Drawdown: {performance_summary['current_drawdown']}")
    print(f"Active Alerts: {performance_summary['active_alerts']}")
    
    print("\nRisk Management Framework demonstration completed!")


if __name__ == "__main__":
    main()