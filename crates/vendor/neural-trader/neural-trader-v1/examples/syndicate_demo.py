#!/usr/bin/env python3
"""
Syndicate Investment System Demo
Demonstrates the key features of collaborative investment management
"""

from decimal import Decimal
from datetime import timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.syndicate.member_management import SyndicateMemberManager, MemberRole
from src.syndicate.capital_management import (
    FundAllocationEngine, ProfitDistributionSystem, WithdrawalManager,
    BettingOpportunity, AllocationStrategy, DistributionModel
)


def demo_syndicate_creation():
    """Demonstrate creating a trading syndicate"""
    print("=== Creating Trading Syndicate ===\n")
    
    # Initialize syndicate
    syndicate = SyndicateMemberManager("demo-syndicate-001")
    
    # Add members with different roles
    members = [
        ("Alice Johnson", "alice@syndicate.com", MemberRole.LEAD_INVESTOR, "100000"),
        ("Bob Smith", "bob@syndicate.com", MemberRole.SENIOR_ANALYST, "50000"),
        ("Charlie Brown", "charlie@syndicate.com", MemberRole.JUNIOR_ANALYST, "25000"),
        ("Diana Prince", "diana@syndicate.com", MemberRole.CONTRIBUTING_MEMBER, "10000"),
        ("Eve Wilson", "eve@syndicate.com", MemberRole.OBSERVER, "5000"),
    ]
    
    for name, email, role, contribution in members:
        member = syndicate.add_member(name, email, role, Decimal(contribution))
        print(f"Added {name} as {role.value} with ${contribution} contribution")
    
    total_capital = syndicate.get_total_capital()
    print(f"\nTotal syndicate capital: ${total_capital:,}")
    
    return syndicate


def demo_fund_allocation(syndicate):
    """Demonstrate automated fund allocation"""
    print("\n\n=== Fund Allocation Demo ===\n")
    
    # Initialize allocation engine
    allocator = FundAllocationEngine(
        syndicate_id="demo-syndicate-001",
        total_bankroll=syndicate.get_total_capital()
    )
    
    # Create betting opportunities
    opportunities = [
        BettingOpportunity(
            sport="NBA",
            event="Lakers vs Celtics",
            bet_type="spread",
            selection="Lakers -3.5",
            odds=1.91,
            probability=0.58,
            edge=0.05,
            confidence=0.75,
            model_agreement=0.82,
            time_until_event=timedelta(hours=24),
            liquidity=50000
        ),
        BettingOpportunity(
            sport="NFL", 
            event="Patriots vs Jets",
            bet_type="moneyline",
            selection="Patriots ML",
            odds=2.20,
            probability=0.52,
            edge=0.074,
            confidence=0.85,
            model_agreement=0.90,
            time_until_event=timedelta(hours=48),
            liquidity=100000
        ),
        BettingOpportunity(
            sport="NBA",
            event="Warriors vs Nets",
            bet_type="total",
            selection="Over 220.5",
            odds=1.87,
            probability=0.56,
            edge=0.023,
            confidence=0.60,
            model_agreement=0.65,
            time_until_event=timedelta(hours=6),
            liquidity=30000
        )
    ]
    
    # Get allocation recommendations
    for opp in opportunities:
        result = allocator.allocate_funds(opp, AllocationStrategy.KELLY_CRITERION)
        
        print(f"\n{opp.sport} - {opp.event}")
        print(f"Selection: {opp.selection} @ {opp.odds}")
        print(f"Edge: {opp.edge:.1%}, Confidence: {opp.confidence:.0%}")
        print(f"Recommended bet: ${result.amount}")
        print(f"Percentage of bankroll: {result.percentage_of_bankroll:.2%}")
        
        if result.warnings:
            print(f"Warnings: {', '.join(result.warnings)}")
        
        if result.approval_required:
            print("⚠️  Requires lead investor approval")


def demo_performance_tracking(syndicate):
    """Demonstrate member performance tracking"""
    print("\n\n=== Performance Tracking Demo ===\n")
    
    # Simulate some bet outcomes
    bet_results = [
        {
            "member_id": list(syndicate.members.values())[1].id,  # Senior Analyst
            "bet_id": "BET-001",
            "sport": "NBA",
            "bet_type": "spread",
            "odds": 1.91,
            "stake": 1000,
            "outcome": "won",
            "profit": 910,
            "confidence": 0.75
        },
        {
            "member_id": list(syndicate.members.values())[1].id,
            "bet_id": "BET-002", 
            "sport": "NFL",
            "bet_type": "moneyline",
            "odds": 2.20,
            "stake": 1500,
            "outcome": "won",
            "profit": 1800,
            "confidence": 0.85
        },
        {
            "member_id": list(syndicate.members.values())[2].id,  # Junior Analyst
            "bet_id": "BET-003",
            "sport": "NBA",
            "bet_type": "total",
            "odds": 1.87,
            "stake": 500,
            "outcome": "lost",
            "profit": -500,
            "confidence": 0.60
        }
    ]
    
    # Track outcomes
    for result in bet_results:
        syndicate.performance_tracker.track_bet_outcome(
            member_id=result["member_id"],
            bet_details=result
        )
    
    # Show performance reports
    for member_id, member in list(syndicate.members.items())[:3]:
        report = syndicate.get_member_performance_report(member_id)
        
        print(f"\n{report['member_info']['name']} ({report['member_info']['role']})")
        print(f"Capital: ${report['financial_summary']['capital_contribution']}")
        print(f"Win Rate: {report['betting_performance']['win_rate']:.1%}")
        print(f"ROI: {report['financial_summary']['roi']:.1%}")


def demo_profit_distribution(syndicate):
    """Demonstrate profit distribution"""
    print("\n\n=== Profit Distribution Demo ===\n")
    
    # Initialize distribution system
    distributor = ProfitDistributionSystem("demo-syndicate-001")
    
    # Calculate weekly profit distribution
    weekly_profit = Decimal("15000")
    
    print(f"Distributing weekly profit: ${weekly_profit:,}")
    print("\nUsing Hybrid Model (50% capital, 30% performance, 20% equal)\n")
    
    distributions = distributor.calculate_distribution(
        total_profit=weekly_profit,
        members=list(syndicate.members.values()),
        model=DistributionModel.HYBRID
    )
    
    # Show distributions
    for member_id, details in distributions.items():
        member = syndicate.members[member_id]
        if member.is_active:
            print(f"{member.name} ({member.role.value}):")
            print(f"  Gross: ${details['gross_amount']}")
            print(f"  Tax: ${details['tax_withheld']}")
            print(f"  Net: ${details['net_amount']}")


def demo_voting_system(syndicate):
    """Demonstrate democratic voting"""
    print("\n\n=== Voting System Demo ===\n")
    
    # Create a proposal
    vote_id = syndicate.voting_system.create_vote(
        proposal_type="strategy_change",
        proposal_details={
            "proposal": "Increase NBA exposure limit from 40% to 50%",
            "reason": "Strong performance in NBA predictions",
            "proposed_limit": "50%",
            "current_limit": "40%"
        },
        proposed_by=list(syndicate.members.values())[1].id,  # Senior Analyst
        voting_period_hours=48
    )
    
    print("Proposal: Increase NBA exposure limit from 40% to 50%")
    print("Voting period: 48 hours\n")
    
    # Cast some votes
    votes = [
        (0, "approve"),  # Lead Investor
        (1, "approve"),  # Senior Analyst
        (2, "approve"),  # Junior Analyst
        (3, "reject"),   # Contributing Member
    ]
    
    for idx, decision in votes:
        member = list(syndicate.members.values())[idx]
        syndicate.voting_system.cast_vote(vote_id, member.id, decision)
        weight = member.calculate_voting_weight(syndicate.get_total_capital())
        print(f"{member.name} voted: {decision} (weight: {weight:.2f})")
    
    # Get results
    results = syndicate.voting_system.get_vote_results(vote_id)
    print(f"\nVoting Results:")
    print(f"Approval: {results['approval_percentage']:.1%}")
    print(f"Participation: {results['participation_rate']:.1%}")
    print(f"Status: {'PASSED' if results['approval_percentage'] > 50 else 'FAILED'}")


def main():
    """Run the complete syndicate demo"""
    print("""
    ╔═══════════════════════════════════════════════╗
    ║   AI News Trader - Syndicate System Demo      ║
    ║   Collaborative Investment Management         ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    # Run demos
    syndicate = demo_syndicate_creation()
    demo_fund_allocation(syndicate)
    demo_performance_tracking(syndicate)
    demo_profit_distribution(syndicate)
    demo_voting_system(syndicate)
    
    print("\n\n✅ Demo complete! The syndicate system provides:")
    print("   • Collaborative capital pooling")
    print("   • Automated fund allocation with risk management")
    print("   • Performance tracking and member analytics")
    print("   • Fair profit distribution with tax handling")
    print("   • Democratic governance through weighted voting")
    print("\nReady for production use in trading and sports betting!")


if __name__ == "__main__":
    main()