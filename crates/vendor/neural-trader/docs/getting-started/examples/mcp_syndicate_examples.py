#!/usr/bin/env python3
"""
MCP Syndicate Tools - Comprehensive Examples
Demonstrates all 17 syndicate management tools with real-world scenarios
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Note: In Claude Code, these tools are available as mcp__ai_news_trader__<tool_name>
# This example shows the conceptual usage patterns

async def example_1_create_professional_syndicate():
    """Example 1: Creating a Professional Sports Betting Syndicate"""
    print("=== Example 1: Creating Professional Syndicate ===\n")
    
    # Create the syndicate with advanced configuration
    syndicate = await mcp__ai_news_trader__syndicate_create(
        name="Elite Sports Analytics Group",
        description="Professional syndicate focused on NBA and NFL with quantitative strategies",
        initial_capital=1000000,  # $1M initial capital
        allocation_strategy="kelly_criterion",
        risk_limits={
            "max_single_bet": 0.03,      # 3% max per bet
            "daily_exposure": 0.15,       # 15% daily limit
            "sport_concentration": 0.40,   # 40% max in one sport
            "stop_loss": 0.08,            # 8% daily stop loss
            "cash_reserve": 0.30          # 30% minimum cash
        }
    )
    
    print(f"Created syndicate: {syndicate['name']}")
    print(f"Syndicate ID: {syndicate['syndicate_id']}")
    print(f"Initial capital: ${syndicate['initial_capital']:,}")
    print(f"Risk limits configured: {syndicate['risk_limits']}")
    
    return syndicate['syndicate_id']


async def example_2_build_syndicate_team():
    """Example 2: Building a Diverse Syndicate Team"""
    print("\n=== Example 2: Building Syndicate Team ===\n")
    
    syndicate_id = "syn_20240626_001"
    
    # Define team members with different roles and contributions
    team_members = [
        {
            "name": "Alice Johnson",
            "email": "alice@elitesports.com",
            "role": "lead_investor",
            "contribution": 400000,
            "background": "Hedge fund manager, 15 years experience"
        },
        {
            "name": "Dr. Bob Chen",
            "email": "bob@elitesports.com",
            "role": "senior_analyst",
            "contribution": 200000,
            "background": "PhD in Statistics, sports modeling expert"
        },
        {
            "name": "Charlie Williams",
            "email": "charlie@elitesports.com",
            "role": "senior_analyst",
            "contribution": 150000,
            "background": "Former professional sports analyst"
        },
        {
            "name": "Diana Martinez",
            "email": "diana@elitesports.com",
            "role": "junior_analyst",
            "contribution": 100000,
            "background": "Data scientist, ML specialist"
        },
        {
            "name": "Erik Thompson",
            "email": "erik@elitesports.com",
            "role": "contributing_member",
            "contribution": 75000,
            "background": "Sports journalist, insider knowledge"
        },
        {
            "name": "Fiona O'Brien",
            "email": "fiona@elitesports.com",
            "role": "contributing_member",
            "contribution": 50000,
            "background": "Risk management consultant"
        },
        {
            "name": "George Kim",
            "email": "george@elitesports.com",
            "role": "observer",
            "contribution": 25000,
            "background": "New investor, learning phase"
        }
    ]
    
    # Add each member to the syndicate
    for member_data in team_members:
        member = await mcp__ai_news_trader__syndicate_add_member(
            syndicate_id=syndicate_id,
            name=member_data["name"],
            email=member_data["email"],
            role=member_data["role"],
            contribution=member_data["contribution"]
        )
        
        print(f"Added {member['name']}:")
        print(f"  Role: {member['role']}")
        print(f"  Contribution: ${member['contribution']:,}")
        print(f"  Investment Tier: {member['investment_tier']}")
        print(f"  Voting Weight: {member['voting_weight']:.2%}\n")
    
    # Get syndicate summary
    members = await mcp__ai_news_trader__syndicate_get_members(
        syndicate_id=syndicate_id,
        include_performance=True
    )
    
    print(f"Total Members: {members['total_members']}")
    print(f"Total Capital: ${members['total_capital']:,}")


async def example_3_democratic_governance():
    """Example 3: Democratic Voting on Strategy Changes"""
    print("\n=== Example 3: Democratic Governance ===\n")
    
    syndicate_id = "syn_20240626_001"
    
    # Senior analyst proposes strategy change
    proposal = await mcp__ai_news_trader__syndicate_create_proposal(
        syndicate_id=syndicate_id,
        proposal_type="strategy_change",
        title="Increase NBA exposure limit from 40% to 50%",
        details={
            "current_limit": "40%",
            "proposed_limit": "50%",
            "rationale": "Our NBA models have shown 73% accuracy over the past 3 months",
            "risk_analysis": "Backtesting shows acceptable risk with increased limit",
            "implementation_date": "2024-07-01"
        },
        proposed_by="mem_20240626_002",  # Dr. Bob Chen
        voting_period_hours=72  # 3 days for important decisions
    )
    
    print(f"Proposal Created: {proposal['title']}")
    print(f"Proposal ID: {proposal['proposal_id']}")
    print(f"Voting ends: {proposal['voting_ends']}")
    
    # Simulate voting by different members
    votes = [
        ("mem_20240626_001", "approve", "Strong NBA performance justifies increase"),
        ("mem_20240626_002", "approve", "My analysis supports this change"),
        ("mem_20240626_003", "approve", "Agree with the data"),
        ("mem_20240626_004", "approve", "Models look solid"),
        ("mem_20240626_005", "reject", "Concerned about concentration risk"),
        ("mem_20240626_006", "abstain", "Need more information")
    ]
    
    for member_id, decision, reason in votes:
        vote = await mcp__ai_news_trader__syndicate_cast_vote(
            syndicate_id=syndicate_id,
            proposal_id=proposal['proposal_id'],
            member_id=member_id,
            vote=decision
        )
        print(f"\nVote cast: {decision} (weight: {vote['voting_weight']:.2%})")
        print(f"Current approval: {vote['current_results']['approval_percentage']:.1%}")
    
    # Get final results
    results = await mcp__ai_news_trader__syndicate_get_proposal_results(
        syndicate_id=syndicate_id,
        proposal_id=proposal['proposal_id']
    )
    
    print(f"\n=== Voting Results ===")
    print(f"Status: {results['status'].upper()}")
    print(f"Approval: {results['final_results']['approval_percentage']:.1%}")
    print(f"Participation: {results['final_results']['participation_rate']:.1%}")
    
    if results['status'] == 'passed':
        print(f"Implementation Date: {results['implementation_date']}")


async def example_4_intelligent_fund_allocation():
    """Example 4: AI-Driven Fund Allocation with Risk Management"""
    print("\n=== Example 4: Intelligent Fund Allocation ===\n")
    
    syndicate_id = "syn_20240626_001"
    
    # Multiple betting opportunities to evaluate
    opportunities = [
        {
            "sport": "NBA",
            "event": "Lakers vs Warriors - Game 5 Playoffs",
            "bet_type": "spread",
            "selection": "Lakers -3.5",
            "odds": 1.91,
            "probability": 0.58,  # 58% win probability
            "edge": 0.05,         # 5% expected edge
            "confidence": 0.82,   # High confidence
            "model_agreement": 0.88,  # Multiple models agree
            "time_until_event": timedelta(hours=24),
            "liquidity": 150000
        },
        {
            "sport": "NFL",
            "event": "Chiefs vs Bills - AFC Championship",
            "bet_type": "moneyline",
            "selection": "Chiefs ML",
            "odds": 2.15,
            "probability": 0.52,
            "edge": 0.062,
            "confidence": 0.75,
            "model_agreement": 0.79,
            "time_until_event": timedelta(hours=48),
            "liquidity": 250000
        },
        {
            "sport": "NBA",
            "event": "Celtics vs Heat",
            "bet_type": "total",
            "selection": "Over 215.5",
            "odds": 1.87,
            "probability": 0.56,
            "edge": 0.023,        # Lower edge
            "confidence": 0.65,   # Medium confidence
            "model_agreement": 0.62,
            "time_until_event": timedelta(hours=6),
            "liquidity": 75000
        },
        {
            "sport": "NFL",
            "event": "Ravens vs Steelers",
            "bet_type": "spread",
            "selection": "Ravens -7.5",
            "odds": 2.05,
            "probability": 0.45,  # Negative EV
            "edge": -0.03,
            "confidence": 0.55,
            "model_agreement": 0.48,
            "time_until_event": timedelta(hours=72),
            "liquidity": 100000
        }
    ]
    
    # Check current positions first
    positions = await mcp__ai_news_trader__syndicate_get_positions(
        syndicate_id=syndicate_id
    )
    
    print(f"Current Exposure: ${positions['total_exposure']:,} ({positions['exposure_percentage']:.1%})")
    print(f"Active Positions: {len(positions['active_positions'])}\n")
    
    # Evaluate each opportunity
    for opp in opportunities:
        print(f"\n{opp['sport']} - {opp['event']}")
        print(f"Selection: {opp['selection']} @ {opp['odds']}")
        print(f"Edge: {opp['edge']:.1%}, Confidence: {opp['confidence']:.0%}")
        
        # Get allocation recommendation
        allocation = await mcp__ai_news_trader__syndicate_allocate_funds(
            syndicate_id=syndicate_id,
            opportunity=opp,
            strategy="kelly_criterion"  # Can also use "fixed_percentage", "dynamic_confidence"
        )
        
        print(f"Recommended: ${allocation['recommended_amount']:,} ({allocation['percentage_of_bankroll']:.2%})")
        
        if allocation['warnings']:
            print(f"⚠️  Warnings: {', '.join(allocation['warnings'])}")
        
        if allocation['approval_required']:
            print("🔒 Requires lead investor approval")
        
        # Execute high-confidence bets
        if opp['confidence'] >= 0.75 and allocation['risk_assessment']['within_limits']:
            bet = await mcp__ai_news_trader__syndicate_execute_bet(
                syndicate_id=syndicate_id,
                allocation_id=allocation['allocation_id'],
                approved_by="mem_20240626_001"  # Lead investor
            )
            print(f"✅ BET PLACED: ${bet['amount']:,} - Potential return: ${bet['potential_return']:,}")


async def example_5_performance_tracking():
    """Example 5: Comprehensive Performance Tracking and Analytics"""
    print("\n=== Example 5: Performance Tracking ===\n")
    
    syndicate_id = "syn_20240626_001"
    
    # Get individual member performance
    members = ["mem_20240626_002", "mem_20240626_003", "mem_20240626_004"]
    
    for member_id in members:
        performance = await mcp__ai_news_trader__syndicate_member_performance(
            syndicate_id=syndicate_id,
            member_id=member_id,
            period_days=30
        )
        
        print(f"\n{performance['member_name']} - 30 Day Performance:")
        print(f"  Win Rate: {performance['betting_performance']['win_rate']:.1%}")
        print(f"  ROI: {performance['betting_performance']['roi']:.1%}")
        print(f"  Total P&L: ${performance['betting_performance']['profit_loss']:,}")
        print(f"  Alpha: {performance['skill_metrics']['alpha']:.3f}")
        print(f"  Consistency: {performance['skill_metrics']['consistency_score']:.2f}")
        print(f"  Overall Rank: #{performance['ranking']['overall_rank']}")
    
    # Generate syndicate-wide report
    report = await mcp__ai_news_trader__syndicate_performance_report(
        syndicate_id=syndicate_id,
        period="monthly",
        include_member_details=True
    )
    
    print(f"\n=== Syndicate Monthly Report ===")
    print(f"Total Return: {report['summary']['total_return']:.1%}")
    print(f"Total Profit: ${report['summary']['total_profit']:,}")
    print(f"Win Rate: {report['summary']['win_rate']:.1%}")
    print(f"Sharpe Ratio: {report['summary']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {report['summary']['max_drawdown']:.1%}")
    
    print(f"\nSport Performance:")
    for sport, stats in report['sport_breakdown'].items():
        print(f"  {sport}: ${stats['profit']:,} ({stats['roi']:.1%} ROI)")
    
    print(f"\nRisk Assessment:")
    print(f"  VaR (95%): {report['risk_metrics']['var_95']:.1%}")
    print(f"  Expected Shortfall: {report['risk_metrics']['expected_shortfall']:.1%}")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")


async def example_6_profit_distribution():
    """Example 6: Fair and Transparent Profit Distribution"""
    print("\n=== Example 6: Profit Distribution ===\n")
    
    syndicate_id = "syn_20240626_001"
    
    # Calculate weekly profit distribution
    weekly_profit = 85000  # $85,000 profit this week
    
    # Try different distribution models
    models = ["hybrid", "proportional", "performance_weighted", "tiered"]
    
    for model in models:
        print(f"\n--- {model.upper()} Distribution Model ---")
        
        distribution = await mcp__ai_news_trader__syndicate_calculate_distribution(
            syndicate_id=syndicate_id,
            total_profit=weekly_profit,
            distribution_model=model,
            period="weekly"
        )
        
        # Show top 3 distributions
        for i, (member_id, details) in enumerate(list(distribution['distributions'].items())[:3]):
            print(f"\n{details['member_name']}:")
            print(f"  Gross: ${details['gross_amount']:,}")
            print(f"  Tax: ${details['tax_withheld']:,} ({details['tax_rate']:.0%})")
            print(f"  Net: ${details['net_amount']:,}")
            
            if 'calculation_breakdown' in details:
                print(f"  Breakdown:")
                for component, amount in details['calculation_breakdown'].items():
                    print(f"    {component}: ${amount:,}")
    
    # Process the hybrid model distribution
    print(f"\n=== Processing Hybrid Model Distribution ===")
    
    result = await mcp__ai_news_trader__syndicate_process_distribution(
        syndicate_id=syndicate_id,
        distribution_id=distribution['distribution_id'],
        authorized_by="mem_20240626_001"  # Lead investor
    )
    
    print(f"Status: {result['status']}")
    print(f"Total Distributed: ${result['summary']['total_distributed']:,}")
    print(f"Tax Withheld: ${result['summary']['total_tax_withheld']:,}")
    print(f"Successful Transactions: {result['summary']['successful_transactions']}")


async def example_7_risk_management_scenario():
    """Example 7: Advanced Risk Management in Action"""
    print("\n=== Example 7: Risk Management Scenario ===\n")
    
    syndicate_id = "syn_20240626_001"
    
    # Simulate a high-risk scenario
    print("Scenario: Multiple large betting opportunities on the same day")
    
    # High-value opportunities
    opportunities = [
        {
            "sport": "NBA", 
            "event": "Finals Game 7",
            "odds": 2.20, 
            "probability": 0.55, 
            "edge": 0.10,
            "confidence": 0.85,
            "requested_amount": 50000  # Large bet
        },
        {
            "sport": "NBA",
            "event": "Conference Finals",
            "odds": 1.95,
            "probability": 0.54,
            "edge": 0.04,
            "confidence": 0.75,
            "requested_amount": 40000
        },
        {
            "sport": "NFL",
            "event": "Super Bowl",
            "odds": 2.50,
            "probability": 0.48,
            "edge": 0.08,
            "confidence": 0.80,
            "requested_amount": 60000
        }
    ]
    
    running_exposure = 0
    approved_bets = []
    rejected_bets = []
    
    for opp in opportunities:
        # Check if bet would exceed limits
        allocation = await mcp__ai_news_trader__syndicate_allocate_funds(
            syndicate_id=syndicate_id,
            opportunity=opp
        )
        
        print(f"\n{opp['event']} - Requested: ${opp['requested_amount']:,}")
        print(f"Kelly Recommendation: ${allocation['recommended_amount']:,}")
        
        if allocation['risk_assessment']['within_limits']:
            # Simulate execution
            running_exposure += allocation['recommended_amount']
            approved_bets.append({
                "event": opp['event'],
                "amount": allocation['recommended_amount'],
                "exposure_after": running_exposure
            })
            print(f"✅ APPROVED - Exposure after: ${running_exposure:,}")
        else:
            rejected_bets.append({
                "event": opp['event'],
                "reason": allocation['warnings'][0] if allocation['warnings'] else "Risk limit exceeded"
            })
            print(f"❌ REJECTED - {allocation['warnings'][0]}")
    
    print(f"\n=== Risk Management Summary ===")
    print(f"Approved Bets: {len(approved_bets)}")
    print(f"Rejected Bets: {len(rejected_bets)}")
    print(f"Total Exposure: ${running_exposure:,}")
    print(f"Risk Compliance: {'✅ Within Limits' if running_exposure < 200000 else '❌ Overleveraged'}")


async def example_8_member_withdrawal():
    """Example 8: Managing Member Withdrawals"""
    print("\n=== Example 8: Member Withdrawal Process ===\n")
    
    syndicate_id = "syn_20240626_001"
    member_id = "mem_20240626_005"  # Contributing member
    
    # Regular withdrawal request
    withdrawal = await mcp__ai_news_trader__syndicate_request_withdrawal(
        syndicate_id=syndicate_id,
        member_id=member_id,
        amount=25000,
        is_emergency=False
    )
    
    print(f"Regular Withdrawal Request:")
    print(f"  Requested: ${withdrawal['requested_amount']:,}")
    print(f"  Available Balance: ${withdrawal['available_balance']:,}")
    print(f"  Penalty: ${withdrawal['penalty']:,}")
    print(f"  Net Amount: ${withdrawal['net_amount']:,}")
    print(f"  Scheduled Date: {withdrawal['scheduled_date']}")
    print(f"  Status: {withdrawal['status']}")
    
    # Emergency withdrawal example
    emergency_withdrawal = await mcp__ai_news_trader__syndicate_request_withdrawal(
        syndicate_id=syndicate_id,
        member_id="mem_20240626_006",
        amount=30000,
        is_emergency=True
    )
    
    print(f"\nEmergency Withdrawal Request:")
    print(f"  Requested: ${emergency_withdrawal['requested_amount']:,}")
    print(f"  Penalty (10%): ${emergency_withdrawal['penalty']:,}")
    print(f"  Net Amount: ${emergency_withdrawal['net_amount']:,}")
    print(f"  Status: {emergency_withdrawal['status']}")


async def example_9_multi_syndicate_comparison():
    """Example 9: Comparing Multiple Syndicates"""
    print("\n=== Example 9: Multi-Syndicate Comparison ===\n")
    
    # List all syndicates
    syndicates = await mcp__ai_news_trader__syndicate_list(
        status="active",
        min_capital=100000,
        sort_by="total_return"
    )
    
    print(f"Active Syndicates with >$100k capital:\n")
    
    for syn in syndicates['syndicates']:
        print(f"{syn['name']}:")
        print(f"  Capital: ${syn['total_capital']:,}")
        print(f"  Members: {syn['member_count']}")
        print(f"  Return: {syn['total_return']:.1%}")
        print(f"  Status: {syn['status']}")
        
        # Get detailed info for top syndicate
        if syn == syndicates['syndicates'][0]:
            info = await mcp__ai_news_trader__syndicate_get_info(
                syndicate_id=syn['syndicate_id']
            )
            
            print(f"\nTop Performer Details:")
            print(f"  Sharpe Ratio: {info['performance']['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {info['performance']['max_drawdown']:.1%}")
            print(f"  Active Positions: {info['active_positions']}")


async def example_10_claude_conversation_flow():
    """Example 10: Natural Claude Conversation Flow"""
    print("\n=== Example 10: Claude Conversation Flow ===\n")
    
    print("User: I want to create a sports betting syndicate with my friends. We have about $250k to start.")
    print("\nClaude: I'll help you create a professional sports betting syndicate. Let me set that up for you.\n")
    
    # Create syndicate
    syndicate = await mcp__ai_news_trader__syndicate_create(
        name="Friends Sports Investment Group",
        description="Collaborative sports betting syndicate focused on data-driven strategies",
        initial_capital=250000,
        allocation_strategy="kelly_criterion"
    )
    
    print(f"I've created your syndicate: {syndicate['name']}")
    print(f"Syndicate ID: {syndicate['syndicate_id']}")
    
    print("\nUser: Great! Can you add me as the lead and my two friends as analysts?")
    print("\nClaude: Of course! I'll add you as the lead investor and your friends as analysts.\n")
    
    # Add members
    members_data = [
        ("You", "lead@syndicate.com", "lead_investor", 100000),
        ("Friend 1", "friend1@syndicate.com", "senior_analyst", 75000),
        ("Friend 2", "friend2@syndicate.com", "senior_analyst", 75000)
    ]
    
    for name, email, role, amount in members_data:
        member = await mcp__ai_news_trader__syndicate_add_member(
            syndicate_id=syndicate['syndicate_id'],
            name=name,
            email=email,
            role=role,
            contribution=amount
        )
        print(f"Added {name} as {role} with ${amount:,} contribution")
    
    print("\nUser: What NBA games should we bet on today?")
    print("\nClaude: Let me analyze today's NBA games and find the best opportunities for your syndicate.\n")
    
    # Simulate opportunity analysis
    opportunity = {
        "sport": "NBA",
        "event": "Lakers vs Celtics",
        "bet_type": "spread",
        "selection": "Lakers -3.5",
        "odds": 1.91,
        "probability": 0.58,
        "edge": 0.05,
        "confidence": 0.78
    }
    
    allocation = await mcp__ai_news_trader__syndicate_allocate_funds(
        syndicate_id=syndicate['syndicate_id'],
        opportunity=opportunity
    )
    
    print(f"I found a strong opportunity:")
    print(f"• {opportunity['event']}: {opportunity['selection']} @ {opportunity['odds']}")
    print(f"• Our models show a {opportunity['edge']:.1%} edge with {opportunity['confidence']:.0%} confidence")
    print(f"• Recommended bet: ${allocation['recommended_amount']:,} ({allocation['percentage_of_bankroll']:.1%} of bankroll)")
    print(f"\nThis bet is within your risk limits and doesn't require additional approval. Would you like to place it?")


async def main():
    """Run all examples"""
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║     AI News Trader - MCP Syndicate Tools Examples    ║
    ║     Comprehensive Guide to All 17 Syndicate Tools    ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Run examples sequentially
    syndicate_id = await example_1_create_professional_syndicate()
    await example_2_build_syndicate_team()
    await example_3_democratic_governance()
    await example_4_intelligent_fund_allocation()
    await example_5_performance_tracking()
    await example_6_profit_distribution()
    await example_7_risk_management_scenario()
    await example_8_member_withdrawal()
    await example_9_multi_syndicate_comparison()
    await example_10_claude_conversation_flow()
    
    print("\n\n✅ All examples completed successfully!")
    print("\nKey Takeaways:")
    print("• Syndicates enable collaborative investment with proper governance")
    print("• Risk management is built into every allocation decision")
    print("• Democratic voting ensures fair decision-making")
    print("• Performance tracking identifies top contributors")
    print("• Profit distribution is transparent and customizable")
    print("• The system scales from small groups to large investment pools")


if __name__ == "__main__":
    # Note: In real usage through Claude Code, these would be called directly
    # This example demonstrates the usage patterns
    asyncio.run(main())