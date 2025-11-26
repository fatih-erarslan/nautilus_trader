"""
Comprehensive Example Usage of Syndicate Management System

This file demonstrates how to use all the syndicate management components together:
- Capital Management
- Voting and Consensus
- Member Management
- Collaboration Tools
- Smart Contract Integration

This example shows a complete workflow from syndicate creation to betting operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal

from syndicate.capital_manager import CapitalManager, AllocationMethod
from syndicate.voting_system import VotingSystem, ProposalType, VotingMethod, VoteType
from syndicate.member_manager import MemberManager, MemberRole, ExpertiseArea
from syndicate.collaboration import CollaborationManager, ChannelType, DocumentType, MessageType
from syndicate.smart_contracts import SmartContractManager, ContractType, GovernanceAction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyndicateDemo:
    """
    Demonstration class showing complete syndicate management workflow
    """
    
    def __init__(self, syndicate_id: str = "DEMO_SYNDICATE_001"):
        self.syndicate_id = syndicate_id
        
        # Initialize all managers
        self.capital_manager = CapitalManager(pool_id=f"POOL_{syndicate_id}")
        self.voting_system = VotingSystem(syndicate_id=syndicate_id)
        self.member_manager = MemberManager(syndicate_id=syndicate_id)
        self.collaboration_manager = CollaborationManager(syndicate_id=syndicate_id)
        self.smart_contract_manager = SmartContractManager(syndicate_id=syndicate_id)
        
        # Demo data
        self.demo_members = {}
        self.demo_bets = []

    async def run_complete_demo(self):
        """Run a complete demonstration of all syndicate features"""
        logger.info("=== Starting Comprehensive Syndicate Management Demo ===")
        
        try:
            # 1. Setup and member onboarding
            await self.demo_member_onboarding()
            
            # 2. Capital management demonstration
            await self.demo_capital_management()
            
            # 3. Voting and governance demonstration
            await self.demo_voting_and_governance()
            
            # 4. Collaboration features demonstration
            await self.demo_collaboration_features()
            
            # 5. Smart contract integration demonstration
            await self.demo_smart_contract_integration()
            
            # 6. Full betting workflow demonstration
            await self.demo_betting_workflow()
            
            # 7. Generate comprehensive reports
            await self.generate_final_reports()
            
            logger.info("=== Demo Completed Successfully ===")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise

    async def demo_member_onboarding(self):
        """Demonstrate member registration and role assignment"""
        logger.info("\n--- Member Onboarding Demo ---")
        
        # Create founder
        founder_invitation = await self.member_manager.create_invitation(
            inviter_id="system",
            role=MemberRole.FOUNDER
        )
        
        founder_id = await self.member_manager.register_member(
            username="alice_founder",
            email="alice@syndicate.com",
            invitation_code=founder_invitation,
            full_name="Alice Johnson",
            bio="Experienced sports bettor and syndicate founder"
        )
        
        await self.member_manager.approve_member(founder_id, "system")
        self.demo_members["founder"] = founder_id
        
        # Add founder to capital and voting systems
        await self.capital_manager.add_member(founder_id, Decimal('50000'))
        await self.voting_system.register_voter(founder_id, Decimal('50000'), 
                                               expertise_score=9.5, performance_score=8.8)
        
        # Create other members
        members_to_create = [
            ("bob_analyst", "bob@syndicate.com", "Bob Smith", "Statistical analysis expert", MemberRole.ANALYST, Decimal('25000'), 8.5, 7.2),
            ("charlie_lead", "charlie@syndicate.com", "Charlie Brown", "Sports betting strategist", MemberRole.LEAD, Decimal('35000'), 8.0, 8.5),
            ("diana_contributor", "diana@syndicate.com", "Diana Wilson", "Football specialist", MemberRole.CONTRIBUTOR, Decimal('15000'), 7.0, 6.8),
            ("evan_analyst", "evan@syndicate.com", "Evan Davis", "Basketball expert", MemberRole.ANALYST, Decimal('20000'), 7.8, 7.5)
        ]
        
        for username, email, full_name, bio, role, capital, expertise, performance in members_to_create:
            # Create invitation
            invitation = await self.member_manager.create_invitation(
                inviter_id=founder_id,
                role=role
            )
            
            # Register member
            member_id = await self.member_manager.register_member(
                username=username,
                email=email,
                invitation_code=invitation,
                full_name=full_name,
                bio=bio
            )
            
            # Approve member
            await self.member_manager.approve_member(member_id, founder_id)
            
            # Add to capital system
            await self.capital_manager.add_member(member_id, capital)
            
            # Register as voter
            await self.voting_system.register_voter(member_id, capital, 
                                                   expertise_score=expertise, 
                                                   performance_score=performance)
            
            # Update expertise
            if "analyst" in username or "football" in bio.lower():
                await self.member_manager.update_expertise_score(
                    member_id, ExpertiseArea.FOOTBALL, 85.0, founder_id
                )
            if "basketball" in bio.lower():
                await self.member_manager.update_expertise_score(
                    member_id, ExpertiseArea.BASKETBALL, 90.0, founder_id
                )
            
            self.demo_members[username.split('_')[0]] = member_id
            
        logger.info(f"Onboarded {len(self.demo_members)} members successfully")

    async def demo_capital_management(self):
        """Demonstrate capital management features"""
        logger.info("\n--- Capital Management Demo ---")
        
        # Show initial pool status
        pool_summary = self.capital_manager.get_pool_summary()
        logger.info(f"Initial pool capital: ${pool_summary['total_capital']}")
        
        # Process additional deposits
        await self.capital_manager.deposit_funds(
            self.demo_members["bob"], Decimal('10000'), "Additional investment"
        )
        
        # Demonstrate different allocation methods
        bet_amount = Decimal('5000')
        
        # Test proportional allocation
        logger.info("Testing proportional capital allocation...")
        allocations = await self.capital_manager.allocate_capital_for_bet(
            bet_id="BET_001_DEMO", 
            required_amount=bet_amount,
            allocation_method=AllocationMethod.PROPORTIONAL
        )
        
        logger.info(f"Proportional allocations: {allocations}")
        
        # Simulate bet settlement (winning bet)
        await self.capital_manager.process_bet_settlement(
            bet_id="BET_001_DEMO",
            result="win",
            payout=Decimal('7500')  # 50% profit
        )
        
        # Test performance-weighted allocation
        logger.info("Testing performance-weighted allocation...")
        allocations = await self.capital_manager.allocate_capital_for_bet(
            bet_id="BET_002_DEMO",
            required_amount=bet_amount,
            allocation_method=AllocationMethod.PERFORMANCE_WEIGHTED
        )
        
        # Simulate bet settlement (losing bet)
        await self.capital_manager.process_bet_settlement(
            bet_id="BET_002_DEMO",
            result="loss",
            payout=Decimal('0')
        )
        
        # Rebalance pool
        await self.capital_manager.rebalance_pool()
        
        # Show updated pool summary
        updated_summary = self.capital_manager.get_pool_summary()
        logger.info(f"Updated pool capital: ${updated_summary['total_capital']}")
        logger.info(f"Pool ROI: {updated_summary['total_roi_percentage']:.2f}%")

    async def demo_voting_and_governance(self):
        """Demonstrate voting and governance features"""
        logger.info("\n--- Voting and Governance Demo ---")
        
        # Create a proposal for a large bet
        proposal_id = await self.voting_system.create_proposal(
            proposer_id=self.demo_members["charlie"],
            proposal_type=ProposalType.LARGE_BET,
            title="High-Stakes Super Bowl Bet",
            description="Proposal to allocate $25,000 on Super Bowl champion bet with 3.5 odds",
            details={
                "game": "Super Bowl LVIII",
                "bet_type": "moneyline",
                "team": "Kansas City Chiefs",
                "odds": 3.5,
                "amount": 25000,
                "expected_roi": 250,
                "risk_level": "high"
            }
        )
        
        # Start voting
        await self.voting_system.start_voting(proposal_id)
        
        # Members cast votes
        votes = [
            (self.demo_members["founder"], VoteType.YES, "Strong team, good value odds"),
            (self.demo_members["bob"], VoteType.YES, "Statistical analysis supports this bet"),
            (self.demo_members["charlie"], VoteType.YES, "Proposer vote - confident in analysis"),
            (self.demo_members["diana"], VoteType.NO, "Too much risk for single bet"),
            (self.demo_members["evan"], VoteType.ABSTAIN, "Need more analysis time")
        ]
        
        for voter_id, vote_type, rationale in votes:
            await self.voting_system.cast_vote(
                proposal_id=proposal_id,
                voter_id=voter_id,
                vote_type=vote_type,
                rationale=rationale
            )
        
        # Create a strategy change proposal
        strategy_proposal_id = await self.voting_system.create_proposal(
            proposer_id=self.demo_members["bob"],
            proposal_type=ProposalType.STRATEGY_CHANGE,
            title="Implement Advanced ML Betting Strategy",
            description="Proposal to integrate machine learning algorithms for bet selection",
            details={
                "strategy_name": "ML-Enhanced Analysis",
                "implementation_cost": 15000,
                "expected_improvement": "15-20% ROI increase",
                "timeline": "3 months",
                "risk_assessment": "medium"
            }
        )
        
        await self.voting_system.start_voting(strategy_proposal_id)
        
        # Show voting results
        proposal_summary = self.voting_system.get_proposal_summary(proposal_id)
        logger.info(f"Large bet proposal status: {proposal_summary['status']}")
        logger.info(f"Participation rate: {proposal_summary['voting_statistics']['participation_rate']:.1%}")

    async def demo_collaboration_features(self):
        """Demonstrate collaboration and communication features"""
        logger.info("\n--- Collaboration Features Demo ---")
        
        # Create specialized channels
        research_channel = await self.collaboration_manager.create_channel(
            creator_id=self.demo_members["bob"],
            name="nfl-analysis",
            description="NFL game analysis and research",
            channel_type=ChannelType.RESEARCH
        )
        
        strategy_channel = await self.collaboration_manager.create_channel(
            creator_id=self.demo_members["charlie"],
            name="betting-strategy",
            description="Strategy development and optimization",
            channel_type=ChannelType.STRATEGY
        )
        
        # Send messages with analysis
        await self.collaboration_manager.send_message(
            channel_id=research_channel,
            sender_id=self.demo_members["bob"],
            content="Latest injury report shows Chiefs QB is 100% healthy. This significantly improves our Super Bowl bet odds.",
            message_type=MessageType.ANALYSIS
        )
        
        await self.collaboration_manager.send_message(
            channel_id=strategy_channel,
            sender_id=self.demo_members["charlie"],
            content="Proposing Kelly Criterion implementation for optimal bet sizing. @alice @bob thoughts?",
            message_type=MessageType.TEXT
        )
        
        # Create collaborative documents
        game_analysis_doc = await self.collaboration_manager.create_document(
            creator_id=self.demo_members["diana"],
            title="Super Bowl LVIII Complete Analysis",
            content="""
# Super Bowl LVIII Analysis

## Team Comparison
- Kansas City Chiefs: 14-3 regular season, strong offense
- Philadelphia Eagles: 11-6 regular season, improved defense

## Key Factors
1. Quarterback performance under pressure
2. Injury reports and player availability
3. Historical Super Bowl performance
4. Weather conditions (dome game)

## Betting Recommendation
Based on comprehensive analysis, recommend Chiefs moneyline at +120 odds.
Expected value calculation shows 15% positive EV.

## Risk Assessment
- Medium-high risk due to playoff volatility
- Recommend 3-5% of bankroll allocation
            """,
            document_type=DocumentType.GAME_PREVIEW,
            collaborators={self.demo_members["bob"], self.demo_members["charlie"]}
        )
        
        # Edit document collaboratively
        await self.collaboration_manager.edit_document(
            document_id=game_analysis_doc,
            editor_id=self.demo_members["bob"],
            new_content="""
# Super Bowl LVIII Analysis

## Team Comparison
- Kansas City Chiefs: 14-3 regular season, strong offense, elite playoff experience
- Philadelphia Eagles: 11-6 regular season, improved defense, young core

## Key Factors
1. Quarterback performance under pressure (UPDATED: Mahomes 8-2 in playoffs)
2. Injury reports and player availability
3. Historical Super Bowl performance
4. Weather conditions (dome game)
5. **NEW**: Coaching matchups favor Chiefs

## Statistical Analysis
- Chiefs cover spread 68% in playoff games
- Eagles struggle against mobile QBs (Chiefs advantage)
- O/U trends suggest UNDER 47.5 points

## Betting Recommendation
Based on comprehensive analysis, recommend Chiefs moneyline at +120 odds.
Expected value calculation shows 18% positive EV (updated with new data).

## Risk Assessment
- Medium-high risk due to playoff volatility
- Recommend 4-6% of bankroll allocation (increased confidence)
            """,
            summary="Added statistical analysis and coaching matchup insights"
        )
        
        # Create a research project
        ml_project = await self.collaboration_manager.create_project(
            lead_id=self.demo_members["bob"],
            name="ML Betting Algorithm Development",
            description="Develop machine learning algorithms for automated bet analysis",
            project_type="research_development",
            deadline=datetime.now() + timedelta(days=90)
        )
        
        # Add project tasks
        tasks = [
            "Data collection and preprocessing",
            "Feature engineering for sports metrics",
            "Model training and validation",
            "Backtesting on historical data",
            "Integration with existing systems"
        ]
        
        for task_name in tasks:
            await self.collaboration_manager.add_project_task(
                project_id=ml_project,
                task_name=task_name,
                description=f"Complete {task_name.lower()} phase",
                assigned_to=self.demo_members["bob"] if "training" in task_name.lower() else None,
                due_date=datetime.now() + timedelta(days=30)
            )
        
        # Add knowledge base items
        await self.collaboration_manager.add_knowledge_item(
            creator_id=self.demo_members["charlie"],
            title="Kelly Criterion for Optimal Bet Sizing",
            content="""
The Kelly Criterion is a mathematical formula for determining optimal bet size:

f = (bp - q) / b

Where:
- f = fraction of bankroll to bet
- b = odds received (decimal odds - 1)
- p = probability of winning
- q = probability of losing (1-p)

Example:
If you believe a team has 60% chance to win at +150 odds:
f = (1.5 × 0.6 - 0.4) / 1.5 = 0.327

Bet 32.7% of bankroll for optimal growth.

## Advantages
- Maximizes long-term growth
- Prevents overbetting
- Mathematically optimal

## Limitations
- Requires accurate probability estimates
- Can suggest large bets with high confidence
- Doesn't account for risk tolerance
            """,
            category="strategy",
            subcategory="bet_sizing"
        )
        
        logger.info("Created collaboration channels, documents, and knowledge base items")

    async def demo_smart_contract_integration(self):
        """Demonstrate smart contract and blockchain integration"""
        logger.info("\n--- Smart Contract Integration Demo ---")
        
        # Create escrow for the large bet
        escrow_participants = {
            self.demo_members["founder"],
            self.demo_members["charlie"],
            self.demo_members["bob"]
        }
        
        escrow_amounts = {
            self.demo_members["founder"]: Decimal('10000'),
            self.demo_members["charlie"]: Decimal('8000'),
            self.demo_members["bob"]: Decimal('7000')
        }
        
        release_conditions = [
            {
                "condition_id": "game_completion",
                "description": "Super Bowl game completed",
                "auto_trigger": True
            },
            {
                "condition_id": "bet_settlement",
                "description": "Sportsbook confirms bet result",
                "verification_required": True
            }
        ]
        
        escrow_id = await self.smart_contract_manager.create_escrow(
            creator_id=self.demo_members["founder"],
            participants=escrow_participants,
            amounts=escrow_amounts,
            release_conditions=release_conditions,
            expiry_days=14
        )
        
        # Execute governance action through smart contract
        governance_action = await self.smart_contract_manager.execute_governance_action(
            action=GovernanceAction.PROFIT_DISTRIBUTION,
            parameters={
                "distribution_method": "proportional",
                "profit_amount": 12500,
                "minimum_stake": 1000,
                "initiator_stake": 50000
            },
            initiator_id=self.demo_members["founder"]
        )
        
        # Create dispute (example scenario)
        dispute_id = await self.smart_contract_manager.create_dispute(
            initiator_id=self.demo_members["diana"],
            respondent_id=self.demo_members["charlie"],
            dispute_type="bet_allocation",
            title="Unfair Bet Allocation Dispute",
            description="Claim that bet allocation algorithm unfairly reduced my participation in winning bet",
            amount_disputed=Decimal('2500')
        )
        
        # Resolve dispute
        await self.smart_contract_manager.resolve_dispute(
            dispute_id=dispute_id,
            resolver_id=self.demo_members["founder"],
            resolution="Reviewed allocation algorithm. Minor adjustment needed. Compensating 500 from syndicate reserves.",
            compensation_amount=Decimal('500')
        )
        
        logger.info(f"Created escrow {escrow_id} and processed governance actions")

    async def demo_betting_workflow(self):
        """Demonstrate complete betting workflow"""
        logger.info("\n--- Complete Betting Workflow Demo ---")
        
        # 1. Research and analysis (collaboration)
        research_doc = await self.collaboration_manager.create_document(
            creator_id=self.demo_members["bob"],
            title="NBA Finals Game 4 Analysis",
            content="Comprehensive analysis suggesting Lakers +4.5 has value",
            document_type=DocumentType.BET_ANALYSIS
        )
        
        # 2. Create betting proposal (voting)
        bet_proposal = await self.voting_system.create_proposal(
            proposer_id=self.demo_members["bob"],
            proposal_type=ProposalType.LARGE_BET,
            title="NBA Finals Game 4 - Lakers +4.5",
            description="High-confidence bet based on statistical analysis",
            details={
                "game": "NBA Finals Game 4",
                "bet_type": "spread",
                "team": "Los Angeles Lakers",
                "line": "+4.5",
                "odds": -110,
                "amount": 8000,
                "confidence": 85
            }
        )
        
        await self.voting_system.start_voting(bet_proposal)
        
        # 3. Vote on proposal
        for member_name, member_id in self.demo_members.items():
            vote = VoteType.YES if member_name in ["founder", "bob", "charlie"] else VoteType.NO
            await self.voting_system.cast_vote(
                proposal_id=bet_proposal,
                voter_id=member_id,
                vote_type=vote,
                rationale=f"Vote from {member_name}"
            )
        
        # 4. Allocate capital (capital management)
        if True:  # Simulate proposal passed
            allocations = await self.capital_manager.allocate_capital_for_bet(
                bet_id="NBA_FINALS_G4",
                required_amount=Decimal('8000'),
                allocation_method=AllocationMethod.HYBRID
            )
            
            logger.info(f"Allocated capital for NBA bet: {allocations}")
        
        # 5. Execute bet through smart contract
        bet_execution_tx = await self.smart_contract_manager.execute_governance_action(
            action=GovernanceAction.CAPITAL_REALLOCATION,
            parameters={
                "bet_id": "NBA_FINALS_G4",
                "amount": 8000,
                "allocations": {k: float(v) for k, v in allocations.items()}
            },
            initiator_id=self.demo_members["bob"]
        )
        
        # 6. Simulate bet result and settlement
        await asyncio.sleep(1)  # Simulate game completion
        
        # Winning bet - Lakers covered the spread
        await self.capital_manager.process_bet_settlement(
            bet_id="NBA_FINALS_G4",
            result="win",
            payout=Decimal('15272')  # 8000 * (1 + 110/100) = 8000 * 1.909
        )
        
        # 7. Update member performance
        for member_id in allocations.keys():
            if member_id in self.demo_members.values():
                member_allocation = allocations[member_id]
                member_payout = member_allocation * Decimal('1.909')
                
                await self.member_manager.update_performance(
                    member_id=member_id,
                    bet_result={
                        "stake": float(member_allocation),
                        "payout": float(member_payout),
                        "won": True,
                        "odds": 1.909
                    }
                )
        
        # 8. Distribute profits through smart contract
        profit_distribution_tx = await self.smart_contract_manager.execute_governance_action(
            action=GovernanceAction.PROFIT_DISTRIBUTION,
            parameters={
                "total_profit": float(Decimal('15272') - Decimal('8000')),
                "distribution_method": "proportional"
            },
            initiator_id=self.demo_members["founder"]
        )
        
        logger.info("Completed full betting workflow from analysis to profit distribution")

    async def generate_final_reports(self):
        """Generate comprehensive reports from all systems"""
        logger.info("\n--- Final System Reports ---")
        
        # Capital management report
        capital_report = self.capital_manager.export_capital_report()
        logger.info(f"Final pool value: ${capital_report['pool_summary']['total_capital']}")
        logger.info(f"Total ROI: {capital_report['pool_summary']['total_roi_percentage']:.2f}%")
        
        # Member analytics
        member_analytics = self.member_manager.get_syndicate_analytics()
        logger.info(f"Active members: {member_analytics['active_members']}")
        logger.info(f"Average member ROI: {member_analytics['performance_summary']['average_roi']:.2f}%")
        
        # Voting analytics
        voting_analytics = self.voting_system.get_governance_analytics()
        logger.info(f"Total proposals: {voting_analytics['total_proposals']}")
        logger.info(f"Average participation: {voting_analytics['average_participation_rate']:.1%}")
        
        # Collaboration analytics
        collab_analytics = self.collaboration_manager.get_collaboration_analytics()
        logger.info(f"Total documents: {collab_analytics['documents']['total_documents']}")
        logger.info(f"Active projects: {collab_analytics['projects']['projects_by_status'].get('active', 0)}")
        
        # Blockchain summary
        blockchain_summary = self.smart_contract_manager.get_syndicate_blockchain_summary()
        logger.info(f"Smart contracts deployed: {blockchain_summary['contracts']['total_contracts']}")
        logger.info(f"Total transactions: {blockchain_summary['transactions']['total_transactions']}")
        
        # Generate member rankings
        top_performers = self.member_manager.get_member_rankings(sort_by="roi", limit=5)
        logger.info("\nTop Performers by ROI:")
        for rank, performer in enumerate(top_performers, 1):
            logger.info(f"{rank}. {performer['username']}: {performer['metric_value']:.2f}%")
        
        return {
            "capital_report": capital_report,
            "member_analytics": member_analytics,
            "voting_analytics": voting_analytics,
            "collaboration_analytics": collab_analytics,
            "blockchain_summary": blockchain_summary,
            "top_performers": top_performers
        }


async def main():
    """Run the complete syndicate management demo"""
    demo = SyndicateDemo()
    
    try:
        reports = await demo.run_complete_demo()
        
        print("\n" + "="*60)
        print("SYNDICATE MANAGEMENT SYSTEM DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Print summary statistics
        print(f"\nSYNDICATE STATISTICS:")
        print(f"├── Pool Value: ${reports['capital_report']['pool_summary']['total_capital']}")
        print(f"├── Total Members: {reports['member_analytics']['total_members']}")
        print(f"├── Active Proposals: {reports['voting_analytics']['active_proposals']}")
        print(f"├── Documents Created: {reports['collaboration_analytics']['documents']['total_documents']}")
        print(f"├── Smart Contracts: {reports['blockchain_summary']['contracts']['total_contracts']}")
        print(f"└── Average ROI: {reports['member_analytics']['performance_summary']['average_roi']:.2f}%")
        
        print(f"\nTOP PERFORMERS:")
        for performer in reports['top_performers'][:3]:
            print(f"  {performer['rank']}. {performer['username']}: {performer['metric_value']:.2f}% ROI")
        
        print(f"\nAll systems integrated successfully! 🎉")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())