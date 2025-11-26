"""
Integration Tests for Syndicate MCP Tools with Trading System

This module tests the integration between syndicate management tools
and the existing trading system, including:
- Syndicate-based trading workflows
- Integration with sports betting APIs
- Multi-user trading scenarios
- Real-time collaboration during trades
"""

import pytest
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import json

# Import required components
from src.sports_betting.syndicate.capital_manager import CapitalManager
from src.sports_betting.syndicate.voting_system import VotingSystem, ProposalType, VoteType
from src.sports_betting.syndicate.member_manager import MemberManager, MemberRole


@pytest.mark.integration
class TestSyndicateTradingIntegration:
    """Test syndicate integration with trading tools"""
    
    @pytest.mark.asyncio
    async def test_syndicate_sports_betting_workflow(self, mock_mcp_server, mock_members, syndicate_test_config):
        """Test complete workflow from syndicate proposal to bet execution"""
        # Step 1: Member creates analysis using trading tools
        analysis_result = await mock_mcp_server.call_tool(
            "analyze_betting_market_depth",
            {
                "market_id": "NFL_SUPERBOWL_2025",
                "sport": "NFL",
                "use_gpu": True
            }
        )
        
        # Step 2: Create betting proposal based on analysis
        proposal_result = await mock_mcp_server.call_tool(
            "syndicate_create_proposal",
            {
                "proposer_id": mock_members[1]["member_id"],  # Lead member
                "type": ProposalType.LARGE_BET.value,
                "title": "Super Bowl 2025 - Chiefs ML",
                "description": f"Based on market analysis showing value at current odds",
                "details": {
                    "market_id": "NFL_SUPERBOWL_2025",
                    "bet_type": "moneyline",
                    "team": "Kansas City Chiefs",
                    "odds": 2.5,
                    "recommended_stake": 25000,
                    "analysis_confidence": 0.85,
                    "expected_value": 1.15
                }
            }
        )
        
        # Step 3: Members vote on proposal
        votes = []
        for member in mock_members[:4]:  # Voting members
            vote_result = await mock_mcp_server.call_tool(
                "syndicate_cast_vote",
                {
                    "proposal_id": proposal_result["proposal_id"],
                    "voter_id": member["member_id"],
                    "vote": VoteType.YES.value if member["performance_score"] > 0.7 else VoteType.NO.value,
                    "rationale": "Based on performance history"
                }
            )
            votes.append(vote_result)
        
        # Step 4: Tally votes and check approval
        tally_result = await mock_mcp_server.call_tool(
            "syndicate_tally_votes",
            {
                "proposal_id": proposal_result["proposal_id"],
                "close_voting": True
            }
        )
        
        assert tally_result["outcome"] in ["approved", "rejected"]
        
        if tally_result["outcome"] == "approved":
            # Step 5: Allocate syndicate funds
            allocation_result = await mock_mcp_server.call_tool(
                "syndicate_allocate_funds",
                {
                    "bet_id": f"BET_FROM_{proposal_result['proposal_id']}",
                    "total_amount": 25000,
                    "allocation_method": "hybrid",
                    "participating_members": [m["member_id"] for m in mock_members[:4]]
                }
            )
            
            # Step 6: Execute sports bet using allocated funds
            bet_result = await mock_mcp_server.call_tool(
                "execute_sports_bet",
                {
                    "market_id": "NFL_SUPERBOWL_2025",
                    "selection": "Kansas City Chiefs",
                    "stake": 25000,
                    "odds": 2.5,
                    "bet_type": "back",
                    "validate_only": False,
                    "syndicate_id": syndicate_test_config["syndicate_id"],
                    "allocations": allocation_result["allocations"]
                }
            )
            
            # Step 7: Track bet in syndicate system
            tracking_result = await mock_mcp_server.call_tool(
                "syndicate_bet_history",
                {
                    "bet_ids": [bet_result["bet_id"]],
                    "include_live_status": True
                }
            )
            
            assert tracking_result["bets"][0]["status"] == "active"
            assert tracking_result["bets"][0]["syndicate_id"] == syndicate_test_config["syndicate_id"]
    
    @pytest.mark.asyncio
    async def test_multi_syndicate_arbitrage_coordination(self, mock_mcp_server, syndicate_test_config):
        """Test multiple syndicates coordinating on arbitrage opportunities"""
        # Create two test syndicates
        syndicates = []
        for i in range(2):
            syndicate_id = f"SYNDICATE_{i+1:03d}"
            syndicate = {
                "syndicate_id": syndicate_id,
                "name": f"Test Syndicate {i+1}",
                "capital": Decimal("50000"),
                "members": []
            }
            
            # Add members to each syndicate
            for j in range(3):
                member = await mock_mcp_server.call_tool(
                    "syndicate_create_member",
                    {
                        "syndicate_id": syndicate_id,
                        "username": f"syndicate{i+1}_member{j+1}",
                        "email": f"s{i+1}m{j+1}@test.com",
                        "role": MemberRole.LEAD.value if j == 0 else MemberRole.CONTRIBUTOR.value,
                        "initial_contribution": 15000 if j == 0 else 10000
                    }
                )
                syndicate["members"].append(member)
            
            syndicates.append(syndicate)
        
        # Find arbitrage opportunity
        arb_result = await mock_mcp_server.call_tool(
            "find_sports_arbitrage",
            {
                "sport": "NBA",
                "min_profit_margin": 0.02,
                "use_gpu": True
            }
        )
        
        if arb_result.get("opportunities"):
            opportunity = arb_result["opportunities"][0]
            
            # Each syndicate creates proposal for their side of arbitrage
            for i, syndicate in enumerate(syndicates):
                side = "back" if i == 0 else "lay"
                
                # Create proposal in syndicate
                proposal = await mock_mcp_server.call_tool(
                    "syndicate_create_proposal",
                    {
                        "syndicate_id": syndicate["syndicate_id"],
                        "proposer_id": syndicate["members"][0]["member_id"],
                        "type": ProposalType.LARGE_BET.value,
                        "title": f"Arbitrage Opportunity - {side.upper()}",
                        "description": f"Guaranteed profit opportunity: {opportunity['profit_margin']:.2%}",
                        "details": {
                            "opportunity_id": opportunity["id"],
                            "side": side,
                            "stake_required": opportunity[f"{side}_stake"],
                            "guaranteed_profit": opportunity["guaranteed_profit"]
                        }
                    }
                )
                
                # Fast-track voting for arbitrage
                for member in syndicate["members"]:
                    await mock_mcp_server.call_tool(
                        "syndicate_cast_vote",
                        {
                            "proposal_id": proposal["proposal_id"],
                            "voter_id": member["member_id"],
                            "vote": VoteType.YES.value,
                            "rationale": "Arbitrage opportunity"
                        }
                    )
                
                # Execute syndicate's side
                await mock_mcp_server.call_tool(
                    "syndicate_allocate_funds",
                    {
                        "bet_id": f"ARB_{opportunity['id']}_{side}",
                        "total_amount": opportunity[f"{side}_stake"],
                        "allocation_method": "proportional",
                        "urgent": True  # Fast allocation for arbitrage
                    }
                )
    
    @pytest.mark.asyncio
    async def test_syndicate_neural_forecast_integration(self, mock_mcp_server, mock_members):
        """Test syndicate using neural forecasting for decisions"""
        # Step 1: Generate neural forecast
        forecast_result = await mock_mcp_server.call_tool(
            "neural_forecast",
            {
                "symbol": "NBA_LAKERS_WINS",
                "horizon": 10,  # Next 10 games
                "confidence_level": 0.95,
                "use_gpu": True
            }
        )
        
        # Step 2: Analyze forecast for betting opportunities
        high_confidence_games = [
            game for game in forecast_result["predictions"]
            if game["confidence"] > 0.8 and game["expected_value"] > 1.1
        ]
        
        # Step 3: Create batch proposal for multiple bets
        if high_confidence_games:
            batch_proposal = await mock_mcp_server.call_tool(
                "syndicate_create_proposal",
                {
                    "proposer_id": mock_members[1]["member_id"],  # Analyst
                    "type": ProposalType.STRATEGY_CHANGE.value,
                    "title": "ML-Driven Betting Strategy for Lakers Games",
                    "description": f"Neural model identified {len(high_confidence_games)} high-value opportunities",
                    "details": {
                        "strategy": "neural_forecast_based",
                        "games": high_confidence_games,
                        "total_allocation": len(high_confidence_games) * 5000,
                        "model_confidence": forecast_result["model_metrics"]["accuracy"],
                        "backtested_roi": forecast_result["model_metrics"]["historical_roi"]
                    }
                }
            )
            
            # Automated voting based on model confidence
            for member in mock_members[:4]:
                vote = VoteType.YES if forecast_result["model_metrics"]["accuracy"] > 0.75 else VoteType.ABSTAIN
                await mock_mcp_server.call_tool(
                    "syndicate_cast_vote",
                    {
                        "proposal_id": batch_proposal["proposal_id"],
                        "voter_id": member["member_id"],
                        "vote": vote.value,
                        "rationale": f"Model accuracy: {forecast_result['model_metrics']['accuracy']:.2%}"
                    }
                )
    
    @pytest.mark.asyncio
    async def test_syndicate_risk_management_integration(self, mock_mcp_server, mock_members, syndicate_test_config):
        """Test syndicate risk management with trading limits"""
        # Get current syndicate exposure
        current_positions = await mock_mcp_server.call_tool(
            "get_betting_portfolio_status",
            {
                "include_risk_analysis": True,
                "syndicate_filter": syndicate_test_config["syndicate_id"]
            }
        )
        
        # Calculate available risk budget
        total_capital = sum(Decimal(str(m["capital_contribution"])) for m in mock_members)
        max_risk = total_capital * Decimal(str(syndicate_test_config["max_bet_percentage"]))
        current_exposure = Decimal(str(current_positions.get("total_exposure", 0)))
        available_budget = max_risk - current_exposure
        
        # Propose new bet with risk check
        new_bet_amount = 15000
        
        proposal_result = await mock_mcp_server.call_tool(
            "syndicate_create_proposal",
            {
                "proposer_id": mock_members[2]["member_id"],
                "type": ProposalType.LARGE_BET.value,
                "title": "NHL Stanley Cup Finals Bet",
                "description": "High-value opportunity with risk assessment",
                "details": {
                    "amount": new_bet_amount,
                    "current_exposure": float(current_exposure),
                    "available_budget": float(available_budget),
                    "risk_score": 0.7,
                    "kelly_criterion_size": 12000
                }
            }
        )
        
        # System should enforce risk limits
        if new_bet_amount > available_budget:
            # Proposal should be flagged or auto-adjusted
            assert proposal_result.get("risk_warning") is not None or \
                   proposal_result.get("adjusted_amount") is not None
        
        # Test portfolio rebalancing after multiple bets
        rebalance_result = await mock_mcp_server.call_tool(
            "portfolio_rebalance",
            {
                "target_allocations": {
                    "NFL": 0.3,
                    "NBA": 0.25,
                    "NHL": 0.2,
                    "MLB": 0.15,
                    "CASH": 0.1
                },
                "syndicate_id": syndicate_test_config["syndicate_id"],
                "rebalance_threshold": 0.05
            }
        )
        
        assert rebalance_result["status"] in ["rebalanced", "no_action_needed"]


@pytest.mark.integration
class TestSyndicateCollaborationIntegration:
    """Test real-time collaboration during trading"""
    
    @pytest.mark.asyncio
    async def test_live_betting_collaboration(self, mock_mcp_server, mock_members, mock_websocket):
        """Test real-time collaboration during live betting"""
        # Create live betting session
        session_id = "LIVE_SESSION_001"
        
        # Members join live session
        for member in mock_members[:3]:
            await mock_mcp_server.call_tool(
                "syndicate_join_live_session",
                {
                    "session_id": session_id,
                    "member_id": member["member_id"],
                    "role": "analyst" if member["role"] == MemberRole.ANALYST else "observer"
                }
            )
        
        # Simulate live game events
        game_events = [
            {"type": "injury", "player": "Star Player", "team": "Home", "impact": "high"},
            {"type": "score", "team": "Away", "points": 7, "time": "Q1 5:23"},
            {"type": "momentum_shift", "direction": "Away", "confidence": 0.75}
        ]
        
        for event in game_events:
            # Broadcast event to session
            await mock_websocket.send_json({
                "type": "game_event",
                "session_id": session_id,
                "event": event,
                "timestamp": datetime.now().isoformat()
            })
            
            # Members react and discuss
            if event["type"] == "injury" and event["impact"] == "high":
                # Quick proposal for hedge bet
                proposal = await mock_mcp_server.call_tool(
                    "syndicate_create_proposal",
                    {
                        "proposer_id": mock_members[1]["member_id"],
                        "type": ProposalType.LARGE_BET.value,
                        "title": "Hedge Bet - Injury Impact",
                        "description": f"Hedge position due to {event['player']} injury",
                        "details": {
                            "original_position": "Home -3.5",
                            "hedge_position": "Away +3.5",
                            "hedge_amount": 5000,
                            "reason": "significant_injury"
                        },
                        "expedited": True,  # Fast decision needed
                        "voting_deadline_minutes": 5
                    }
                )
                
                # Quick voting
                for member in mock_members[:3]:
                    await mock_mcp_server.call_tool(
                        "syndicate_cast_vote",
                        {
                            "proposal_id": proposal["proposal_id"],
                            "voter_id": member["member_id"],
                            "vote": VoteType.YES.value,
                            "rationale": "Agree with hedge due to injury impact"
                        }
                    )
    
    @pytest.mark.asyncio
    async def test_syndicate_knowledge_sharing(self, mock_mcp_server, mock_members):
        """Test knowledge base integration for syndicate decisions"""
        # Member adds analysis to knowledge base
        knowledge_item = await mock_mcp_server.call_tool(
            "syndicate_add_knowledge",
            {
                "creator_id": mock_members[1]["member_id"],
                "title": "NBA Playoff Trends Analysis",
                "content": "Historical data shows home teams cover 68% in Game 7s",
                "category": "strategy",
                "tags": ["NBA", "playoffs", "trends", "game7"],
                "confidence_score": 0.85
            }
        )
        
        # Search knowledge base during proposal creation
        search_result = await mock_mcp_server.call_tool(
            "syndicate_search_knowledge",
            {
                "query": "NBA playoffs game 7",
                "categories": ["strategy", "analysis"],
                "min_confidence": 0.7
            }
        )
        
        # Use knowledge in proposal
        if search_result["items"]:
            relevant_knowledge = search_result["items"][0]
            
            proposal = await mock_mcp_server.call_tool(
                "syndicate_create_proposal",
                {
                    "proposer_id": mock_members[2]["member_id"],
                    "type": ProposalType.LARGE_BET.value,
                    "title": "NBA Game 7 - Home Team Cover",
                    "description": "Based on syndicate knowledge base analysis",
                    "details": {
                        "knowledge_ref": relevant_knowledge["id"],
                        "historical_win_rate": 0.68,
                        "confidence": relevant_knowledge["confidence_score"],
                        "proposed_stake": 8000
                    }
                }
            )


@pytest.mark.integration
class TestSyndicatePerformanceTracking:
    """Test performance tracking and analytics integration"""
    
    @pytest.mark.asyncio
    async def test_member_performance_impact_on_allocation(self, mock_mcp_server, mock_members):
        """Test how member performance affects fund allocation"""
        # Simulate historical performance
        performance_data = {
            mock_members[0]["member_id"]: {"roi": 0.25, "win_rate": 0.70, "bets": 50},
            mock_members[1]["member_id"]: {"roi": 0.15, "win_rate": 0.65, "bets": 45},
            mock_members[2]["member_id"]: {"roi": -0.05, "win_rate": 0.45, "bets": 40},
            mock_members[3]["member_id"]: {"roi": 0.10, "win_rate": 0.60, "bets": 30}
        }
        
        # Update member performance scores
        for member_id, perf in performance_data.items():
            await mock_mcp_server.call_tool(
                "syndicate_update_member",
                {
                    "member_id": member_id,
                    "performance_metrics": perf,
                    "recalculate_weight": True
                }
            )
        
        # Test performance-weighted allocation
        allocation_result = await mock_mcp_server.call_tool(
            "syndicate_allocate_funds",
            {
                "bet_id": "PERF_TEST_BET",
                "total_amount": 20000,
                "allocation_method": "performance_weighted",
                "participating_members": list(performance_data.keys())
            }
        )
        
        # Verify top performer gets larger allocation
        allocations = allocation_result["allocations"]
        top_performer = mock_members[0]["member_id"]
        
        assert allocations[top_performer] > allocations[mock_members[2]["member_id"]]
        assert allocations[top_performer] > allocations[mock_members[3]["member_id"]]
    
    @pytest.mark.asyncio
    async def test_syndicate_benchmark_comparison(self, mock_mcp_server, syndicate_test_config):
        """Test syndicate performance against benchmarks"""
        # Get syndicate performance
        syndicate_perf = await mock_mcp_server.call_tool(
            "syndicate_performance_metrics",
            {
                "syndicate_id": syndicate_test_config["syndicate_id"],
                "period_days": 90,
                "include_breakdown": True
            }
        )
        
        # Compare against market benchmarks
        benchmark_comparison = await mock_mcp_server.call_tool(
            "run_benchmark",
            {
                "strategy": "syndicate_managed",
                "benchmark_type": "performance",
                "compare_to": ["market_average", "top_10_percent", "index_fund"],
                "use_gpu": True
            }
        )
        
        # Generate performance report
        report = await mock_mcp_server.call_tool(
            "performance_report",
            {
                "strategy": "syndicate_managed",
                "period_days": 90,
                "include_benchmark": True,
                "syndicate_data": {
                    "id": syndicate_test_config["syndicate_id"],
                    "performance": syndicate_perf["overall_metrics"],
                    "benchmark_results": benchmark_comparison
                }
            }
        )
        
        assert report["sections"]["benchmark_comparison"] is not None
        assert report["sections"]["risk_adjusted_returns"] is not None


@pytest.mark.integration
@pytest.mark.slow
class TestSyndicateStressScenarios:
    """Test syndicate behavior under stress conditions"""
    
    @pytest.mark.asyncio
    async def test_mass_withdrawal_handling(self, mock_mcp_server, mock_members):
        """Test handling multiple withdrawal requests"""
        # Simulate market downturn triggering withdrawals
        withdrawal_requests = []
        
        for member in mock_members[1:4]:  # 3 members request withdrawal
            request = await mock_mcp_server.call_tool(
                "syndicate_process_withdrawal",
                {
                    "member_id": member["member_id"],
                    "amount": float(member["capital_contribution"]) * 0.5,
                    "reason": "Market uncertainty",
                    "urgent": True
                }
            )
            withdrawal_requests.append(request)
        
        # System should handle liquidity management
        liquidity_check = await mock_mcp_server.call_tool(
            "syndicate_liquidity_analysis",
            {
                "pending_withdrawals": [r["withdrawal_id"] for r in withdrawal_requests],
                "include_recommendations": True
            }
        )
        
        assert liquidity_check["status"] in ["sufficient", "requires_position_closure", "staged_processing"]
        
        if liquidity_check["status"] == "requires_position_closure":
            # System should recommend which positions to close
            assert "recommended_closures" in liquidity_check
            assert len(liquidity_check["recommended_closures"]) > 0
    
    @pytest.mark.asyncio
    async def test_dispute_resolution_workflow(self, mock_mcp_server, mock_members):
        """Test dispute resolution between members"""
        # Create a disputed bet allocation
        dispute = await mock_mcp_server.call_tool(
            "syndicate_create_dispute",
            {
                "initiator_id": mock_members[2]["member_id"],
                "respondent_id": mock_members[1]["member_id"],
                "type": "allocation_fairness",
                "description": "Unfair allocation in recent high-value bet",
                "evidence": {
                    "bet_id": "BET_DISPUTED_001",
                    "expected_allocation": 5000,
                    "actual_allocation": 3000,
                    "basis": "performance_history"
                }
            }
        )
        
        # Automated mediation process
        mediation = await mock_mcp_server.call_tool(
            "syndicate_mediate_dispute",
            {
                "dispute_id": dispute["dispute_id"],
                "mediator_id": mock_members[0]["member_id"],  # Founder mediates
                "review_data": True,
                "apply_rules": True
            }
        )
        
        # Vote on resolution if needed
        if mediation["requires_vote"]:
            for member in mock_members[:4]:
                if member["member_id"] not in [dispute["initiator_id"], dispute["respondent_id"]]:
                    await mock_mcp_server.call_tool(
                        "syndicate_cast_vote",
                        {
                            "proposal_id": mediation["resolution_proposal_id"],
                            "voter_id": member["member_id"],
                            "vote": VoteType.YES.value if mediation["recommended_action"] else VoteType.NO.value
                        }
                    )


# Helper function to run integration test suite
async def run_integration_tests():
    """Run all integration tests with detailed reporting"""
    print("Starting Syndicate MCP Integration Tests...")
    print("=" * 60)
    
    # Run pytest with integration marker
    import subprocess
    result = subprocess.run(
        [
            "pytest",
            __file__,
            "-v",
            "-m", "integration",
            "--tb=short",
            "--maxfail=5"
        ],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run integration tests
    import sys
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)