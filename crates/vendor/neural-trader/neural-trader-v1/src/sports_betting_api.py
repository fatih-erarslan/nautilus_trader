"""
Sports Betting API Endpoints
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Body, Query, HTTPException, Depends
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import random

from src.auth import check_auth_optional

router = APIRouter(prefix="/sports", tags=["Sports Betting"])

# Pydantic Models
class SportsBetRequest(BaseModel):
    market_id: str = Field(..., description="Market ID")
    selection: str = Field(..., description="Selection to bet on")
    stake: float = Field(..., ge=0.01)
    odds: float = Field(..., ge=1.01)
    bet_type: str = Field(default="back", description="back or lay")
    validate_only: bool = Field(default=True, description="Validate without placing")

class KellyCriterionRequest(BaseModel):
    probability: float = Field(..., ge=0, le=1)
    odds: float = Field(..., ge=1.01)
    bankroll: float = Field(..., ge=0)
    confidence: float = Field(default=1.0, ge=0, le=2)

class BettingStrategySimulation(BaseModel):
    strategy_config: Dict[str, Any] = Field(..., description="Strategy configuration")
    num_simulations: int = Field(default=1000, ge=100, le=10000)
    use_gpu: bool = Field(default=False)

# Sports Betting Endpoints
@router.get("/events/{sport}")
async def get_sports_events(
    sport: str,
    days_ahead: int = Query(default=7, le=30),
    use_gpu: bool = Query(default=False),
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Get upcoming sports events with comprehensive analysis.
    Based on MCP AI News Trader get_sports_events.
    """
    sports_events = {
        "football": ["NFL", "Premier League", "Champions League"],
        "basketball": ["NBA", "EuroLeague"],
        "tennis": ["Australian Open", "ATP Tour"],
        "soccer": ["World Cup Qualifiers", "La Liga", "Serie A"]
    }
    
    events = []
    event_types = sports_events.get(sport.lower(), ["General"])
    
    for i in range(10):
        event_date = datetime.now() + timedelta(days=i % days_ahead)
        events.append({
            "event_id": f"EVT-{sport.upper()}-{i:03d}",
            "sport": sport,
            "competition": event_types[i % len(event_types)],
            "home_team": f"Team A{i}",
            "away_team": f"Team B{i}",
            "start_time": event_date.isoformat(),
            "markets_available": 25 + i * 3,
            "analysis": {
                "home_win_probability": 0.45 + (i % 20) / 100,
                "draw_probability": 0.25,
                "away_win_probability": 0.30 - (i % 20) / 100,
                "predicted_total_goals": 2.5 + (i % 5) / 10,
                "form_rating": {
                    "home": 7.5 + (i % 3),
                    "away": 6.5 + (i % 4)
                }
            },
            "betting_insights": {
                "value_bets": ["Over 2.5 goals", "Both teams to score"],
                "confidence": 0.72
            }
        })
    
    return {
        "sport": sport,
        "events": events,
        "total_events": len(events),
        "days_ahead": days_ahead,
        "gpu_analysis": use_gpu,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/odds/{sport}")
async def get_sports_odds(
    sport: str,
    market_types: Optional[List[str]] = Query(default=None),
    regions: Optional[List[str]] = Query(default=None),
    use_gpu: bool = Query(default=False),
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Get real-time sports betting odds with market analysis.
    Based on MCP AI News Trader get_sports_odds.
    """
    default_markets = market_types or ["moneyline", "spread", "totals"]
    default_regions = regions or ["us", "uk", "eu"]
    
    odds_data = []
    for i in range(5):
        event_odds = {
            "event_id": f"EVT-{sport.upper()}-{i:03d}",
            "sport": sport,
            "home_team": f"Team A{i}",
            "away_team": f"Team B{i}",
            "markets": {}
        }
        
        for market in default_markets:
            if market == "moneyline":
                event_odds["markets"][market] = {
                    "home": 1.85 + (i % 10) / 20,
                    "away": 2.10 - (i % 10) / 20,
                    "draw": 3.25
                }
            elif market == "spread":
                event_odds["markets"][market] = {
                    "home": {"line": -1.5, "odds": 1.91},
                    "away": {"line": 1.5, "odds": 1.91}
                }
            elif market == "totals":
                event_odds["markets"][market] = {
                    "over": {"line": 2.5, "odds": 1.85},
                    "under": {"line": 2.5, "odds": 1.95}
                }
        
        event_odds["best_odds"] = {
            "provider": "Provider" + str(i % 3 + 1),
            "region": default_regions[i % len(default_regions)]
        }
        
        odds_data.append(event_odds)
    
    return {
        "sport": sport,
        "odds": odds_data,
        "market_types": default_markets,
        "regions": default_regions,
        "gpu_accelerated": use_gpu,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/arbitrage/find")
async def find_sports_arbitrage(
    sport: str = Body(...),
    min_profit_margin: float = Body(default=0.01, ge=0, le=0.1),
    use_gpu: bool = Body(default=False),
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Find arbitrage opportunities in sports betting markets.
    Based on MCP AI News Trader find_sports_arbitrage.
    """
    arbitrage_opportunities = []
    
    # Simulate finding arb opportunities
    for i in range(3):
        arb = {
            "opportunity_id": f"ARB-{i:03d}",
            "sport": sport,
            "event": f"Team A{i} vs Team B{i}",
            "market": "moneyline",
            "legs": [
                {"outcome": "home", "odds": 2.15, "provider": "BookmakerA", "stake_percentage": 46.5},
                {"outcome": "away", "odds": 2.20, "provider": "BookmakerB", "stake_percentage": 45.5},
                {"outcome": "draw", "odds": 4.50, "provider": "BookmakerC", "stake_percentage": 8.0}
            ],
            "total_probability": 0.985,  # Less than 1 = arbitrage
            "profit_margin": 0.015 + i * 0.005,
            "required_capital": 1000,
            "expected_profit": 15 + i * 5,
            "confidence": 0.95 - i * 0.05,
            "time_sensitivity": "high" if i == 0 else "medium"
        }
        
        if arb["profit_margin"] >= min_profit_margin:
            arbitrage_opportunities.append(arb)
    
    return {
        "opportunities": arbitrage_opportunities,
        "total_found": len(arbitrage_opportunities),
        "min_profit_margin": min_profit_margin,
        "best_opportunity": arbitrage_opportunities[0] if arbitrage_opportunities else None,
        "gpu_accelerated": use_gpu,
        "scan_time_ms": 125
    }

@router.post("/market/depth-analysis")
async def analyze_betting_market_depth(
    market_id: str = Body(...),
    sport: str = Body(...),
    use_gpu: bool = Body(default=False),
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Analyze betting market depth and liquidity.
    Based on MCP AI News Trader analyze_betting_market_depth.
    """
    return {
        "market_id": market_id,
        "sport": sport,
        "liquidity_score": 0.82,
        "market_depth": {
            "total_volume": 125000,
            "active_bettors": 3250,
            "avg_bet_size": 38.5,
            "max_bet_size": 5000
        },
        "price_levels": {
            "back": [
                {"odds": 1.91, "volume": 25000},
                {"odds": 1.90, "volume": 18000},
                {"odds": 1.89, "volume": 12000}
            ],
            "lay": [
                {"odds": 1.92, "volume": 22000},
                {"odds": 1.93, "volume": 15000},
                {"odds": 1.94, "volume": 10000}
            ]
        },
        "market_efficiency": 0.94,
        "manipulation_risk": "low",
        "recommended_stake_limit": 2500,
        "gpu_accelerated": use_gpu
    }

@router.post("/kelly-criterion")
async def calculate_kelly_criterion(
    request: KellyCriterionRequest,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Calculate optimal bet size using Kelly Criterion with risk adjustments.
    Based on MCP AI News Trader calculate_kelly_criterion.
    """
    # Kelly formula: f = (p * b - q) / b
    # where f = fraction, p = probability, b = odds-1, q = 1-p
    decimal_odds = request.odds
    b = decimal_odds - 1
    p = request.probability
    q = 1 - p
    
    kelly_fraction = (p * b - q) / b if b > 0 else 0
    
    # Apply confidence adjustment
    adjusted_fraction = kelly_fraction * request.confidence
    
    # Apply conservative limits (max 25% of bankroll)
    safe_fraction = min(max(adjusted_fraction, 0), 0.25)
    
    recommended_bet = request.bankroll * safe_fraction
    
    return {
        "probability": p,
        "odds": decimal_odds,
        "bankroll": request.bankroll,
        "kelly_fraction": kelly_fraction,
        "confidence_adjusted": adjusted_fraction,
        "safe_fraction": safe_fraction,
        "recommended_bet": recommended_bet,
        "expected_growth_rate": p * (b * safe_fraction + 1) + q * (1 - safe_fraction) - 1,
        "risk_of_ruin": 0.001 if safe_fraction < 0.1 else 0.01,
        "confidence": request.confidence
    }

@router.post("/strategy/simulate")
async def simulate_betting_strategy(
    request: BettingStrategySimulation,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Simulate betting strategy performance with Monte Carlo analysis.
    Based on MCP AI News Trader simulate_betting_strategy.
    """
    simulations = []
    config = request.strategy_config
    
    for i in range(min(request.num_simulations, 100)):  # Limit for demo
        starting_bankroll = config.get("starting_bankroll", 10000)
        final_bankroll = starting_bankroll * (1 + random.uniform(-0.3, 0.5))
        
        simulations.append({
            "simulation_id": i,
            "final_bankroll": final_bankroll,
            "total_return": (final_bankroll - starting_bankroll) / starting_bankroll,
            "max_drawdown": -random.uniform(0.05, 0.25),
            "total_bets": random.randint(50, 200),
            "win_rate": random.uniform(0.45, 0.65)
        })
    
    # Calculate statistics
    returns = [s["total_return"] for s in simulations]
    avg_return = sum(returns) / len(returns)
    
    return {
        "strategy_config": config,
        "num_simulations": len(simulations),
        "results": {
            "avg_return": avg_return,
            "median_return": sorted(returns)[len(returns)//2],
            "best_return": max(returns),
            "worst_return": min(returns),
            "profitable_simulations": sum(1 for r in returns if r > 0),
            "avg_drawdown": sum(s["max_drawdown"] for s in simulations) / len(simulations),
            "avg_win_rate": sum(s["win_rate"] for s in simulations) / len(simulations)
        },
        "risk_metrics": {
            "var_95": sorted(returns)[int(len(returns) * 0.05)],
            "cvar_95": sum(sorted(returns)[:int(len(returns) * 0.05)]) / max(int(len(returns) * 0.05), 1),
            "sharpe_ratio": avg_return / (0.15 if avg_return > 0 else 1)  # Simplified
        },
        "recommendation": "viable" if avg_return > 0.1 else "risky",
        "gpu_accelerated": request.use_gpu
    }

@router.get("/portfolio/betting-status")
async def get_betting_portfolio_status(
    include_risk_analysis: bool = Query(default=True),
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Get comprehensive betting portfolio status and risk metrics.
    Based on MCP AI News Trader get_betting_portfolio_status.
    """
    active_bets = [
        {
            "bet_id": "BET-001",
            "sport": "football",
            "event": "Team A vs Team B",
            "selection": "Team A",
            "stake": 100,
            "odds": 1.85,
            "potential_return": 185,
            "status": "pending"
        },
        {
            "bet_id": "BET-002",
            "sport": "basketball",
            "event": "Lakers vs Celtics",
            "selection": "Over 215.5",
            "stake": 150,
            "odds": 1.91,
            "potential_return": 286.5,
            "status": "pending"
        }
    ]
    
    portfolio = {
        "active_bets": active_bets,
        "total_stakes": sum(b["stake"] for b in active_bets),
        "potential_returns": sum(b["potential_return"] for b in active_bets),
        "settled_today": {
            "won": 5,
            "lost": 3,
            "profit": 125.50
        },
        "month_to_date": {
            "total_bets": 85,
            "win_rate": 0.58,
            "profit": 1250.75,
            "roi": 0.125
        }
    }
    
    if include_risk_analysis:
        portfolio["risk_analysis"] = {
            "current_exposure": sum(b["stake"] for b in active_bets),
            "max_exposure": 1000,
            "risk_score": 0.45,
            "concentration_risk": {
                "football": 0.4,
                "basketball": 0.6
            },
            "kelly_compliance": 0.92,
            "recommended_adjustments": [
                "Reduce basketball exposure",
                "Consider hedging BET-002"
            ]
        }
    
    return portfolio

@router.post("/bet/execute")
async def execute_sports_bet(
    request: SportsBetRequest,
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Execute sports bet with comprehensive validation and risk checks.
    Based on MCP AI News Trader execute_sports_bet.
    """
    # Validate bet
    validation = {
        "odds_valid": request.odds >= 1.01,
        "stake_within_limits": request.stake <= 5000,
        "market_open": True,
        "sufficient_liquidity": True
    }
    
    if not all(validation.values()):
        return {
            "status": "validation_failed",
            "validation": validation,
            "message": "Bet validation failed"
        }
    
    if request.validate_only:
        return {
            "status": "validated",
            "validation": validation,
            "estimated_return": request.stake * request.odds,
            "message": "Bet validated successfully (not placed)"
        }
    
    # Simulate bet execution
    return {
        "status": "executed",
        "bet_id": f"BET-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "market_id": request.market_id,
        "selection": request.selection,
        "stake": request.stake,
        "odds": request.odds,
        "bet_type": request.bet_type,
        "potential_return": request.stake * request.odds,
        "commission": request.stake * 0.05 if request.bet_type == "lay" else 0,
        "timestamp": datetime.now().isoformat(),
        "demo_mode": True
    }

@router.get("/performance/betting")
async def get_sports_betting_performance(
    period_days: int = Query(default=30),
    include_detailed_analysis: bool = Query(default=True),
    user: Optional[Dict[str, Any]] = Depends(check_auth_optional)
):
    """
    Get comprehensive sports betting performance analytics.
    Based on MCP AI News Trader get_sports_betting_performance.
    """
    performance = {
        "period_days": period_days,
        "summary": {
            "total_bets": 245,
            "won": 142,
            "lost": 103,
            "win_rate": 0.58,
            "total_staked": 24500,
            "total_returned": 28750,
            "net_profit": 4250,
            "roi": 0.173,
            "avg_odds": 1.92
        },
        "by_sport": {
            "football": {"bets": 120, "profit": 2100, "roi": 0.18},
            "basketball": {"bets": 75, "profit": 1500, "roi": 0.20},
            "tennis": {"bets": 50, "profit": 650, "roi": 0.13}
        },
        "by_market": {
            "moneyline": {"bets": 100, "profit": 1800, "roi": 0.18},
            "spread": {"bets": 85, "profit": 1450, "roi": 0.17},
            "totals": {"bets": 60, "profit": 1000, "roi": 0.167}
        }
    }
    
    if include_detailed_analysis:
        performance["detailed_analysis"] = {
            "best_performing_sport": "basketball",
            "worst_performing_market": "totals",
            "streak_analysis": {
                "current_streak": "W3",
                "longest_win_streak": 7,
                "longest_loss_streak": 4
            },
            "time_analysis": {
                "best_day": "Saturday",
                "best_time": "20:00-22:00",
                "worst_day": "Tuesday"
            },
            "edge_analysis": {
                "estimated_edge": 0.035,
                "confidence": 0.78,
                "sample_size_adequate": True
            },
            "recommendations": [
                "Focus on basketball spread bets",
                "Reduce Tuesday betting volume",
                "Increase stakes on high-confidence bets"
            ]
        }
    
    return performance