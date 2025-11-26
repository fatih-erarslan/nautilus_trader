# Sports Betting API Integration Guide for AI News Trading Platform

## Integration Architecture

### 1. Extending the MCP Server

The AI News Trading platform already has a robust MCP (Model Context Protocol) server. Here's how to extend it with sports betting capabilities:

```python
# src/mcp/sports_betting_tools.py

from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
from datetime import datetime
import redis
from dataclasses import dataclass

@dataclass
class SportsBettingConfig:
    """Configuration for sports betting API providers"""
    theoddsapi_key: str
    betfair_app_key: str
    betfair_cert_path: str
    pinnacle_username: str
    pinnacle_password: str
    cache_ttl: int = 300  # 5 minutes
    
class SportsBettingAggregator:
    """Aggregates odds from multiple sports betting providers"""
    
    def __init__(self, config: SportsBettingConfig):
        self.config = config
        self.redis_client = redis.Redis(decode_responses=True)
        self.providers = {
            'theoddsapi': TheOddsAPIProvider(config.theoddsapi_key),
            'betfair': BetfairProvider(config.betfair_app_key, config.betfair_cert_path),
            'pinnacle': PinnacleProvider(config.pinnacle_username, config.pinnacle_password)
        }
        
    async def get_aggregated_odds(
        self,
        sport: str,
        event_id: Optional[str] = None,
        market_type: str = 'h2h',
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Get aggregated odds from all providers"""
        
        cache_key = f"odds:{sport}:{event_id}:{market_type}"
        
        if use_cache:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Fetch from all providers concurrently
        tasks = []
        for provider_name, provider in self.providers.items():
            if provider.supports_sport(sport):
                tasks.append(
                    self._fetch_provider_odds(
                        provider_name, provider, sport, event_id, market_type
                    )
                )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        aggregated = self._aggregate_odds_data(results)
        
        # Cache results
        self.redis_client.setex(
            cache_key,
            self.config.cache_ttl,
            json.dumps(aggregated)
        )
        
        return aggregated
        
    async def _fetch_provider_odds(
        self,
        provider_name: str,
        provider: Any,
        sport: str,
        event_id: Optional[str],
        market_type: str
    ) -> Dict[str, Any]:
        """Fetch odds from a single provider"""
        try:
            start_time = datetime.now()
            odds = await provider.get_odds(sport, event_id, market_type)
            latency = (datetime.now() - start_time).total_seconds()
            
            return {
                'provider': provider_name,
                'odds': odds,
                'latency': latency,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        except Exception as e:
            return {
                'provider': provider_name,
                'error': str(e),
                'status': 'error'
            }
    
    def _aggregate_odds_data(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate odds data from multiple providers"""
        successful_results = [r for r in results if isinstance(r, dict) and r.get('status') == 'success']
        
        if not successful_results:
            return {'status': 'error', 'message': 'No providers returned data'}
        
        # Find best odds for each outcome
        best_odds = {}
        all_odds = {}
        
        for result in successful_results:
            provider = result['provider']
            for event in result['odds'].get('events', []):
                event_id = event['id']
                
                if event_id not in all_odds:
                    all_odds[event_id] = {
                        'event': event['name'],
                        'sport': event['sport'],
                        'start_time': event['commence_time'],
                        'odds_by_provider': {}
                    }
                
                # Store odds by provider
                all_odds[event_id]['odds_by_provider'][provider] = {
                    'outcomes': event['outcomes'],
                    'last_update': result['timestamp'],
                    'latency': result['latency']
                }
                
                # Track best odds
                for outcome in event['outcomes']:
                    key = f"{event_id}:{outcome['name']}"
                    current_best = best_odds.get(key, {'odds': 0})
                    
                    if outcome['odds'] > current_best['odds']:
                        best_odds[key] = {
                            'odds': outcome['odds'],
                            'provider': provider,
                            'outcome': outcome['name'],
                            'event_id': event_id
                        }
        
        return {
            'status': 'success',
            'aggregated_at': datetime.now().isoformat(),
            'providers_queried': len(results),
            'providers_successful': len(successful_results),
            'events': all_odds,
            'best_odds': best_odds,
            'arbitrage_opportunities': self._find_arbitrage(all_odds)
        }
    
    def _find_arbitrage(self, all_odds: Dict) -> List[Dict]:
        """Identify arbitrage opportunities across providers"""
        opportunities = []
        
        for event_id, event_data in all_odds.items():
            providers_odds = event_data['odds_by_provider']
            
            if len(providers_odds) < 2:
                continue
                
            # For each outcome combination, check for arbitrage
            outcome_names = set()
            for provider_data in providers_odds.values():
                for outcome in provider_data['outcomes']:
                    outcome_names.add(outcome['name'])
            
            # Calculate arbitrage for binary markets (2 outcomes)
            if len(outcome_names) == 2:
                outcome_list = list(outcome_names)
                best_odds = {}
                
                for outcome in outcome_list:
                    best_odds[outcome] = {
                        'odds': 0,
                        'provider': None
                    }
                    
                    for provider, data in providers_odds.items():
                        for provider_outcome in data['outcomes']:
                            if provider_outcome['name'] == outcome:
                                if provider_outcome['odds'] > best_odds[outcome]['odds']:
                                    best_odds[outcome] = {
                                        'odds': provider_outcome['odds'],
                                        'provider': provider
                                    }
                
                # Calculate arbitrage percentage
                implied_prob_sum = sum(1/odd['odds'] for odd in best_odds.values() if odd['odds'] > 0)
                
                if implied_prob_sum < 1:  # Arbitrage opportunity exists
                    profit_percentage = (1 - implied_prob_sum) * 100
                    
                    opportunities.append({
                        'event': event_data['event'],
                        'event_id': event_id,
                        'profit_percentage': profit_percentage,
                        'best_odds': best_odds,
                        'implied_probability_sum': implied_prob_sum
                    })
        
        return sorted(opportunities, key=lambda x: x['profit_percentage'], reverse=True)


# MCP Tool Definitions for Sports Betting

async def get_sports_odds_tool(
    sport: str,
    market_type: str = "h2h",
    event_id: Optional[str] = None,
    providers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get aggregated sports betting odds from multiple providers
    
    Parameters:
    - sport: Sport key (e.g., "americanfootball_nfl", "basketball_nba")
    - market_type: Market type (h2h, spreads, totals)
    - event_id: Specific event ID (optional)
    - providers: List of providers to query (optional, defaults to all)
    
    Returns aggregated odds with best prices highlighted
    """
    aggregator = get_sports_betting_aggregator()
    return await aggregator.get_aggregated_odds(sport, event_id, market_type)


async def find_arbitrage_opportunities_tool(
    sport: str,
    min_profit_percentage: float = 1.0,
    max_opportunities: int = 10
) -> Dict[str, Any]:
    """
    Find arbitrage opportunities across different bookmakers
    
    Parameters:
    - sport: Sport to analyze
    - min_profit_percentage: Minimum profit percentage to report
    - max_opportunities: Maximum number of opportunities to return
    
    Returns list of arbitrage opportunities sorted by profit potential
    """
    aggregator = get_sports_betting_aggregator()
    all_odds = await aggregator.get_aggregated_odds(sport)
    
    opportunities = all_odds.get('arbitrage_opportunities', [])
    filtered = [
        opp for opp in opportunities 
        if opp['profit_percentage'] >= min_profit_percentage
    ][:max_opportunities]
    
    return {
        'status': 'success',
        'sport': sport,
        'opportunities_found': len(filtered),
        'opportunities': filtered,
        'timestamp': datetime.now().isoformat()
    }


async def analyze_betting_value_tool(
    sport: str,
    event_id: str,
    model_probability: float,
    outcome: str
) -> Dict[str, Any]:
    """
    Analyze betting value by comparing model probabilities with market odds
    
    Parameters:
    - sport: Sport key
    - event_id: Event identifier
    - model_probability: Your model's probability for the outcome (0-1)
    - outcome: Outcome to analyze
    
    Returns value analysis including expected value and Kelly criterion
    """
    aggregator = get_sports_betting_aggregator()
    odds_data = await aggregator.get_aggregated_odds(sport, event_id)
    
    if odds_data['status'] != 'success':
        return odds_data
    
    event_odds = odds_data['events'].get(event_id)
    if not event_odds:
        return {'status': 'error', 'message': 'Event not found'}
    
    # Find best odds for the outcome
    best_odds = 0
    best_provider = None
    
    for provider, data in event_odds['odds_by_provider'].items():
        for provider_outcome in data['outcomes']:
            if provider_outcome['name'] == outcome:
                if provider_outcome['odds'] > best_odds:
                    best_odds = provider_outcome['odds']
                    best_provider = provider
    
    if best_odds == 0:
        return {'status': 'error', 'message': 'Outcome not found'}
    
    # Calculate value metrics
    market_implied_prob = 1 / best_odds
    edge = model_probability - market_implied_prob
    expected_value = (model_probability * (best_odds - 1)) - (1 - model_probability)
    
    # Kelly Criterion calculation
    kelly_fraction = 0
    if edge > 0:
        kelly_fraction = (model_probability * (best_odds - 1) - (1 - model_probability)) / (best_odds - 1)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% of bankroll
    
    return {
        'status': 'success',
        'event': event_odds['event'],
        'outcome': outcome,
        'best_odds': best_odds,
        'best_provider': best_provider,
        'model_probability': model_probability,
        'market_implied_probability': market_implied_prob,
        'edge': edge,
        'expected_value': expected_value,
        'kelly_fraction': kelly_fraction,
        'recommendation': 'bet' if expected_value > 0.05 else 'no_bet',
        'confidence': 'high' if edge > 0.1 else 'medium' if edge > 0.05 else 'low'
    }


async def stream_live_odds_tool(
    sport: str,
    event_ids: List[str],
    providers: Optional[List[str]] = None,
    callback_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Subscribe to live odds streaming for specific events
    
    Parameters:
    - sport: Sport key
    - event_ids: List of event IDs to stream
    - providers: Providers to stream from (supports WebSocket providers only)
    - callback_url: Webhook URL for odds updates (optional)
    
    Returns subscription details and WebSocket connection info
    """
    # Implementation would establish WebSocket connections
    # This is a placeholder showing the structure
    
    streaming_providers = ['betfair', 'lsports']  # Providers that support streaming
    selected_providers = providers if providers else streaming_providers
    
    subscriptions = []
    for provider in selected_providers:
        if provider in streaming_providers:
            # In real implementation, establish WebSocket connection
            subscriptions.append({
                'provider': provider,
                'status': 'subscribed',
                'events': event_ids,
                'connection': f'wss://{provider}.stream/odds'
            })
    
    return {
        'status': 'success',
        'subscriptions': subscriptions,
        'callback_url': callback_url,
        'message': 'Streaming connections established'
    }
```

### 2. Compliance Integration Layer

```python
# src/compliance/sports_betting_compliance.py

from typing import Dict, Optional, List, Tuple
import geoip2.database
from datetime import datetime, timedelta
import hashlib
import requests

class SportsBettingCompliance:
    """Handles compliance requirements for sports betting operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.geoip_reader = geoip2.database.Reader(config['geoip_db_path'])
        self.excluded_users = set()  # Self-exclusion list
        self.kyc_provider = KYCProvider(config['kyc_api_key'])
        
    async def verify_user_eligibility(
        self,
        user_id: str,
        ip_address: str,
        user_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive user eligibility check for sports betting
        
        Returns: (is_eligible, list_of_failed_checks)
        """
        failed_checks = []
        
        # 1. Geographic location check
        location = self.get_user_location(ip_address)
        if not self.is_location_permitted(location):
            failed_checks.append(f"Sports betting not permitted in {location['state']}")
        
        # 2. Age verification
        if not self.verify_age(user_data.get('date_of_birth')):
            failed_checks.append("User under 21 years old")
        
        # 3. KYC status check
        kyc_status = await self.check_kyc_status(user_id)
        if kyc_status != 'verified':
            failed_checks.append(f"KYC status: {kyc_status}")
        
        # 4. Self-exclusion check
        if self.is_self_excluded(user_id):
            failed_checks.append("User is self-excluded")
        
        # 5. AML risk check
        aml_risk = await self.check_aml_risk(user_id, user_data)
        if aml_risk == 'high':
            failed_checks.append("High AML risk score")
        
        # 6. Responsible gambling limits
        if not self.check_gambling_limits(user_id):
            failed_checks.append("Exceeded responsible gambling limits")
        
        is_eligible = len(failed_checks) == 0
        
        # Log compliance check
        self.log_compliance_check(user_id, is_eligible, failed_checks)
        
        return is_eligible, failed_checks
    
    def get_user_location(self, ip_address: str) -> Dict[str, str]:
        """Get user location from IP address"""
        try:
            response = self.geoip_reader.city(ip_address)
            return {
                'country': response.country.iso_code,
                'state': response.subdivisions.most_specific.iso_code,
                'city': response.city.name,
                'latitude': response.location.latitude,
                'longitude': response.location.longitude
            }
        except Exception as e:
            return {'country': 'unknown', 'state': 'unknown'}
    
    def is_location_permitted(self, location: Dict[str, str]) -> bool:
        """Check if sports betting is permitted in user's location"""
        # US states where online sports betting is legal (as of 2024)
        permitted_states = {
            'AZ', 'CO', 'CT', 'DC', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'NV', 'NH', 'NJ', 'NY', 'NC', 'OH',
            'OR', 'PA', 'RI', 'TN', 'VT', 'VA', 'WV', 'WY'
        }
        
        if location['country'] == 'US':
            return location['state'] in permitted_states
        
        # European countries generally permit with proper licensing
        permitted_countries = {
            'GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'DK', 'SE', 'NO',
            'FI', 'AT', 'CH', 'IE', 'PT', 'GR', 'RO', 'BG', 'HR', 'CZ',
            'HU', 'PL', 'SK', 'SI'
        }
        
        return location['country'] in permitted_countries
    
    def verify_age(self, date_of_birth: Optional[str]) -> bool:
        """Verify user is 21 or older"""
        if not date_of_birth:
            return False
            
        try:
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d')
            age = (datetime.now() - dob).days // 365
            return age >= 21
        except:
            return False
    
    async def check_kyc_status(self, user_id: str) -> str:
        """Check KYC verification status"""
        # In production, this would call the KYC provider API
        return await self.kyc_provider.get_verification_status(user_id)
    
    def is_self_excluded(self, user_id: str) -> bool:
        """Check if user has self-excluded"""
        return user_id in self.excluded_users
    
    async def check_aml_risk(
        self,
        user_id: str,
        user_data: Dict[str, Any]
    ) -> str:
        """Perform AML risk assessment"""
        risk_factors = []
        
        # Check transaction patterns
        recent_deposits = user_data.get('recent_deposits', [])
        if len(recent_deposits) > 10:
            risk_factors.append('high_transaction_frequency')
        
        total_deposited = sum(d['amount'] for d in recent_deposits)
        if total_deposited > 10000:
            risk_factors.append('high_transaction_volume')
        
        # Check for suspicious patterns
        if self.has_suspicious_patterns(recent_deposits):
            risk_factors.append('suspicious_patterns')
        
        # Check sanctions lists
        if await self.check_sanctions_list(user_data.get('name', '')):
            risk_factors.append('sanctions_match')
        
        if len(risk_factors) >= 2:
            return 'high'
        elif len(risk_factors) == 1:
            return 'medium'
        else:
            return 'low'
    
    def check_gambling_limits(self, user_id: str) -> bool:
        """Check responsible gambling limits"""
        # Check daily, weekly, monthly limits
        # This would query the user's betting history
        return True  # Placeholder
    
    def has_suspicious_patterns(self, transactions: List[Dict]) -> bool:
        """Detect suspicious transaction patterns"""
        if not transactions:
            return False
        
        # Check for rapid deposits and withdrawals
        deposits = [t for t in transactions if t['type'] == 'deposit']
        withdrawals = [t for t in transactions if t['type'] == 'withdrawal']
        
        # Suspicious if depositing and withdrawing frequently
        if len(deposits) > 5 and len(withdrawals) > 5:
            time_diff = (deposits[-1]['timestamp'] - deposits[0]['timestamp']).days
            if time_diff < 7:  # Multiple deposits and withdrawals within a week
                return True
        
        return False
    
    async def check_sanctions_list(self, name: str) -> bool:
        """Check if user appears on sanctions lists"""
        # In production, integrate with sanctions screening API
        # Example: ComplyAdvantage, Dow Jones, etc.
        return False  # Placeholder
    
    def log_compliance_check(
        self,
        user_id: str,
        is_eligible: bool,
        failed_checks: List[str]
    ):
        """Log compliance check for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'is_eligible': is_eligible,
            'failed_checks': failed_checks,
            'check_id': hashlib.sha256(
                f"{user_id}{datetime.now()}".encode()
            ).hexdigest()
        }
        
        # In production, store in compliance database
        print(f"Compliance log: {log_entry}")


# MCP Tools for Compliance

async def verify_betting_eligibility_tool(
    user_id: str,
    ip_address: str,
    date_of_birth: str,
    recent_transactions: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Verify user eligibility for sports betting
    
    Parameters:
    - user_id: User identifier
    - ip_address: User's IP address for geolocation
    - date_of_birth: User's date of birth (YYYY-MM-DD)
    - recent_transactions: List of recent transactions for AML check
    
    Returns compliance check results
    """
    compliance = get_compliance_checker()
    
    user_data = {
        'date_of_birth': date_of_birth,
        'recent_deposits': recent_transactions or []
    }
    
    is_eligible, failed_checks = await compliance.verify_user_eligibility(
        user_id, ip_address, user_data
    )
    
    return {
        'status': 'success',
        'user_id': user_id,
        'is_eligible': is_eligible,
        'failed_checks': failed_checks,
        'timestamp': datetime.now().isoformat(),
        'next_steps': 'Proceed with betting' if is_eligible else 'Address compliance issues'
    }


async def set_responsible_gambling_limits_tool(
    user_id: str,
    daily_limit: Optional[float] = None,
    weekly_limit: Optional[float] = None,
    monthly_limit: Optional[float] = None,
    loss_limit: Optional[float] = None,
    time_limit_minutes: Optional[int] = None
) -> Dict[str, Any]:
    """
    Set responsible gambling limits for a user
    
    Parameters:
    - user_id: User identifier
    - daily_limit: Maximum daily betting amount
    - weekly_limit: Maximum weekly betting amount
    - monthly_limit: Maximum monthly betting amount
    - loss_limit: Maximum acceptable loss
    - time_limit_minutes: Maximum daily betting time in minutes
    
    Returns confirmation of limits set
    """
    limits = {
        'daily_limit': daily_limit,
        'weekly_limit': weekly_limit,
        'monthly_limit': monthly_limit,
        'loss_limit': loss_limit,
        'time_limit_minutes': time_limit_minutes
    }
    
    # Remove None values
    limits = {k: v for k, v in limits.items() if v is not None}
    
    # In production, store in database
    # For now, return confirmation
    
    return {
        'status': 'success',
        'user_id': user_id,
        'limits_set': limits,
        'effective_date': datetime.now().isoformat(),
        'message': 'Responsible gambling limits set successfully'
    }
```

### 3. Integration with Existing AI News Trading Platform

```python
# src/mcp/enhanced_mcp_server.py
# Add these tools to the existing MCP server

@with_error_handler
async def get_sports_news_betting_correlation_tool(
    symbol: str,
    sport: str,
    lookback_hours: int = 24,
    correlation_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Analyze correlation between news sentiment and sports betting odds movements
    
    Parameters:
    - symbol: Stock symbol (e.g., team owner company)
    - sport: Sport to analyze
    - lookback_hours: Hours of historical data
    - correlation_threshold: Minimum correlation to report
    
    Returns correlation analysis between news and betting odds
    """
    # Get news sentiment
    news_sentiment = await analyze_news(
        symbol=symbol,
        lookback_hours=lookback_hours,
        sentiment_model="enhanced",
        use_gpu=True
    )
    
    # Get historical odds movements
    aggregator = get_sports_betting_aggregator()
    odds_history = await aggregator.get_historical_odds(
        sport=sport,
        lookback_hours=lookback_hours
    )
    
    # Calculate correlations
    correlations = calculate_news_odds_correlation(
        news_sentiment['articles'],
        odds_history
    )
    
    significant_correlations = [
        corr for corr in correlations
        if abs(corr['correlation']) >= correlation_threshold
    ]
    
    return {
        'status': 'success',
        'symbol': symbol,
        'sport': sport,
        'analysis_period_hours': lookback_hours,
        'total_news_items': len(news_sentiment['articles']),
        'total_odds_movements': len(odds_history),
        'significant_correlations': significant_correlations,
        'average_correlation': np.mean([c['correlation'] for c in correlations]),
        'recommendation': generate_correlation_recommendation(significant_correlations),
        'timestamp': datetime.now().isoformat()
    }


@with_error_handler
async def create_sports_arbitrage_strategy_tool(
    name: str,
    sports: List[str],
    min_profit_percentage: float = 1.0,
    max_stake: float = 1000.0,
    auto_execute: bool = False
) -> Dict[str, Any]:
    """
    Create an automated sports arbitrage trading strategy
    
    Parameters:
    - name: Strategy name
    - sports: List of sports to monitor
    - min_profit_percentage: Minimum arbitrage profit to act on
    - max_stake: Maximum stake per arbitrage opportunity
    - auto_execute: Whether to automatically execute trades
    
    Returns strategy creation confirmation
    """
    strategy = SportsBettingStrategy(
        name=name,
        strategy_type='arbitrage',
        config={
            'sports': sports,
            'min_profit_percentage': min_profit_percentage,
            'max_stake': max_stake,
            'auto_execute': auto_execute
        }
    )
    
    # Register strategy
    strategy_id = await register_strategy(strategy)
    
    # Start monitoring
    if auto_execute:
        await start_strategy_monitoring(strategy_id)
    
    return {
        'status': 'success',
        'strategy_id': strategy_id,
        'name': name,
        'type': 'sports_arbitrage',
        'config': strategy.config,
        'monitoring_active': auto_execute,
        'message': f"Sports arbitrage strategy '{name}' created successfully"
    }


# Add to MCP tool registry
SPORTS_BETTING_TOOLS = [
    {
        "name": "get_sports_odds",
        "description": "Get aggregated sports betting odds from multiple providers",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sport": {"type": "string", "description": "Sport key (e.g., 'americanfootball_nfl')"},
                "market_type": {"type": "string", "default": "h2h"},
                "event_id": {"type": "string", "description": "Specific event ID"},
                "providers": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["sport"]
        }
    },
    {
        "name": "find_arbitrage_opportunities",
        "description": "Find arbitrage opportunities across bookmakers",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sport": {"type": "string"},
                "min_profit_percentage": {"type": "number", "default": 1.0},
                "max_opportunities": {"type": "integer", "default": 10}
            },
            "required": ["sport"]
        }
    },
    {
        "name": "analyze_betting_value",
        "description": "Analyze betting value using AI model probabilities",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sport": {"type": "string"},
                "event_id": {"type": "string"},
                "model_probability": {"type": "number", "minimum": 0, "maximum": 1},
                "outcome": {"type": "string"}
            },
            "required": ["sport", "event_id", "model_probability", "outcome"]
        }
    },
    {
        "name": "verify_betting_eligibility",
        "description": "Verify user eligibility for sports betting (compliance)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "ip_address": {"type": "string"},
                "date_of_birth": {"type": "string", "format": "date"},
                "recent_transactions": {"type": "array"}
            },
            "required": ["user_id", "ip_address", "date_of_birth"]
        }
    },
    {
        "name": "get_sports_news_betting_correlation",
        "description": "Analyze correlation between news and betting odds",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "sport": {"type": "string"},
                "lookback_hours": {"type": "integer", "default": 24},
                "correlation_threshold": {"type": "number", "default": 0.3}
            },
            "required": ["symbol", "sport"]
        }
    }
]
```

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. Set up API credentials for TheOddsAPI
2. Implement basic odds aggregation
3. Create caching layer with Redis
4. Build compliance checking framework

### Phase 2: Integration (Weeks 3-4)
1. Extend MCP server with sports betting tools
2. Implement arbitrage detection
3. Add value betting analysis
4. Create correlation analysis with news

### Phase 3: Advanced Features (Weeks 5-6)
1. Add Betfair Exchange integration
2. Implement WebSocket streaming
3. Build automated strategy execution
4. Enhanced compliance monitoring

### Phase 4: Production Ready (Weeks 7-8)
1. Comprehensive testing
2. Performance optimization
3. Security audit
4. Documentation and training

## Security Considerations

1. **API Key Management**
   - Store credentials in secure vault
   - Rotate keys regularly
   - Use environment-specific keys

2. **Data Encryption**
   - Encrypt sensitive user data
   - Use TLS for all API calls
   - Secure WebSocket connections

3. **Access Control**
   - Role-based permissions
   - Audit trail for all betting activities
   - Two-factor authentication for traders

4. **Compliance Logging**
   - Immutable audit logs
   - Regular compliance reports
   - Automated suspicious activity alerts

This integration guide provides a complete framework for adding sports betting capabilities to your AI News Trading platform while maintaining compliance and security standards.