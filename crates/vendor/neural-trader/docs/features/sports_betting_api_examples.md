# Sports Betting API Integration Examples

## 1. TheOddsAPI Integration

### Basic Setup and Authentication

```python
# theoddsapi_client.py
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

class TheOddsAPIClient:
    """Client for TheOddsAPI integration"""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_sports(self) -> List[Dict[str, Any]]:
        """Get list of available sports"""
        url = f"{self.BASE_URL}/sports"
        params = {"apiKey": self.api_key}
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_odds(
        self,
        sport: str,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        date_format: str = "iso"
    ) -> Dict[str, Any]:
        """Get odds for a specific sport"""
        url = f"{self.BASE_URL}/sports/{sport}/odds"
        
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format
        }
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Add request metadata
            remaining_requests = response.headers.get('x-requests-remaining', 'unknown')
            used_requests = response.headers.get('x-requests-used', 'unknown')
            
            return {
                "data": data,
                "metadata": {
                    "remaining_requests": remaining_requests,
                    "used_requests": used_requests,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    async def get_historical_odds(
        self,
        sport: str,
        days_from: int,
        date_format: str = "iso"
    ) -> Dict[str, Any]:
        """Get historical odds data"""
        url = f"{self.BASE_URL}/historical/sports/{sport}/odds"
        
        params = {
            "apiKey": self.api_key,
            "daysFrom": days_from,
            "dateFormat": date_format
        }
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()


# Example usage
async def theoddsapi_example():
    async with TheOddsAPIClient(api_key="your_api_key") as client:
        # Get available sports
        sports = await client.get_sports()
        print(f"Available sports: {len(sports)}")
        
        # Get NFL odds
        nfl_odds = await client.get_odds(
            sport="americanfootball_nfl",
            regions="us",
            markets="h2h,spreads"
        )
        
        print(f"NFL games with odds: {len(nfl_odds['data'])}")
        
        # Process odds for arbitrage
        for game in nfl_odds['data']:
            print(f"\nGame: {game['home_team']} vs {game['away_team']}")
            
            # Find best odds for each outcome
            best_home_odds = 0
            best_away_odds = 0
            best_home_book = ""
            best_away_book = ""
            
            for bookmaker in game['bookmakers']:
                for market in bookmaker['markets']:
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == game['home_team']:
                                if outcome['price'] > best_home_odds:
                                    best_home_odds = outcome['price']
                                    best_home_book = bookmaker['title']
                            elif outcome['name'] == game['away_team']:
                                if outcome['price'] > best_away_odds:
                                    best_away_odds = outcome['price']
                                    best_away_book = bookmaker['title']
            
            # Calculate arbitrage
            if best_home_odds > 0 and best_away_odds > 0:
                implied_prob = (1/best_home_odds) + (1/best_away_odds)
                if implied_prob < 1:
                    profit = (1 - implied_prob) * 100
                    print(f"  ARBITRAGE OPPORTUNITY: {profit:.2f}% profit")
                    print(f"  Home: {best_home_book} @ {best_home_odds}")
                    print(f"  Away: {best_away_book} @ {best_away_odds}")
```

## 2. Betfair Exchange Integration

### Authentication and Session Management

```python
# betfair_client.py
import requests
from betfairlightweight import APIClient
from betfairlightweight.streaming import StreamListener
import betfairlightweight.filters as filters
from typing import Dict, List, Optional, Any
import json
from datetime import datetime, timedelta

class BetfairClient:
    """Client for Betfair Exchange API"""
    
    def __init__(self, username: str, password: str, app_key: str, cert_path: str):
        self.username = username
        self.password = password
        self.app_key = app_key
        self.cert_path = cert_path
        self.client = None
        
    def login(self):
        """Authenticate with Betfair"""
        self.client = APIClient(
            username=self.username,
            password=self.password,
            app_key=self.app_key,
            certs=self.cert_path
        )
        self.client.login()
        
    def get_market_catalogue(
        self,
        event_type_ids: List[str],
        market_projection: List[str] = None,
        max_results: int = 100
    ) -> List[Dict]:
        """Get market catalogue for specific event types"""
        if market_projection is None:
            market_projection = [
                "COMPETITION",
                "EVENT",
                "EVENT_TYPE",
                "MARKET_START_TIME",
                "MARKET_DESCRIPTION",
                "RUNNER_DESCRIPTION",
                "RUNNER_METADATA"
            ]
        
        market_filter = filters.market_filter(
            event_type_ids=event_type_ids,
            market_start_time={
                'from': datetime.now(),
                'to': datetime.now() + timedelta(days=7)
            }
        )
        
        return self.client.betting.list_market_catalogue(
            filter=market_filter,
            market_projection=market_projection,
            max_results=max_results
        )
    
    def get_market_book(
        self,
        market_ids: List[str],
        price_projection: Dict[str, Any] = None
    ) -> List[Dict]:
        """Get current market prices"""
        if price_projection is None:
            price_projection = {
                'priceData': ['EX_BEST_OFFERS', 'EX_TRADED'],
                'virtualise': True
            }
        
        return self.client.betting.list_market_book(
            market_ids=market_ids,
            price_projection=price_projection
        )
    
    def place_order(
        self,
        market_id: str,
        selection_id: int,
        side: str,
        order_type: str,
        size: float,
        price: float = None,
        persistence_type: str = "LAPSE"
    ) -> Dict:
        """Place an order on the exchange"""
        instructions = [{
            'selectionId': selection_id,
            'handicap': 0,
            'side': side,
            'orderType': order_type,
            'limitOrder': {
                'size': size,
                'price': price,
                'persistenceType': persistence_type
            }
        }]
        
        return self.client.betting.place_orders(
            market_id=market_id,
            instructions=instructions
        )
    
    def stream_market_data(self, market_ids: List[str]):
        """Stream live market data"""
        # Create a stream listener
        listener = StreamListener(
            max_latency=0.5,
        )
        
        # Create stream
        stream = self.client.streaming.create_stream(
            listener=listener,
            description="Market Stream"
        )
        
        # Subscribe to markets
        market_filter = filters.streaming_market_filter(
            market_ids=market_ids
        )
        
        market_data_filter = filters.streaming_market_data_filter(
            fields=['EX_BEST_OFFERS', 'EX_MARKET_DEF'],
            ladder_levels=3
        )
        
        stream.subscribe_to_markets(
            market_filter=market_filter,
            market_data_filter=market_data_filter,
            conflate_ms=1000
        )
        
        # Start stream
        stream.start()
        
        return stream


# Example usage
def betfair_example():
    client = BetfairClient(
        username="your_username",
        password="your_password",
        app_key="your_app_key",
        cert_path="/path/to/cert"
    )
    
    # Login
    client.login()
    
    # Get football markets
    football_markets = client.get_market_catalogue(
        event_type_ids=['1'],  # 1 = Soccer/Football
        max_results=50
    )
    
    print(f"Found {len(football_markets)} football markets")
    
    # Get prices for first market
    if football_markets:
        market_id = football_markets[0]['marketId']
        market_book = client.get_market_book([market_id])
        
        for runner in market_book[0]['runners']:
            back_price = runner['ex']['availableToBack'][0]['price'] if runner['ex']['availableToBack'] else None
            lay_price = runner['ex']['availableToLay'][0]['price'] if runner['ex']['availableToLay'] else None
            
            print(f"Selection {runner['selectionId']}: Back={back_price}, Lay={lay_price}")
    
    # Example: Place a back bet (demo - be careful with real money!)
    # order_result = client.place_order(
    #     market_id=market_id,
    #     selection_id=12345,
    #     side="BACK",
    #     order_type="LIMIT",
    #     size=10.0,
    #     price=2.0
    # )
```

## 3. Pinnacle API Integration

### Direct API Access

```python
# pinnacle_client.py
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
import base64
from datetime import datetime

class PinnacleClient:
    """Client for Pinnacle Sports API"""
    
    BASE_URL = "https://api.pinnacle.com"
    
    def __init__(self, username: str, password: str):
        self.auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        return {
            "Authorization": f"Basic {self.auth}",
            "Content-Type": "application/json"
        }
    
    async def get_sports(self) -> List[Dict[str, Any]]:
        """Get list of sports"""
        url = f"{self.BASE_URL}/v2/sports"
        
        async with self.session.get(url, headers=self._get_headers()) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_fixtures(
        self,
        sport_id: int,
        is_live: bool = False,
        since: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get fixtures for a sport"""
        url = f"{self.BASE_URL}/v1/fixtures"
        
        params = {
            "sportId": sport_id,
            "isLive": 1 if is_live else 0
        }
        
        if since:
            params["since"] = since
            
        async with self.session.get(
            url,
            headers=self._get_headers(),
            params=params
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_odds(
        self,
        sport_id: int,
        is_live: bool = False,
        since: Optional[int] = None,
        event_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Get odds for events"""
        url = f"{self.BASE_URL}/v1/odds"
        
        params = {
            "sportId": sport_id,
            "isLive": 1 if is_live else 0,
            "oddsFormat": "AMERICAN"
        }
        
        if since:
            params["since"] = since
            
        if event_ids:
            params["eventIds"] = ",".join(map(str, event_ids))
            
        async with self.session.get(
            url,
            headers=self._get_headers(),
            params=params
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_line(
        self,
        sport_id: int,
        league_id: int,
        event_id: int,
        period_number: int,
        bet_type: str,
        team: Optional[str] = None,
        side: Optional[str] = None,
        handicap: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get specific betting line"""
        url = f"{self.BASE_URL}/v1/line"
        
        params = {
            "sportId": sport_id,
            "leagueId": league_id,
            "eventId": event_id,
            "periodNumber": period_number,
            "betType": bet_type,
            "oddsFormat": "AMERICAN"
        }
        
        if team:
            params["team"] = team
        if side:
            params["side"] = side
        if handicap is not None:
            params["handicap"] = handicap
            
        async with self.session.get(
            url,
            headers=self._get_headers(),
            params=params
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def place_bet(
        self,
        sport_id: int,
        event_id: int,
        line_id: int,
        period_number: int,
        bet_type: str,
        stake: float,
        win_risk_stake: str = "RISK",
        team: Optional[str] = None,
        side: Optional[str] = None
    ) -> Dict[str, Any]:
        """Place a bet"""
        url = f"{self.BASE_URL}/v2/bets"
        
        bet_data = {
            "uniqueRequestId": str(datetime.now().timestamp()),
            "acceptBetterLine": True,
            "winRiskStake": win_risk_stake,
            "stake": stake,
            "lineId": line_id,
            "sportId": sport_id,
            "eventId": event_id,
            "periodNumber": period_number,
            "betType": bet_type
        }
        
        if team:
            bet_data["team"] = team
        if side:
            bet_data["side"] = side
            
        async with self.session.post(
            url,
            headers=self._get_headers(),
            json=bet_data
        ) as response:
            response.raise_for_status()
            return await response.json()


# Example usage
async def pinnacle_example():
    async with PinnacleClient(
        username="your_username",
        password="your_password"
    ) as client:
        # Get sports
        sports = await client.get_sports()
        
        # Find NFL
        nfl_sport = next((s for s in sports if s['name'] == 'Football'), None)
        if nfl_sport:
            sport_id = nfl_sport['id']
            
            # Get fixtures
            fixtures = await client.get_fixtures(sport_id)
            
            # Get odds for all events
            odds = await client.get_odds(sport_id)
            
            # Process for best lines
            for league in odds.get('leagues', []):
                print(f"\nLeague: {league['id']}")
                
                for event in league.get('events', []):
                    print(f"\nEvent: {event['id']}")
                    
                    for period in event.get('periods', []):
                        if period['number'] == 0:  # Full game
                            
                            # Money line
                            if 'moneyline' in period:
                                ml = period['moneyline']
                                print(f"  Moneyline - Home: {ml.get('home')}, Away: {ml.get('away')}")
                            
                            # Spreads
                            if 'spreads' in period:
                                for spread in period['spreads']:
                                    print(f"  Spread {spread['hdp']} - Home: {spread.get('home')}, Away: {spread.get('away')}")
                            
                            # Totals
                            if 'totals' in period:
                                for total in period['totals']:
                                    print(f"  Total {total['points']} - Over: {total.get('over')}, Under: {total.get('under')}")
```

## 4. Aggregated Odds Comparison

### Multi-Provider Aggregation

```python
# odds_aggregator.py
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OddsData:
    provider: str
    event_id: str
    home_team: str
    away_team: str
    home_odds: float
    away_odds: float
    draw_odds: Optional[float]
    timestamp: datetime
    
class MultiProviderAggregator:
    """Aggregate odds from multiple providers"""
    
    def __init__(self, providers: Dict[str, Any]):
        self.providers = providers
        
    async def get_best_odds(self, sport: str, event_id: str) -> Dict[str, Any]:
        """Get best odds from all providers"""
        
        # Fetch from all providers concurrently
        tasks = []
        for provider_name, provider_client in self.providers.items():
            if provider_name == "theoddsapi":
                tasks.append(self._get_theoddsapi_odds(provider_client, sport))
            elif provider_name == "betfair":
                tasks.append(self._get_betfair_odds(provider_client, sport))
            elif provider_name == "pinnacle":
                tasks.append(self._get_pinnacle_odds(provider_client, sport))
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process and find best odds
        best_odds = self._find_best_odds(results)
        arbitrage_opps = self._calculate_arbitrage(best_odds)
        
        return {
            "best_odds": best_odds,
            "arbitrage_opportunities": arbitrage_opps,
            "providers_queried": len(self.providers),
            "timestamp": datetime.now().isoformat()
        }
    
    def _find_best_odds(self, results: List[Any]) -> Dict[str, Any]:
        """Find best odds across all providers"""
        best_by_event = {}
        
        for result in results:
            if isinstance(result, Exception):
                continue
                
            for event in result.get('events', []):
                event_key = f"{event['home_team']} vs {event['away_team']}"
                
                if event_key not in best_by_event:
                    best_by_event[event_key] = {
                        'home_team': event['home_team'],
                        'away_team': event['away_team'],
                        'best_home': {'odds': 0, 'provider': None},
                        'best_away': {'odds': 0, 'provider': None},
                        'best_draw': {'odds': 0, 'provider': None}
                    }
                
                # Update best odds
                if event['home_odds'] > best_by_event[event_key]['best_home']['odds']:
                    best_by_event[event_key]['best_home'] = {
                        'odds': event['home_odds'],
                        'provider': event['provider']
                    }
                    
                if event['away_odds'] > best_by_event[event_key]['best_away']['odds']:
                    best_by_event[event_key]['best_away'] = {
                        'odds': event['away_odds'],
                        'provider': event['provider']
                    }
                    
                if event.get('draw_odds', 0) > best_by_event[event_key]['best_draw']['odds']:
                    best_by_event[event_key]['best_draw'] = {
                        'odds': event['draw_odds'],
                        'provider': event['provider']
                    }
                    
        return best_by_event
    
    def _calculate_arbitrage(self, best_odds: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate arbitrage opportunities"""
        opportunities = []
        
        for event, odds_data in best_odds.items():
            # Two-way arbitrage (no draw)
            if odds_data['best_home']['odds'] > 0 and odds_data['best_away']['odds'] > 0:
                home_prob = 1 / odds_data['best_home']['odds']
                away_prob = 1 / odds_data['best_away']['odds']
                total_prob = home_prob + away_prob
                
                if total_prob < 1:
                    profit = (1 - total_prob) * 100
                    opportunities.append({
                        'type': 'two_way',
                        'event': event,
                        'profit_percentage': profit,
                        'home_bet': {
                            'provider': odds_data['best_home']['provider'],
                            'odds': odds_data['best_home']['odds'],
                            'stake_percentage': away_prob / total_prob * 100
                        },
                        'away_bet': {
                            'provider': odds_data['best_away']['provider'],
                            'odds': odds_data['best_away']['odds'],
                            'stake_percentage': home_prob / total_prob * 100
                        }
                    })
                    
            # Three-way arbitrage (with draw)
            if (odds_data['best_home']['odds'] > 0 and 
                odds_data['best_away']['odds'] > 0 and 
                odds_data['best_draw']['odds'] > 0):
                
                home_prob = 1 / odds_data['best_home']['odds']
                away_prob = 1 / odds_data['best_away']['odds']
                draw_prob = 1 / odds_data['best_draw']['odds']
                total_prob = home_prob + away_prob + draw_prob
                
                if total_prob < 1:
                    profit = (1 - total_prob) * 100
                    opportunities.append({
                        'type': 'three_way',
                        'event': event,
                        'profit_percentage': profit,
                        'home_bet': {
                            'provider': odds_data['best_home']['provider'],
                            'odds': odds_data['best_home']['odds'],
                            'stake_percentage': (away_prob + draw_prob) / total_prob * 100
                        },
                        'away_bet': {
                            'provider': odds_data['best_away']['provider'],
                            'odds': odds_data['best_away']['odds'],
                            'stake_percentage': (home_prob + draw_prob) / total_prob * 100
                        },
                        'draw_bet': {
                            'provider': odds_data['best_draw']['provider'],
                            'odds': odds_data['best_draw']['odds'],
                            'stake_percentage': (home_prob + away_prob) / total_prob * 100
                        }
                    })
                    
        return sorted(opportunities, key=lambda x: x['profit_percentage'], reverse=True)


# Example usage
async def aggregator_example():
    # Initialize providers
    providers = {
        "theoddsapi": TheOddsAPIClient("api_key"),
        "betfair": BetfairClient("user", "pass", "app_key", "cert"),
        "pinnacle": PinnacleClient("user", "pass")
    }
    
    aggregator = MultiProviderAggregator(providers)
    
    # Get best odds and arbitrage opportunities
    result = await aggregator.get_best_odds(
        sport="soccer_epl",
        event_id="match_123"
    )
    
    print(f"Found {len(result['arbitrage_opportunities'])} arbitrage opportunities")
    
    for opp in result['arbitrage_opportunities'][:5]:  # Top 5
        print(f"\nEvent: {opp['event']}")
        print(f"Profit: {opp['profit_percentage']:.2f}%")
        print(f"Betting strategy:")
        
        if opp['type'] == 'two_way':
            print(f"  - Home: ${opp['home_bet']['stake_percentage']:.2f} at {opp['home_bet']['provider']}")
            print(f"  - Away: ${opp['away_bet']['stake_percentage']:.2f} at {opp['away_bet']['provider']}")
        else:
            print(f"  - Home: ${opp['home_bet']['stake_percentage']:.2f} at {opp['home_bet']['provider']}")
            print(f"  - Away: ${opp['away_bet']['stake_percentage']:.2f} at {opp['away_bet']['provider']}")
            print(f"  - Draw: ${opp['draw_bet']['stake_percentage']:.2f} at {opp['draw_bet']['provider']}")
```

## 5. WebSocket Streaming Implementation

### Real-time Odds Streaming

```python
# websocket_streaming.py
import asyncio
import websockets
import json
from typing import Dict, Callable, Any
from datetime import datetime

class OddsStreamManager:
    """Manage WebSocket connections for real-time odds"""
    
    def __init__(self):
        self.connections = {}
        self.callbacks = {}
        
    async def connect_betfair_stream(
        self,
        session_token: str,
        app_key: str,
        callback: Callable
    ):
        """Connect to Betfair streaming API"""
        uri = "wss://stream-api.betfair.com:443/stream"
        
        async with websockets.connect(uri) as websocket:
            # Authenticate
            auth_message = {
                "op": "authentication",
                "appKey": app_key,
                "session": session_token
            }
            await websocket.send(json.dumps(auth_message))
            
            # Handle messages
            async for message in websocket:
                data = json.loads(message)
                await callback(data)
                
    async def connect_lsports_stream(
        self,
        token: str,
        sports: List[str],
        callback: Callable
    ):
        """Connect to LSports WebSocket feed"""
        uri = f"wss://api.lsports.eu/ws?token={token}"
        
        async with websockets.connect(uri) as websocket:
            # Subscribe to sports
            subscribe_message = {
                "action": "subscribe",
                "sports": sports,
                "markets": ["1X2", "Handicap", "Total"]
            }
            await websocket.send(json.dumps(subscribe_message))
            
            # Handle messages
            async for message in websocket:
                data = json.loads(message)
                await callback(data)
                
    async def process_odds_update(self, data: Dict[str, Any]):
        """Process incoming odds updates"""
        update_type = data.get('type', 'unknown')
        
        if update_type == 'odds_change':
            # Process odds change
            event_id = data['eventId']
            market = data['market']
            odds = data['odds']
            
            # Check for significant movements
            if self.is_significant_movement(event_id, market, odds):
                await self.alert_significant_movement(event_id, market, odds)
                
        elif update_type == 'market_suspend':
            # Handle market suspension
            await self.handle_market_suspension(data)
            
    def is_significant_movement(
        self,
        event_id: str,
        market: str,
        new_odds: Dict
    ) -> bool:
        """Check if odds movement is significant"""
        # Get previous odds from cache
        previous = self.get_cached_odds(event_id, market)
        
        if not previous:
            return False
            
        # Calculate percentage change
        for outcome, odds in new_odds.items():
            prev_odds = previous.get(outcome, 0)
            if prev_odds > 0:
                change = abs((odds - prev_odds) / prev_odds)
                if change > 0.1:  # 10% change
                    return True
                    
        return False
        
    async def alert_significant_movement(
        self,
        event_id: str,
        market: str,
        odds: Dict
    ):
        """Alert when significant odds movement detected"""
        alert = {
            "type": "significant_movement",
            "event_id": event_id,
            "market": market,
            "odds": odds,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all registered callbacks
        for callback in self.callbacks.values():
            await callback(alert)


# Example WebSocket consumer
async def odds_stream_consumer():
    manager = OddsStreamManager()
    
    async def handle_odds_update(data):
        """Handle incoming odds updates"""
        print(f"Received update: {data['type']}")
        
        if data['type'] == 'significant_movement':
            print(f"ALERT: Significant movement in {data['event_id']}")
            print(f"Market: {data['market']}")
            print(f"New odds: {data['odds']}")
            
            # Could trigger automated trading here
            
    # Register callback
    manager.callbacks['main'] = handle_odds_update
    
    # Connect to streams
    await asyncio.gather(
        manager.connect_betfair_stream(
            session_token="token",
            app_key="app_key",
            callback=manager.process_odds_update
        ),
        manager.connect_lsports_stream(
            token="token",
            sports=["soccer", "basketball"],
            callback=manager.process_odds_update
        )
    )
```

## 6. Compliance Integration

### KYC and Geolocation Example

```python
# compliance_integration.py
import aiohttp
from typing import Dict, Any, Tuple
import ipinfo
from datetime import datetime

class ComplianceManager:
    """Manage compliance checks for sports betting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.ipinfo_handler = ipinfo.getHandler(config['ipinfo_token'])
        self.jumio_api_key = config['jumio_api_key']
        self.jumio_api_secret = config['jumio_api_secret']
        
    async def perform_kyc_check(
        self,
        user_id: str,
        id_front_image: bytes,
        id_back_image: bytes,
        selfie_image: bytes
    ) -> Dict[str, Any]:
        """Perform KYC verification using Jumio"""
        
        url = "https://netverify.com/api/v4/initiate"
        
        headers = {
            "Authorization": f"Bearer {self.jumio_api_key}:{self.jumio_api_secret}",
            "Content-Type": "application/json"
        }
        
        # Create verification request
        data = {
            "customerInternalReference": user_id,
            "userReference": user_id,
            "reportingCriteria": "sports_betting_kyc",
            "callbackUrl": "https://yourapi.com/kyc/callback",
            "workflowId": 100,  # Your workflow ID
            "type": "ID_VERIFICATION",
            "country": "USA",
            "documents": [
                {
                    "type": "DRIVING_LICENSE",
                    "country": "USA"
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                
                # Upload images
                await self._upload_kyc_images(
                    session,
                    result['transactionReference'],
                    id_front_image,
                    id_back_image,
                    selfie_image
                )
                
                return {
                    "status": "initiated",
                    "transaction_reference": result['transactionReference'],
                    "verification_url": result['redirectUrl']
                }
    
    def check_geolocation(self, ip_address: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if user is in permitted location"""
        details = self.ipinfo_handler.getDetails(ip_address)
        
        location = {
            "country": details.country,
            "region": details.region,
            "city": details.city,
            "latitude": details.latitude,
            "longitude": details.longitude
        }
        
        # Check if location is permitted
        permitted = self._is_location_permitted(location)
        
        return permitted, location
    
    def _is_location_permitted(self, location: Dict[str, Any]) -> bool:
        """Check if location allows sports betting"""
        
        # US states where online sports betting is legal
        us_permitted_states = {
            'Arizona', 'Colorado', 'Connecticut', 'Illinois', 'Indiana',
            'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
            'Massachusetts', 'Michigan', 'Nevada', 'New Hampshire',
            'New Jersey', 'New York', 'North Carolina', 'Ohio', 'Oregon',
            'Pennsylvania', 'Rhode Island', 'Tennessee', 'Vermont',
            'Virginia', 'West Virginia', 'Wyoming'
        }
        
        if location['country'] == 'US':
            return location['region'] in us_permitted_states
            
        # European countries (simplified - check actual regulations)
        eu_permitted = {
            'GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'DK', 'SE'
        }
        
        return location['country'] in eu_permitted
    
    async def check_responsible_gambling_limits(
        self,
        user_id: str,
        bet_amount: float,
        bet_type: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if bet complies with responsible gambling limits"""
        
        # Get user's limits from database
        limits = await self.get_user_limits(user_id)
        
        # Get user's current activity
        activity = await self.get_user_activity(user_id)
        
        # Check daily limit
        if activity['daily_total'] + bet_amount > limits['daily_limit']:
            return False, "Daily betting limit exceeded"
            
        # Check loss limit
        if activity['daily_loss'] > limits['loss_limit']:
            return False, "Daily loss limit reached"
            
        # Check time limit
        if activity['session_duration'] > limits['time_limit_minutes'] * 60:
            return False, "Session time limit exceeded"
            
        # Check cool-off period
        if self.in_cool_off_period(user_id):
            return False, "User in cool-off period"
            
        return True, None


# Example usage
async def compliance_example():
    compliance = ComplianceManager({
        'ipinfo_token': 'your_token',
        'jumio_api_key': 'your_key',
        'jumio_api_secret': 'your_secret'
    })
    
    # Check geolocation
    ip_address = "1.2.3.4"
    permitted, location = compliance.check_geolocation(ip_address)
    
    if not permitted:
        print(f"Betting not permitted in {location['region']}, {location['country']}")
        return
        
    # Check betting limits
    can_bet, reason = await compliance.check_responsible_gambling_limits(
        user_id="user123",
        bet_amount=100.0,
        bet_type="single"
    )
    
    if not can_bet:
        print(f"Bet rejected: {reason}")
        return
        
    print("All compliance checks passed!")
```

These examples provide comprehensive integration patterns for the major sports betting APIs, including authentication, real-time streaming, compliance checks, and multi-provider aggregation. Each can be adapted to your specific requirements and integrated with the existing AI News Trading platform.