"""
The Odds API Client
Handles authentication, rate limiting, and API communication
"""

import os
import time
import logging
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class OddsAPIClient:
    """
    Client for The Odds API v4
    Handles authentication, rate limiting, and error handling
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Odds API client

        Args:
            api_key: The Odds API key. If None, will try to load from environment
        """
        self.api_key = api_key or os.getenv('THE_ODDS_API_KEY')
        if not self.api_key:
            raise ValueError(
                "The Odds API key not found. Please set THE_ODDS_API_KEY environment variable "
                "or pass api_key parameter. Get your free key at: https://the-odds-api.com/"
            )

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Neural-Trader/1.0.0'
        })

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        self.requests_remaining = None
        self.requests_used = None

        # Cache for sports list (rarely changes)
        self._sports_cache = None
        self._sports_cache_time = None
        self._sports_cache_duration = 3600  # 1 hour

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and extract rate limit info

        Args:
            response: The HTTP response object

        Returns:
            JSON response data

        Raises:
            requests.HTTPError: If the request failed
        """
        # Extract rate limit headers
        self.requests_remaining = response.headers.get('x-requests-remaining')
        self.requests_used = response.headers.get('x-requests-used')

        # Log rate limit info
        if self.requests_remaining:
            logger.info(f"API requests remaining: {self.requests_remaining}")

        # Handle different response codes
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise requests.HTTPError("Invalid API key. Check your THE_ODDS_API_KEY")
        elif response.status_code == 422:
            raise requests.HTTPError(f"Invalid request parameters: {response.text}")
        elif response.status_code == 429:
            raise requests.HTTPError("Rate limit exceeded. Please wait before making more requests")
        else:
            response.raise_for_status()

    def make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make a request to The Odds API

        Args:
            endpoint: API endpoint (e.g., 'sports', 'sports/upcoming/odds')
            params: Query parameters
            use_cache: Whether to use cached responses for eligible endpoints

        Returns:
            JSON response data
        """
        # Check cache for sports endpoint
        if endpoint == 'sports' and use_cache and self._sports_cache:
            cache_age = time.time() - (self._sports_cache_time or 0)
            if cache_age < self._sports_cache_duration:
                logger.info("Using cached sports list")
                return self._sports_cache

        # Prepare request
        url = f"{self.BASE_URL}/{endpoint}"
        request_params = params.copy() if params else {}
        request_params['apiKey'] = self.api_key

        # Rate limiting
        self._wait_for_rate_limit()

        try:
            logger.info(f"Making request to: {endpoint}")
            response = self.session.get(url, params=request_params, timeout=30)
            data = self._handle_response(response)

            # Cache sports list
            if endpoint == 'sports' and use_cache:
                self._sports_cache = data
                self._sports_cache_time = time.time()

            return data

        except requests.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {e}")
            raise

    def get_sports(self) -> List[Dict[str, Any]]:
        """
        Get list of available sports

        Returns:
            List of sports with keys: key, group, title, description, active, has_outrights
        """
        return self.make_request('sports')

    def get_odds(
        self,
        sport: str,
        regions: str = 'us',
        markets: str = 'h2h',
        odds_format: str = 'decimal',
        date_format: str = 'iso',
        bookmakers: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get odds for a sport

        Args:
            sport: Sport key (e.g., 'americanfootball_nfl', 'basketball_nba')
            regions: Comma-separated list of regions (us, uk, au, eu)
            markets: Comma-separated list of markets (h2h, spreads, totals, outrights)
            odds_format: 'decimal' or 'american'
            date_format: 'iso' or 'unix'
            bookmakers: Comma-separated list of bookmaker keys (optional)

        Returns:
            List of events with odds
        """
        params = {
            'regions': regions,
            'markets': markets,
            'oddsFormat': odds_format,
            'dateFormat': date_format
        }

        if bookmakers:
            params['bookmakers'] = bookmakers

        return self.make_request(f'sports/{sport}/odds', params, use_cache=False)

    def get_event_odds(
        self,
        sport: str,
        event_id: str,
        regions: str = 'us',
        markets: str = 'h2h',
        odds_format: str = 'decimal',
        date_format: str = 'iso',
        bookmakers: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get odds for a specific event

        Args:
            sport: Sport key
            event_id: Event ID
            regions: Comma-separated list of regions
            markets: Comma-separated list of markets
            odds_format: 'decimal' or 'american'
            date_format: 'iso' or 'unix'
            bookmakers: Comma-separated list of bookmaker keys (optional)

        Returns:
            Event with odds data
        """
        params = {
            'regions': regions,
            'markets': markets,
            'oddsFormat': odds_format,
            'dateFormat': date_format
        }

        if bookmakers:
            params['bookmakers'] = bookmakers

        return self.make_request(f'sports/{sport}/events/{event_id}/odds', params, use_cache=False)

    def get_usage_info(self) -> Dict[str, Any]:
        """
        Get current API usage information

        Returns:
            Dictionary with requests_remaining and requests_used
        """
        return {
            'requests_remaining': self.requests_remaining,
            'requests_used': self.requests_used,
            'last_updated': datetime.now().isoformat()
        }

    def validate_connection(self) -> Dict[str, Any]:
        """
        Validate API connection and credentials

        Returns:
            Validation result with status and details
        """
        try:
            sports = self.get_sports()
            return {
                'status': 'success',
                'api_key_valid': True,
                'sports_available': len(sports),
                'requests_remaining': self.requests_remaining,
                'message': 'Successfully connected to The Odds API'
            }
        except Exception as e:
            return {
                'status': 'error',
                'api_key_valid': False,
                'error': str(e),
                'message': 'Failed to connect to The Odds API'
            }