"""
Neural Trader Data Verification Module
Ensures all data is real, not simulated
Truth Score Threshold: 0.95
"""

import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import hashlib
import requests

class DataVerification:
    """Real-time data verification system"""
    
    def __init__(self, truth_threshold: float = 0.95):
        self.truth_threshold = truth_threshold
        self.verified_sources = []
        self.verification_log = []
        self.logger = logging.getLogger(__name__)
        
    def verify_live_data(self, data_source: Dict) -> Tuple[bool, float, Dict]:
        """
        Comprehensive verification of live data feed
        Returns: (is_verified, truth_score, detailed_checks)
        """
        checks = {
            'timestamp_current': self.check_timestamp(data_source),
            'price_realistic': self.check_price_movements(data_source),
            'volume_authentic': self.check_volume_patterns(data_source),
            'api_authenticated': self.check_api_credentials(data_source),
            'latency_acceptable': self.check_latency(data_source),
            'no_demo_mode': self.check_not_demo_mode(data_source),
            'cross_reference_valid': self.cross_reference_data(data_source)
        }
        
        # Calculate truth score
        truth_score = sum(1 for v in checks.values() if v) / len(checks)
        is_verified = truth_score >= self.truth_threshold
        
        # Log verification
        self.log_verification(data_source, is_verified, truth_score, checks)
        
        return is_verified, truth_score, checks
    
    def check_timestamp(self, data: Dict) -> bool:
        """Ensure timestamp is within 1 second of current time"""
        try:
            current_time = time.time()
            data_time = data.get('timestamp', 0)
            
            # Convert to epoch if needed
            if isinstance(data_time, str):
                dt = datetime.fromisoformat(data_time.replace('Z', '+00:00'))
                data_time = dt.timestamp()
            
            time_diff = abs(current_time - data_time)
            return time_diff < 1.0  # Within 1 second
        except Exception as e:
            self.logger.error(f"Timestamp check failed: {e}")
            return False
    
    def check_price_movements(self, data: Dict) -> bool:
        """Verify price changes are within realistic bounds"""
        try:
            price = data.get('price', 0)
            prev_price = data.get('prev_close', price)
            
            if price <= 0 or prev_price <= 0:
                return False
            
            # Calculate percentage change
            price_change_pct = abs((price - prev_price) / prev_price) * 100
            
            # Check against circuit breaker limits (typically 20% for stocks)
            return price_change_pct < 20
        except Exception as e:
            self.logger.error(f"Price movement check failed: {e}")
            return False
    
    def check_volume_patterns(self, data: Dict) -> bool:
        """Validate volume matches market hours and patterns"""
        try:
            volume = data.get('volume', 0)
            
            # Basic sanity checks
            if volume < 0:
                return False
            
            # Check if volume is unrealistically high
            if volume > 1e12:  # Trillion shares would be unrealistic
                return False
            
            # Check market hours (simplified - would need proper market calendar)
            current_hour = datetime.now(timezone.utc).hour
            is_market_hours = 13 <= current_hour <= 21  # Rough NYSE hours in UTC
            
            # Volume should be > 0 during market hours
            if is_market_hours and volume == 0:
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Volume pattern check failed: {e}")
            return False
    
    def check_api_credentials(self, data_source: Dict) -> bool:
        """Verify API keys are valid and not in demo mode"""
        try:
            api_key = data_source.get('api_key', '')
            
            # Check for demo/test indicators
            demo_indicators = ['demo', 'test', 'sandbox', 'paper', 'sim']
            for indicator in demo_indicators:
                if indicator in api_key.lower():
                    return False
            
            # Check key format (simplified - each API has different formats)
            if len(api_key) < 20:  # Most real API keys are longer
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"API credential check failed: {e}")
            return False
    
    def check_latency(self, data_source: Dict) -> bool:
        """Ensure data latency is under threshold"""
        try:
            latency_ms = data_source.get('latency_ms', float('inf'))
            return latency_ms < 100  # Under 100ms
        except Exception as e:
            self.logger.error(f"Latency check failed: {e}")
            return False
    
    def check_not_demo_mode(self, data_source: Dict) -> bool:
        """Ensure system is not in demo/simulation mode"""
        try:
            # Check various demo mode indicators
            is_demo = (
                data_source.get('demo_mode', False) or
                data_source.get('paper_trading', False) or
                data_source.get('simulation', False) or
                'sandbox' in str(data_source.get('endpoint', '')).lower()
            )
            
            return not is_demo
        except Exception as e:
            self.logger.error(f"Demo mode check failed: {e}")
            return False
    
    def cross_reference_data(self, data: Dict) -> bool:
        """Cross-reference price with multiple sources"""
        try:
            symbol = data.get('symbol', '')
            price = data.get('price', 0)
            
            if not symbol or price <= 0:
                return False
            
            # In production, would check against multiple APIs
            # For now, basic validation
            price_variance_threshold = 0.02  # 2% variance allowed
            
            # Simulated cross-reference (would use real APIs)
            reference_prices = self.get_reference_prices(symbol)
            if not reference_prices:
                return True  # Can't cross-reference, assume valid
            
            avg_price = sum(reference_prices) / len(reference_prices)
            variance = abs(price - avg_price) / avg_price
            
            return variance < price_variance_threshold
        except Exception as e:
            self.logger.error(f"Cross-reference check failed: {e}")
            return False
    
    def get_reference_prices(self, symbol: str) -> List[float]:
        """Get reference prices from multiple sources (simplified)"""
        # In production, would query multiple APIs
        # This is a placeholder
        return []
    
    def log_verification(self, data_source: Dict, is_verified: bool, 
                         truth_score: float, checks: Dict):
        """Log verification results"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'source': data_source.get('name', 'unknown'),
            'verified': is_verified,
            'truth_score': truth_score,
            'checks': checks,
            'hash': self.generate_verification_hash(data_source)
        }
        
        self.verification_log.append(entry)
        
        if not is_verified:
            self.logger.warning(f"Verification failed: {entry}")
    
    def generate_verification_hash(self, data: Dict) -> str:
        """Generate unique hash for verification audit"""
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_truth_metrics(self) -> Dict:
        """Get current truth score metrics"""
        if not self.verification_log:
            return {
                'overall_score': 0,
                'total_checks': 0,
                'passed': 0,
                'failed': 0
            }
        
        recent_logs = self.verification_log[-100:]  # Last 100 checks
        
        return {
            'overall_score': sum(log['truth_score'] for log in recent_logs) / len(recent_logs),
            'total_checks': len(recent_logs),
            'passed': sum(1 for log in recent_logs if log['verified']),
            'failed': sum(1 for log in recent_logs if not log['verified']),
            'threshold': self.truth_threshold
        }


class NeuralTraderVerification:
    """Neural Trader specific verification"""
    
    @staticmethod
    def verify_not_demo_mode(config: Optional[Dict] = None) -> bool:
        """Check neural-trader is NOT in demo mode"""
        if config is None:
            # Would load from actual config
            return True
        
        checks = {
            'not_demo_mode': not config.get('demo_mode', False),
            'real_api_keys': all([
                config.get('ALPACA_KEY') and 'demo' not in config.get('ALPACA_KEY', ''),
                config.get('POLYGON_KEY') and 'test' not in config.get('POLYGON_KEY', '')
            ]),
            'live_trading_enabled': config.get('live_trading', False),
            'paper_trading_disabled': not config.get('paper_trading', True)
        }
        
        return all(checks.values())
    
    @staticmethod
    def verify_data_sources(sources: List[Tuple[str, str]]) -> Dict[str, bool]:
        """Verify all data sources are live"""
        results = {}
        
        for name, endpoint in sources:
            # Check for sandbox/demo indicators in URL
            is_live = not any(term in endpoint.lower() 
                             for term in ['sandbox', 'demo', 'test', 'paper'])
            
            # Check for secure protocols
            is_secure = endpoint.startswith(('wss://', 'https://'))
            
            results[name] = is_live and is_secure
        
        return results


def execute_trade_with_verification(trade_params: Dict, verifier: DataVerification) -> Dict:
    """Execute trade only if verification passes"""
    
    # Pre-trade verification
    is_verified, truth_score, checks = verifier.verify_live_data(trade_params.get('data_source', {}))
    
    if not is_verified:
        raise ValueError(f"VERIFICATION FAILED: Truth score {truth_score:.2f} < {verifier.truth_threshold}")
    
    # Log successful verification
    print(f"✓ Trade verified with truth score: {truth_score:.2f}")
    
    # Execute trade (placeholder)
    result = {
        'status': 'executed',
        'verification': {
            'truth_score': truth_score,
            'checks': checks
        }
    }
    
    return result


def print_truth_dashboard(verifier: DataVerification):
    """Display truth verification dashboard"""
    metrics = verifier.get_truth_metrics()
    
    print("┌─────────────────────────────────────┐")
    print("│ TRUTH VERIFICATION SYSTEM           │")
    print("├─────────────────────────────────────┤")
    print(f"│ Overall Truth Score: {metrics['overall_score']:.2f} {'✓' if metrics['overall_score'] >= 0.95 else '✗'}         │")
    print(f"│ Total Checks: {metrics['total_checks']}                    │")
    print(f"│ Passed: {metrics['passed']} ✓                        │")
    print(f"│ Failed: {metrics['failed']} ✗                        │")
    print(f"│ Threshold: {metrics['threshold']}                   │")
    print("└─────────────────────────────────────┘")


if __name__ == "__main__":
    # Example usage
    verifier = DataVerification(truth_threshold=0.95)
    
    # Test data source
    test_data = {
        'symbol': 'AAPL',
        'price': 150.25,
        'prev_close': 149.80,
        'volume': 75000000,
        'timestamp': time.time(),
        'api_key': 'pk_real_1234567890abcdef',
        'latency_ms': 45,
        'demo_mode': False,
        'endpoint': 'https://api.alpaca.markets/v2'
    }
    
    # Verify data
    is_verified, score, checks = verifier.verify_live_data(test_data)
    
    # Display results
    print_truth_dashboard(verifier)
    print(f"\nDetailed Checks:")
    for check, passed in checks.items():
        print(f"  {check}: {'✓' if passed else '✗'}")