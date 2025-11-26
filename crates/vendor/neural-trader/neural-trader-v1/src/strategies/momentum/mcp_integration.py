"""
MCP Integration Module for Neural Momentum Strategy
Integrates with AI News Trader MCP tools for real market data and analysis
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class MCPIntegration:
    """
    Integration layer with AI News Trader MCP tools
    Provides real market data, news analysis, and neural predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes
        
    async def get_market_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market analysis using MCP tools"""
        try:
            # Check cache first
            cache_key = f"market_analysis_{symbol}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # Get quick market analysis
            from mcp__ai_news_trader import quick_analysis
            analysis = await quick_analysis(symbol=symbol, use_gpu=True)
            
            # Cache the result
            self._cache_data(cache_key, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting market analysis for {symbol}: {e}")
            return self._get_fallback_analysis(symbol)
    
    async def get_news_sentiment(self, symbol: str, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get news sentiment analysis using MCP tools"""
        try:
            cache_key = f"news_sentiment_{symbol}_{lookback_hours}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # Get news sentiment analysis
            from mcp__ai_news_trader import analyze_news
            sentiment = await analyze_news(
                symbol=symbol,
                lookback_hours=lookback_hours,
                sentiment_model="enhanced",
                use_gpu=True
            )
            
            self._cache_data(cache_key, sentiment)
            return sentiment
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return self._get_fallback_sentiment(symbol)
    
    async def get_neural_forecast(self, symbol: str, horizon: int = 5) -> Dict[str, Any]:
        """Get neural price forecast using MCP tools"""
        try:
            cache_key = f"neural_forecast_{symbol}_{horizon}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # Get neural forecast
            from mcp__ai_news_trader import neural_forecast
            forecast = await neural_forecast(
                symbol=symbol,
                horizon=horizon,
                confidence_level=0.95,
                use_gpu=True
            )
            
            self._cache_data(cache_key, forecast)
            return forecast
            
        except Exception as e:
            logger.error(f"Error getting neural forecast for {symbol}: {e}")
            return self._get_fallback_forecast(symbol)
    
    async def get_correlation_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Get correlation analysis between symbols"""
        try:
            cache_key = f"correlation_{'-'.join(sorted(symbols))}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # Get correlation analysis
            from mcp__ai_news_trader import correlation_analysis
            correlations = await correlation_analysis(
                symbols=symbols,
                period_days=90,
                use_gpu=True
            )
            
            self._cache_data(cache_key, correlations)
            return correlations
            
        except Exception as e:
            logger.error(f"Error getting correlation analysis: {e}")
            return self._get_fallback_correlations(symbols)
    
    async def run_risk_analysis(self, portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run portfolio risk analysis using MCP tools"""
        try:
            # Get risk analysis
            from mcp__ai_news_trader import risk_analysis
            risk_metrics = await risk_analysis(
                portfolio=portfolio,
                time_horizon=1,
                use_gpu=True,
                use_monte_carlo=True,
                var_confidence=0.05
            )
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error running risk analysis: {e}")
            return self._get_fallback_risk_metrics()
    
    async def simulate_strategy(self, strategy: str, symbol: str, action: str) -> Dict[str, Any]:
        """Simulate trading strategy using MCP tools"""
        try:
            # Simulate trade
            from mcp__ai_news_trader import simulate_trade
            simulation = await simulate_trade(
                strategy=strategy,
                symbol=symbol,
                action=action,
                use_gpu=True
            )
            
            return simulation
            
        except Exception as e:
            logger.error(f"Error simulating strategy: {e}")
            return self._get_fallback_simulation()
    
    async def get_strategy_performance(self, strategy: str) -> Dict[str, Any]:
        """Get strategy performance metrics using MCP tools"""
        try:
            # Get performance report
            from mcp__ai_news_trader import performance_report
            performance = await performance_report(
                strategy=strategy,
                period_days=30,
                include_benchmark=True,
                use_gpu=False
            )
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return self._get_fallback_performance()
    
    async def get_market_data_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market data for multiple symbols efficiently"""
        try:
            results = {}
            
            # Process symbols in parallel
            tasks = []
            for symbol in symbols:
                task = self.get_market_analysis(symbol)
                tasks.append((symbol, task))
            
            # Wait for all tasks to complete
            completed_tasks = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            # Process results
            for (symbol, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"Error getting data for {symbol}: {result}")
                    results[symbol] = self._get_fallback_analysis(symbol)
                else:
                    results[symbol] = result
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting batch market data: {e}")
            return {symbol: self._get_fallback_analysis(symbol) for symbol in symbols}
    
    async def get_momentum_signals(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get momentum signals for symbols using integrated analysis"""
        signals = []
        
        try:
            # Get market data for all symbols
            market_data = await self.get_market_data_batch(symbols)
            
            # Get correlations between symbols
            correlations = await self.get_correlation_analysis(symbols)
            
            for symbol in symbols:
                try:
                    # Get individual symbol data
                    analysis = market_data.get(symbol, {})
                    sentiment = await self.get_news_sentiment(symbol)
                    forecast = await self.get_neural_forecast(symbol)
                    
                    # Calculate momentum signal
                    signal = self._calculate_momentum_signal(
                        symbol, analysis, sentiment, forecast, correlations
                    )
                    
                    if signal:
                        signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error processing signal for {symbol}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting momentum signals: {e}")
            return []
    
    def _calculate_momentum_signal(self, symbol: str, analysis: Dict[str, Any], 
                                 sentiment: Dict[str, Any], forecast: Dict[str, Any],
                                 correlations: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate momentum signal from integrated data"""
        try:
            # Extract key metrics
            price = analysis.get('analysis', {}).get('price', 0)
            trend = analysis.get('analysis', {}).get('trend', 'neutral')
            volatility = analysis.get('analysis', {}).get('volatility', 'medium')
            rsi = analysis.get('analysis', {}).get('rsi', 50)
            macd = analysis.get('analysis', {}).get('macd', 0)
            
            # Sentiment metrics
            overall_sentiment = sentiment.get('overall_sentiment', 0)
            sentiment_confidence = max([
                article.get('confidence', 0) 
                for article in sentiment.get('articles', [])
            ]) if sentiment.get('articles') else 0
            
            # Forecast metrics
            forecast_direction = forecast.get('forecast_direction', 0) if forecast else 0
            forecast_confidence = forecast.get('confidence', 0) if forecast else 0
            
            # Calculate composite momentum score
            technical_score = self._calculate_technical_score(rsi, macd, trend, volatility)
            sentiment_score = self._calculate_sentiment_score(overall_sentiment, sentiment_confidence)
            neural_score = self._calculate_neural_score(forecast_direction, forecast_confidence)
            
            # Combine scores with weights
            momentum_score = (
                technical_score * 0.4 +
                sentiment_score * 0.3 +
                neural_score * 0.3
            )
            
            # Determine signal direction and strength
            if momentum_score > 0.6:
                direction = 'long'
                strength = min((momentum_score - 0.6) / 0.4, 1.0)
            elif momentum_score < 0.4:
                direction = 'short' 
                strength = min((0.4 - momentum_score) / 0.4, 1.0)
            else:
                return None  # No clear signal
            
            # Calculate overall confidence
            confidence = np.mean([
                min(abs(technical_score - 0.5) * 4, 1.0),
                sentiment_confidence,
                forecast_confidence
            ])
            
            return {
                'symbol': symbol,
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'momentum_score': momentum_score,
                'technical_score': technical_score,
                'sentiment_score': sentiment_score,
                'neural_score': neural_score,
                'price': price,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum signal for {symbol}: {e}")
            return None
    
    def _calculate_technical_score(self, rsi: float, macd: float, trend: str, volatility: str) -> float:
        """Calculate technical analysis component score"""
        score = 0.5  # Neutral starting point
        
        # RSI component
        if rsi > 70:
            score -= 0.2  # Overbought
        elif rsi < 30:
            score += 0.3  # Oversold momentum
        elif 40 < rsi < 60:
            score += 0.1  # Neutral momentum
        
        # MACD component
        if macd > 0:
            score += 0.2
        elif macd < 0:
            score -= 0.1
        
        # Trend component
        if trend == 'bullish':
            score += 0.2
        elif trend == 'bearish':
            score -= 0.2
        
        # Volatility adjustment
        if volatility == 'high':
            score *= 0.9  # Reduce confidence in high volatility
        
        return max(0, min(1, score))
    
    def _calculate_sentiment_score(self, sentiment: float, confidence: float) -> float:
        """Calculate sentiment component score"""
        # Normalize sentiment from [-1, 1] to [0, 1]
        base_score = (sentiment + 1) / 2
        
        # Weight by confidence
        weighted_score = base_score * confidence + 0.5 * (1 - confidence)
        
        return max(0, min(1, weighted_score))
    
    def _calculate_neural_score(self, forecast_direction: float, forecast_confidence: float) -> float:
        """Calculate neural prediction component score"""
        # Normalize forecast direction to [0, 1]
        base_score = (forecast_direction + 1) / 2
        
        # Weight by confidence
        weighted_score = base_score * forecast_confidence + 0.5 * (1 - forecast_confidence)
        
        return max(0, min(1, weighted_score))
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and still valid"""
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key]['timestamp']
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl
    
    def _cache_data(self, key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    # Fallback methods for when MCP tools are unavailable
    def _get_fallback_analysis(self, symbol: str) -> Dict[str, Any]:
        """Fallback market analysis"""
        return {
            'analysis': {
                'price': 100.0,
                'trend': 'neutral',
                'volatility': 'medium',
                'rsi': 50.0,
                'macd': 0.0,
                'recommendation': 'hold'
            },
            'processing': {'method': 'fallback'},
            'status': 'fallback'
        }
    
    def _get_fallback_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fallback sentiment analysis"""
        return {
            'overall_sentiment': 0.0,
            'articles': [],
            'processing': {'method': 'fallback'},
            'status': 'fallback'
        }
    
    def _get_fallback_forecast(self, symbol: str) -> Dict[str, Any]:
        """Fallback neural forecast"""
        return {
            'forecast_direction': 0.0,
            'confidence': 0.5,
            'processing': {'method': 'fallback'},
            'status': 'fallback'
        }
    
    def _get_fallback_correlations(self, symbols: List[str]) -> Dict[str, Any]:
        """Fallback correlation analysis"""
        return {
            'correlations': {f"{s1}-{s2}": 0.3 for s1 in symbols for s2 in symbols if s1 != s2},
            'status': 'fallback'
        }
    
    def _get_fallback_risk_metrics(self) -> Dict[str, Any]:
        """Fallback risk metrics"""
        return {
            'var_95': 0.02,
            'expected_shortfall': 0.03,
            'volatility': 0.2,
            'status': 'fallback'
        }
    
    def _get_fallback_simulation(self) -> Dict[str, Any]:
        """Fallback simulation results"""
        return {
            'expected_return': 0.02,
            'risk': 0.05,
            'sharpe_ratio': 0.4,
            'status': 'fallback'
        }
    
    def _get_fallback_performance(self) -> Dict[str, Any]:
        """Fallback performance metrics"""
        return {
            'total_return': 0.05,
            'sharpe_ratio': 1.0,
            'max_drawdown': 0.08,
            'win_rate': 0.6,
            'status': 'fallback'
        }