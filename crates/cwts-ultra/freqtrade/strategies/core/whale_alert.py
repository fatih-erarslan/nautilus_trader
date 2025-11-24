#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:40:17 2025

@author: ashina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Whale Alert System

This system continuously monitors cryptocurrency markets for large transactions
and unusual trading activity that may indicate whale movements. It leverages
the existing WhaleDetector and CDFA components to generate reliable alerts.

Author: Created on May 20, 2025
"""

import os
import time
import logging
import json
import threading
import queue
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
import websocket
import requests
import warnings

# Import core components from the CDFA suite
from advanced_cdfa import AdvancedCDFA, AdvancedCDFAConfig
from cdfa_extensions.cdfa_integration import CDFAIntegration
from cdfa_extensions.detectors.whale_detector import WhaleDetector, WhaleParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('whale_alert.log')
    ]
)

logger = logging.getLogger('WhaleAlertSystem')

class MarketDataSource:
    """
    Base class for market data sources.
    Implementations can connect to exchanges via websockets or REST APIs.
    """
    
    def __init__(self, symbols: List[str], timeframe: str = '1m'):
        """Initialize the data source"""
        self.symbols = symbols
        self.timeframe = timeframe
        self.latest_data: Dict[str, pd.DataFrame] = {}
        self.callbacks: List[callable] = []
        self.running = False
        self._thread = None
        
    def register_callback(self, callback: callable) -> None:
        """Register a callback for new data"""
        self.callbacks.append(callback)
        
    def start(self) -> None:
        """Start the data collection"""
        if self._thread is not None and self._thread.is_alive():
            return
            
        self.running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        
    def stop(self) -> None:
        """Stop the data collection"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get the latest data for a symbol"""
        return self.latest_data.get(symbol)
    
    def _run(self) -> None:
        """Run the data collection (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _run method")
    
    def _notify_callbacks(self, symbol: str, data: pd.DataFrame) -> None:
        """Notify all callbacks of new data"""
        for callback in self.callbacks:
            try:
                callback(symbol, data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")


class BinanceDataSource(MarketDataSource):
    """
    Market data source for Binance exchange.
    Connects to Binance API to get real-time and historical price data.
    """
    
    def __init__(self, symbols: List[str], timeframe: str = '1m', 
                lookback_bars: int = 500):
        """Initialize Binance data source"""
        super().__init__(symbols, timeframe)
        self.base_url = "https://api.binance.com/api/v3"
        self.ws_base_url = "wss://stream.binance.com:9443/ws"
        self.lookback_bars = lookback_bars
        self.ws = None
        self.last_update: Dict[str, float] = {}
        
    def _run(self) -> None:
        """Run the data collection"""
        # First, load historical data for context
        for symbol in self.symbols:
            try:
                hist_data = self._fetch_historical_data(symbol)
                if hist_data is not None:
                    self.latest_data[symbol] = hist_data
                    self.last_update[symbol] = time.time()
                    logger.info(f"Loaded historical data for {symbol}: {len(hist_data)} bars")
            except Exception as e:
                logger.error(f"Error loading historical data for {symbol}: {e}")
        
        # Then start the websocket connection for real-time updates
        self._start_websocket()
        
        # Regular check for missed updates (fallback to REST API)
        while self.running:
            time.sleep(30)  # Check every 30 seconds
            current_time = time.time()
            
            for symbol in self.symbols:
                # If no update in 5 minutes, fetch via REST API
                if symbol not in self.last_update or current_time - self.last_update[symbol] > 300:
                    try:
                        logger.warning(f"No recent updates for {symbol}, fetching via REST API")
                        hist_data = self._fetch_historical_data(symbol, limit=100)
                        if hist_data is not None:
                            self.latest_data[symbol] = hist_data
                            self.last_update[symbol] = current_time
                            self._notify_callbacks(symbol, hist_data)
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {e}")
    
    def _fetch_historical_data(self, symbol: str, limit: int = None) -> Optional[pd.DataFrame]:
        """Fetch historical data from REST API"""
        symbol_formatted = symbol.replace("/", "")
        limit = limit or self.lookback_bars
        endpoint = f"/klines"
        interval = self._convert_timeframe(self.timeframe)
        
        params = {
            "symbol": symbol_formatted,
            "interval": interval,
            "limit": limit
        }
        
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            
            # Parse the response data
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                               'quote_asset_volume', 'taker_buy_base_asset_volume', 
                               'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamp to datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def _start_websocket(self) -> None:
        """Start websocket connection for real-time data"""
        streams = []
        for symbol in self.symbols:
            symbol_formatted = symbol.lower().replace("/", "")
            interval = self._convert_timeframe(self.timeframe)
            streams.append(f"{symbol_formatted}@kline_{interval}")
        
        stream_path = "/".join(streams)
        ws_url = f"{self.ws_base_url}/{stream_path}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Extract the symbol and kline data
                symbol = data['s']
                k = data['k']
                
                # Only process completed candles
                if k['x']:  # Candle closed
                    orig_symbol = self._get_original_symbol(symbol)
                    
                    # Update the dataframe
                    if orig_symbol in self.latest_data:
                        df = self.latest_data[orig_symbol].copy()
                        
                        # Create a new row
                        new_row = pd.DataFrame([{
                            'open': float(k['o']),
                            'high': float(k['h']),
                            'low': float(k['l']),
                            'close': float(k['c']),
                            'volume': float(k['v']),
                            'quote_asset_volume': float(k['q']),
                            'number_of_trades': int(k['n']),
                            'taker_buy_base_asset_volume': float(k['V']),
                            'taker_buy_quote_asset_volume': float(k['Q'])
                        }], index=[pd.to_datetime(k['t'], unit='ms')])
                        
                        # Append the new row
                        df = pd.concat([df, new_row])
                        
                        # Remove duplicates and sort
                        df = df[~df.index.duplicated(keep='last')]
                        df.sort_index(inplace=True)
                        
                        # Update stored data
                        self.latest_data[orig_symbol] = df
                        self.last_update[orig_symbol] = time.time()
                        
                        # Notify callbacks
                        self._notify_callbacks(orig_symbol, df)
            except Exception as e:
                logger.error(f"Error processing websocket message: {e}")
        
        def on_error(ws, error):
            logger.error(f"Websocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"Websocket closed: {close_status_code} - {close_msg}")
            
            # Attempt reconnection after delay if still running
            if self.running:
                time.sleep(5)
                self._start_websocket()
        
        def on_open(ws):
            logger.info("Websocket connection established")
        
        # Create and start the websocket
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert our timeframe format to Binance format"""
        mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1w': '1w'
        }
        return mapping.get(timeframe, '1m')
    
    def _get_original_symbol(self, binance_symbol: str) -> str:
        """Get original symbol format from Binance format"""
        for symbol in self.symbols:
            if symbol.replace("/", "") == binance_symbol:
                return symbol
        return binance_symbol  # Fallback


class WhaleAlertSystem:
    """
    Real-time whale alert system that monitors market data,
    detects whale activity, and sends alerts.
    """
    
    def __init__(self, symbols: List[str], timeframe: str = '1m',
                 data_source_type: str = 'binance',
                 alert_threshold: float = 0.7,
                 min_alert_interval: int = 300,
                 config_path: Optional[str] = None):
        """Initialize the whale alert system"""
        self.symbols = symbols
        self.timeframe = timeframe
        self.alert_threshold = alert_threshold
        self.min_alert_interval = min_alert_interval
        self.last_alerts: Dict[str, float] = {}
        
        # Initialize detector components
        self.whale_detector = WhaleDetector()
        
        # Set whale detector parameters - using parameter names that likely exist
        # in your actual WhaleParameters class
        try:
            whale_params = WhaleParameters(
                volume_threshold=2.5,        # Likely parameter for volume spikes
                price_threshold=0.01,        # Likely parameter for price impact
                time_window=20,              # Time window parameter
                sensitivity=0.8              # General sensitivity parameter
            )
            self.whale_detector.set_parameters(whale_params)
        except TypeError as e:
            # Fallback if parameters still don't match
            logger.warning(f"Could not set whale parameters: {e}")
            
        # Initialize CDFA for signal validation
        cdfa_config = None
        if config_path and os.path.exists(config_path):
            # Load config from file
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                cdfa_config = AdvancedCDFAConfig(**config_data)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        if cdfa_config is None:
            # Use default config
            cdfa_config = AdvancedCDFAConfig()
            cdfa_config.log_level = logging.INFO
            cdfa_config.use_gpu = False  # GPU not needed for real-time alerts
        
        # Initialize the CDFA system
        self.cdfa = AdvancedCDFA(cdfa_config)
        
        # Initialize market data source
        if data_source_type.lower() == 'binance':
            self.data_source = BinanceDataSource(symbols, timeframe)
        else:
            raise ValueError(f"Unsupported data source type: {data_source_type}")
        
        # Register callback for new data
        self.data_source.register_callback(self._on_new_data)
        
        # Alert queue and processing thread
        self.alert_queue = queue.Queue()
        self.running = False
        self.alert_thread = None
        
        logger.info(f"Whale Alert System initialized for {len(symbols)} symbols")
    
    def start(self) -> None:
        """Start the whale alert system"""
        if self.running:
            return
        
        self.running = True
        
        # Start alert processing thread
        self.alert_thread = threading.Thread(target=self._process_alerts)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        # Start data source
        self.data_source.start()
        
        logger.info("Whale Alert System started")
    
    def stop(self) -> None:
        """Stop the whale alert system"""
        self.running = False
        
        # Stop data source
        self.data_source.stop()
        
        # Wait for alert thread to finish
        if self.alert_thread:
            self.alert_thread.join(timeout=5.0)
        
        logger.info("Whale Alert System stopped")
    
    def _on_new_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Handle new market data"""
        try:
            # First, detect whale activity
            whale_signal = self.whale_detector.detect(data)
            
            # If no signal, no need to proceed
            if not whale_signal.iloc[-1]:
                return
            
            # Validate with CDFA for confirmation
            # Prepare input for CDFA
            signals_df = pd.DataFrame(index=data.index)
            signals_df['whale'] = whale_signal
            
            # Add some context signals
            latest_data = data.tail(100)  # Use last 100 bars
            
            # Simple volatility measure
            returns = latest_data['close'].pct_change()
            signals_df['volatility'] = returns.rolling(20).std()
            
            # Volume ratio
            volume_ma = latest_data['volume'].rolling(20).mean()
            signals_df['volume_ratio'] = latest_data['volume'] / volume_ma
            
            # Large candles
            body_size = abs(latest_data['close'] - latest_data['open']) / latest_data['open']
            signals_df['large_candle'] = (body_size > body_size.rolling(50).mean() * 2).astype(float)
            
            # Use CDFA to validate
            fused_signal = self.cdfa.fuse_signals(signals_df)
            
            # Check if the signal exceeds threshold
            signal_value = fused_signal.iloc[-1]
            
            if signal_value >= self.alert_threshold:
                # Check if we should send an alert (not too frequent)
                current_time = time.time()
                if (symbol not in self.last_alerts or 
                    current_time - self.last_alerts[symbol] >= self.min_alert_interval):
                    
                    # Queue the alert
                    alert_data = {
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'price': data['close'].iloc[-1],
                        'volume': data['volume'].iloc[-1],
                        'signal_value': signal_value,
                        'confidence': min(signal_value * 1.2, 1.0)  # Scale up the confidence
                    }
                    
                    self.alert_queue.put(alert_data)
                    self.last_alerts[symbol] = current_time
        
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}")
    
    def _process_alerts(self) -> None:
        """Process alerts from the queue"""
        while self.running:
            try:
                # Get alert data with timeout
                try:
                    alert_data = self.alert_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Send the alert
                self._send_alert(alert_data)
                
                # Mark task as done
                self.alert_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def _send_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send whale activity alert"""
        try:
            # Log the alert
            symbol = alert_data['symbol']
            confidence = alert_data['confidence']
            price = alert_data['price']
            
            logger.info(f"🚨 WHALE ALERT: {symbol} - Confidence: {confidence:.2f} - Price: {price:.2f}")
            
            # Send to PADS if available
            try:
                self.cdfa.report_to_pads(
                    signal_type="whale",
                    result={"fusion_result": {"fused_signal": [alert_data['signal_value']], "confidence": alert_data['confidence']}},
                    symbol=alert_data['symbol'],
                    confidence=alert_data['confidence']
                )
                logger.info(f"Reported whale alert to PADS: {symbol}")
            except Exception as e:
                logger.warning(f"Error reporting to PADS: {e}")
            
            # Here you could add more alert destinations:
            # - Email alerts
            # - SMS alerts
            # - Chat notifications (Telegram, Discord, etc.)
            # - Webhook integrations
            
            # Save alert to file
            self._save_alert(alert_data)
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _save_alert(self, alert_data: Dict[str, Any]) -> None:
        """Save alert to file for historical record"""
        try:
            # Create alerts directory if it doesn't exist
            os.makedirs('alerts', exist_ok=True)
            
            # Generate filename based on date
            date_str = datetime.now().strftime('%Y-%m-%d')
            filename = f"alerts/whale_alerts_{date_str}.json"
            
            # Check if file exists
            file_exists = os.path.isfile(filename)
            
            # Load existing alerts if file exists
            alerts = []
            if file_exists:
                with open(filename, 'r') as f:
                    alerts = json.load(f)
            
            # Add new alert
            alerts.append(alert_data)
            
            # Write back to file
            with open(filename, 'w') as f:
                json.dump(alerts, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving alert: {e}")


def main():
    """Main function to run the whale alert system"""
    parser = argparse.ArgumentParser(description='Real-Time Whale Alert System')
    parser.add_argument('--symbols', type=str, default='BTC/USDT,ETH/USDT,BNB/USDT',
                      help='Comma-separated list of symbols to monitor')
    parser.add_argument('--timeframe', type=str, default='5m',
                      help='Timeframe for analysis (1m, 5m, 15m, 1h, etc.)')
    parser.add_argument('--threshold', type=float, default=0.75,
                      help='Alert threshold (0.0-1.0)')
    parser.add_argument('--interval', type=int, default=300,
                      help='Minimum time between alerts for the same symbol (seconds)')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to CDFA configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Create and start the whale alert system
    whale_alert = WhaleAlertSystem(
        symbols=symbols,
        timeframe=args.timeframe,
        alert_threshold=args.threshold,
        min_alert_interval=args.interval,
        config_path=args.config
    )
    
    try:
        whale_alert.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        whale_alert.stop()


if __name__ == "__main__":
    main()