"""
GPU-Accelerated Signal Generator using CuPy

This module provides GPU-optimized signal generation for trading strategies
using CuPy arrays for maximum parallel processing performance.
"""

import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import warnings

try:
    import cupy as cp
    import numba.cuda as cuda
    from numba import cuda as nb_cuda
    import cudf
    import pandas as pd
    import numpy as np
    GPU_AVAILABLE = cuda.is_available()
except ImportError as e:
    GPU_AVAILABLE = False
    import pandas as pd
    import numpy as np
    warnings.warn(f"GPU libraries not available: {e}. Using CPU fallback.")

logger = logging.getLogger(__name__)


class GPUSignalGenerator:
    """
    GPU-accelerated signal generator using CuPy for parallel processing.
    
    Features:
    - CuPy array operations for massive parallelization
    - Vectorized signal scoring across multiple assets
    - GPU-accelerated pattern recognition
    - Real-time signal filtering and ranking
    - Memory-efficient batch signal generation
    
    Performance Targets:
    - Process 10,000+ assets simultaneously
    - Generate signals in <10ms
    - Memory efficiency >70%
    """
    
    def __init__(self,
                 enable_fallback: bool = True,
                 precision: str = "float32"):
        """
        Initialize GPU Signal Generator.
        
        Args:
            enable_fallback: Enable CPU fallback if GPU unavailable
            precision: Floating point precision ('float32' or 'float64')
        """
        self.gpu_available = GPU_AVAILABLE
        self.enable_fallback = enable_fallback
        self.precision = getattr(cp, precision) if GPU_AVAILABLE else getattr(np, precision)
        self.signal_cache = {}
        self.performance_stats = {
            "signals_generated": 0,
            "avg_generation_time": 0.0,
            "cache_hit_rate": 0.0,
            "gpu_utilization": 0.0
        }
        
        if self.gpu_available:
            logger.info("GPU Signal Generator initialized with CuPy acceleration")
        else:
            logger.info("Using CPU fallback mode for signal generation")
            
    def generate_momentum_signals(self,
                                price_data: Union[cp.ndarray, np.ndarray],
                                volume_data: Union[cp.ndarray, np.ndarray],
                                lookback_periods: List[int] = [5, 10, 20],
                                threshold: float = 0.02) -> Dict[str, Union[cp.ndarray, np.ndarray]]:
        """
        Generate momentum signals using GPU-accelerated calculations.
        
        Args:
            price_data: 2D array of price data [assets x time_periods]
            volume_data: 2D array of volume data [assets x time_periods]
            lookback_periods: List of lookback periods for momentum calculation
            threshold: Minimum momentum threshold for signal generation
            
        Returns:
            Dictionary containing momentum signals and scores
        """
        start_time = time.time()
        
        try:
            if self.gpu_available:
                # Convert to CuPy arrays if needed
                prices = cp.asarray(price_data, dtype=self.precision)
                volumes = cp.asarray(volume_data, dtype=self.precision)
            else:
                prices = np.asarray(price_data, dtype=self.precision)
                volumes = np.asarray(volume_data, dtype=self.precision)
                
            signals = {}
            
            # Calculate momentum for each lookback period
            for period in lookback_periods:
                momentum_score = self._calculate_momentum_vectorized(prices, period)
                volume_confirmation = self._calculate_volume_confirmation(volumes, period)
                
                # Combined momentum signal
                combined_signal = momentum_score * volume_confirmation
                
                # Apply threshold filter
                if self.gpu_available:
                    signal_mask = cp.abs(combined_signal) > threshold
                    filtered_signals = cp.where(signal_mask, combined_signal, 0)
                else:
                    signal_mask = np.abs(combined_signal) > threshold
                    filtered_signals = np.where(signal_mask, combined_signal, 0)
                    
                signals[f'momentum_{period}d'] = filtered_signals
                
            # Generate composite momentum signal
            signals['composite_momentum'] = self._calculate_composite_momentum(signals)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self._update_signal_stats(len(price_data), processing_time)
            
            logger.debug(f"Generated momentum signals for {len(price_data)} assets in {processing_time:.4f}s")
            return signals
            
        except Exception as e:
            logger.error(f"Momentum signal generation failed: {e}")
            if self.enable_fallback and self.gpu_available:
                return self._generate_momentum_signals_cpu_fallback(price_data, volume_data, lookback_periods, threshold)
            raise
            
    def _calculate_momentum_vectorized(self, 
                                     prices: Union[cp.ndarray, np.ndarray], 
                                     period: int) -> Union[cp.ndarray, np.ndarray]:
        """Calculate momentum using vectorized operations."""
        if self.gpu_available:
            # GPU-accelerated momentum calculation
            current_prices = prices[:, -1]
            past_prices = prices[:, -period-1] if prices.shape[1] > period else prices[:, 0]
            
            # Handle division by zero
            momentum = cp.where(past_prices != 0, 
                              (current_prices - past_prices) / past_prices,
                              0)
            
            # Normalize momentum scores
            momentum_std = cp.std(momentum)
            momentum_mean = cp.mean(momentum)
            if momentum_std > 0:
                normalized_momentum = (momentum - momentum_mean) / momentum_std
            else:
                normalized_momentum = momentum
                
        else:
            # CPU fallback
            current_prices = prices[:, -1]
            past_prices = prices[:, -period-1] if prices.shape[1] > period else prices[:, 0]
            
            momentum = np.where(past_prices != 0,
                              (current_prices - past_prices) / past_prices,
                              0)
                              
            momentum_std = np.std(momentum)
            momentum_mean = np.mean(momentum)
            if momentum_std > 0:
                normalized_momentum = (momentum - momentum_mean) / momentum_std
            else:
                normalized_momentum = momentum
                
        return normalized_momentum
        
    def _calculate_volume_confirmation(self,
                                     volumes: Union[cp.ndarray, np.ndarray],
                                     period: int) -> Union[cp.ndarray, np.ndarray]:
        """Calculate volume confirmation using vectorized operations."""
        if self.gpu_available:
            # Calculate average volume over period
            if volumes.shape[1] > period:
                recent_vol_avg = cp.mean(volumes[:, -period:], axis=1)
                historical_vol_avg = cp.mean(volumes[:, :-period], axis=1)
            else:
                recent_vol_avg = cp.mean(volumes, axis=1)
                historical_vol_avg = recent_vol_avg
                
            # Volume confirmation ratio
            vol_confirmation = cp.where(historical_vol_avg > 0,
                                      recent_vol_avg / historical_vol_avg,
                                      1.0)
                                      
            # Normalize and bound between 0.5 and 2.0
            vol_confirmation = cp.clip(vol_confirmation, 0.5, 2.0)
            vol_confirmation = (vol_confirmation - 0.5) / 1.5  # Scale to 0-1
            
        else:
            # CPU fallback
            if volumes.shape[1] > period:
                recent_vol_avg = np.mean(volumes[:, -period:], axis=1)
                historical_vol_avg = np.mean(volumes[:, :-period], axis=1)
            else:
                recent_vol_avg = np.mean(volumes, axis=1)
                historical_vol_avg = recent_vol_avg
                
            vol_confirmation = np.where(historical_vol_avg > 0,
                                      recent_vol_avg / historical_vol_avg,
                                      1.0)
                                      
            vol_confirmation = np.clip(vol_confirmation, 0.5, 2.0)
            vol_confirmation = (vol_confirmation - 0.5) / 1.5
            
        return vol_confirmation
        
    def _calculate_composite_momentum(self, signals: Dict) -> Union[cp.ndarray, np.ndarray]:
        """Calculate composite momentum signal from multiple timeframes."""
        # Weight different timeframes
        weights = {'momentum_5d': 0.2, 'momentum_10d': 0.3, 'momentum_20d': 0.5}
        
        if self.gpu_available:
            composite = cp.zeros_like(list(signals.values())[0])
        else:
            composite = np.zeros_like(list(signals.values())[0])
            
        for signal_key, weight in weights.items():
            if signal_key in signals:
                composite += signals[signal_key] * weight
                
        return composite
        
    def generate_reversal_signals(self,
                                price_data: Union[cp.ndarray, np.ndarray],
                                rsi_data: Union[cp.ndarray, np.ndarray],
                                bollinger_data: Dict[str, Union[cp.ndarray, np.ndarray]],
                                oversold_threshold: float = 0.3,
                                overbought_threshold: float = 0.7) -> Dict[str, Union[cp.ndarray, np.ndarray]]:
        """
        Generate mean reversion signals using GPU acceleration.
        
        Args:
            price_data: Price data array
            rsi_data: RSI indicator values
            bollinger_data: Dictionary with 'upper', 'middle', 'lower' bands
            oversold_threshold: RSI oversold threshold
            overbought_threshold: RSI overbought threshold
            
        Returns:
            Dictionary containing reversal signals
        """
        start_time = time.time()
        
        try:
            if self.gpu_available:
                prices = cp.asarray(price_data, dtype=self.precision)
                rsi = cp.asarray(rsi_data, dtype=self.precision)
                bb_upper = cp.asarray(bollinger_data['upper'], dtype=self.precision)
                bb_lower = cp.asarray(bollinger_data['lower'], dtype=self.precision)
                bb_middle = cp.asarray(bollinger_data['middle'], dtype=self.precision)
            else:
                prices = np.asarray(price_data, dtype=self.precision)
                rsi = np.asarray(rsi_data, dtype=self.precision)
                bb_upper = np.asarray(bollinger_data['upper'], dtype=self.precision)
                bb_lower = np.asarray(bollinger_data['lower'], dtype=self.precision)
                bb_middle = np.asarray(bollinger_data['middle'], dtype=self.precision)
                
            signals = {}
            
            # RSI-based reversal signals
            if self.gpu_available:
                rsi_oversold = (rsi < oversold_threshold).astype(self.precision)
                rsi_overbought = (rsi > overbought_threshold).astype(self.precision)
            else:
                rsi_oversold = (rsi < oversold_threshold).astype(self.precision)
                rsi_overbought = (rsi > overbought_threshold).astype(self.precision)
                
            # Bollinger Band reversal signals
            if self.gpu_available:
                bb_oversold = (prices < bb_lower).astype(self.precision)
                bb_overbought = (prices > bb_upper).astype(self.precision)
                bb_mean_revert = cp.where(prices > bb_middle, -1, 1) * cp.abs(prices - bb_middle) / (bb_upper - bb_lower)
            else:
                bb_oversold = (prices < bb_lower).astype(self.precision)
                bb_overbought = (prices > bb_upper).astype(self.precision)
                bb_mean_revert = np.where(prices > bb_middle, -1, 1) * np.abs(prices - bb_middle) / (bb_upper - bb_lower)
                
            # Combined reversal signals
            signals['rsi_reversal'] = rsi_oversold - rsi_overbought
            signals['bb_reversal'] = bb_oversold - bb_overbought
            signals['mean_reversion'] = bb_mean_revert
            
            # Composite reversal signal
            signals['composite_reversal'] = (
                signals['rsi_reversal'] * 0.4 +
                signals['bb_reversal'] * 0.3 +
                signals['mean_reversion'] * 0.3
            )
            
            processing_time = time.time() - start_time
            self._update_signal_stats(len(price_data), processing_time)
            
            logger.debug(f"Generated reversal signals for {len(price_data)} assets in {processing_time:.4f}s")
            return signals
            
        except Exception as e:
            logger.error(f"Reversal signal generation failed: {e}")
            raise
            
    def generate_breakout_signals(self,
                                price_data: Union[cp.ndarray, np.ndarray],
                                volume_data: Union[cp.ndarray, np.ndarray],
                                resistance_levels: Union[cp.ndarray, np.ndarray],
                                support_levels: Union[cp.ndarray, np.ndarray],
                                volume_threshold: float = 1.5) -> Dict[str, Union[cp.ndarray, np.ndarray]]:
        """
        Generate breakout signals using GPU acceleration.
        
        Args:
            price_data: Current price data
            volume_data: Current volume data
            resistance_levels: Resistance level data
            support_levels: Support level data
            volume_threshold: Minimum volume surge for valid breakout
            
        Returns:
            Dictionary containing breakout signals
        """
        start_time = time.time()
        
        try:
            if self.gpu_available:
                prices = cp.asarray(price_data, dtype=self.precision)
                volumes = cp.asarray(volume_data, dtype=self.precision)
                resistance = cp.asarray(resistance_levels, dtype=self.precision)
                support = cp.asarray(support_levels, dtype=self.precision)
            else:
                prices = np.asarray(price_data, dtype=self.precision)
                volumes = np.asarray(volume_data, dtype=self.precision)
                resistance = np.asarray(resistance_levels, dtype=self.precision)
                support = np.asarray(support_levels, dtype=self.precision)
                
            signals = {}
            
            # Calculate volume surge
            if volumes.ndim > 1:
                avg_volume = volumes.mean(axis=1) if self.gpu_available else volumes.mean(axis=1)
                current_volume = volumes[:, -1] if volumes.shape[1] > 1 else volumes.flatten()
            else:
                avg_volume = volumes
                current_volume = volumes
                
            if self.gpu_available:
                volume_surge = cp.where(avg_volume > 0, current_volume / avg_volume, 1.0)
                volume_confirmed = volume_surge > volume_threshold
            else:
                volume_surge = np.where(avg_volume > 0, current_volume / avg_volume, 1.0)
                volume_confirmed = volume_surge > volume_threshold
                
            # Breakout detection
            if self.gpu_available:
                resistance_breakout = (prices > resistance) & volume_confirmed
                support_breakdown = (prices < support) & volume_confirmed
                
                # Calculate breakout strength
                resistance_strength = cp.where(resistance > 0, (prices - resistance) / resistance, 0)
                support_strength = cp.where(support > 0, (support - prices) / support, 0)
            else:
                resistance_breakout = (prices > resistance) & volume_confirmed
                support_breakdown = (prices < support) & volume_confirmed
                
                resistance_strength = np.where(resistance > 0, (prices - resistance) / resistance, 0)
                support_strength = np.where(support > 0, (support - prices) / support, 0)
                
            # Generate signals
            signals['resistance_breakout'] = resistance_breakout.astype(self.precision) * resistance_strength
            signals['support_breakdown'] = support_breakdown.astype(self.precision) * support_strength
            signals['volume_surge'] = volume_surge
            
            # Composite breakout signal
            signals['composite_breakout'] = signals['resistance_breakout'] - signals['support_breakdown']
            
            processing_time = time.time() - start_time
            self._update_signal_stats(len(price_data), processing_time)
            
            logger.debug(f"Generated breakout signals for {len(price_data)} assets in {processing_time:.4f}s")
            return signals
            
        except Exception as e:
            logger.error(f"Breakout signal generation failed: {e}")
            raise
            
    def rank_signals(self,
                    signals: Dict[str, Union[cp.ndarray, np.ndarray]],
                    ranking_method: str = "composite",
                    top_n: Optional[int] = None) -> Dict[str, Union[cp.ndarray, np.ndarray]]:
        """
        Rank and filter signals using GPU acceleration.
        
        Args:
            signals: Dictionary of signal arrays
            ranking_method: Method for ranking ('composite', 'momentum', 'reversal', 'breakout')
            top_n: Number of top signals to return
            
        Returns:
            Dictionary containing ranked signals and indices
        """
        start_time = time.time()
        
        try:
            # Select ranking signal
            if ranking_method == "composite":
                if 'composite_momentum' in signals:
                    ranking_signal = signals['composite_momentum']
                elif 'composite_reversal' in signals:
                    ranking_signal = signals['composite_reversal']
                elif 'composite_breakout' in signals:
                    ranking_signal = signals['composite_breakout']
                else:
                    # Fallback to first available signal
                    ranking_signal = list(signals.values())[0]
            else:
                signal_key = f'composite_{ranking_method}'
                ranking_signal = signals.get(signal_key, list(signals.values())[0])
                
            # Rank signals
            if self.gpu_available:
                # Get absolute values for ranking
                abs_signals = cp.abs(ranking_signal)
                
                # Sort indices by signal strength
                sorted_indices = cp.argsort(abs_signals)[::-1]  # Descending order
                
                if top_n:
                    top_indices = sorted_indices[:top_n]
                else:
                    top_indices = sorted_indices
                    
            else:
                abs_signals = np.abs(ranking_signal)
                sorted_indices = np.argsort(abs_signals)[::-1]
                
                if top_n:
                    top_indices = sorted_indices[:top_n]
                else:
                    top_indices = sorted_indices
                    
            # Create ranked signals dictionary
            ranked_signals = {
                'ranking_indices': top_indices,
                'ranking_scores': ranking_signal[top_indices],
                'ranking_method': ranking_method
            }
            
            # Add filtered signals for top assets
            for signal_name, signal_values in signals.items():
                ranked_signals[f'top_{signal_name}'] = signal_values[top_indices]
                
            processing_time = time.time() - start_time
            logger.debug(f"Ranked {len(ranking_signal)} signals in {processing_time:.4f}s")
            
            return ranked_signals
            
        except Exception as e:
            logger.error(f"Signal ranking failed: {e}")
            raise
            
    def batch_generate_signals(self,
                             price_batches: List[Union[cp.ndarray, np.ndarray]],
                             volume_batches: List[Union[cp.ndarray, np.ndarray]],
                             signal_types: List[str] = ["momentum", "reversal", "breakout"]) -> List[Dict]:
        """
        Generate signals for multiple batches in parallel.
        
        Args:
            price_batches: List of price data batches
            volume_batches: List of volume data batches
            signal_types: Types of signals to generate
            
        Returns:
            List of signal dictionaries for each batch
        """
        start_time = time.time()
        results = []
        
        try:
            for i, (prices, volumes) in enumerate(zip(price_batches, volume_batches)):
                batch_signals = {}
                
                if "momentum" in signal_types:
                    momentum_signals = self.generate_momentum_signals(prices, volumes)
                    batch_signals.update(momentum_signals)
                    
                # Note: reversal and breakout would need additional data (RSI, Bollinger Bands, etc.)
                # This is a simplified implementation
                
                results.append(batch_signals)
                
                # Memory cleanup for GPU
                if self.gpu_available and i % 5 == 0:
                    cp.get_default_memory_pool().free_all_blocks()
                    
            processing_time = time.time() - start_time
            total_assets = sum(len(batch) for batch in price_batches)
            
            logger.info(f"Generated signals for {total_assets} assets across {len(price_batches)} batches in {processing_time:.4f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch signal generation failed: {e}")
            raise
            
    def filter_signals_by_criteria(self,
                                 signals: Dict[str, Union[cp.ndarray, np.ndarray]],
                                 min_strength: float = 0.1,
                                 max_assets: Optional[int] = None,
                                 exclude_indices: Optional[List[int]] = None) -> Dict[str, Union[cp.ndarray, np.ndarray]]:
        """
        Filter signals based on various criteria.
        
        Args:
            signals: Input signals dictionary
            min_strength: Minimum signal strength threshold
            max_assets: Maximum number of assets to return
            exclude_indices: Asset indices to exclude
            
        Returns:
            Filtered signals dictionary
        """
        try:
            # Get main ranking signal
            if 'composite_momentum' in signals:
                main_signal = signals['composite_momentum']
            elif 'composite_reversal' in signals:
                main_signal = signals['composite_reversal']
            else:
                main_signal = list(signals.values())[0]
                
            # Apply strength filter
            if self.gpu_available:
                strength_mask = cp.abs(main_signal) >= min_strength
            else:
                strength_mask = np.abs(main_signal) >= min_strength
                
            # Apply exclusion filter
            if exclude_indices:
                if self.gpu_available:
                    exclude_mask = cp.ones(len(main_signal), dtype=bool)
                    exclude_mask[exclude_indices] = False
                    combined_mask = strength_mask & exclude_mask
                else:
                    exclude_mask = np.ones(len(main_signal), dtype=bool)
                    exclude_mask[exclude_indices] = False
                    combined_mask = strength_mask & exclude_mask
            else:
                combined_mask = strength_mask
                
            # Get valid indices
            if self.gpu_available:
                valid_indices = cp.where(combined_mask)[0]
            else:
                valid_indices = np.where(combined_mask)[0]
                
            # Apply max assets limit
            if max_assets and len(valid_indices) > max_assets:
                # Sort by signal strength and take top N
                signal_strengths = main_signal[valid_indices]
                if self.gpu_available:
                    top_indices = cp.argsort(cp.abs(signal_strengths))[-max_assets:]
                else:
                    top_indices = np.argsort(np.abs(signal_strengths))[-max_assets:]
                valid_indices = valid_indices[top_indices]
                
            # Filter all signals
            filtered_signals = {}
            for signal_name, signal_values in signals.items():
                filtered_signals[signal_name] = signal_values[valid_indices]
                
            filtered_signals['filtered_indices'] = valid_indices
            filtered_signals['filter_criteria'] = {
                'min_strength': min_strength,
                'max_assets': max_assets,
                'excluded_count': len(exclude_indices) if exclude_indices else 0
            }
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Signal filtering failed: {e}")
            raise
            
    def _generate_momentum_signals_cpu_fallback(self, price_data, volume_data, lookback_periods, threshold):
        """CPU fallback for momentum signal generation."""
        logger.info("Using CPU fallback for momentum signal generation")
        
        # Temporarily disable GPU mode
        original_gpu_state = self.gpu_available
        self.gpu_available = False
        
        try:
            result = self.generate_momentum_signals(price_data, volume_data, lookback_periods, threshold)
            return result
        finally:
            self.gpu_available = original_gpu_state
            
    def _update_signal_stats(self, assets_processed: int, processing_time: float):
        """Update signal generation statistics."""
        self.performance_stats["signals_generated"] += assets_processed
        
        # Update average processing time (exponential moving average)
        alpha = 0.1
        self.performance_stats["avg_generation_time"] = (
            alpha * processing_time + 
            (1 - alpha) * self.performance_stats["avg_generation_time"]
        )
        
        # Update GPU utilization estimate
        if self.gpu_available:
            throughput = assets_processed / processing_time if processing_time > 0 else 0
            self.performance_stats["gpu_utilization"] = min(throughput / 50000, 1.0)  # Normalize to 50k assets/sec
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()
        stats.update({
            "gpu_available": self.gpu_available,
            "enable_fallback": self.enable_fallback,
            "precision": str(self.precision),
            "cache_size": len(self.signal_cache)
        })
        
        return stats
        
    def clear_cache(self):
        """Clear signal cache to free memory."""
        self.signal_cache.clear()
        if self.gpu_available:
            cp.get_default_memory_pool().free_all_blocks()
            
    def benchmark_signal_generation(self,
                                  num_assets: int = 10000,
                                  time_periods: int = 252,
                                  iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark signal generation performance.
        
        Args:
            num_assets: Number of assets to simulate
            time_periods: Number of time periods (trading days)
            iterations: Number of benchmark iterations
            
        Returns:
            Performance benchmark results
        """
        logger.info(f"Benchmarking signal generation with {num_assets} assets")
        
        # Generate test data
        np.random.seed(42)
        test_prices = np.random.uniform(50, 200, (num_assets, time_periods)).astype(np.float32)
        test_volumes = np.random.uniform(1000, 100000, (num_assets, time_periods)).astype(np.float32)
        
        # GPU benchmark
        gpu_times = []
        if self.gpu_available:
            for i in range(iterations):
                start_time = time.time()
                signals = self.generate_momentum_signals(test_prices, test_volumes)
                gpu_times.append(time.time() - start_time)
                
                # Memory cleanup
                self.clear_cache()
                
        # CPU benchmark
        cpu_times = []
        original_gpu_state = self.gpu_available
        self.gpu_available = False
        
        for i in range(iterations):
            start_time = time.time()
            signals = self.generate_momentum_signals(test_prices, test_volumes)
            cpu_times.append(time.time() - start_time)
            
        self.gpu_available = original_gpu_state
        
        # Calculate metrics
        avg_gpu_time = np.mean(gpu_times) if gpu_times else float('inf')
        avg_cpu_time = np.mean(cpu_times)
        speedup = avg_cpu_time / avg_gpu_time if gpu_times else 0
        
        gpu_throughput = num_assets / avg_gpu_time if gpu_times else 0
        cpu_throughput = num_assets / avg_cpu_time
        
        results = {
            "num_assets": num_assets,
            "time_periods": time_periods,
            "iterations": iterations,
            "avg_gpu_time": avg_gpu_time,
            "avg_cpu_time": avg_cpu_time,
            "speedup_factor": speedup,
            "gpu_throughput": gpu_throughput,
            "cpu_throughput": cpu_throughput,
            "target_latency_achieved": avg_gpu_time < 0.01,  # <10ms target
            "target_throughput_achieved": gpu_throughput >= 10000  # 10k assets/sec target
        }
        
        logger.info(f"Signal Generation Benchmark Results:")
        logger.info(f"  GPU Time: {avg_gpu_time:.4f}s")
        logger.info(f"  CPU Time: {avg_cpu_time:.4f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  GPU Throughput: {gpu_throughput:.0f} assets/sec")
        
        return results