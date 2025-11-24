import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any
import logging
from enum import Enum, auto


logger = logging.getLogger(__name__)


class PatternRecWindow(Enum):
    SHORT = 14
    MEDIUM = 30
    LONG = 50

class PatternRecognizer:
    """
    Pattern recognition component using Dynamic Time Warping (DTW) 
    and other techniques to identify temporal patterns in market data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the pattern recognizer with configuration parameters."""
        self.config = config if config else {}
        self.lbk_window = self.config.get("lbk_window", 3)  # LB_Keogh lower bounding window
        
    def _lb_keogh_bound(self, series: np.ndarray, window: int) -> tuple:
        """
        LB_Keogh lower bounding technique to accelerate DTW.
        Creates envelope of upper and lower bounds for fast early pruning.
        
        Args:
            series: Time series to create the envelope for
            window: Window size for the envelope
            
        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        n = len(series)
        upper = np.zeros(n)
        lower = np.zeros(n)
        
        for i in range(n):
            window_start = max(0, i - window)
            window_end = min(n-1, i + window)
            lower[i] = min(series[window_start:window_end+1])
            upper[i] = max(series[window_start:window_end+1])
        
        return lower, upper
    
    def _can_prune(self, query: np.ndarray, 
                  template: np.ndarray, 
                  template_lower: np.ndarray, 
                  template_upper: np.ndarray,
                  best_so_far: float) -> bool:
        """
        Check if we can prune this template comparison using LB_Keogh.
        
        Args:
            query: Normalized input series
            template: Normalized template
            template_lower: Lower bound for template
            template_upper: Upper bound for template
            best_so_far: Best DTW distance encountered so far
            
        Returns:
            True if comparison can be pruned, False otherwise
        """
        lb_dist = 0.0
        
        for i in range(len(query)):
            # If query is above upper bound
            if query[i] > template_upper[i]:
                lb_dist += (query[i] - template_upper[i]) ** 2
            # If query is below lower bound
            elif query[i] < template_lower[i]:
                lb_dist += (template_lower[i] - query[i]) ** 2
                
            # Early abandoning
            if lb_dist >= best_so_far:
                return True
                
        return False
    
    def _dtw_distance(self, s1: np.ndarray, s2: np.ndarray, window: int) -> float:
        """
        Calculate DTW distance between two time series with a window constraint.
        Uses a more memory-efficient implementation for large series.
        
        Args:
            s1: First time series
            s2: Second time series
            window: Sakoe-Chiba band width
            
        Returns:
            DTW distance between s1 and s2
        """
        n, m = len(s1), len(s2)
        
        # For very small series, use the full matrix approach
        if max(n, m) <= 100:
            # Initialize distance matrix with infinity
            dtw_matrix = np.full((n+1, m+1), np.inf)
            dtw_matrix[0, 0] = 0
            
            # Fill the matrix
            for i in range(1, n+1):
                # Apply window constraint
                window_start = max(1, i - window)
                window_end = min(m, i + window)
                
                for j in range(window_start, window_end + 1):
                    cost = abs(s1[i-1] - s2[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],     # insertion
                        dtw_matrix[i, j-1],     # deletion
                        dtw_matrix[i-1, j-1]    # match
                    )
            
            # Return the final distance
            return dtw_matrix[n, m]
        else:
            # For larger series, use a memory-efficient approach with only two rows
            # This reduces memory from O(n*m) to O(m)
            prev_row = np.full(m+1, np.inf)
            curr_row = np.full(m+1, np.inf)
            prev_row[0] = 0
            
            for i in range(1, n+1):
                curr_row[0] = np.inf
                window_start = max(1, i - window)
                window_end = min(m, i + window)
                
                for j in range(window_start, window_end + 1):
                    cost = abs(s1[i-1] - s2[j-1])
                    curr_row[j] = cost + min(
                        prev_row[j],     # insertion
                        curr_row[j-1],   # deletion
                        prev_row[j-1]    # match
                    )
                
                # Swap rows
                prev_row, curr_row = curr_row, prev_row
            
            # Final result is in prev_row (due to the final swap)
            return prev_row[m]
    
    def _normalize_series(self, s: np.ndarray) -> np.ndarray:
        """
        Normalize series to [0,1] range for comparison.
        Handles edge cases like constant series.
        
        Args:
            s: Input time series
            
        Returns:
            Normalized time series in [0,1] range
        """
        if len(s) == 0:
            return s
            
        s_min, s_max = np.min(s), np.max(s)
        if s_max == s_min:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    def detect_dtw_patterns(self, 
                           series: Union[np.ndarray, List[float]], 
                           templates: Dict[str, Union[np.ndarray, List[float]]],
                           window_size: int = 20) -> Dict[str, float]:
        """
        Detect patterns using Dynamic Time Warping with optimizations.
        
        Implements:
        - Efficient DTW algorithm with Sakoe-Chiba band constraint
        - Lower-bounding technique (LB_Keogh) for fast pruning
        - Variable-length pattern handling
        - Memory-efficient implementation for large series
        
        Args:
            series: Time series data
            templates: Dictionary of pattern templates
            window_size: Window size for comparison (Sakoe-Chiba band)
            
        Returns:
            Dict[str, float]: Match scores for templates (higher = better match)
        """
        # Ensure inputs are numpy arrays
        series = np.asarray(series, dtype=float)
        if len(series) == 0:
            return {}
        
        # Normalize the input series
        norm_series = self._normalize_series(series)
        
        # Prepare result dictionary and tracking for best matches
        results = {}
        best_distance = float('inf')
        
        # Process templates in order of ascending size for faster pruning
        sorted_templates = sorted(
            templates.items(), 
            key=lambda x: len(np.asarray(x[1])) if hasattr(x[1], '__len__') else 0
        )
        
        for name, template in sorted_templates:
            template = np.asarray(template, dtype=float)
            
            # Skip invalid templates
            if len(template) == 0:
                results[name] = 0.0
                continue
                
            # Handle variable-length patterns
            if len(template) > len(series):
                # Template longer than series - calculate partial match
                template = template[:len(series)]
                penalty = 0.2  # Penalty for partial matching
            else:
                penalty = 0.0
            
            # Normalize the template
            norm_template = self._normalize_series(template)
            
            # Calculate lower bounds for early pruning
            lower, upper = self._lb_keogh_bound(norm_template, self.lbk_window)
            
            # Check if we can prune this template using LB_Keogh
            if self._can_prune(norm_series[:len(norm_template)], norm_template, 
                              lower, upper, best_distance):
                # Set a default low similarity score for pruned templates
                similarity = 0.3
            else:
                # Calculate full DTW distance
                adjusted_window = min(window_size, len(norm_template) // 2)
                distance = self._dtw_distance(norm_series[:len(norm_template)], 
                                             norm_template, adjusted_window)
                
                # Normalize by path length to make comparable across different lengths
                distance = distance / (len(norm_template) + len(norm_series[:len(norm_template)]))
                
                # Update best distance for pruning
                best_distance = min(best_distance, distance)
                
                # Convert distance to similarity score (0-1, higher is better match)
                # Apply length penalty for partial matches
                similarity = max(0.0, 1.0 - (distance / 2.0) - penalty)
            
            results[name] = similarity
        
        return results

    def detect(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Detect patterns in the DataFrame and return a series of pattern strength signals.
        Expected interface for CDFA server integration.
        
        Args:
            dataframe: Input DataFrame with OHLCV data
            
        Returns:
            pd.Series: Pattern strength signals (0-1 range)
        """
        try:
            # Use close prices for pattern detection
            if 'close' not in dataframe.columns:
                return pd.Series(0.5, index=dataframe.index)
            
            prices = dataframe['close'].values
            
            # Define simple pattern templates for recognition
            patterns = {
                'uptrend': [1, 1.1, 1.2, 1.3, 1.4],
                'downtrend': [1, 0.9, 0.8, 0.7, 0.6],
                'consolidation': [1, 1.05, 0.95, 1.02, 0.98],
                'reversal_up': [1, 0.8, 0.7, 0.9, 1.2],
                'reversal_down': [1, 1.2, 1.3, 1.1, 0.8]
            }
            
            # Calculate rolling pattern similarities
            window_size = min(20, len(prices) // 4)
            pattern_scores = pd.Series(0.5, index=dataframe.index)
            
            if len(prices) >= 10:
                # Calculate pattern matches for recent data
                recent_prices = prices[-window_size:] if len(prices) >= window_size else prices
                similarities = self.detect_dtw_patterns(recent_prices, patterns)
                
                # Combine pattern similarities into a single score
                max_similarity = max(similarities.values()) if similarities else 0.5
                pattern_scores.iloc[-1] = max_similarity
                
                # Propagate the score to recent values with decay
                if len(pattern_scores) > 1:
                    for i in range(min(5, len(pattern_scores) - 1)):
                        idx = len(pattern_scores) - 2 - i
                        if idx >= 0:
                            decay = 0.9 ** (i + 1)
                            pattern_scores.iloc[idx] = max_similarity * decay
            
            return pattern_scores
            
        except Exception as e:
            # Return neutral pattern score on error
            return pd.Series(0.5, index=dataframe.index)

    def detect_signals(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """
        Detect patterns in price/volume arrays and return signal information.
        Expected interface for CDFA server integration.
        
        Args:
            prices: Array of price values
            volumes: Array of volume values
            
        Returns:
            Dict containing signal, confidence, and pattern details
        """
        try:
            # Create DataFrame for internal processing
            df = pd.DataFrame({
                'close': prices,
                'volume': volumes
            })
            
            # Use the detect method to get pattern signals
            pattern_series = self.detect(df)
            
            # Get latest pattern signal
            latest_signal = pattern_series.iloc[-1] if len(pattern_series) > 0 else 0.5
            
            # Calculate confidence based on signal strength and consistency
            if len(pattern_series) >= 5:
                recent_signals = pattern_series.tail(5)
                signal_consistency = 1.0 - np.std(recent_signals)
                confidence = min(1.0, max(0.1, signal_consistency * latest_signal))
            else:
                confidence = 0.5
            
            # Determine if a strong pattern is detected
            pattern_detected = latest_signal > 0.7
            
            return {
                "signal": float(latest_signal),
                "confidence": float(confidence),
                "detected": bool(pattern_detected),
                "pattern_strength": float(latest_signal),
                "analysis_type": "pattern_recognition",
                "data_points": len(prices)
            }
            
        except Exception as e:
            return {
                "signal": 0.5,
                "confidence": 0.0,
                "detected": False,
                "error": str(e),
                "analysis_type": "pattern_recognition"
            }