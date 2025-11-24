#!/usr/bin/env python3
"""
CDFA Data Pipeline Activation Solution
=====================================

This script provides the missing link to activate CDFA analyzers and detectors
in the main trading system, resolving empty analyzer_scores and detector_signals.

INTEGRATION POINTS:
1. Enhanced CDFA system
2. Advanced CDFA system  
3. Real-time market data fetchers
4. Comprehensive profit score calculation

USAGE:
Add this activation code to your main CDFA initialization to connect analyzers.
"""

import logging
from typing import Dict, Any, Optional

def activate_cdfa_analyzers(cdfa_instance, market_data_source=None):
    """
    Activate CDFA analyzers and detectors for real data processing
    
    Args:
        cdfa_instance: Main CDFA instance (Enhanced or Advanced)
        market_data_source: Optional external market data source
        
    Returns:
        bool: True if activation successful
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Import CDFA components
        from cdfa_extensions.realtime_market_analyzer import RealtimeMarketAnalyzer
        from cdfa_extensions.analyzer_connector import AnalyzerConnector
        
        # Import specific analyzers
        from cdfa_extensions.analyzers.antifragility_analyzer import AntifragilityAnalyzer
        from cdfa_extensions.analyzers.fibonacci_analyzer import FibonacciAnalyzer
        from cdfa_extensions.analyzers.panarchy_analyzer import PanarchyAnalyzer
        from cdfa_extensions.analyzers.soc_analyzer import SOCAnalyzer
        
        # Import specific detectors
        from cdfa_extensions.detectors.black_swan_detector import BlackSwanDetector
        from cdfa_extensions.detectors.whale_detector import WhaleDetector
        from cdfa_extensions.detectors.fibonacci_pattern_detector import FibonacciPatternDetector
        from cdfa_extensions.detectors.pattern_recognizer import PatternRecognizer
        
        # Create or use existing market data analyzer
        if market_data_source is None:
            market_data_config = {
                "exchange": "binance",
                "base_currencies": ["BTC", "ETH", "SOL", "AVAX", "ADA", "DOT", "LINK", "UNI"],
                "quote_currencies": ["USDT", "BUSD", "USDC"],
                "max_pairs": 50,
                "auto_update": True,
                "update_interval": 300  # 5 minutes
            }
            
            market_analyzer = RealtimeMarketAnalyzer(
                config=market_data_config,
                log_level="INFO"
            )
        else:
            market_analyzer = market_data_source
        
        # Initialize CDFA analyzers
        analyzers = {
            'antifragility': AntifragilityAnalyzer(),
            'fibonacci': FibonacciAnalyzer(),
            'panarchy': PanarchyAnalyzer(),
            'soc': SOCAnalyzer()
        }
        
        # Initialize CDFA detectors
        detectors = {
            'black_swan': BlackSwanDetector(),
            'whale': WhaleDetector(),
            'fibonacci_pattern': FibonacciPatternDetector(),
            'pattern_recognizer': PatternRecognizer()
        }
        
        # Connect analyzers to market data
        for name, analyzer in analyzers.items():
            market_analyzer.add_analyzer(name, analyzer)
            logger.info(f"✓ Connected analyzer: {name}")
        
        # Connect detectors to market data (with adapter)
        for name, detector in detectors.items():
            # Create detector adapter
            class DetectorAdapter:
                def __init__(self, detector):
                    self.detector = detector
                
                def analyze(self, dataframe, metadata):
                    try:
                        if hasattr(self.detector, 'detect'):
                            signals = self.detector.detect(dataframe, metadata)
                        else:
                            signals = self.detector.analyze(dataframe, metadata)
                        
                        if signals:
                            return {
                                'signals': signals,
                                'score': self._calculate_score(signals) if isinstance(signals, list) else 0.5
                            }
                        return None
                    except Exception as e:
                        logger.error(f"Error in detector {name}: {e}")
                        return None
                
                def _calculate_score(self, signals):
                    if not signals:
                        return 0.0
                    total_strength = sum(s.get('strength', 0.5) for s in signals)
                    return min(1.0, total_strength / len(signals))
            
            adapter = DetectorAdapter(detector)
            market_analyzer.add_analyzer(name, adapter)
            logger.info(f"✓ Connected detector: {name}")
        
        # Store reference in CDFA instance
        if hasattr(cdfa_instance, 'market_analyzer'):
            cdfa_instance.market_analyzer = market_analyzer
        
        if hasattr(cdfa_instance, 'analyzers'):
            cdfa_instance.analyzers.update(analyzers)
            
        if hasattr(cdfa_instance, 'detectors'):
            cdfa_instance.detectors.update(detectors)
        
        # Add method to get analysis results
        def get_pair_analysis(symbol):
            """Get analysis results for a trading pair"""
            metadata = market_analyzer.get_pair_metadata(symbol)
            if metadata:
                return {
                    'analyzer_scores': dict(metadata.analyzer_scores),
                    'detector_signals': dict(metadata.detector_signals),
                    'regime_state': metadata.regime_state,
                    'opportunity_score': metadata.opportunity_score
                }
            return {}
        
        cdfa_instance.get_pair_analysis = get_pair_analysis
        
        # Start auto-updates
        if market_data_source is None:
            # Only start if we created the analyzer
            market_analyzer.update_market_data()
        
        logger.info("✓ CDFA data pipeline activated successfully")
        logger.info(f"✓ Connected {len(analyzers)} analyzers and {len(detectors)} detectors")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ CDFA activation failed: {e}")
        return False

# Integration example for Enhanced CDFA
def integrate_with_enhanced_cdfa():
    """Example integration with enhanced CDFA system"""
    try:
        from enhanced_cdfa import CognitiveDiversityFusionAnalysis
        
        # Create CDFA instance
        cdfa = CognitiveDiversityFusionAnalysis()
        
        # Activate analyzers and detectors
        success = activate_cdfa_analyzers(cdfa)
        
        if success:
            print("✓ Enhanced CDFA system activated with real data pipeline")
            
            # Test with sample pairs
            sample_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            for pair in sample_pairs:
                analysis = cdfa.get_pair_analysis(pair)
                print(f"{pair}: analyzer_scores={len(analysis.get('analyzer_scores', {}))}, "
                      f"detector_signals={len(analysis.get('detector_signals', {}))}")
        
        return cdfa
        
    except Exception as e:
        print(f"Integration failed: {e}")
        return None

if __name__ == "__main__":
    # Test integration
    cdfa_instance = integrate_with_enhanced_cdfa()
