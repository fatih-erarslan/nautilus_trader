#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDFA Extensions Module

This module extends the Cognitive Diversity Fusion Analysis (CDFA) system with:
1. Cross-asset analysis capabilities
2. Advanced visualization framework
3. Pulsar module communication interface
4. PADS reporting functionality
5. Hardware-agnostic acceleration utilities

Author: Created on May 6, 2025
"""
"""
CDFA Extensions Package

This package contains extensions and utilities for the Cognitive Diversity Fusion Analysis system.
"""
    
__version__ = '1.0.0'

# Import hardware acceleration first to avoid circular imports
from .hw_acceleration import HardwareAccelerator, HardwareInfo, AcceleratorType, NumbaDict, NumbaList

# Then import the rest of the modules
from .pulsar_connector import PulsarConnector, PulsarMessage, MessagePriority, CommunicationMode
from .pads_reporter import PADSReporter, PADSFeedback, PADSSignal, SignalConfidence, SignalDirection, SignalTimeframe, SignalType
from .cross_asset_analyzer import CrossAssetAnalyzer, CorrelationMethod, CrossAssetRelationship
from .advanced_visualization import VisualizationEngine, VisualizationType
from .sentiment_analyzer import SentimentAnalyzer
from .crypto_data_fetcher import MarketDataFetcher
from .wavelet_processor import WaveletProcessor, WaveletAnalysisResult, WaveletDecompResult, WaveletDenoiseResult, WaveletFamily
from .torchscript_fusion import TorchScriptFusion, FusionResult, FusionType
from .neuromorphic_analyzer import NeuromorphicAnalyzer, NeuromorphicEngine
from .mra_analyzer import MultiResolutionAnalyzer, MRADecomposition, MRAMode
from .stdp_optimizer import STDPOptimizer, STDPMode, STDPParameters, STDPResult
# Lazy import to avoid circular import
def get_cdfa_integration():
    """Lazy import function for CDFAIntegration to avoid circular imports."""
    from .cdfa_integration import CDFAIntegration
    return CDFAIntegration
from .cdfa_conf import CDFAConfigManager, CDFAFreqTradeStrategy, CDFAFreqTradeIntegration
from .cdfa_optimizer import CDFAOptimizer, ModelFormat, OptimizationLevel
from .redis_connector import RedisConnector


__all__ = [
    # From .hw_acceleration
    'HardwareAccelerator', 'HardwareInfo', 'AcceleratorType', 'NumbaDict', 'NumbaList',

    # From .pulsar_connector
    'PulsarConnector', 'PulsarMessage', 'MessagePriority', 'CommunicationMode',

    # From .pads_reporter
    'PADSReporter', 'PADSFeedback', 'PADSSignal', 'SignalConfidence', 'SignalDirection', 'SignalTimeframe', 'SignalType',

    # From .cross_asset_analyzer
    'CrossAssetAnalyzer', 'CorrelationMethod', 'CrossAssetRelationship',

    # From .advanced_visualization
    'VisualizationEngine', 'VisualizationType',

    # From .sentiment_analyzer
    'SentimentAnalyzer',

    # From .crypto_data_fetcher
    'MarketDataFetcher',

    # From .wavelet_processor
    'WaveletProcessor', 'WaveletAnalysisResult', 'WaveletDecompResult', 'WaveletDenoiseResult', 'WaveletFamily',

    # From .torchscript_fusion
    'TorchScriptFusion', 'FusionResult', 'FusionType',

    # From .neuromorphic_analyzer
    'NeuromorphicAnalyzer', 'NeuromorphicEngine',

    # From .mra_analyzer
    'MultiResolutionAnalyzer', 'MRADecomposition', 'MRAMode',

    # From .stdp_optimizer
    'STDPOptimizer', 'STDPMode', 'STDPParameters', 'STDPResult',
    
    # From .cdfa_integration (available via get_cdfa_integration() function)
    # Access via: cdfa_extensions.get_cdfa_integration()
        
    # From .cdfa_conf
    'CDFAConfigManager', 'CDFAFreqTradeStrategy', 'CDFAFreqTradeIntegration',

    # From .cdfa_optimizer
    'CDFAOptimizer', 'ModelFormat', 'OptimizationLevel',

    # From .redis_connector
    'RedisConnector',
]