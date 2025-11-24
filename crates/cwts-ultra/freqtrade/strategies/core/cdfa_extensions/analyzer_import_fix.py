#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyzer Import Fix

Provides a wrapper to properly import the analyzers and detectors based on the
current directory structure, ensuring compatibility with the CDFA system.

Created on May 20, 2025
"""

import os
import sys
import logging
import importlib.util
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

# Setup logging
logger = logging.getLogger(__name__)

# Path to analyzers and detectors
CORE_PATH = os.path.dirname(os.path.abspath(__file__))
ANALYZERS_PATH = os.path.join(CORE_PATH, "analyzers")
DETECTORS_PATH = os.path.join(CORE_PATH, "detectors")

# Create dummy analyzer and detector classes for fallback
class DummyAnalyzer:
    """Dummy analyzer for when real analyzer can't be loaded"""
    
    def __init__(self, name="DummyAnalyzer"):
        self.name = name
        logger.warning(f"Using dummy analyzer: {name}")
    
    def analyze(self, dataframe, metadata):
        """Dummy analyze method"""
        logger.debug(f"Dummy analyze for {self.name}")
        return {
            'score': 0.5,
            'regime': 'unknown',
            'confidence': 0.0,
            'is_trending': False,
            'trend_strength': 0.0
        }

class DummyDetector:
    """Dummy detector for when real detector can't be loaded"""
    
    def __init__(self, name="DummyDetector"):
        self.name = name
        logger.warning(f"Using dummy detector: {name}")
    
    def detect(self, dataframe, metadata):
        """Dummy detect method"""
        logger.debug(f"Dummy detect for {self.name}")
        return []

class DummyMRAAnalyzer:
    """Dummy MRA analyzer for when real MRA analyzer can't be loaded"""
    
    def __init__(self):
        logger.warning("Using dummy MRA analyzer")
    
    def decompose(self, data):
        """Dummy decompose method"""
        return {'scales': [], 'wavelet_coeffs': []}
    
    def analyze(self, dataframe, metadata):
        """Dummy analyze method"""
        return {
            'score': 0.5,
            'regime': 'unknown',
            'confidence': 0.0
        }
    
    def analyze_regimes(self, dataframe):
        """Dummy analyze_regimes method"""
        return {'regimes': [0] * len(dataframe)}

# Create fallback analyzers and detectors
FALLBACK_ANALYZERS = {
    'antifragilityanalyzer': DummyAnalyzer('antifragilityanalyzer'),
    'panarchyanalyzer': DummyAnalyzer('panarchyanalyzer'),
    'socanalyzer': DummyAnalyzer('socanalyzer'),
    'fibonaccianalyzer': DummyAnalyzer('fibonaccianalyzer'),
    'mra_analyzer': DummyAnalyzer('mra_analyzer')
}

FALLBACK_DETECTORS = {
    'blackswandetector': DummyDetector('blackswandetector'),
    'fibonaccipatterndetector': DummyDetector('fibonaccipatterndetector'),
    'patternrecognizer': DummyDetector('patternrecognizer'),
    'whaledetector': DummyDetector('whaledetector')
}

# Create dummy MRA analyzer
FALLBACK_MRA_ANALYZER = DummyMRAAnalyzer()

# Function to get real analyzers and detectors if possible
def get_real_components():
    """
    Attempt to get real analyzers and detectors.
    
    Returns:
        Tuple of (analyzers, detectors, mra_analyzer)
    """
    # Try to import real components
    try:
        # Check if analyzers directory exists
        if os.path.isdir(ANALYZERS_PATH):
            analyzers = import_analyzers()
        else:
            logger.warning(f"Analyzers directory not found: {ANALYZERS_PATH}")
            analyzers = {}
        
        # Check if detectors directory exists
        if os.path.isdir(DETECTORS_PATH):
            detectors = import_detectors()
        else:
            logger.warning(f"Detectors directory not found: {DETECTORS_PATH}")
            detectors = {}
        
        # Import MRA analyzer
        mra_analyzer = import_mra_analyzer()
        
        return analyzers, detectors, mra_analyzer
        
    except Exception as e:
        logger.error(f"Error getting real components: {e}")
        return {}, {}, None

def import_analyzers():
    """
    Import all analyzers.
    
    Returns:
        Dictionary of analyzer_name -> analyzer_class
    """
    analyzers = {}
    
    try:
        # Find all analyzer files
        analyzer_files = []
        for root, dirs, files in os.walk(ANALYZERS_PATH):
            for file in files:
                if file.endswith("_analyzer.py") and not file.startswith("__"):
                    analyzer_files.append((os.path.join(root, file), file[:-3]))
        
        # Import each analyzer
        for path, name in analyzer_files:
            try:
                spec = importlib.util.spec_from_file_location(name, path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    
                    # Find analyzer class
                    for attr_name in dir(module):
                        if attr_name.endswith("Analyzer") and hasattr(module, attr_name):
                            try:
                                analyzer_class = getattr(module, attr_name)
                                analyzer_instance = analyzer_class()
                                if hasattr(analyzer_instance, "analyze"):
                                    analyzer_name = attr_name.lower()
                                    analyzers[analyzer_name] = analyzer_instance
                                    logger.info(f"Imported analyzer: {analyzer_name}")
                            except Exception as e:
                                logger.error(f"Error instantiating analyzer {attr_name}: {e}")
            
            except Exception as e:
                logger.error(f"Error importing analyzer {name}: {e}")
        
        logger.info(f"Imported {len(analyzers)} analyzers")
        
    except Exception as e:
        logger.error(f"Error importing analyzers: {e}")
    
    return analyzers

def import_detectors():
    """
    Import all detectors.
    
    Returns:
        Dictionary of detector_name -> detector_class
    """
    detectors = {}
    
    try:
        # Find all detector files
        detector_files = []
        for root, dirs, files in os.walk(DETECTORS_PATH):
            for file in files:
                if (file.endswith("_detector.py") or file.endswith("_recognizer.py")) and not file.startswith("__"):
                    detector_files.append((os.path.join(root, file), file[:-3]))
        
        # Import each detector
        for path, name in detector_files:
            try:
                spec = importlib.util.spec_from_file_location(name, path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    
                    # Find detector class
                    for attr_name in dir(module):
                        if (attr_name.endswith("Detector") or attr_name.endswith("Recognizer")) and hasattr(module, attr_name):
                            try:
                                detector_class = getattr(module, attr_name)
                                detector_instance = detector_class()
                                if hasattr(detector_instance, "detect"):
                                    detector_name = attr_name.lower()
                                    detectors[detector_name] = detector_instance
                                    logger.info(f"Imported detector: {detector_name}")
                            except Exception as e:
                                logger.error(f"Error instantiating detector {attr_name}: {e}")
            
            except Exception as e:
                logger.error(f"Error importing detector {name}: {e}")
        
        logger.info(f"Imported {len(detectors)} detectors")
        
    except Exception as e:
        logger.error(f"Error importing detectors: {e}")
    
    return detectors

# Import MRA analyzer separately (often needed by other components)
def import_mra_analyzer():
    """
    Import MRA analyzer.
    
    Returns:
        MRA analyzer class or None if not found
    """
    try:
        path = os.path.join(CORE_PATH, "mra_analyzer.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("mra_analyzer", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["mra_analyzer"] = module
                spec.loader.exec_module(module)
                
                # Find MRA analyzer class
                for attr_name in dir(module):
                    if "MultiResolution" in attr_name and hasattr(module, attr_name):
                        try:
                            mra_class = getattr(module, attr_name)
                            mra_instance = mra_class()
                            logger.info(f"Imported MRA analyzer: {attr_name}")
                            return mra_instance
                        except Exception as e:
                            logger.error(f"Error instantiating MRA analyzer {attr_name}: {e}")
        
        logger.warning("MRA analyzer not found")
        return None
        
    except Exception as e:
        logger.error(f"Error importing MRA analyzer: {e}")
        return None

# Try to get real components, fall back to dummy implementations
real_analyzers, real_detectors, real_mra = get_real_components()

# Merge real and fallback components
ANALYZERS = {**FALLBACK_ANALYZERS, **real_analyzers}
DETECTORS = {**FALLBACK_DETECTORS, **real_detectors}
MRA_ANALYZER = real_mra if real_mra is not None else FALLBACK_MRA_ANALYZER

# Export to make available to importers
__all__ = ["ANALYZERS", "DETECTORS", "MRA_ANALYZER", "import_analyzers", "import_detectors", "import_mra_analyzer"]