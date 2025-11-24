#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDFA Interface Definitions
==========================

This module provides abstract base classes and interfaces to break circular dependencies
in the CDFA system while maintaining full functionality. All components implement these
interfaces to enable dependency injection and lazy loading.

Author: Agent 6 - Circular Import Resolution Specialist
Date: 2025-06-29
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import pandas as pd


# ============================================================================
# Core Type Definitions (Isolated from circular dependencies)
# ============================================================================

class FusionType(Enum):
    """Signal fusion strategies"""
    SCORE = "score"
    RANK = "rank"
    HYBRID = "hybrid"
    LAYERED = "layered"
    ADAPTIVE = "adaptive"


class SignalType(Enum):
    """Signal type classifications"""
    BINARY = "binary"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    RANKING = "ranking"


class DiversityMethod(Enum):
    """Diversity calculation methods"""
    KENDALL = "kendall"
    SPEARMAN = "spearman" 
    PEARSON = "pearson"
    HAMMING = "hamming"
    JACCARD = "jaccard"
    KL_DIVERGENCE = "kl_divergence"
    JSK_DIVERGENCE = "jsk_divergence"


@dataclass
class CDFAResult:
    """Standard result structure for CDFA operations"""
    fused_signal: np.ndarray
    confidence: float
    weights: Dict[str, float]
    diversity_matrix: Optional[np.ndarray] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class MarketRegime:
    """Market regime information"""
    regime: str
    trend_strength: float
    volatility: float
    energy_distribution: Dict[str, float] = field(default_factory=dict)
    cyclicality: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Core Abstract Interfaces
# ============================================================================

class ICDFACore(ABC):
    """Core CDFA functionality interface"""
    
    @abstractmethod
    def process_signals_from_dataframe(self, dataframe: pd.DataFrame, symbol: str = None,
                                     calculate_fusion: bool = True) -> Dict[str, Any]:
        """Process signals from DataFrame input"""
        pass
    
    @abstractmethod
    def adaptive_fusion(self, system_scores: Dict[str, List[float]], 
                       performance_metrics: Dict[str, float],
                       market_regime: str = "mixed",
                       volatility: float = 0.5) -> List[float]:
        """Perform adaptive signal fusion"""
        pass
    
    @abstractmethod
    def fuse_signals(self, signals_df: pd.DataFrame) -> pd.Series:
        """Fuse multiple signals into single output"""
        pass


class IHardwareAccelerator(ABC):
    """Hardware acceleration interface"""
    
    @abstractmethod
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available"""
        pass
    
    @abstractmethod
    def get_compute_capability(self) -> Dict[str, Any]:
        """Get hardware compute capabilities"""
        pass
    
    @abstractmethod
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Hardware-accelerated score normalization"""
        pass
    
    @abstractmethod
    def calculate_diversity_matrix(self, signals: np.ndarray) -> np.ndarray:
        """Hardware-accelerated diversity matrix calculation"""
        pass


class IWaveletProcessor(ABC):
    """Wavelet processing interface"""
    
    @abstractmethod
    def denoise_signal(self, signal: np.ndarray) -> np.ndarray:
        """Denoise signal using wavelets"""
        pass
    
    @abstractmethod
    def detect_cycles(self, signal: np.ndarray) -> Dict[str, Any]:
        """Detect cyclic patterns in signal"""
        pass
    
    @abstractmethod
    def analyze_market_regime(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market regime using wavelet decomposition"""
        pass


class INeuromorphicAnalyzer(ABC):
    """Neuromorphic processing interface"""
    
    @abstractmethod
    def process_with_snn(self, features: np.ndarray) -> Dict[str, Any]:
        """Process features with spiking neural network"""
        pass


class ICrossAssetAnalyzer(ABC):
    """Cross-asset analysis interface"""
    
    @abstractmethod
    def add_asset_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Add asset data for cross-asset analysis"""
        pass
    
    @abstractmethod
    def calculate_correlation_matrix(self, method: str = "pearson") -> pd.DataFrame:
        """Calculate cross-asset correlations"""
        pass
    
    @abstractmethod
    def calculate_lead_lag_relationships(self) -> pd.DataFrame:
        """Calculate lead-lag relationships between assets"""
        pass


class IVisualizationEngine(ABC):
    """Visualization interface"""
    
    @abstractmethod
    def create_fusion_visualization(self, signals: Dict[str, np.ndarray], 
                                  fusion_result: np.ndarray,
                                  timestamps: Optional[List] = None) -> Dict[str, Any]:
        """Create fusion result visualization"""
        pass
    
    @abstractmethod
    def create_diversity_matrix_visualization(self, diversity_matrix: np.ndarray,
                                            labels: List[str]) -> Dict[str, Any]:
        """Create diversity matrix heatmap"""
        pass


class IRedisConnector(ABC):
    """Redis communication interface"""
    
    @abstractmethod
    def publish_to_pads(self, signal_type: str, data: Dict[str, Any],
                       confidence: float, priority: int = 0) -> bool:
        """Publish signal to PADS via Redis"""
        pass
    
    @abstractmethod
    def subscribe(self, channel: str, callback: Callable) -> None:
        """Subscribe to Redis channel"""
        pass


class ITorchScriptFusion(ABC):
    """TorchScript optimization interface"""
    
    @abstractmethod
    def get_cached_model(self, model_name: str) -> Optional[Any]:
        """Get cached TorchScript model"""
        pass
    
    @abstractmethod
    def compile_model(self, model: Any, example_input: Any, model_name: str) -> Any:
        """Compile and cache TorchScript model"""
        pass


# ============================================================================
# Component Factory Interface
# ============================================================================

class ICDFAComponentFactory(ABC):
    """Factory interface for creating CDFA components"""
    
    @abstractmethod
    def create_hardware_accelerator(self, config: Dict[str, Any]) -> IHardwareAccelerator:
        """Create hardware accelerator instance"""
        pass
    
    @abstractmethod
    def create_wavelet_processor(self, config: Dict[str, Any]) -> IWaveletProcessor:
        """Create wavelet processor instance"""
        pass
    
    @abstractmethod
    def create_neuromorphic_analyzer(self, hardware: IHardwareAccelerator,
                                   config: Dict[str, Any]) -> INeuromorphicAnalyzer:
        """Create neuromorphic analyzer instance"""
        pass
    
    @abstractmethod
    def create_cross_asset_analyzer(self, config: Dict[str, Any]) -> ICrossAssetAnalyzer:
        """Create cross-asset analyzer instance"""
        pass
    
    @abstractmethod
    def create_visualization_engine(self, config: Dict[str, Any]) -> IVisualizationEngine:
        """Create visualization engine instance"""
        pass
    
    @abstractmethod
    def create_redis_connector(self, config: Any) -> IRedisConnector:
        """Create Redis connector instance"""
        pass
    
    @abstractmethod
    def create_torchscript_fusion(self, hardware: IHardwareAccelerator,
                                config: Dict[str, Any]) -> ITorchScriptFusion:
        """Create TorchScript fusion instance"""
        pass


# ============================================================================
# Dependency Injection Container
# ============================================================================

class CDFADependencyContainer:
    """
    Dependency injection container for CDFA components.
    Breaks circular dependencies through lazy initialization and interface-based design.
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
    
    def register_factory(self, interface_name: str, factory: Callable) -> None:
        """Register a factory function for creating instances"""
        self._factories[interface_name] = factory
    
    def register_singleton(self, interface_name: str, instance: Any) -> None:
        """Register a singleton instance"""
        self._singletons[interface_name] = instance
    
    def register_config(self, config: Dict[str, Any]) -> None:
        """Register configuration for dependency injection"""
        self._config.update(config)
    
    def get(self, interface_name: str) -> Any:
        """Get service instance, creating if necessary"""
        # Check singletons first
        if interface_name in self._singletons:
            return self._singletons[interface_name]
        
        # Check already created services
        if interface_name in self._services:
            return self._services[interface_name]
        
        # Create new instance using factory
        if interface_name in self._factories:
            factory = self._factories[interface_name]
            instance = factory(self._config)
            self._services[interface_name] = instance
            return instance
        
        raise ValueError(f"No registration found for {interface_name}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary"""
        return self._config.copy()
    
    def clear(self) -> None:
        """Clear all registrations (for testing)"""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._config.clear()


# ============================================================================
# Global Container Instance
# ============================================================================

# Global dependency injection container
cdfa_container = CDFADependencyContainer()


# ============================================================================
# Lazy Import Utilities
# ============================================================================

def lazy_import(module_name: str, attribute_name: str = None):
    """
    Lazy import decorator to defer imports until runtime.
    Breaks circular import cycles by deferring import resolution.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                module = __import__(module_name, fromlist=[attribute_name] if attribute_name else [])
                if attribute_name:
                    imported_obj = getattr(module, attribute_name)
                else:
                    imported_obj = module
                return func(imported_obj, *args, **kwargs)
            except ImportError as e:
                raise ImportError(f"Failed to import {module_name}.{attribute_name or ''}: {e}")
        return wrapper
    return decorator


def runtime_import(module_path: str, class_name: str = None):
    """
    Runtime import function for delayed module loading.
    Returns the imported class/module or None if import fails.
    """
    try:
        if class_name:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        else:
            return __import__(module_path)
    except ImportError:
        return None


# ============================================================================
# Configuration Utilities
# ============================================================================

@dataclass
class CDFAConfig:
    """
    Configuration class isolated from circular dependencies.
    Contains all parameters needed by CDFA components.
    """
    # Core configuration
    diversity_threshold: float = 0.3
    performance_threshold: float = 0.6
    default_diversity_method: DiversityMethod = DiversityMethod.KENDALL
    default_fusion_type: FusionType = FusionType.HYBRID
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 1000
    parallelization_threshold: int = 5
    min_signals_required: int = 2
    max_workers: int = 8
    use_numba: bool = True
    use_vectorization: bool = True
    
    # Signal processing
    rsc_scale_factor: float = 4.0
    expansion_factor: int = 2
    reduction_ratio: float = 0.5
    kl_epsilon: float = 1e-9
    kl_num_bins: int = 10
    
    # Adaptive learning
    adaptive_alpha_vol_sensitivity: float = 0.4
    diversity_weighting_scheme: str = "multiplicative"
    additive_weighting_perf_bias: float = 0.6
    enable_adaptive_learning: bool = False
    feedback_window: int = 100
    learning_rate: float = 0.05
    performance_decay: float = 0.95
    
    # Logging and monitoring
    enable_logging: bool = True
    log_level: int = 20  # INFO level
    
    # Redis configuration
    enable_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_channel_prefix: str = "cdfa:"
    signal_ttl: int = 3600
    update_queue_size: int = 100
    
    # Machine learning
    enable_ml: bool = False
    ml_model_type: str = "rf"
    ml_update_interval: int = 300
    ml_batch_size: int = 64
    ml_learning_rate: float = 0.01
    ml_update_strategy: str = "sample"
    
    # Visualization
    enable_visualization: bool = False
    plot_style: str = "darkgrid"
    max_plots_history: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        for field_name, field_obj in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if isinstance(value, Enum):
                result[field_name] = value.value
            else:
                result[field_name] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CDFAConfig':
        """Create configuration from dictionary"""
        # Handle enum conversions
        processed_dict = config_dict.copy()
        
        if 'default_diversity_method' in processed_dict:
            if isinstance(processed_dict['default_diversity_method'], str):
                processed_dict['default_diversity_method'] = DiversityMethod(processed_dict['default_diversity_method'])
        
        if 'default_fusion_type' in processed_dict:
            if isinstance(processed_dict['default_fusion_type'], str):
                processed_dict['default_fusion_type'] = FusionType(processed_dict['default_fusion_type'])
        
        return cls(**processed_dict)