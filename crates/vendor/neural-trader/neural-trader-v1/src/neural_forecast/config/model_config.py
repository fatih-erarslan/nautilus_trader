"""
Model Configuration Manager

Provides model-specific configuration templates and management for neural forecasting
models including NHITS, NBEATS, TFT, and PatchTST with optimization presets.
"""

import json
import yaml
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported neural forecasting model types."""
    NBEATS = "NBEATS"
    NHITS = "NHITS"
    TFT = "TFT"
    PATCHTST = "PatchTST"


class OptimizationPreset(Enum):
    """Optimization presets for different use cases."""
    FAST = "fast"           # Quick training, lower accuracy
    BALANCED = "balanced"   # Good balance of speed and accuracy
    ACCURATE = "accurate"   # Best accuracy, slower training
    PRODUCTION = "production"  # Production-optimized settings


@dataclass
class ModelTemplate:
    """Template for model configuration."""
    model_type: ModelType
    preset: OptimizationPreset
    config: Dict[str, Any]
    description: str
    use_cases: List[str]
    expected_accuracy: str
    training_time: str
    memory_usage: str


class ModelConfigManager:
    """
    Manages neural forecasting model configurations with templates,
    optimization presets, and automatic parameter tuning.
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize model configuration manager.
        
        Args:
            templates_dir: Directory containing model templates
        """
        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent / "templates"
        self.templates = {}
        self.custom_templates = {}
        
        self._ensure_templates_directory()
        self._create_default_templates()
        self._load_custom_templates()
        
        logger.info("Model configuration manager initialized")
    
    def _ensure_templates_directory(self):
        """Ensure templates directory exists."""
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        (self.templates_dir / "custom").mkdir(exist_ok=True)
    
    def _create_default_templates(self):
        """Create default model configuration templates."""
        
        # NBEATS Templates
        self.templates[f"{ModelType.NBEATS.value}_{OptimizationPreset.FAST.value}"] = ModelTemplate(
            model_type=ModelType.NBEATS,
            preset=OptimizationPreset.FAST,
            config={
                "input_size": 72,  # 3 days of hourly data
                "h": 24,           # 24-hour forecast
                "max_steps": 200,
                "learning_rate": 0.003,
                "batch_size": 64,
                "windows_batch_size": 256,
                "stack_types": ["trend", "seasonality"],
                "n_blocks": [2, 2],
                "hidden_size": 256,
                "dropout": 0.0,
                "enable_progress_bar": False,
                "early_stopping_patience": 10
            },
            description="Fast NBEATS configuration for quick prototyping",
            use_cases=["Development", "Quick testing", "Real-time inference"],
            expected_accuracy="Good (MAPE: 3-5%)",
            training_time="2-5 minutes",
            memory_usage="Low (1-2GB)"
        )
        
        self.templates[f"{ModelType.NBEATS.value}_{OptimizationPreset.BALANCED.value}"] = ModelTemplate(
            model_type=ModelType.NBEATS,
            preset=OptimizationPreset.BALANCED,
            config={
                "input_size": 168,  # 1 week of hourly data
                "h": 24,
                "max_steps": 500,
                "learning_rate": 0.001,
                "batch_size": 32,
                "windows_batch_size": 128,
                "stack_types": ["trend", "seasonality"],
                "n_blocks": [3, 3],
                "hidden_size": 512,
                "dropout": 0.1,
                "enable_progress_bar": False,
                "early_stopping_patience": 20
            },
            description="Balanced NBEATS configuration for general use",
            use_cases=["Production inference", "General forecasting", "A/B testing"],
            expected_accuracy="Very Good (MAPE: 2-4%)",
            training_time="5-15 minutes",
            memory_usage="Medium (2-4GB)"
        )
        
        self.templates[f"{ModelType.NBEATS.value}_{OptimizationPreset.ACCURATE.value}"] = ModelTemplate(
            model_type=ModelType.NBEATS,
            preset=OptimizationPreset.ACCURATE,
            config={
                "input_size": 336,  # 2 weeks of hourly data
                "h": 24,
                "max_steps": 1000,
                "learning_rate": 0.0005,
                "batch_size": 16,
                "windows_batch_size": 64,
                "stack_types": ["trend", "seasonality", "generic"],
                "n_blocks": [4, 4, 2],
                "hidden_size": 1024,
                "dropout": 0.2,
                "enable_progress_bar": False,
                "early_stopping_patience": 30
            },
            description="High-accuracy NBEATS configuration for maximum performance",
            use_cases=["Critical forecasting", "Research", "Benchmark studies"],
            expected_accuracy="Excellent (MAPE: 1-3%)",
            training_time="15-45 minutes",
            memory_usage="High (4-8GB)"
        )
        
        # NHITS Templates
        self.templates[f"{ModelType.NHITS.value}_{OptimizationPreset.FAST.value}"] = ModelTemplate(
            model_type=ModelType.NHITS,
            preset=OptimizationPreset.FAST,
            config={
                "input_size": 72,
                "h": 24,
                "max_steps": 200,
                "learning_rate": 0.003,
                "batch_size": 64,
                "windows_batch_size": 256,
                "n_freq_downsample": [12, 6, 1],
                "pooling_sizes": [2, 2, 1],
                "n_pool_kernel_size": [2, 2, 1],
                "interpolation_mode": "linear",
                "dropout": 0.0
            },
            description="Fast NHITS configuration optimized for speed",
            use_cases=["Real-time systems", "High-frequency trading", "Quick experiments"],
            expected_accuracy="Good (MAPE: 3-5%)",
            training_time="1-3 minutes",
            memory_usage="Low (1-2GB)"
        )
        
        self.templates[f"{ModelType.NHITS.value}_{OptimizationPreset.BALANCED.value}"] = ModelTemplate(
            model_type=ModelType.NHITS,
            preset=OptimizationPreset.BALANCED,
            config={
                "input_size": 168,
                "h": 24,
                "max_steps": 500,
                "learning_rate": 0.001,
                "batch_size": 32,
                "windows_batch_size": 128,
                "n_freq_downsample": [24, 12, 1],
                "pooling_sizes": [2, 2, 1],
                "n_pool_kernel_size": [4, 2, 1],
                "interpolation_mode": "linear",
                "dropout": 0.1
            },
            description="Balanced NHITS configuration for production use",
            use_cases=["Production systems", "Financial forecasting", "Resource planning"],
            expected_accuracy="Very Good (MAPE: 2-4%)",
            training_time="3-10 minutes",
            memory_usage="Medium (2-4GB)"
        )
        
        # TFT Templates
        self.templates[f"{ModelType.TFT.value}_{OptimizationPreset.BALANCED.value}"] = ModelTemplate(
            model_type=ModelType.TFT,
            preset=OptimizationPreset.BALANCED,
            config={
                "input_size": 168,
                "h": 24,
                "max_steps": 500,
                "learning_rate": 0.001,
                "batch_size": 32,
                "hidden_size": 128,
                "n_heads": 4,
                "dropout": 0.1,
                "add_relative_index": True
            },
            description="Temporal Fusion Transformer for complex patterns",
            use_cases=["Multi-variate forecasting", "Complex dependencies", "Attention analysis"],
            expected_accuracy="Excellent (MAPE: 1-3%)",
            training_time="10-30 minutes",
            memory_usage="High (3-6GB)"
        )
        
        # PatchTST Templates
        self.templates[f"{ModelType.PATCHTST.value}_{OptimizationPreset.BALANCED.value}"] = ModelTemplate(
            model_type=ModelType.PATCHTST,
            preset=OptimizationPreset.BALANCED,
            config={
                "input_size": 168,
                "h": 24,
                "max_steps": 500,
                "learning_rate": 0.001,
                "batch_size": 32,
                "patch_len": 16,
                "stride": 8,
                "hidden_size": 128,
                "n_heads": 4,
                "e_layers": 3,
                "dropout": 0.1
            },
            description="Patch-based Time Series Transformer",
            use_cases=["Long sequence forecasting", "Efficient training", "State-of-the-art accuracy"],
            expected_accuracy="Excellent (MAPE: 1-3%)",
            training_time="5-20 minutes",
            memory_usage="Medium (2-5GB)"
        )
        
        # Production Templates (optimized for deployment)
        self.templates[f"{ModelType.NHITS.value}_{OptimizationPreset.PRODUCTION.value}"] = ModelTemplate(
            model_type=ModelType.NHITS,
            preset=OptimizationPreset.PRODUCTION,
            config={
                "input_size": 168,
                "h": 24,
                "max_steps": 1000,
                "learning_rate": 0.0008,
                "batch_size": 64,  # Optimized for GPU utilization
                "windows_batch_size": 256,
                "n_freq_downsample": [24, 12, 1],
                "pooling_sizes": [2, 2, 1],
                "n_pool_kernel_size": [4, 2, 1],
                "interpolation_mode": "linear",
                "dropout": 0.15,
                "enable_progress_bar": False,
                "save_checkpoints": True,
                "checkpoint_interval": 100,
                "early_stopping_patience": 50
            },
            description="Production-optimized NHITS with checkpointing and monitoring",
            use_cases=["Production deployment", "Critical systems", "24/7 operations"],
            expected_accuracy="Excellent (MAPE: 1.5-3%)",
            training_time="10-25 minutes",
            memory_usage="Medium-High (3-5GB)"
        )
    
    def _load_custom_templates(self):
        """Load custom templates from files."""
        custom_dir = self.templates_dir / "custom"
        
        for template_file in custom_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                
                template_name = template_file.stem
                self.custom_templates[template_name] = template_data
                
                logger.debug(f"Loaded custom template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading custom template {template_file}: {e}")
    
    def get_template(self, model_type: Union[str, ModelType], 
                    preset: Union[str, OptimizationPreset]) -> Optional[ModelTemplate]:
        """
        Get model template by type and preset.
        
        Args:
            model_type: Model type
            preset: Optimization preset
            
        Returns:
            Model template or None if not found
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        if isinstance(preset, str):
            preset = OptimizationPreset(preset)
        
        template_key = f"{model_type.value}_{preset.value}"
        return self.templates.get(template_key)
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available templates."""
        templates_list = []
        
        for template_name, template in self.templates.items():
            templates_list.append({
                'name': template_name,
                'model_type': template.model_type.value,
                'preset': template.preset.value,
                'description': template.description,
                'use_cases': template.use_cases,
                'expected_accuracy': template.expected_accuracy,
                'training_time': template.training_time,
                'memory_usage': template.memory_usage
            })
        
        return sorted(templates_list, key=lambda x: (x['model_type'], x['preset']))
    
    def create_model_config(self, 
                          model_type: Union[str, ModelType],
                          preset: Union[str, OptimizationPreset] = OptimizationPreset.BALANCED,
                          overrides: Optional[Dict[str, Any]] = None,
                          horizon: Optional[int] = None,
                          input_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Create model configuration with optional overrides.
        
        Args:
            model_type: Model type
            preset: Optimization preset
            overrides: Parameter overrides
            horizon: Forecast horizon override
            input_size: Input size override
            
        Returns:
            Model configuration dictionary
        """
        template = self.get_template(model_type, preset)
        
        if template is None:
            raise ValueError(f"Template not found for {model_type} with preset {preset}")
        
        config = copy.deepcopy(template.config)
        
        # Apply standard overrides
        if horizon is not None:
            config['h'] = horizon
        
        if input_size is not None:
            config['input_size'] = input_size
        
        # Apply custom overrides
        if overrides:
            config.update(overrides)
        
        return config
    
    def create_ensemble_config(self, 
                             models: List[Tuple[Union[str, ModelType], Union[str, OptimizationPreset]]],
                             horizon: int = 24,
                             weighting_strategy: str = "equal") -> Dict[str, Any]:
        """
        Create ensemble configuration from multiple models.
        
        Args:
            models: List of (model_type, preset) tuples
            horizon: Forecast horizon
            weighting_strategy: Strategy for combining predictions
            
        Returns:
            Ensemble configuration
        """
        ensemble_config = {
            'ensemble': {
                'enabled': True,
                'weighting_strategy': weighting_strategy,
                'models': {}
            },
            'training': {
                'frequency': 'H',
                'validation_split': 0.2,
                'early_stopping': True
            }
        }
        
        for model_type, preset in models:
            model_config = self.create_model_config(model_type, preset, horizon=horizon)
            
            if isinstance(model_type, ModelType):
                model_name = model_type.value
            else:
                model_name = model_type
            
            ensemble_config['ensemble']['models'][model_name] = model_config
        
        return ensemble_config
    
    def optimize_config_for_data(self, 
                                model_type: Union[str, ModelType],
                                data_characteristics: Dict[str, Any],
                                performance_target: str = "balanced") -> Dict[str, Any]:
        """
        Optimize model configuration based on data characteristics.
        
        Args:
            model_type: Model type
            data_characteristics: Data statistics and characteristics
            performance_target: Performance target (speed, accuracy, balanced)
            
        Returns:
            Optimized model configuration
        """
        # Start with base template
        if performance_target == "speed":
            preset = OptimizationPreset.FAST
        elif performance_target == "accuracy":
            preset = OptimizationPreset.ACCURATE
        else:
            preset = OptimizationPreset.BALANCED
        
        config = self.create_model_config(model_type, preset)
        
        # Optimize based on data characteristics
        data_points = data_characteristics.get('data_points', 1000)
        seasonality_strength = data_characteristics.get('seasonality_strength', 0.5)
        trend_strength = data_characteristics.get('trend_strength', 0.5)
        noise_level = data_characteristics.get('noise_level', 0.1)
        frequency = data_characteristics.get('frequency', 'H')
        
        # Adjust input size based on data points
        if data_points < 500:
            config['input_size'] = min(config['input_size'], 72)  # 3 days
        elif data_points > 5000:
            config['input_size'] = max(config['input_size'], 336)  # 2 weeks
        
        # Adjust model complexity based on data characteristics
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        if model_type == ModelType.NBEATS:
            # Adjust NBEATS stack configuration
            if trend_strength > 0.7:
                config['stack_types'] = ["trend", "seasonality"]
                config['n_blocks'] = [4, 2]
            elif seasonality_strength > 0.7:
                config['stack_types'] = ["seasonality", "trend"]
                config['n_blocks'] = [4, 2]
            else:
                config['stack_types'] = ["trend", "seasonality", "generic"]
                config['n_blocks'] = [2, 2, 2]
        
        elif model_type == ModelType.NHITS:
            # Adjust NHITS frequency downsampling
            if frequency == 'H':  # Hourly data
                config['n_freq_downsample'] = [24, 12, 1]
            elif frequency == 'D':  # Daily data
                config['n_freq_downsample'] = [7, 1]
            elif frequency == 'M':  # Monthly data
                config['n_freq_downsample'] = [12, 1]
        
        # Adjust regularization based on noise level
        if noise_level > 0.2:
            config['dropout'] = min(config.get('dropout', 0.1) + 0.1, 0.3)
        elif noise_level < 0.05:
            config['dropout'] = max(config.get('dropout', 0.1) - 0.05, 0.0)
        
        # Adjust training parameters
        if data_points < 1000:
            config['max_steps'] = min(config['max_steps'], 300)
            config['early_stopping_patience'] = 10
        elif data_points > 10000:
            config['max_steps'] = max(config['max_steps'], 800)
            config['early_stopping_patience'] = 30
        
        return config
    
    def save_custom_template(self, 
                           name: str,
                           model_type: Union[str, ModelType],
                           config: Dict[str, Any],
                           description: str = "",
                           use_cases: Optional[List[str]] = None) -> bool:
        """
        Save custom model template.
        
        Args:
            name: Template name
            model_type: Model type
            config: Model configuration
            description: Template description
            use_cases: List of use cases
            
        Returns:
            True if saved successfully
        """
        try:
            if isinstance(model_type, ModelType):
                model_type = model_type.value
            
            template_data = {
                'model_type': model_type,
                'config': config,
                'description': description,
                'use_cases': use_cases or [],
                'created_at': datetime.now().isoformat()
            }
            
            template_file = self.templates_dir / "custom" / f"{name}.yaml"
            
            with open(template_file, 'w') as f:
                yaml.dump(template_data, f, default_flow_style=False, sort_keys=False)
            
            # Add to custom templates cache
            self.custom_templates[name] = template_data
            
            logger.info(f"Saved custom template: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving custom template {name}: {e}")
            return False
    
    def get_recommended_config(self, 
                             use_case: str,
                             constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get recommended configuration based on use case and constraints.
        
        Args:
            use_case: Use case description
            constraints: Performance/resource constraints
            
        Returns:
            Recommended configuration
        """
        constraints = constraints or {}
        
        # Define use case mappings
        use_case_mappings = {
            'real_time': (ModelType.NHITS, OptimizationPreset.FAST),
            'high_frequency_trading': (ModelType.NHITS, OptimizationPreset.FAST),
            'production_forecasting': (ModelType.NHITS, OptimizationPreset.PRODUCTION),
            'research': (ModelType.TFT, OptimizationPreset.ACCURATE),
            'quick_prototype': (ModelType.NBEATS, OptimizationPreset.FAST),
            'financial_planning': (ModelType.NBEATS, OptimizationPreset.BALANCED),
            'long_term_forecasting': (ModelType.TFT, OptimizationPreset.ACCURATE),
            'multi_variate': (ModelType.TFT, OptimizationPreset.BALANCED),
            'general_purpose': (ModelType.NHITS, OptimizationPreset.BALANCED)
        }
        
        # Get base recommendation
        model_type, preset = use_case_mappings.get(
            use_case.lower().replace(' ', '_'), 
            (ModelType.NHITS, OptimizationPreset.BALANCED)
        )
        
        # Adjust based on constraints
        max_memory_gb = constraints.get('max_memory_gb')
        max_training_time_minutes = constraints.get('max_training_time_minutes')
        min_accuracy = constraints.get('min_accuracy')
        
        if max_memory_gb and max_memory_gb < 3:
            preset = OptimizationPreset.FAST
        elif max_training_time_minutes and max_training_time_minutes < 5:
            preset = OptimizationPreset.FAST
        elif min_accuracy and min_accuracy > 0.95:
            preset = OptimizationPreset.ACCURATE
        
        config = self.create_model_config(model_type, preset)
        
        # Apply constraint-specific adjustments
        if max_memory_gb:
            if max_memory_gb < 2:
                config['batch_size'] = min(config['batch_size'], 16)
                config['hidden_size'] = min(config.get('hidden_size', 256), 256)
            elif max_memory_gb > 8:
                config['batch_size'] = max(config['batch_size'], 64)
        
        return {
            'recommended_model': model_type.value,
            'recommended_preset': preset.value,
            'config': config,
            'rationale': f"Recommended for {use_case} with given constraints"
        }
    
    def compare_templates(self, 
                        template_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple templates side by side.
        
        Args:
            template_names: List of template names to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            'templates': {},
            'parameter_comparison': {},
            'performance_comparison': {}
        }
        
        all_params = set()
        
        # Collect template information
        for name in template_names:
            if name in self.templates:
                template = self.templates[name]
                comparison['templates'][name] = {
                    'model_type': template.model_type.value,
                    'preset': template.preset.value,
                    'config': template.config,
                    'expected_accuracy': template.expected_accuracy,
                    'training_time': template.training_time,
                    'memory_usage': template.memory_usage
                }
                all_params.update(template.config.keys())
        
        # Create parameter comparison matrix
        for param in sorted(all_params):
            comparison['parameter_comparison'][param] = {}
            for name in template_names:
                if name in comparison['templates']:
                    comparison['parameter_comparison'][param][name] = \
                        comparison['templates'][name]['config'].get(param, 'N/A')
        
        return comparison