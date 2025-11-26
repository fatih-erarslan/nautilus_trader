"""
Data Augmenter for Time Series Neural Training

This module provides advanced data augmentation techniques specifically
designed for time series neural forecasting models to improve generalization
and model robustness.

Key Features:
- Time series specific augmentations
- Noise injection techniques
- Temporal transformations
- Magnitude warping
- Pattern mixing
- Synthetic data generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import warnings

logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """Types of data augmentation."""
    NOISE_INJECTION = "noise_injection"
    JITTERING = "jittering"
    TIME_WARPING = "time_warping"
    MAGNITUDE_WARPING = "magnitude_warping"
    WINDOW_SLICING = "window_slicing"
    PERMUTATION = "permutation"
    SCALING = "scaling"
    ROTATION = "rotation"
    FLIPPING = "flipping"
    CROPPING = "cropping"
    SPAWNING = "spawning"  # Generate new samples from existing
    MIXING = "mixing"      # Mix multiple samples


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    # Augmentation types to apply
    enabled_augmentations: List[AugmentationType] = field(
        default_factory=lambda: [
            AugmentationType.NOISE_INJECTION,
            AugmentationType.JITTERING,
            AugmentationType.SCALING
        ]
    )
    
    # Noise injection parameters
    noise_ratio: float = 0.01  # Relative noise level
    noise_type: str = 'gaussian'  # 'gaussian', 'uniform', 'laplace'
    
    # Jittering parameters
    jitter_ratio: float = 0.02  # Relative jitter amount
    
    # Time warping parameters
    time_warping: bool = True
    warp_strength: float = 0.1  # Warping strength
    warp_knots: int = 4  # Number of warping points
    
    # Magnitude warping parameters
    magnitude_warping: bool = True
    magnitude_strength: float = 0.1
    magnitude_knots: int = 4
    
    # Scaling parameters
    scaling_range: Tuple[float, float] = (0.9, 1.1)
    
    # Window operations
    window_slice_ratio: float = 0.1  # Fraction of window to slice
    permutation_segments: int = 4     # Number of segments for permutation
    
    # Mixing parameters
    mixup_alpha: float = 0.2  # Mixup interpolation parameter
    cutmix_ratio: float = 0.1  # CutMix ratio
    
    # Generation parameters
    augmentation_factor: float = 2.0  # How much to augment (2.0 = double data)
    preserve_trends: bool = True      # Preserve overall trends
    preserve_seasonality: bool = True # Preserve seasonal patterns
    
    # Quality control
    max_deviation: float = 0.3  # Maximum allowed deviation from original
    validation_ratio: float = 0.1  # Fraction to use for validation


@dataclass
class AugmentationResult:
    """Result of data augmentation."""
    success: bool
    data: Optional[Dict[str, np.ndarray]] = None
    augmentation_info: Dict[str, Any] = field(default_factory=dict)
    original_size: int = 0
    augmented_size: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DataAugmenter:
    """
    Advanced data augmenter for time series neural training.
    
    This class provides sophisticated augmentation techniques specifically
    designed for financial time series data, helping improve model
    generalization while preserving important temporal characteristics.
    """
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Augmentation statistics
        self.augmentation_stats = {
            'total_augmented': 0,
            'augmentations_applied': {aug.value: 0 for aug in AugmentationType},
            'generated_samples': 0,
            'quality_scores': []
        }
        
        self.logger.info("DataAugmenter initialized")
    
    async def augment(self, data: Dict[str, np.ndarray]) -> AugmentationResult:
        """
        Augment time series data for neural training.
        
        Args:
            data: Dictionary containing 'X' and 'y' arrays
        
        Returns:
            AugmentationResult with augmented data
        """
        try:
            self.logger.info("Starting data augmentation")
            
            # Validate input
            validation_result = self._validate_input(data)
            if not validation_result['valid']:
                return AugmentationResult(
                    success=False,
                    errors=validation_result['errors']
                )
            
            X = data['X']  # Shape: (n_samples, n_timesteps, n_features)
            y = data['y']  # Shape: (n_samples, forecast_horizon)
            
            original_size = len(X)
            target_size = int(original_size * self.config.augmentation_factor)
            augmented_samples = target_size - original_size
            
            if augmented_samples <= 0:
                return AugmentationResult(
                    success=True,
                    data=data,
                    original_size=original_size,
                    augmented_size=original_size,
                    augmentation_info={'factor': 1.0, 'methods': []}
                )
            
            # Initialize augmented data lists
            augmented_X = [X]
            augmented_y = [y]
            applied_methods = []
            warnings = []
            
            # Apply different augmentation techniques
            samples_needed = augmented_samples
            samples_generated = 0
            
            for augmentation_type in self.config.enabled_augmentations:
                if samples_generated >= samples_needed:
                    break
                
                batch_size = min(samples_needed - samples_generated, original_size)
                
                aug_result = await self._apply_augmentation(
                    X[:batch_size], y[:batch_size], augmentation_type
                )
                
                if aug_result['success']:
                    augmented_X.append(aug_result['X_aug'])
                    augmented_y.append(aug_result['y_aug'])
                    samples_generated += len(aug_result['X_aug'])
                    applied_methods.append(augmentation_type.value)
                    
                    self.augmentation_stats['augmentations_applied'][augmentation_type.value] += len(aug_result['X_aug'])
                else:
                    warnings.extend(aug_result['warnings'])
            
            # Combine all augmented data
            final_X = np.concatenate(augmented_X, axis=0)
            final_y = np.concatenate(augmented_y, axis=0)
            
            # Shuffle the combined data
            indices = np.random.permutation(len(final_X))
            final_X = final_X[indices]
            final_y = final_y[indices]
            
            # Quality validation
            quality_score = self._assess_augmentation_quality(X, y, final_X, final_y)
            
            # Update statistics
            self.augmentation_stats['total_augmented'] += 1
            self.augmentation_stats['generated_samples'] += samples_generated
            self.augmentation_stats['quality_scores'].append(quality_score)
            
            # Prepare result
            augmented_data = {'X': final_X, 'y': final_y}
            
            # Copy other data if present
            for key, value in data.items():
                if key not in ['X', 'y']:
                    # Extend other arrays to match new size
                    if isinstance(value, np.ndarray) and len(value) == original_size:
                        # Repeat the array to match new size
                        repeat_factor = len(final_X) // len(value)
                        remainder = len(final_X) % len(value)
                        extended_value = np.tile(value, repeat_factor)
                        if remainder > 0:
                            extended_value = np.concatenate([extended_value, value[:remainder]])
                        augmented_data[key] = extended_value[indices]
                    else:
                        augmented_data[key] = value
            
            augmentation_info = {
                'original_size': original_size,
                'augmented_size': len(final_X),
                'augmentation_factor': len(final_X) / original_size,
                'methods_applied': applied_methods,
                'quality_score': quality_score
            }
            
            return AugmentationResult(
                success=True,
                data=augmented_data,
                augmentation_info=augmentation_info,
                original_size=original_size,
                augmented_size=len(final_X),
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Error in data augmentation: {e}")
            return AugmentationResult(
                success=False,
                errors=[str(e)]
            )
    
    def _validate_input(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate input data."""
        errors = []
        
        if 'X' not in data or 'y' not in data:
            errors.append("Data must contain 'X' and 'y' arrays")
            return {'valid': False, 'errors': errors}
        
        X = data['X']
        y = data['y']
        
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            errors.append("X and y must be numpy arrays")
        
        if len(X) != len(y):
            errors.append("X and y must have same number of samples")
        
        if X.ndim != 3:
            errors.append("X must be 3D array (samples, timesteps, features)")
        
        if y.ndim != 2:
            errors.append("y must be 2D array (samples, forecast_horizon)")
        
        if len(X) < 10:
            errors.append("Need at least 10 samples for augmentation")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _apply_augmentation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmentation_type: AugmentationType
    ) -> Dict[str, Any]:
        """Apply specific augmentation technique."""
        try:
            if augmentation_type == AugmentationType.NOISE_INJECTION:
                return self._noise_injection(X, y)
            elif augmentation_type == AugmentationType.JITTERING:
                return self._jittering(X, y)
            elif augmentation_type == AugmentationType.TIME_WARPING:
                return self._time_warping(X, y)
            elif augmentation_type == AugmentationType.MAGNITUDE_WARPING:
                return self._magnitude_warping(X, y)
            elif augmentation_type == AugmentationType.SCALING:
                return self._scaling(X, y)
            elif augmentation_type == AugmentationType.WINDOW_SLICING:
                return self._window_slicing(X, y)
            elif augmentation_type == AugmentationType.PERMUTATION:
                return self._permutation(X, y)
            elif augmentation_type == AugmentationType.MIXING:
                return self._mixing(X, y)
            elif augmentation_type == AugmentationType.SPAWNING:
                return self._spawning(X, y)
            else:
                return {
                    'success': False,
                    'warnings': [f"Unsupported augmentation type: {augmentation_type.value}"]
                }
                
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Augmentation {augmentation_type.value} failed: {str(e)}"]
            }
    
    def _noise_injection(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Add noise to the data."""
        X_aug = X.copy()
        y_aug = y.copy()
        
        # Calculate noise level based on data scale
        data_std = np.std(X)
        noise_level = data_std * self.config.noise_ratio
        
        if self.config.noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, X.shape)
        elif self.config.noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, X.shape)
        elif self.config.noise_type == 'laplace':
            noise = np.random.laplace(0, noise_level / np.sqrt(2), X.shape)
        else:
            noise = np.random.normal(0, noise_level, X.shape)
        
        X_aug += noise
        
        # Add smaller noise to targets
        target_noise_level = np.std(y) * self.config.noise_ratio * 0.5
        target_noise = np.random.normal(0, target_noise_level, y.shape)
        y_aug += target_noise
        
        return {
            'success': True,
            'X_aug': X_aug,
            'y_aug': y_aug
        }
    
    def _jittering(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Apply jittering to the data."""
        X_aug = X.copy()
        y_aug = y.copy()
        
        # Apply random small shifts
        data_range = np.ptp(X)  # Peak-to-peak range
        jitter_amount = data_range * self.config.jitter_ratio
        
        jitter = np.random.uniform(-jitter_amount, jitter_amount, X.shape)
        X_aug += jitter
        
        return {
            'success': True,
            'X_aug': X_aug,
            'y_aug': y_aug
        }
    
    def _time_warping(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Apply time warping to the sequences."""
        if not self.config.time_warping:
            return self._noise_injection(X, y)  # Fallback
        
        X_aug = np.zeros_like(X)
        y_aug = y.copy()
        
        n_samples, n_timesteps, n_features = X.shape
        
        for i in range(n_samples):
            for j in range(n_features):
                # Create warping function
                warp_points = np.linspace(0, n_timesteps - 1, self.config.warp_knots)
                warp_values = warp_points + np.random.uniform(
                    -self.config.warp_strength * n_timesteps,
                    self.config.warp_strength * n_timesteps,
                    self.config.warp_knots
                )
                
                # Ensure monotonic warping
                warp_values = np.sort(warp_values)
                warp_values = np.clip(warp_values, 0, n_timesteps - 1)
                
                # Interpolate warping function
                warp_func = interpolate.interp1d(
                    warp_points, warp_values, kind='linear',
                    bounds_error=False, fill_value='extrapolate'
                )
                
                # Apply warping
                original_indices = np.arange(n_timesteps)
                warped_indices = warp_func(original_indices)
                warped_indices = np.clip(warped_indices, 0, n_timesteps - 1)
                
                # Interpolate the sequence
                interp_func = interpolate.interp1d(
                    original_indices, X[i, :, j], kind='linear',
                    bounds_error=False, fill_value='extrapolate'
                )
                
                X_aug[i, :, j] = interp_func(warped_indices)
        
        return {
            'success': True,
            'X_aug': X_aug,
            'y_aug': y_aug
        }
    
    def _magnitude_warping(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Apply magnitude warping to the sequences."""
        if not self.config.magnitude_warping:
            return self._scaling(X, y)  # Fallback
        
        X_aug = X.copy()
        y_aug = y.copy()
        
        n_samples, n_timesteps, n_features = X.shape
        
        for i in range(n_samples):
            for j in range(n_features):
                # Create magnitude warping curve
                warp_points = np.linspace(0, n_timesteps - 1, self.config.magnitude_knots)
                warp_multipliers = 1 + np.random.uniform(
                    -self.config.magnitude_strength,
                    self.config.magnitude_strength,
                    self.config.magnitude_knots
                )
                
                # Interpolate warping multipliers
                warp_func = interpolate.interp1d(
                    warp_points, warp_multipliers, kind='cubic',
                    bounds_error=False, fill_value='extrapolate'
                )
                
                # Apply magnitude warping
                timestep_indices = np.arange(n_timesteps)
                multipliers = warp_func(timestep_indices)
                
                X_aug[i, :, j] *= multipliers
        
        return {
            'success': True,
            'X_aug': X_aug,
            'y_aug': y_aug
        }
    
    def _scaling(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Apply random scaling to the data."""
        scale_factor = np.random.uniform(*self.config.scaling_range)
        
        X_aug = X * scale_factor
        y_aug = y * scale_factor
        
        return {
            'success': True,
            'X_aug': X_aug,
            'y_aug': y_aug
        }
    
    def _window_slicing(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Apply window slicing augmentation."""
        n_samples, n_timesteps, n_features = X.shape
        slice_size = int(n_timesteps * self.config.window_slice_ratio)
        
        if slice_size < 1:
            return self._noise_injection(X, y)  # Fallback
        
        X_aug = np.zeros_like(X)
        y_aug = y.copy()
        
        for i in range(n_samples):
            # Random start position for slice
            start_pos = np.random.randint(0, max(1, n_timesteps - slice_size))
            
            # Copy original data
            X_aug[i] = X[i].copy()
            
            # Remove the slice and interpolate
            mask = np.ones(n_timesteps, dtype=bool)
            mask[start_pos:start_pos + slice_size] = False
            
            for j in range(n_features):
                valid_indices = np.where(mask)[0]
                valid_values = X[i, mask, j]
                
                if len(valid_values) > 1:
                    # Interpolate missing values
                    interp_func = interpolate.interp1d(
                        valid_indices, valid_values, kind='linear',
                        bounds_error=False, fill_value='extrapolate'
                    )
                    
                    missing_indices = np.where(~mask)[0]
                    X_aug[i, missing_indices, j] = interp_func(missing_indices)
        
        return {
            'success': True,
            'X_aug': X_aug,
            'y_aug': y_aug
        }
    
    def _permutation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Apply segment permutation."""
        n_samples, n_timesteps, n_features = X.shape
        segment_size = n_timesteps // self.config.permutation_segments
        
        if segment_size < 1:
            return self._noise_injection(X, y)  # Fallback
        
        X_aug = np.zeros_like(X)
        y_aug = y.copy()
        
        for i in range(n_samples):
            # Create segments
            segments = []
            for s in range(self.config.permutation_segments):
                start = s * segment_size
                end = min((s + 1) * segment_size, n_timesteps)
                segments.append(X[i, start:end])
            
            # Permute segments
            perm_indices = np.random.permutation(len(segments))
            
            # Reconstruct sequence
            current_pos = 0
            for seg_idx in perm_indices:
                segment = segments[seg_idx]
                end_pos = min(current_pos + len(segment), n_timesteps)
                X_aug[i, current_pos:end_pos] = segment[:end_pos - current_pos]
                current_pos = end_pos
        
        return {
            'success': True,
            'X_aug': X_aug,
            'y_aug': y_aug
        }
    
    def _mixing(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Apply mixup augmentation."""
        n_samples = len(X)
        
        # Random pairs for mixing
        indices1 = np.arange(n_samples)
        indices2 = np.random.permutation(n_samples)
        
        # Mixup parameter
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha, n_samples)
        lam = lam.reshape(-1, 1, 1)
        
        # Mix the data
        X_aug = lam * X[indices1] + (1 - lam) * X[indices2]
        
        # Mix targets
        lam_y = lam.squeeze().reshape(-1, 1)
        y_aug = lam_y * y[indices1] + (1 - lam_y) * y[indices2]
        
        return {
            'success': True,
            'X_aug': X_aug,
            'y_aug': y_aug
        }
    
    def _spawning(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Generate new samples by combining existing patterns."""
        n_samples, n_timesteps, n_features = X.shape
        
        # Generate new samples by interpolating between existing ones
        X_aug = np.zeros_like(X)
        y_aug = np.zeros_like(y)
        
        for i in range(n_samples):
            # Select two random samples to interpolate between
            idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
            
            # Random interpolation weight
            alpha = np.random.beta(2, 2)  # Beta distribution for smooth interpolation
            
            # Interpolate
            X_aug[i] = alpha * X[idx1] + (1 - alpha) * X[idx2]
            y_aug[i] = alpha * y[idx1] + (1 - alpha) * y[idx2]
            
            # Add small amount of noise for diversity
            noise_level = np.std(X_aug[i]) * 0.01
            X_aug[i] += np.random.normal(0, noise_level, X_aug[i].shape)
        
        return {
            'success': True,
            'X_aug': X_aug,
            'y_aug': y_aug
        }
    
    def _assess_augmentation_quality(
        self,
        X_orig: np.ndarray,
        y_orig: np.ndarray,
        X_aug: np.ndarray,
        y_aug: np.ndarray
    ) -> float:
        """Assess the quality of augmentation."""
        try:
            # Compare statistical properties
            orig_mean = np.mean(X_orig)
            aug_mean = np.mean(X_aug)
            
            orig_std = np.std(X_orig)
            aug_std = np.std(X_aug)
            
            # Calculate quality metrics
            mean_preservation = 1 - abs(orig_mean - aug_mean) / (abs(orig_mean) + 1e-8)
            std_preservation = 1 - abs(orig_std - aug_std) / (orig_std + 1e-8)
            
            # Check for extreme deviations
            max_deviation = np.max(np.abs(X_aug - X_orig)) / (np.std(X_orig) + 1e-8)
            deviation_score = max(0, 1 - max_deviation / 10)  # Penalize large deviations
            
            # Overall quality score
            quality_score = 0.4 * mean_preservation + 0.4 * std_preservation + 0.2 * deviation_score
            
            return np.clip(quality_score, 0, 1)
            
        except Exception:
            return 0.5  # Neutral score if assessment fails
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get augmentation statistics."""
        stats = self.augmentation_stats.copy()
        
        if stats['quality_scores']:
            stats['avg_quality'] = np.mean(stats['quality_scores'])
            stats['min_quality'] = np.min(stats['quality_scores'])
        else:
            stats['avg_quality'] = 0
            stats['min_quality'] = 0
        
        return stats
    
    def get_config(self) -> AugmentationConfig:
        """Get current configuration."""
        return self.config