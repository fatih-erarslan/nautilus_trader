"""
TensorRT Optimization for Maximum Production Performance
Provides INT8 quantization, kernel fusion, and optimized inference for production deployment.
"""

import logging
import time
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)

# TensorRT imports with fallback
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
    logger.info(f"TensorRT {trt.__version__} available")
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available - install TensorRT for maximum performance")

try:
    from torch2trt import torch2trt, TRTModule
    TORCH2TRT_AVAILABLE = True
except ImportError:
    TORCH2TRT_AVAILABLE = False
    logger.warning("torch2trt not available - limited TensorRT functionality")


@dataclass
class TensorRTConfig:
    """Configuration for TensorRT optimization."""
    # Precision settings
    use_fp16: bool = True
    use_int8: bool = False
    use_tf32: bool = True
    
    # Optimization settings
    max_workspace_size: int = 1 << 30  # 1GB
    max_batch_size: int = 64
    enable_dla: bool = False  # Deep Learning Accelerator
    
    # Calibration for INT8
    calibration_dataset_size: int = 1000
    calibration_cache_file: Optional[str] = None
    
    # Build settings
    strict_type_constraints: bool = False
    enable_tactic_sources: bool = True
    profiling_verbosity: str = "DETAILED"  # LAYER_NAMES_ONLY, DETAILED, NONE
    
    # Dynamic shapes
    enable_dynamic_shapes: bool = True
    min_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    opt_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    max_shapes: Optional[Dict[str, Tuple[int, ...]]] = None


@dataclass
class TensorRTOptimizationResult:
    """Result of TensorRT optimization."""
    success: bool
    engine_path: Optional[str] = None
    optimization_time: float = 0.0
    speedup_factor: float = 1.0
    memory_reduction_mb: float = 0.0
    model_size_mb: float = 0.0
    error_message: Optional[str] = None
    
    # Performance metrics
    original_latency_ms: float = 0.0
    optimized_latency_ms: float = 0.0
    throughput_improvement: float = 0.0


class TensorRTCalibrator:
    """INT8 calibration for TensorRT."""
    
    def __init__(self, 
                 calibration_data: List[torch.Tensor],
                 cache_file: Optional[str] = None):
        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.current_index = 0
        
        # Allocate GPU memory for calibration
        self.device_inputs = []
        self.bindings = []
        
    def get_batch_size(self) -> int:
        """Get calibration batch size."""
        return self.calibration_data[0].shape[0] if self.calibration_data else 1
    
    def get_batch(self, names: List[str]) -> List[int]:
        """Get next calibration batch."""
        if self.current_index >= len(self.calibration_data):
            return None
        
        # Copy data to GPU
        batch = self.calibration_data[self.current_index]
        
        # Allocate device memory if needed
        if not self.device_inputs:
            for i, data in enumerate([batch]):
                device_mem = cuda.mem_alloc(data.nbytes)
                self.device_inputs.append(device_mem)
                self.bindings.append(int(device_mem))
        
        # Copy host to device
        cuda.memcpy_htod(self.device_inputs[0], batch.cpu().numpy())
        
        self.current_index += 1
        return self.bindings
    
    def read_calibration_cache(self) -> bytes:
        """Read calibration cache if available."""
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache: bytes):
        """Write calibration cache."""
        if self.cache_file:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                f.write(cache)


class TensorRTOptimizer:
    """Advanced TensorRT optimizer for maximum performance."""
    
    def __init__(self, config: TensorRTConfig = None):
        self.config = config or TensorRTConfig()
        
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        
        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = None
        self.engine = None
        self.context = None
        
        # Performance tracking
        self.optimization_history = []
        
        logger.info(f"TensorRT optimizer initialized with TensorRT {trt.__version__}")
    
    def optimize_model(self, 
                      model: nn.Module,
                      sample_input: torch.Tensor,
                      output_path: str,
                      calibration_data: Optional[List[torch.Tensor]] = None) -> TensorRTOptimizationResult:
        """Optimize PyTorch model with TensorRT."""
        start_time = time.time()
        
        try:
            # Benchmark original model
            original_latency = self._benchmark_model(model, sample_input)
            
            # Create TensorRT engine
            engine_path = self._build_tensorrt_engine(
                model, sample_input, output_path, calibration_data
            )
            
            if not engine_path:
                return TensorRTOptimizationResult(
                    success=False,
                    error_message="Failed to build TensorRT engine",
                    optimization_time=time.time() - start_time
                )
            
            # Load and benchmark optimized engine
            trt_model = self._load_tensorrt_engine(engine_path)
            optimized_latency = self._benchmark_tensorrt_model(trt_model, sample_input)
            
            # Calculate metrics
            speedup = original_latency / optimized_latency if optimized_latency > 0 else 1.0
            optimization_time = time.time() - start_time
            
            # Get model size
            model_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
            
            result = TensorRTOptimizationResult(
                success=True,
                engine_path=engine_path,
                optimization_time=optimization_time,
                speedup_factor=speedup,
                model_size_mb=model_size_mb,
                original_latency_ms=original_latency,
                optimized_latency_ms=optimized_latency,
                throughput_improvement=speedup - 1.0
            )
            
            # Store optimization history
            self.optimization_history.append(result)
            
            logger.info(f"TensorRT optimization complete: {speedup:.2f}x speedup, "
                       f"{optimization_time:.1f}s build time, {model_size_mb:.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            return TensorRTOptimizationResult(
                success=False,
                error_message=str(e),
                optimization_time=time.time() - start_time
            )
    
    def _build_tensorrt_engine(self,
                              model: nn.Module,
                              sample_input: torch.Tensor,
                              output_path: str,
                              calibration_data: Optional[List[torch.Tensor]] = None) -> Optional[str]:
        """Build TensorRT engine from PyTorch model."""
        try:
            # Method 1: Try torch2trt first (faster)
            if TORCH2TRT_AVAILABLE:
                engine_path = self._build_with_torch2trt(model, sample_input, output_path)
                if engine_path:
                    return engine_path
            
            # Method 2: Use ONNX export + TensorRT (more compatible)
            return self._build_with_onnx(model, sample_input, output_path, calibration_data)
            
        except Exception as e:
            logger.error(f"Engine building failed: {e}")
            return None
    
    def _build_with_torch2trt(self,
                             model: nn.Module,
                             sample_input: torch.Tensor,
                             output_path: str) -> Optional[str]:
        """Build engine using torch2trt (faster method)."""
        try:
            logger.info("Building TensorRT engine with torch2trt...")
            
            model.eval()
            
            # Convert to TensorRT
            trt_model = torch2trt(
                model,
                [sample_input],
                fp16_mode=self.config.use_fp16,
                int8_mode=self.config.use_int8,
                max_workspace_size=self.config.max_workspace_size,
                max_batch_size=self.config.max_batch_size,
                strict_type_constraints=self.config.strict_type_constraints
            )
            
            # Save engine
            engine_path = f"{output_path}.trt"
            torch.save(trt_model.state_dict(), engine_path)
            
            logger.info(f"torch2trt engine saved to {engine_path}")
            return engine_path
            
        except Exception as e:
            logger.warning(f"torch2trt build failed: {e}")
            return None
    
    def _build_with_onnx(self,
                        model: nn.Module,
                        sample_input: torch.Tensor,
                        output_path: str,
                        calibration_data: Optional[List[torch.Tensor]] = None) -> Optional[str]:
        """Build engine using ONNX export (more compatible)."""
        try:
            logger.info("Building TensorRT engine via ONNX...")
            
            # Export to ONNX
            onnx_path = f"{output_path}.onnx"
            
            model.eval()
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                } if self.config.enable_dynamic_shapes else None,
                do_constant_folding=True,
                opset_version=17
            )
            
            # Build TensorRT engine from ONNX
            engine_path = self._build_engine_from_onnx(onnx_path, calibration_data)
            
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            
            return engine_path
            
        except Exception as e:
            logger.error(f"ONNX-based build failed: {e}")
            return None
    
    def _build_engine_from_onnx(self,
                               onnx_path: str,
                               calibration_data: Optional[List[torch.Tensor]] = None) -> Optional[str]:
        """Build TensorRT engine from ONNX model."""
        try:
            # Create builder configuration
            config = self.builder.create_builder_config()
            config.max_workspace_size = self.config.max_workspace_size
            
            # Set precision flags
            if self.config.use_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            if self.config.use_int8 and calibration_data:
                config.set_flag(trt.BuilderFlag.INT8)
                
                # Setup calibrator
                calibrator = TensorRTCalibrator(
                    calibration_data,
                    self.config.calibration_cache_file
                )
                config.int8_calibrator = calibrator
            
            if self.config.use_tf32:
                config.set_flag(trt.BuilderFlag.TF32)
            
            # Enable tactic sources
            if self.config.enable_tactic_sources:
                config.set_tactic_sources(
                    1 << int(trt.TacticSource.CUBLAS) |
                    1 << int(trt.TacticSource.CUBLAS_LT) |
                    1 << int(trt.TacticSource.CUDNN)
                )
            
            # Set profiling verbosity
            config.profiling_verbosity = getattr(
                trt.ProfilingVerbosity, 
                self.config.profiling_verbosity,
                trt.ProfilingVerbosity.DETAILED
            )
            
            # Parse ONNX model
            network = self.builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            
            parser = trt.OnnxParser(network, self.logger)
            
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX parse error: {parser.get_error(error)}")
                    return None
            
            # Setup dynamic shapes if enabled
            if self.config.enable_dynamic_shapes:
                self._setup_dynamic_shapes(config, network)
            
            # Build engine
            logger.info("Building TensorRT engine (this may take several minutes)...")
            
            engine = self.builder.build_engine(network, config)
            
            if not engine:
                logger.error("Failed to build TensorRT engine")
                return None
            
            # Serialize and save engine
            engine_path = f"{onnx_path.replace('.onnx', '.trt')}"
            
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT engine saved to {engine_path}")
            return engine_path
            
        except Exception as e:
            logger.error(f"Engine building from ONNX failed: {e}")
            return None
    
    def _setup_dynamic_shapes(self, config, network):
        """Setup dynamic shape optimization profiles."""
        try:
            profile = self.builder.create_optimization_profile()
            
            # Get input tensor
            input_tensor = network.get_input(0)
            input_shape = input_tensor.shape
            
            # Define shape ranges
            min_shape = self.config.min_shapes or {
                'input': (1, *input_shape[1:])
            }
            opt_shape = self.config.opt_shapes or {
                'input': (self.config.max_batch_size // 2, *input_shape[1:])
            }
            max_shape = self.config.max_shapes or {
                'input': (self.config.max_batch_size, *input_shape[1:])
            }
            
            # Set profile
            profile.set_shape(
                input_tensor.name,
                min_shape['input'],
                opt_shape['input'],
                max_shape['input']
            )
            
            config.add_optimization_profile(profile)
            
        except Exception as e:
            logger.warning(f"Dynamic shapes setup failed: {e}")
    
    def _load_tensorrt_engine(self, engine_path: str):
        """Load TensorRT engine for inference."""
        try:
            if TORCH2TRT_AVAILABLE and engine_path.endswith('.trt'):
                # Load torch2trt model
                trt_model = TRTModule()
                trt_model.load_state_dict(torch.load(engine_path))
                return trt_model
            else:
                # Load native TensorRT engine
                return self._load_native_tensorrt_engine(engine_path)
                
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            return None
    
    def _load_native_tensorrt_engine(self, engine_path: str):
        """Load native TensorRT engine."""
        try:
            # Load serialized engine
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            # Create runtime and deserialize engine
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            if not engine:
                raise RuntimeError("Failed to deserialize engine")
            
            # Create execution context
            context = engine.create_execution_context()
            
            return {'engine': engine, 'context': context}
            
        except Exception as e:
            logger.error(f"Native TensorRT engine loading failed: {e}")
            return None
    
    def _benchmark_model(self, model: nn.Module, sample_input: torch.Tensor, iterations: int = 100) -> float:
        """Benchmark original PyTorch model."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        if sample_input.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(sample_input)
        
        if sample_input.device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.perf_counter() - start_time
        return (total_time / iterations) * 1000  # Convert to ms
    
    def _benchmark_tensorrt_model(self, trt_model, sample_input: torch.Tensor, iterations: int = 100) -> float:
        """Benchmark TensorRT optimized model."""
        if isinstance(trt_model, dict):
            # Native TensorRT engine
            return self._benchmark_native_tensorrt(trt_model, sample_input, iterations)
        else:
            # torch2trt model
            return self._benchmark_torch2trt(trt_model, sample_input, iterations)
    
    def _benchmark_torch2trt(self, trt_model, sample_input: torch.Tensor, iterations: int = 100) -> float:
        """Benchmark torch2trt model."""
        trt_model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = trt_model(sample_input)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = trt_model(sample_input)
        
        torch.cuda.synchronize()
        
        total_time = time.perf_counter() - start_time
        return (total_time / iterations) * 1000
    
    def _benchmark_native_tensorrt(self, trt_model: Dict, sample_input: torch.Tensor, iterations: int = 100) -> float:
        """Benchmark native TensorRT engine."""
        engine = trt_model['engine']
        context = trt_model['context']
        
        # Allocate GPU memory
        input_data = sample_input.cpu().numpy()
        
        # Get binding info
        input_binding = engine.get_binding_index('input')
        output_binding = engine.get_binding_index('output')
        
        # Allocate device memory
        d_input = cuda.mem_alloc(input_data.nbytes)
        
        # Get output shape and allocate
        output_shape = engine.get_binding_shape(output_binding)
        output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize
        d_output = cuda.mem_alloc(output_size)
        
        bindings = [int(d_input), int(d_output)]
        
        # Warmup
        for _ in range(10):
            cuda.memcpy_htod(d_input, input_data)
            context.execute_v2(bindings)
        
        cuda.Context.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            cuda.memcpy_htod(d_input, input_data)
            context.execute_v2(bindings)
        
        cuda.Context.synchronize()
        
        total_time = time.perf_counter() - start_time
        return (total_time / iterations) * 1000
    
    def create_calibration_dataset(self, 
                                 model: nn.Module,
                                 data_loader,
                                 max_samples: int = None) -> List[torch.Tensor]:
        """Create calibration dataset for INT8 quantization."""
        calibration_data = []
        max_samples = max_samples or self.config.calibration_dataset_size
        
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= max_samples:
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                calibration_data.append(inputs.cuda())
        
        logger.info(f"Created calibration dataset with {len(calibration_data)} samples")
        return calibration_data
    
    def analyze_optimization_opportunities(self, 
                                         model: nn.Module,
                                         sample_input: torch.Tensor) -> Dict[str, Any]:
        """Analyze model for TensorRT optimization opportunities."""
        analysis = {
            'model_info': {},
            'optimization_potential': {},
            'recommendations': []
        }
        
        # Model analysis
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming FP32
        
        analysis['model_info'] = {
            'parameter_count': param_count,
            'model_size_mb': model_size_mb,
            'input_shape': list(sample_input.shape),
            'device': str(sample_input.device)
        }
        
        # Optimization potential
        if model_size_mb > 100:
            analysis['recommendations'].append(
                "Large model (>100MB) - TensorRT optimization highly recommended"
            )
        
        if sample_input.device.type == 'cuda':
            gpu_props = torch.cuda.get_device_properties(sample_input.device)
            
            if gpu_props.major >= 8:  # Ampere+
                analysis['recommendations'].append(
                    "Ampere+ GPU detected - INT8 quantization available for maximum speedup"
                )
            elif gpu_props.major >= 7:  # Turing+
                analysis['recommendations'].append(
                    "Turing+ GPU detected - FP16 optimization recommended"
                )
        
        # Layer analysis
        conv_layers = sum(1 for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)))
        linear_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        
        analysis['model_info']['conv_layers'] = conv_layers
        analysis['model_info']['linear_layers'] = linear_layers
        
        if conv_layers > 5:
            analysis['recommendations'].append(
                f"Many convolution layers ({conv_layers}) - excellent TensorRT optimization candidate"
            )
        
        if linear_layers > 10:
            analysis['recommendations'].append(
                f"Many linear layers ({linear_layers}) - good TensorRT optimization potential"
            )
        
        return analysis
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        if not self.optimization_history:
            return {'error': 'No optimizations performed yet'}
        
        speedups = [r.speedup_factor for r in self.optimization_history if r.success]
        optimization_times = [r.optimization_time for r in self.optimization_history]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'successful_optimizations': len(speedups),
            'success_rate': len(speedups) / len(self.optimization_history),
            'avg_speedup': np.mean(speedups) if speedups else 0,
            'max_speedup': np.max(speedups) if speedups else 0,
            'avg_optimization_time': np.mean(optimization_times),
            'tensorrt_version': trt.__version__ if TENSORRT_AVAILABLE else 'N/A',
            'torch2trt_available': TORCH2TRT_AVAILABLE
        }


# Convenience functions
def optimize_model_with_tensorrt(model: nn.Module,
                                sample_input: torch.Tensor,
                                output_path: str,
                                use_int8: bool = False,
                                calibration_data: Optional[List[torch.Tensor]] = None) -> TensorRTOptimizationResult:
    """Quick TensorRT optimization with default settings."""
    config = TensorRTConfig(
        use_fp16=True,
        use_int8=use_int8,
        max_batch_size=sample_input.shape[0] * 4  # Allow 4x batch size
    )
    
    optimizer = TensorRTOptimizer(config)
    return optimizer.optimize_model(model, sample_input, output_path, calibration_data)


def create_int8_calibration_dataset(model: nn.Module,
                                   data_loader,
                                   num_samples: int = 1000) -> List[torch.Tensor]:
    """Create calibration dataset for INT8 quantization."""
    optimizer = TensorRTOptimizer()
    return optimizer.create_calibration_dataset(model, data_loader, num_samples)