#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 10:33:40 2025

@author: ashina
"""

import logging
import os
import sys
import platform
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import pennylane as qml
import threading
from functools import lru_cache
import time

try:
    # Optional CUDA imports
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    # Optional AMD GPU imports
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

try:
    # Optional quantum library import
    import pennylane as qml
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("hw_manager.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Hardware Manager")

class HardwareManager:
    """
    Hardware Manager for detecting, managing and optimizing hardware resources.
    Supports CPU, NVIDIA GPUs, AMD GPUs, and quantum devices.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_manager(cls, **kwargs):
        """
        Singleton pattern implementation for hardware manager.
        
        Args:
            **kwargs: Arguments to pass to constructor
            
        Returns:
            HardwareManager: Singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance
    
    def __init__(self, force_cpu: bool = False, max_gpus: Optional[int] = None,
                use_jit: bool = True, multi_gpu: bool = True,
                quantum_shots: Optional[int] = None, default_quantum_wires: int = 12,
                log_level: int = logging.INFO):
        """
        Initialize the hardware manager.
        
        Args:
            force_cpu (bool): Force CPU usage even when GPUs are available
            max_gpus (int, optional): Maximum number of GPUs to use
            use_jit (bool): Enable JIT compilation for improved performance
            multi_gpu (bool): Allow multiple GPU usage if available
            quantum_shots (int, optional): Number of shots for quantum simulation
            default_quantum_wires (int): Default number of qubits/wires for quantum circuits
            log_level (int): Logging level
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Configuration
        self.force_cpu = force_cpu
        self.max_gpus = max_gpus
        self.use_jit = use_jit
        self.multi_gpu = multi_gpu
        self.quantum_shots = quantum_shots
        self.default_quantum_wires = default_quantum_wires
        self.default_num_wires = default_quantum_wires  # Add alias for compatibility
        
        # State variables
        self._is_initialized = False
        self.gpu_available = False
        self.quantum_available = QUANTUM_AVAILABLE
        self.devices = {
            'cpu': {'available': True, 'count': 1, 'memory': None},
            'nvidia_gpu': {'available': False, 'count': 0, 'devices': []},
            'amd_gpu': {'available': False, 'count': 0, 'devices': []},
            'quantum': {'available': QUANTUM_AVAILABLE, 'devices': []}
        }
        
        # Resource tracking
        self.memory_usage = {}
        self.active_contexts = {}
        
        # Cache for resource-intensive operations (LRU with size limit)
        self.circuit_cache = {}
        self.max_cache_size = 100
        self.cache_locks = {}
        
        # Resource allocation
        self.device_assignments = {}
        
        # Performance metrics
        self.execution_times = {}
        
        # Add missing attributes for compatibility
        self.device_name = None
        self.window_size = 100
        self.forecast_horizon = 10
        self.n_features = 10
        self.annealing_steps = 100
        self.kernel_size = 3
        self.max_qubits = 12
        self.learning_rate = 0.01
        self.current_regime = "normal"
        self.weights = None
        self.prediction_errors = []
        self.training_history = []
        self.hw_accelerator = None
    
    def initialize_hardware(self) -> bool:
        """
        Detect and initialize hardware resources.
        
        Returns:
            bool: True if initialization succeeded
        """
        if self._is_initialized:
            self.logger.info("Hardware already initialized")
            return True
        
        try:
            self.logger.info("Initializing hardware resources")
            
            # Detect CPU resources
            self._detect_cpu_resources()
            
            # Detect GPU resources
            if not self.force_cpu:
                self._detect_gpu_resources()
            
            # Configure quantum resources
            self._configure_quantum_resources()
            
            # Initialize resource tracking
            self._initialize_resource_tracking()
            
            self._is_initialized = True
            self.logger.info(f"Hardware initialization complete. "
                           f"CPU: {self.devices['cpu']['count']} cores, "
                           f"NVIDIA GPU: {self.devices['nvidia_gpu']['count']}, "
                           f"AMD GPU: {self.devices['amd_gpu']['count']}, "
                           f"Quantum: {self.quantum_available}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {str(e)}", exc_info=True)
            return False
    
    def _detect_cpu_resources(self) -> None:
        """Detect CPU resources and capabilities."""
        import multiprocessing
        
        try:
            cpu_count = multiprocessing.cpu_count()
            self.devices['cpu']['count'] = cpu_count
            self.devices['cpu']['model'] = platform.processor()
            
            # Set memory info if psutil is available
            try:
                import psutil
                mem = psutil.virtual_memory()
                self.devices['cpu']['memory'] = mem.total
                self.logger.info(f"Detected {cpu_count} CPU cores with {mem.total / (1024**3):.2f} GB RAM")
            except ImportError:
                self.logger.info(f"Detected {cpu_count} CPU cores, memory info unavailable")
                
        except Exception as e:
            self.logger.warning(f"CPU detection error: {str(e)}")
            self.devices['cpu']['count'] = 1
    
    # ----- Hardware Management Methods -----#


# --- Replace _detect_gpu_resources in hardware_manager.py ---
    def _detect_gpu_resources(self) -> None:
        """Detect NVIDIA and AMD GPU resources using multiple methods."""
        self.logger.info("Detecting GPU resources...")
        nvidia_detected = False
        amd_detected = False

        # --- Method 1: PyTorch Checks ---
        if TORCH_AVAILABLE:
            try:
                # 1a: Check NVIDIA CUDA
                if torch.cuda.is_available():
                    self.devices['nvidia_gpu']['count'] = torch.cuda.device_count()
                    self.devices['nvidia_gpu']['available'] = True
                    self.gpu_available = True
                    nvidia_detected = True
                    self.devices['nvidia_gpu']['devices'] = []
                    for i in range(self.devices['nvidia_gpu']['count']):
                        try:
                            props = torch.cuda.get_device_properties(i)
                            device_info = {'index': i, 'name': props.name, 'memory': props.total_memory}
                            self.devices['nvidia_gpu']['devices'].append(device_info)
                        except Exception as e_prop:
                            self.logger.warning(f"Could not get properties for NVIDIA GPU {i}: {e_prop}")
                    self.logger.info(f"Detected {self.devices['nvidia_gpu']['count']} NVIDIA GPU(s) via PyTorch CUDA.")
                    # Log details... (optional)

                # 1b: Check AMD ROCm (via PyTorch compilation flags)
                # This checks if PyTorch itself was built with ROCm support
                elif hasattr(torch, "_C") and hasattr(torch._C, "_rocm_version") and torch._C._rocblas_available():
                     # We know ROCm *should* be usable by PyTorch, but getting device count/names this way is hard.
                     # Assume 1 GPU for now if this path is taken. More robust check needed if count > 1 matters.
                     self.devices['amd_gpu']['count'] = 1 # Assumption
                     self.devices['amd_gpu']['available'] = True
                     self.gpu_available = True
                     amd_detected = True
                     # Cannot easily get name/memory this way, add placeholder
                     self.devices['amd_gpu']['devices'] = [{'index': 0, 'name': 'AMD ROCm GPU (detected via PyTorch)', 'memory': None}]
                     self.logger.info("Detected AMD ROCm support via PyTorch build flags.")

                # 1c: Check Apple MPS (Placeholder, adapt if needed)
                # elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                #     self.logger.info("Detected Apple MPS.")
                #     # Update capabilities accordingly if needed

                else:
                    self.logger.debug("PyTorch found, but no CUDA, ROCm, or MPS backend available through it.")

            except Exception as e_torch:
                self.logger.warning(f"Error during PyTorch hardware detection: {e_torch}")
        else:
            self.logger.info("PyTorch not available for GPU detection.")


        # --- Method 2: Fallback Checks (if PyTorch didn't find anything) ---

        # 2a: Fallback NVIDIA Check (pycuda) - Optional
        if not nvidia_detected:
            try:
                import pycuda.driver as cuda # Local import to avoid hard dependency
                cuda.init()
                count = cuda.Device.count()
                if count > 0:
                    self.devices['nvidia_gpu']['count'] = count
                    self.devices['nvidia_gpu']['available'] = True
                    self.gpu_available = True
                    nvidia_detected = True
                    self.devices['nvidia_gpu']['devices'] = []
                    for i in range(count):
                         dev = cuda.Device(i)
                         device_info = {'index': i, 'name': dev.name(), 'memory': dev.total_memory()}
                         self.devices['nvidia_gpu']['devices'].append(device_info)
                    self.logger.info(f"Detected {count} NVIDIA GPU(s) via pycuda fallback.")
                    # Log details...
            except ImportError:
                self.logger.debug("pycuda not installed, skipping fallback NVIDIA check.")
            except Exception as e_pycuda:
                self.logger.warning(f"pycuda fallback check failed: {e_pycuda}")


        # 2b: Fallback AMD Check (ctypes loading ROCm SMI library)
        if not amd_detected:
             try:
                 import ctypes
                 # Try to load the ROCm SMI library - just checking existence often indicates ROCm is present
                 try:
                     # Default library name on Linux
                     rocm_lib = ctypes.CDLL("librocm_smi64.so")
                     self.devices['amd_gpu']['count'] = 1 # Assume at least 1 if library loads
                     self.devices['amd_gpu']['available'] = True
                     self.gpu_available = True
                     amd_detected = True
                     # Cannot easily get details this way, add placeholder
                     if not self.devices['amd_gpu']['devices']: # Avoid overwriting if PyTorch found it
                          self.devices['amd_gpu']['devices'] = [{'index': 0, 'name': 'AMD ROCm GPU (detected via librocm_smi64.so)', 'memory': None}]
                     self.logger.info("Detected AMD ROCm GPU presence via ctypes library loading.")
                 except OSError:
                     self.logger.debug("librocm_smi64.so not found via ctypes, ROCm likely not present or configured for this method.")
                 except Exception as e_ctypes_load:
                      self.logger.warning(f"ctypes check for librocm_smi64.so failed: {e_ctypes_load}")

             except ImportError:
                  self.logger.debug("ctypes not available for AMD fallback check.")
             except Exception as e_ctypes:
                  self.logger.warning(f"ctypes AMD fallback check failed: {e_ctypes}")


        # --- Method 3: Final AMD Check (rocm-smi command) ---
        # Only run if no AMD GPU detected yet, as it failed before
        if not amd_detected:
             self.logger.info("Attempting AMD GPU detection via rocm-smi command as final fallback...")
             import subprocess
             import json
             try:
                 process = subprocess.run(['rocm-smi', '--showallinfo', '--json'], capture_output=True, text=True, check=True, timeout=10)
                 self.logger.debug(f"rocm-smi raw stdout:\n{process.stdout}") # Keep debug log
                 rocm_info = json.loads(process.stdout)
                 amd_devices_info = []
                 card_count = 0
                 for key, card_data in rocm_info.items():
                     if key.startswith('card'):
                         # ... (keep the JSON parsing logic from the previous version) ...
                         try:
                             gpu_id = card_data.get('GPU ID', 'N/A')
                             model = card_data.get('Card model', 'Unknown AMD GPU')
                             total_mem_str = card_data.get('VRAM Total Memory (B)', card_data.get('Total VRAM (B)'))
                             total_mem = int(total_mem_str) if total_mem_str and total_mem_str.isdigit() else 0
                             if total_mem > 0:
                                 device_info = {'index': card_count, 'gpu_id': gpu_id, 'name': model, 'memory': total_mem}
                                 amd_devices_info.append(device_info)
                                 card_count += 1
                         except Exception as e_parse: self.logger.warning(f"Failed to parse rocm-smi info for {key}: {e_parse}")

                 if card_count > 0:
                      self.devices['amd_gpu']['count'] = card_count
                      self.devices['amd_gpu']['available'] = True
                      self.devices['amd_gpu']['devices'] = amd_devices_info
                      self.gpu_available = True
                      amd_detected = True
                      self.logger.info(f"Detected {card_count} AMD ROCm GPU(s) via rocm-smi fallback.")
                      # Log details...
                 else:
                      self.logger.info("No AMD ROCm GPUs detected via rocm-smi fallback.")

             except Exception as e_rocm_smi:
                 self.logger.info(f"rocm-smi command fallback failed: {e_rocm_smi}")
                 # Ensure state is correct if rocm-smi fails
                 if not self.devices['amd_gpu']['available']: # Check if already detected by other means
                      self.devices['amd_gpu']['available'] = False
                      self.devices['amd_gpu']['count'] = 0

        # Log final detection status
        if not self.gpu_available:
            self.logger.info("No compatible GPUs (NVIDIA CUDA or AMD ROCm) detected by any method.")
        else:
            self.logger.info(f"GPU detection complete. NVIDIA Available: {nvidia_detected}, AMD Available: {amd_detected}")


    def _configure_quantum_resources(self) -> None:
        """Configure quantum resources based on detected hardware."""
        if not QUANTUM_AVAILABLE:
            self.logger.info("Quantum libraries (PennyLane) not available.")
            self.quantum_available = False
            return

        try:
            # Determine preferred backend based on *detected* hardware
            preferred_backend = "lightning.qubit" # Default CPU backend

            if self.devices['nvidia_gpu']['available']:
                 # For GTX 1080 (CUDA 6.1), lightning.gpu is not supported, use lightning.kokkos
                 try:
                      # Try lightning.kokkos first for GTX 1080 compatibility
                      qml.device('lightning.kokkos', wires=1)
                      preferred_backend = 'lightning.kokkos'
                      self.logger.info("NVIDIA GPU detected, using 'lightning.kokkos' for CUDA 6.1 compatibility.")
                 except Exception as e_kokkos:
                      self.logger.warning(f"lightning.kokkos failed ({e_kokkos}), trying lightning.gpu.")
                      try:
                           qml.device("lightning.gpu", wires=1)
                           preferred_backend = "lightning.gpu"
                           self.logger.info("Using lightning.gpu")
                      except Exception as e_lgpu:
                           self.logger.warning(f"Both lightning.kokkos and lightning.gpu failed. Using CPU fallback.")
                           preferred_backend = 'lightning.qubit'


            elif self.devices['amd_gpu']['available']:
                 # Check if lightning.kokkos is available/loadable
                 try:
                      qml.device("lightning.kokkos", wires=1)
                      preferred_backend = "lightning.kokkos"
                      self.logger.info("AMD GPU detected, preferring 'lightning.kokkos'.")
                 except Exception as e_lk:
                      self.logger.warning(f"AMD GPU detected, but 'lightning.kokkos' failed to load ({e_lk}). Falling back to 'lightning.qubit'.")
                      preferred_backend = 'lightning.qubit'


            # Check if the preferred backend actually exists in PennyLane's list
            # Use the corrected way to access device names
            available_device_names = []
            # Check available devices in a way compatible with PennyLane 0.41.0
            try:
                # Try to identify available devices by attempting to create them
                available_device_names = []
                
                # Check for common lightning devices
                for device_name in ["lightning.kokkos", "lightning.gpu", "lightning.qubit", "default.qubit"]:
                    try:
                        # Try to create device with minimal settings
                        qml.device(device_name, wires=1)
                        available_device_names.append(device_name)
                        self.logger.info(f"Found available device: {device_name}")
                    except Exception:
                        # Device not available
                        pass
            except Exception:
                # Fallback check
                common_devices_to_check = ['lightning.gpu', 'lightning.kokkos', 'lightning.qubit']
                for dev_name in common_devices_to_check:
                    try:
                        qml.device(dev_name, wires=1)
                        available_device_names.append(dev_name)
                    except Exception:
                        pass

            if preferred_backend not in available_device_names:
                self.logger.warning(f"Preferred backend '{preferred_backend}' not found in available devices: {available_device_names}. Falling back.")
                # Fallback logic: kokkos -> qubit -> default
                if 'lightning.kokkos' in available_device_names:
                    preferred_backend = 'lightning.kokkos'
                elif 'lightning.qubit' in available_device_names:
                    preferred_backend = 'lightning.qubit'
                else:
                    self.logger.error("No suitable default PennyLane simulators found!")
                    self.devices['quantum']['available'] = False
                    self.quantum_available = False
                    return  # Cannot proceed

            # Set the default device
            self.devices['quantum']['default_device'] = preferred_backend
            self.devices['quantum']['available'] = True
            self.quantum_available = True
            self.devices['quantum']['devices'] = available_device_names # Store all found devices
            self.device_name = preferred_backend  # Set device_name for compatibility
            self.logger.info(f"Selected default quantum device: {self.devices['quantum']['default_device']}")

            # Set environment variables for GPU backends if needed
            if preferred_backend in ['lightning.gpu', 'lightning.kokkos']:
                if self.devices['nvidia_gpu']['available']:
                    os.environ['PENNYLANE_LIGHTNING_GPU_ARCH'] = 'nvidia' # Or specific arch
                    # Set OpenMP environment for better Kokkos performance
                    if 'OMP_PROC_BIND' not in os.environ:
                        os.environ['OMP_PROC_BIND'] = 'spread'
                    if 'OMP_PLACES' not in os.environ:
                        os.environ['OMP_PLACES'] = 'threads'
                elif self.devices['amd_gpu']['available']:
                    os.environ['PENNYLANE_LIGHTNING_GPU_ARCH'] = 'amd'


        except Exception as e:
            self.logger.error(f"Quantum resource configuration error: {str(e)}", exc_info=True)
            self.devices['quantum']['available'] = False
            self.quantum_available = False
            self.devices['quantum']['devices'] = []
            
    def _initialize_resource_tracking(self) -> None:
        """Initialize resource tracking for memory and execution time."""
        # Initialize memory tracking
        self.memory_usage = {
            'cpu': 0,
            'nvidia_gpu': [0] * self.devices['nvidia_gpu']['count'],
            'amd_gpu': [0] * self.devices['amd_gpu']['count'],
            'quantum': 0
        }
        
        # Initialize execution time tracking
        self.execution_times = {
            'cpu': [],
            'nvidia_gpu': [],
            'amd_gpu': [],
            'quantum': []
        }
        
        # Initialize circuit cache
        self.circuit_cache = {}
        self.cache_locks = {}

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information including hardware details."""
        model_info = {
            "model_type": "QuantumAnnealingRegression",
            "hardware_device": self.device_name if hasattr(self, 'device_name') else "none",
            "quantum_available": self.quantum_available,
            "gpu_available": self.gpu_available,
            "window_size": self.window_size,
            "forecast_horizon": self.forecast_horizon,
            "n_features": self.n_features,
            "annealing_steps": self.annealing_steps,
            "kernel_size": self.kernel_size,
            "max_qubits": self.max_qubits,
            "learning_rate": self.learning_rate,
            "current_regime": self.current_regime,
            "weights_shape": None if self.weights is None else self.weights.shape,
            "recent_error": np.mean(np.abs(self.prediction_errors[-10:])) if len(self.prediction_errors) >= 10 else None,
            "training_history_entries": len(self.training_history),
            "quantum_backend": self.device_name if hasattr(self, 'device_name') else "none"
        }
        
        # Add simple accelerator info without trying to access detailed properties
        if self.hw_accelerator is not None:
            try:
                accel_type = "unknown"
                if hasattr(self.hw_accelerator, 'get_accelerator_type'):
                    accel_type = str(self.hw_accelerator.get_accelerator_type())
                model_info["accelerator_type"] = accel_type
            except Exception:
                pass
        
        return model_info
    
    def get_optimal_device(self, required_memory: float = 0, quantum_required: bool = False,
                          qubits_required: int = 0, precision: str = 'float32') -> Dict:
        """
        Get the optimal device for a computation based on requirements.
        
        Args:
            required_memory (float): Required memory in bytes
            quantum_required (bool): Whether quantum capabilities are required
            qubits_required (int): Number of qubits required for quantum computation
            precision (str): Required precision ('float32' or 'float64')
            
        Returns:
            Dict: Device configuration
        """
        if not self._is_initialized:
            self.initialize_hardware()
            
        # If quantum computation is required
        if quantum_required:
            if not self.quantum_available:
                # Fall back to classical device if quantum is required but not available
                self.logger.warning("Quantum computation required but not available, falling back to classical")
                quantum_required = False
            else:
                return self._get_quantum_device(qubits_required)
        
        # For classical computation, check if GPU is available and has enough memory
        if self.gpu_available and not self.force_cpu:
            # Check NVIDIA GPUs first
            if self.devices['nvidia_gpu']['available']:
                for i, device in enumerate(self.devices['nvidia_gpu']['devices']):
                    free_memory = device['memory'] - self.memory_usage['nvidia_gpu'][i]
                    if free_memory >= required_memory:
                        return {
                            'type': 'nvidia_gpu',
                            'index': i,
                            'name': device['name'],
                            'memory': device['memory'],
                            'free_memory': free_memory
                        }
            
            # Check AMD GPUs if no suitable NVIDIA GPU found
            if self.devices['amd_gpu']['available']:
                for i, device in enumerate(self.devices['amd_gpu']['devices']):
                    free_memory = device['memory'] - self.memory_usage['amd_gpu'][i]
                    if free_memory >= required_memory:
                        return {
                            'type': 'amd_gpu',
                            'index': i,
                            'name': device['name'],
                            'memory': device['memory'],
                            'free_memory': free_memory
                        }
        
        # Fall back to CPU
        return {
            'type': 'cpu',
            'cores': self.devices['cpu']['count'],
            'model': self.devices['cpu'].get('model', 'Unknown')
        }
    
# --- Replace _get_quantum_device in hardware_manager.py ---

    def _get_quantum_device(self, qubits_required: int) -> Dict:
        """
        Get the optimal quantum device configuration.
        FIXED: Handles missing 'platform' key for AMD detection methods.
        """
        if not self.quantum_available:
            raise ValueError("Quantum computation not available")

        if qubits_required <= 0: # Ensure positive qubits
            qubits_required = self.default_num_wires # Use attribute from init

        # --- Select Device Name Based on Availability ---
        # Determine the actual device NAME we want to use first
        selected_device_name = self.devices['quantum'].get('default_device', 'lightning.qubit')
        # Ensure the selected default is actually available in the filtered list
        if selected_device_name not in self.devices['quantum'].get('devices', []):
             # If default isn't available, pick the first from the filtered list, or fallback
             available_sims = self.devices['quantum'].get('devices', [])
             if available_sims:
                  selected_device_name = available_sims[0]
                  self.logger.warning(f"Default device '{self.devices['quantum'].get('default_device')}' not in available list {available_sims}. Using '{selected_device_name}'.")
             else:
                  # This case should ideally be caught by _configure_quantum_resources, but as a safeguard:
                  self.logger.error("No quantum simulators seem available despite quantum_available=True. Check config.")
                  raise ValueError("No quantum simulators available.")


        # --- Set Environment Variables based on SELECTED backend and DETECTED hardware ---
        # Only set if using a GPU-accelerated backend
        if selected_device_name in ['lightning.gpu', 'lightning.kokkos']:
            if self.devices['nvidia_gpu']['available']:
                # Set CUDA visible devices if needed (e.g., select specific GPU)
                # os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Example: Use primary GPU
                self.logger.debug("Configuring environment for NVIDIA GPU backend.")
            elif self.devices['amd_gpu']['available']:
                # Set ROCm/HIP environment variables
                # os.environ['HIP_VISIBLE_DEVICES'] = '0' # Example: Use primary GPU
                # We don't need the specific platform info from the dictionary here.
                # The fact that amd_gpu is available is enough.
                # If specific HSA overrides ARE needed based on GFX version,
                # that would require parsing 'rocm-smi -i --showproductname' or similar,
                # which adds complexity. Let's omit it for now.
                # os.environ['HSA_OVERRIDE_GFX_VERSION'] = ... # Omitted for simplicity
                self.logger.debug("Configuring environment for AMD GPU backend.")
            else:
                self.logger.warning(f"GPU backend '{selected_device_name}' selected, but no corresponding GPU detected in hardware state.")


        # Configure shots for simulation
        shots_config = self.quantum_shots # Use value set during manager init

        # Return the final configuration dictionary
        device_config_result = {
            'type': 'quantum',
            'device': selected_device_name, # Use the determined name
            'wires': qubits_required,
            'shots': shots_config
        }
        self.logger.debug(f"Returning quantum device config: {device_config_result}")
        return device_config_result
    
    def get_quantum_circuit(self, circuit_id: str, num_wires: int) -> qml.device:
        """
        Get a quantum circuit, from cache if available or create new.
        
        Args:
            circuit_id (str): Unique identifier for the circuit
            num_wires (int): Number of qubits/wires
            
        Returns:
            qml.device: Quantum device
        """
        if not QUANTUM_AVAILABLE:
            raise RuntimeError("Quantum libraries not available")
            
        # Check if circuit is in cache
        cache_key = f"{circuit_id}_{num_wires}"
        
        if cache_key in self.circuit_cache:
            self.logger.debug(f"Using cached quantum circuit: {cache_key}")
            return self.circuit_cache[cache_key]
            
        # Get optimal device
        device_config = self._get_quantum_device(num_wires)
        
        # Create new quantum device
        try:
            device = qml.device(
                device_config['device'], 
                wires=num_wires,
                shots=device_config['shots']
            )
            
            # Add to cache
            self._add_to_circuit_cache(cache_key, device)
            
            return device
            
        except Exception as e:
            self.logger.error(f"Error creating quantum circuit: {str(e)}")
            raise
    
    def _add_to_circuit_cache(self, key: str, device) -> None:
        """
        Add a quantum device to the circuit cache with LRU eviction.
        
        Args:
            key (str): Cache key
            device: Quantum device
        """
        # Apply LRU eviction policy if cache is full
        if len(self.circuit_cache) >= self.max_cache_size:
            # Simple LRU: remove oldest entry (first item in dictionary)
            self.circuit_cache.pop(next(iter(self.circuit_cache)))
            
        # Add to cache
        self.circuit_cache[key] = device
    
    def track_execution_time(self, device_type: str, execution_time: float) -> None:
        """
        Track execution time for performance monitoring.
        
        Args:
            device_type (str): Device type ('cpu', 'nvidia_gpu', 'amd_gpu', 'quantum')
            execution_time (float): Execution time in milliseconds
        """
        if device_type in self.execution_times:
            self.execution_times[device_type].append(execution_time)
            
            # Keep only the last 100 execution times
            if len(self.execution_times[device_type]) > 100:
                self.execution_times[device_type] = self.execution_times[device_type][-100:]
    
    def get_average_execution_time(self, device_type: str) -> Optional[float]:
        """
        Get average execution time for a device type.
        
        Args:
            device_type (str): Device type
            
        Returns:
            Optional[float]: Average execution time in milliseconds, or None if no data
        """
        if device_type in self.execution_times and self.execution_times[device_type]:
            return sum(self.execution_times[device_type]) / len(self.execution_times[device_type])
        return None
    
    def cleanup_resources(self) -> None:
        """Release and clean up hardware resources."""
        self.logger.info("Cleaning up hardware resources")
        
        # Clear cache
        self.circuit_cache.clear()
        
        # Reset memory usage tracking
        self._initialize_resource_tracking()
        
        # Additional cleanup for specific devices
        if self.devices['nvidia_gpu']['available'] and TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                self.logger.info("CUDA cache cleared")
            except Exception as e:
                self.logger.warning(f"Error clearing CUDA cache: {str(e)}")
        
        if self.devices['amd_gpu']['available'] and OPENCL_AVAILABLE:
            try:
                for context in self.active_contexts.values():
                    context.release()
                self.active_contexts.clear()
                self.logger.info("OpenCL contexts released")
            except Exception as e:
                self.logger.warning(f"Error releasing OpenCL contexts: {str(e)}")