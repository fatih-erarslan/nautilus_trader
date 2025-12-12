#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 13:45:25 2025

@author: ashina
"""

# --- bluewolf.py ---

import logging
import threading
import time
import gc
from typing import Dict, Any, Optional, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports for type hints
if TYPE_CHECKING:
    from hardware_manager import HardwareManager # Assuming HardwareManager is importable
    from usp import UniversalSignalProcessor # Assuming USP is importable
    from qar import PanarchyAdaptiveDecisionSystem, QuantumAgenticReasoning # bluewolf might monitor PADS
    from soc_analyzer import SOCAnalyzer
    from risk_manager import ViaNegativaFilter, RiskManager
    from cdfa_analyzer import CognitiveDiversityFusionAnalysis
    from iqad import ImmuneQuantumAnomalyDetector
    from qerc import QuantumReservoirComputing
    
    # Import other component types if needed (e.g., PADS, specific Analyzers)
    # from qar import PanarchyAdaptiveDecisionSystem

logger = logging.getLogger("KutlugBlueWolf") # Dedicated logger

# Default component check intervals and timeouts (in seconds)
DEFAULT_PING_TIMEOUT = 600  # 5 minutes for main bot loop ping
DEFAULT_COMPONENT_TIMEOUT = 600 # 10 minutes for individual components
DEFAULT_MONITOR_INTERVAL = 60   # Check every 1 minute

class KutlugBlueWolf:
    """
    Monitors the health and responsiveness of critical trading bot components
    and attempts recovery actions when timeouts occur. Integrates with
    HardwareManager and USP for checking/recovering hardware and processing layers.
    """

    def __init__(self,
                 hw_manager: Optional['HardwareManager'] = None,
                 usp: Optional['UniversalSignalProcessor'] = None,
                 pads: Optional['PanarchyAdaptiveDecisionSystem'] = None, # Monitor PADS
                 # REMOVE qar argument
                 soc_analyzer: Optional['SOCAnalyzer'] = None,
                 risk_manager: Optional['RiskManager'] = None, # Monitor base RiskManager
                 via_negativa: Optional['ViaNegativaFilter'] = None, # Monitor ViaNegativa separately if needed
                 cdfa_analyzer: Optional['CognitiveDiversityFusionAnalysis'] = None,
                 iqad: Optional['ImmuneQuantumAnomalyDetector'] = None,
                 qerc: Optional['QuantumReservoirComputing'] = None,
                 # Add other components if needed
                 ping_timeout: int = DEFAULT_PING_TIMEOUT,
                 component_timeout: int = DEFAULT_COMPONENT_TIMEOUT,
                 monitor_interval: int = DEFAULT_MONITOR_INTERVAL,
                 max_restarts: int = 3,
                 log_level: int = logging.INFO):
        """
        Initializes the bluewolf.

        Args:
            hw_manager: Instance of HardwareManager.
            usp: Instance of UniversalSignalProcessor.
            ping_timeout: Seconds of inactivity from main loop before recovery attempt.
            component_timeout: Seconds of inactivity for a registered component before recovery.
            monitor_interval: How often (in seconds) the bluewolf checks health.
            max_restarts: Maximum number of full system recovery attempts before stopping.
            log_level: Logging level.
        """
        self.logger = logger
        self.logger.setLevel(log_level)

        # Store references to manageable components
        self.hw_manager = hw_manager
        self.usp = usp
        self.pads = pads
        self.soc_analyzer = soc_analyzer
        self.risk_manager = risk_manager
        self.via_negativa = via_negativa # Store if passed
        self.cdfa_analyzer = cdfa_analyzer
        self.iqad = iqad
        self.qerc = qerc
        # self.pads = pads # Store other components if passed

        # --- Component Registry ---
        # Store components that can be checked and recovered
        self.monitored_components: Dict[str, Any] = {}
        if self.hw_manager: self.monitored_components['hardware_manager'] = self.hw_manager
        if self.usp: self.monitored_components['usp'] = self.usp
        if self.pads: self.monitored_components['pads'] = self.pads # PADS includes QAR recovery implicitly
        if self.soc_analyzer: self.monitored_components['soc_analyzer'] = self.soc_analyzer
        if self.risk_manager: self.monitored_components['risk_manager'] = self.risk_manager
        if self.via_negativa: self.monitored_components['via_negativa'] = self.via_negativa # Register if passed
        if self.cdfa_analyzer: self.monitored_components['cdfa_analyzer'] = self.cdfa_analyzer
        if self.iqad: self.monitored_components['iqad'] = self.iqad
        if self.qerc: self.monitored_components['qerc'] = self.qerc

        
        # Add other components passed in the constructor here
        # if self.pads: self.monitored_components['pads'] = self.pads

        # --- State Tracking ---
        self.component_last_active: Dict[str, float] = {name: time.time() for name in self.monitored_components}
        self.main_loop_last_ping: float = time.time()
        self.ping_timeout = max(60, ping_timeout) # Min 1 minute timeout
        self.component_timeout = max(120, component_timeout) # Min 2 minute timeout
        self.monitor_interval = max(30, monitor_interval) # Check at least every 30s
        self.restart_count = 0
        self.max_restarts = max_restarts
        self.is_running = False
        self._stop_event = threading.Event()
        self._bluewolf_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock() # Lock for accessing shared state

        self.logger.info(f"KutlugBlueWolf initialized. Monitoring: {list(self.monitored_components.keys())}")
        self.logger.info(f"  Main Ping Timeout: {self.ping_timeout}s")
        self.logger.info(f"  Component Timeout: {self.component_timeout}s")
        self.logger.info(f"  Monitor Interval: {self.monitor_interval}s")

    def register_component(self, name: str, component_instance: Any):
        """Registers an additional component for monitoring after initialization."""
        if not name or not component_instance:
            self.logger.warning("Cannot register component: invalid name or instance.")
            return
        with self._lock:
            if name in self.monitored_components:
                self.logger.warning(f"Component '{name}' already registered. Overwriting.")
            self.monitored_components[name] = component_instance
            self.component_last_active[name] = time.time() # Initialize timestamp
            self.logger.info(f"Registered additional component for monitoring: '{name}' (Type: {type(component_instance).__name__})")

    def start(self):
        """Starts the bluewolf monitoring thread."""
        if self.is_running:
            self.logger.debug("bluewolf already running.")
            return

        with self._lock:
            if self.is_running: return # Double check
            self._stop_event.clear()
            self._bluewolf_thread = threading.Thread(
                target=self._monitor_loop, daemon=True, name="KutlugBlueWolfThread"
            )
            self.is_running = True
            self._bluewolf_thread.start()
            self.logger.info("KutlugBlueWolf monitoring started.")

    def stop(self):
        """Stops the bluewolf monitoring thread."""
        if not self.is_running:
            self.logger.debug("bluewolf already stopped.")
            return

        with self._lock:
            if not self.is_running: return # Double check
            self.logger.info("Stopping KutlugBlueWolf monitoring...")
            self._stop_event.set()
            if self._bluewolf_thread and self._bluewolf_thread.is_alive():
                 # Give thread time to finish current check
                 self._bluewolf_thread.join(timeout=self.monitor_interval * 1.5)
                 if self._bluewolf_thread.is_alive():
                      self.logger.warning("bluewolf thread did not stop gracefully.")
            self.is_running = False
            self._bluewolf_thread = None
            self.logger.info("KutlugBlueWolf monitoring stopped.")

    def ping(self, component_name: str = "main_loop"):
        """Call this periodically from the main bot loop or component activity."""
        current_time = time.time()
        with self._lock:
            if component_name == "main_loop":
                self.main_loop_last_ping = current_time
                # self.logger.debug("bluewolf received main loop ping.") # Can be noisy
            elif component_name in self.component_last_active:
                self.component_last_active[component_name] = current_time
                # self.logger.debug(f"bluewolf received ping from component: {component_name}")
            else:
                self.logger.warning(f"bluewolf received ping from unregistered component: {component_name}")

    def _monitor_loop(self):
        """Monitoring loop to check component health."""
        self.logger.info("bluewolf monitor loop entered.")
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                is_unresponsive = False

                # 1. Check main process health (using its ping)
                with self._lock: # Access shared state safely
                    main_last_ping = self.main_loop_last_ping
                    ping_timeout = self.ping_timeout
                if current_time - main_last_ping > ping_timeout:
                    self.logger.warning(f"Main loop seems unresponsive (last ping {current_time - main_last_ping:.0f}s ago > {ping_timeout}s). Triggering full recovery.")
                    self._perform_system_recovery() # Attempt full recovery
                    is_unresponsive = True # Skip component checks after full recovery attempt

                # 2. Check individual component health (if main loop is responsive)
                if not is_unresponsive:
                    components_to_check = {}
                    with self._lock: # Get copy of components to check
                         components_to_check = self.monitored_components.copy()

                    for name, component in components_to_check.items():
                        with self._lock: # Get last active time safely
                             last_active = self.component_last_active.get(name, 0)
                             component_timeout = self.component_timeout
                        
                        if current_time - last_active > component_timeout:
                            self.logger.warning(f"Component '{name}' seems unresponsive (last active {current_time - last_active:.0f}s ago > {component_timeout}s).")
                            self._recover_component(name, component)
                        elif hasattr(component, 'check_health') and callable(component.check_health):
                            # Optional: Call component's own health check
                            try:
                                health_status = component.check_health()
                                if not health_status.get("healthy", True):
                                     self.logger.warning(f"Component '{name}' reported unhealthy status: {health_status.get('reason', 'Unknown')}. Attempting recovery.")
                                     self._recover_component(name, component)
                            except Exception as e_health:
                                 self.logger.error(f"Error calling check_health() for component '{name}': {e_health}")

            except Exception as e:
                self.logger.error(f"Error in bluewolf monitoring loop: {e}", exc_info=True)
                # Avoid busy-looping on error
                time.sleep(self.monitor_interval * 2)

            # Wait for the next check interval
            # Check stop event more frequently than the full interval
            stopped = self._stop_event.wait(timeout=min(5.0, self.monitor_interval))
            if stopped:
                break # Exit loop if stop event is set

        self.logger.info("bluewolf monitor loop finished.")


    def _perform_system_recovery(self):
        """Perform system-wide recovery attempt."""
        with self._lock: # Ensure exclusive access for recovery
            self.restart_count += 1
            self.logger.warning(f"Initiating system recovery attempt #{self.restart_count}/{self.max_restarts}...")

            if self.restart_count > self.max_restarts:
                self.logger.critical(f"Maximum system recovery attempts ({self.max_restarts}) exceeded. bluewolf giving up.")
                self.stop() # Stop the bluewolf itself
                # Potentially trigger a more drastic action (e.g., exit bot process)
                # raise SystemExit("bluewolf failed to recover the system.")
                return

            # Attempt to recover all monitored components
            success_count = 0
            for name, component in self.monitored_components.items():
                if self._recover_component(name, component):
                    success_count += 1

            self.logger.info(f"System recovery attempt finished. Recovered {success_count}/{len(self.monitored_components)} components.")

            # Force garbage collection
            try:
                gc.collect()
                self.logger.info("Garbage collection triggered after recovery attempt.")
            except Exception as e_gc:
                self.logger.error(f"Error during garbage collection: {e_gc}")

            # Update ping time to prevent immediate re-recovery loop
            self.main_loop_last_ping = time.time()
            # Reset component times as well after full recovery
            for name in self.component_last_active:
                self.component_last_active[name] = self.main_loop_last_ping


    def _recover_component(self, name: str, component: Any) -> bool:
        """
        Attempts to recover a specific component by calling its 'recover' method.

        Args:
            name: The registered name of the component.
            component: The component instance.

        Returns:
            True if recovery was attempted successfully (or not needed), False otherwise.
        """
        recovered = False
        self.logger.info(f"Attempting recovery for component: '{name}' (Type: {type(component).__name__})...")
        if hasattr(component, 'recover') and callable(component.recover):
            try:
                component.recover() # Call standardized recovery method
                self.logger.info(f"Recovery method called for component '{name}'.")
                recovered = True
            except Exception as e_recover:
                self.logger.error(f"Error calling recover() for component '{name}': {e_recover}", exc_info=True)
        else:
            self.logger.warning(f"Component '{name}' does not have a 'recover' method. Cannot perform specific recovery.")
            # Attempt generic reset if possible? Risky.
            # Example: If it has a reset method?
            # if hasattr(component, 'reset') and callable(component.reset):
            #     try: component.reset(); recovered = True; logger.info("Called generic reset().")
            #     except: pass

        # Update timestamp even if recovery fails to prevent immediate re-triggering loop
        with self._lock:
            self.component_last_active[name] = time.time()

        return recovered

    def __del__(self):
        """Ensure thread is stopped on garbage collection."""
        try:
            self.stop()
        except Exception:
            pass # Ignore errors during cleanup

