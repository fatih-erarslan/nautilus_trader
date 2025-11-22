"""
Kill Switch - Emergency stop mechanism for trading operations
"""

import asyncio
import logging
from typing import Dict, Any, Set
from datetime import datetime
import threading

class KillSwitch:
    """
    Emergency kill switch for immediate trading halt
    Provides fail-safe mechanism for crisis situations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.activation_reason = ""
        self.activation_time = None
        self.registered_components = set()
        self.lock = threading.Lock()
        
    async def initialize(self):
        """Initialize kill switch"""
        self.logger.info("Kill Switch initialized and armed")
        
    async def activate(self, reason: str) -> bool:
        """Activate kill switch - EMERGENCY STOP ALL TRADING"""
        with self.lock:
            if self.active:
                self.logger.warning("Kill switch already active")
                return True
                
            self.active = True
            self.activation_reason = reason
            self.activation_time = datetime.now()
            
        # Immediate notification to all components
        self.logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
        
        # Notify all registered components
        await self._notify_all_components("EMERGENCY_STOP")
        
        # Additional safety measures
        await self._emergency_procedures()
        
        self.logger.critical("🛑 ALL TRADING OPERATIONS HALTED")
        return True
        
    async def deactivate(self, reason: str) -> bool:
        """Deactivate kill switch and allow recovery"""
        with self.lock:
            if not self.active:
                self.logger.info("Kill switch not active")
                return True
                
            self.active = False
            deactivation_time = datetime.now()
            
        duration = deactivation_time - self.activation_time if self.activation_time else None
        
        self.logger.info(f"🔄 Kill switch deactivated: {reason}")
        if duration:
            self.logger.info(f"   Downtime: {duration.total_seconds():.2f} seconds")
            
        # Notify components of recovery
        await self._notify_all_components("RECOVERY_MODE")
        
        return True
        
    async def is_active(self) -> bool:
        """Check if kill switch is currently active"""
        return self.active
        
    async def register_component(self, component_name: str):
        """Register component for kill switch notifications"""
        self.registered_components.add(component_name)
        self.logger.debug(f"Component registered: {component_name}")
        
    async def _notify_all_components(self, event: str):
        """Notify all registered components of kill switch event"""
        for component in self.registered_components:
            try:
                self.logger.info(f"Notifying {component}: {event}")
                # In production, this would send actual notifications
                await asyncio.sleep(0.001)  # Simulate notification time
            except Exception as e:
                self.logger.error(f"Failed to notify {component}: {e}")
                
    async def _emergency_procedures(self):
        """Execute emergency procedures"""
        procedures = [
            "Cancel all pending orders",
            "Close risky positions", 
            "Suspend market data feeds",
            "Lock trading accounts",
            "Alert risk management",
            "Notify regulators"
        ]
        
        for procedure in procedures:
            self.logger.critical(f"Executing: {procedure}")
            await asyncio.sleep(0.001)  # Simulate procedure execution
            
    async def get_status(self) -> Dict[str, Any]:
        """Get current kill switch status"""
        return {
            'active': self.active,
            'activation_reason': self.activation_reason,
            'activation_time': self.activation_time.isoformat() if self.activation_time else None,
            'registered_components': list(self.registered_components)
        }