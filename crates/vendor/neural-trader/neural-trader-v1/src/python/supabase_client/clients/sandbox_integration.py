"""
Sandbox Integration Client
========================

Client for managing E2B sandbox integration with Supabase backend.
Handles sandbox lifecycle, agent deployment, and execution tracking.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..client import AsyncSupabaseClient, SupabaseError
from ..models.database_models import *

logger = logging.getLogger(__name__)

class SandboxStatus(str, Enum):
    """Sandbox status enumeration."""
    CREATING = "creating"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DESTROYED = "destroyed"

class AgentType(str, Enum):
    """Agent types for sandbox deployment."""
    NEURAL_SENTIMENT = "neural_sentiment"
    NEURAL_MOMENTUM = "neural_momentum"
    NEURAL_ARBITRAGE = "neural_arbitrage"
    NEURAL_PAIRS = "neural_pairs"
    NEURAL_RISK = "neural_risk"
    NEURAL_PORTFOLIO = "neural_portfolio"
    TRADING_BOT = "trading_bot"
    CUSTOM = "custom"

@dataclass
class CreateSandboxRequest:
    """Request to create a new sandbox."""
    name: str
    template: str = "trading-base"
    cpu_count: int = 1
    memory_mb: int = 512
    timeout_minutes: int = 60
    environment_vars: Optional[Dict[str, str]] = None
    tags: Optional[Dict[str, str]] = None

@dataclass
class DeployAgentRequest:
    """Request to deploy an agent to sandbox."""
    agent_type: AgentType
    agent_config: Dict[str, Any]
    symbols: List[str]
    strategy_params: Optional[Dict[str, Any]] = None
    auto_start: bool = True

@dataclass
class ExecuteCommandRequest:
    """Request to execute command in sandbox."""
    command: str
    args: Optional[List[str]] = None
    timeout_seconds: int = 30
    capture_output: bool = True

class SandboxIntegrationClient:
    """Client for E2B sandbox integration operations."""
    
    def __init__(self, supabase_client: AsyncSupabaseClient):
        """
        Initialize sandbox integration client.
        
        Args:
            supabase_client: Async Supabase client instance
        """
        self.supabase = supabase_client
        
    async def create_sandbox(
        self, 
        user_id: UUID, 
        request: CreateSandboxRequest
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Create a new E2B sandbox.
        
        Args:
            user_id: User ID creating the sandbox
            request: Sandbox creation request
            
        Returns:
            Tuple of (sandbox_data, error_message)
        """
        try:
            # Validate user permissions
            user_profile = await self.supabase.select(
                "profiles",
                filter_dict={"id": str(user_id)}
            )
            
            if not user_profile:
                return None, "User not found"
                
            # Check sandbox limits
            active_sandboxes = await self.supabase.count(
                "sandboxes",
                filter_dict={
                    "user_id": str(user_id),
                    "status": SandboxStatus.ACTIVE.value
                }
            )
            
            if active_sandboxes >= 5:  # Limit per user
                return None, "Maximum number of active sandboxes reached"
                
            # Create sandbox record
            sandbox_data = {
                "id": str(uuid4()),
                "user_id": str(user_id),
                "name": request.name,
                "template": request.template,
                "cpu_count": request.cpu_count,
                "memory_mb": request.memory_mb,
                "timeout_minutes": request.timeout_minutes,
                "environment_vars": request.environment_vars or {},
                "tags": request.tags or {},
                "status": SandboxStatus.CREATING.value,
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat()
            }
            
            result = await self.supabase.insert("sandboxes", sandbox_data)
            
            # Log sandbox creation
            await self._log_sandbox_event(
                sandbox_data["id"],
                "sandbox_created",
                {"template": request.template, "resources": {"cpu": request.cpu_count, "memory": request.memory_mb}}
            )
            
            logger.info(f"Sandbox created: {sandbox_data['id']}")
            return result[0], None
            
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            return None, f"Sandbox creation failed: {str(e)}"
    
    async def get_sandbox_status(
        self, 
        sandbox_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Get sandbox status and details.
        
        Args:
            sandbox_id: Sandbox ID to query
            
        Returns:
            Tuple of (sandbox_data, error_message)
        """
        try:
            sandbox = await self.supabase.select(
                "sandboxes",
                filter_dict={"id": sandbox_id}
            )
            
            if not sandbox:
                return None, "Sandbox not found"
                
            sandbox_data = sandbox[0]
            
            # Get recent events
            events = await self.supabase.select(
                "sandbox_events",
                filter_dict={"sandbox_id": sandbox_id},
                order_by="-created_at",
                limit=10
            )
            
            # Get deployed agents
            agents = await self.supabase.select(
                "sandbox_agents",
                filter_dict={"sandbox_id": sandbox_id, "status": "active"}
            )
            
            sandbox_data.update({
                "recent_events": events,
                "active_agents": agents
            })
            
            return sandbox_data, None
            
        except Exception as e:
            logger.error(f"Failed to get sandbox status: {e}")
            return None, f"Failed to get sandbox status: {str(e)}"
    
    async def deploy_agent(
        self, 
        sandbox_id: str, 
        request: DeployAgentRequest
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Deploy an agent to a sandbox.
        
        Args:
            sandbox_id: Target sandbox ID
            request: Agent deployment request
            
        Returns:
            Tuple of (agent_data, error_message)
        """
        try:
            # Verify sandbox exists and is active
            sandbox = await self.supabase.select(
                "sandboxes",
                filter_dict={"id": sandbox_id}
            )
            
            if not sandbox:
                return None, "Sandbox not found"
                
            if sandbox[0]["status"] != SandboxStatus.ACTIVE.value:
                return None, "Sandbox is not active"
                
            # Check agent limits per sandbox
            existing_agents = await self.supabase.count(
                "sandbox_agents",
                filter_dict={"sandbox_id": sandbox_id, "status": "active"}
            )
            
            if existing_agents >= 3:  # Limit per sandbox
                return None, "Maximum number of agents per sandbox reached"
                
            # Create agent record
            agent_data = {
                "id": str(uuid4()),
                "sandbox_id": sandbox_id,
                "agent_type": request.agent_type.value,
                "agent_config": request.agent_config,
                "symbols": request.symbols,
                "strategy_params": request.strategy_params or {},
                "status": "deploying",
                "deployed_at": datetime.utcnow().isoformat(),
                "last_heartbeat": datetime.utcnow().isoformat()
            }
            
            result = await self.supabase.insert("sandbox_agents", agent_data)
            
            # Log agent deployment
            await self._log_sandbox_event(
                sandbox_id,
                "agent_deployed",
                {
                    "agent_id": agent_data["id"],
                    "agent_type": request.agent_type.value,
                    "symbols": request.symbols
                }
            )
            
            # Update sandbox last accessed
            await self.supabase.update(
                "sandboxes",
                {"last_accessed": datetime.utcnow().isoformat()},
                {"id": sandbox_id}
            )
            
            logger.info(f"Agent deployed to sandbox {sandbox_id}: {agent_data['id']}")
            return result[0], None
            
        except Exception as e:
            logger.error(f"Failed to deploy agent: {e}")
            return None, f"Agent deployment failed: {str(e)}"
    
    async def execute_command(
        self, 
        sandbox_id: str, 
        request: ExecuteCommandRequest
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Execute a command in the sandbox.
        
        Args:
            sandbox_id: Target sandbox ID
            request: Command execution request
            
        Returns:
            Tuple of (execution_result, error_message)
        """
        try:
            # Verify sandbox exists and is active
            sandbox = await self.supabase.select(
                "sandboxes",
                filter_dict={"id": sandbox_id}
            )
            
            if not sandbox:
                return None, "Sandbox not found"
                
            if sandbox[0]["status"] != SandboxStatus.ACTIVE.value:
                return None, "Sandbox is not active"
                
            # Create execution record
            execution_id = str(uuid4())
            execution_data = {
                "id": execution_id,
                "sandbox_id": sandbox_id,
                "command": request.command,
                "args": request.args or [],
                "timeout_seconds": request.timeout_seconds,
                "status": "executing",
                "started_at": datetime.utcnow().isoformat()
            }
            
            await self.supabase.insert("sandbox_executions", execution_data)
            
            # Log command execution
            await self._log_sandbox_event(
                sandbox_id,
                "command_executed",
                {
                    "execution_id": execution_id,
                    "command": request.command,
                    "timeout": request.timeout_seconds
                }
            )
            
            # Update sandbox last accessed
            await self.supabase.update(
                "sandboxes",
                {"last_accessed": datetime.utcnow().isoformat()},
                {"id": sandbox_id}
            )
            
            # Simulate command execution (in real implementation, this would call E2B API)
            await asyncio.sleep(0.1)  # Simulate execution time
            
            # Update execution result
            result_data = {
                "status": "completed",
                "exit_code": 0,
                "stdout": f"Command '{request.command}' executed successfully",
                "stderr": "",
                "completed_at": datetime.utcnow().isoformat()
            }
            
            await self.supabase.update(
                "sandbox_executions",
                result_data,
                {"id": execution_id}
            )
            
            execution_data.update(result_data)
            
            logger.info(f"Command executed in sandbox {sandbox_id}: {request.command}")
            return execution_data, None
            
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            return None, f"Command execution failed: {str(e)}"
    
    async def stop_sandbox(
        self, 
        sandbox_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Stop a running sandbox.
        
        Args:
            sandbox_id: Sandbox ID to stop
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Verify sandbox exists
            sandbox = await self.supabase.select(
                "sandboxes",
                filter_dict={"id": sandbox_id}
            )
            
            if not sandbox:
                return False, "Sandbox not found"
                
            if sandbox[0]["status"] in [SandboxStatus.STOPPED.value, SandboxStatus.DESTROYED.value]:
                return True, None  # Already stopped
                
            # Stop all active agents
            await self.supabase.update(
                "sandbox_agents",
                {"status": "stopped", "stopped_at": datetime.utcnow().isoformat()},
                {"sandbox_id": sandbox_id, "status": "active"}
            )
            
            # Update sandbox status
            await self.supabase.update(
                "sandboxes",
                {
                    "status": SandboxStatus.STOPPED.value,
                    "stopped_at": datetime.utcnow().isoformat()
                },
                {"id": sandbox_id}
            )
            
            # Log sandbox stop
            await self._log_sandbox_event(
                sandbox_id,
                "sandbox_stopped",
                {"stopped_by": "user_request"}
            )
            
            logger.info(f"Sandbox stopped: {sandbox_id}")
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to stop sandbox: {e}")
            return False, f"Failed to stop sandbox: {str(e)}"
    
    async def destroy_sandbox(
        self, 
        sandbox_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Destroy a sandbox and clean up resources.
        
        Args:
            sandbox_id: Sandbox ID to destroy
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Stop sandbox first if needed
            await self.stop_sandbox(sandbox_id)
            
            # Update sandbox status
            await self.supabase.update(
                "sandboxes",
                {
                    "status": SandboxStatus.DESTROYED.value,
                    "destroyed_at": datetime.utcnow().isoformat()
                },
                {"id": sandbox_id}
            )
            
            # Log sandbox destruction
            await self._log_sandbox_event(
                sandbox_id,
                "sandbox_destroyed",
                {"destroyed_by": "user_request"}
            )
            
            logger.info(f"Sandbox destroyed: {sandbox_id}")
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to destroy sandbox: {e}")
            return False, f"Failed to destroy sandbox: {str(e)}"
    
    async def list_user_sandboxes(
        self, 
        user_id: UUID,
        status_filter: Optional[str] = None,
        limit: int = 50
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        List sandboxes for a user.
        
        Args:
            user_id: User ID to filter by
            status_filter: Optional status filter
            limit: Maximum number of results
            
        Returns:
            Tuple of (sandboxes_list, error_message)
        """
        try:
            filter_dict = {"user_id": str(user_id)}
            if status_filter:
                filter_dict["status"] = status_filter
                
            sandboxes = await self.supabase.select(
                "sandboxes",
                filter_dict=filter_dict,
                order_by="-created_at",
                limit=limit
            )
            
            # Add agent count for each sandbox
            for sandbox in sandboxes:
                agent_count = await self.supabase.count(
                    "sandbox_agents",
                    filter_dict={"sandbox_id": sandbox["id"], "status": "active"}
                )
                sandbox["active_agent_count"] = agent_count
            
            return sandboxes, None
            
        except Exception as e:
            logger.error(f"Failed to list sandboxes: {e}")
            return [], f"Failed to list sandboxes: {str(e)}"
    
    async def get_agent_logs(
        self, 
        agent_id: str,
        lines: int = 100
    ) -> Tuple[List[str], Optional[str]]:
        """
        Get logs for a sandbox agent.
        
        Args:
            agent_id: Agent ID to get logs for
            lines: Number of log lines to retrieve
            
        Returns:
            Tuple of (log_lines, error_message)
        """
        try:
            # Get agent events as logs
            events = await self.supabase.select(
                "sandbox_events",
                filter_dict={"metadata": {"agent_id": agent_id}},
                order_by="-created_at",
                limit=lines
            )
            
            log_lines = [
                f"[{event['created_at']}] {event['event_type']}: {event.get('message', '')}"
                for event in events
            ]
            
            return log_lines, None
            
        except Exception as e:
            logger.error(f"Failed to get agent logs: {e}")
            return [], f"Failed to get agent logs: {str(e)}"
    
    async def cleanup_expired_sandboxes(self) -> Tuple[int, Optional[str]]:
        """
        Clean up expired sandboxes.
        
        Returns:
            Tuple of (cleaned_count, error_message)
        """
        try:
            # Find sandboxes that have exceeded their timeout
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            expired_sandboxes = await self.supabase.select(
                "sandboxes",
                filter_dict={"status": SandboxStatus.ACTIVE.value},
                columns="id,created_at,timeout_minutes"
            )
            
            cleaned_count = 0
            for sandbox in expired_sandboxes:
                created_at = datetime.fromisoformat(sandbox["created_at"].replace("Z", "+00:00"))
                timeout_delta = timedelta(minutes=sandbox["timeout_minutes"])
                
                if created_at + timeout_delta < datetime.utcnow():
                    await self.destroy_sandbox(sandbox["id"])
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired sandboxes")
            return cleaned_count, None
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sandboxes: {e}")
            return 0, f"Cleanup failed: {str(e)}"
    
    async def _log_sandbox_event(
        self, 
        sandbox_id: str, 
        event_type: str, 
        metadata: Dict[str, Any]
    ):
        """Log a sandbox event."""
        try:
            event_data = {
                "id": str(uuid4()),
                "sandbox_id": sandbox_id,
                "event_type": event_type,
                "metadata": metadata,
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self.supabase.insert("sandbox_events", event_data)
            
        except Exception as e:
            logger.error(f"Failed to log sandbox event: {e}")