"""
Process Executor for running arbitrary processes in E2B sandboxes
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from .sandbox_manager import SandboxManager
from .models import (
    ProcessConfig,
    ProcessResult,
    SandboxConfig
)

logger = logging.getLogger(__name__)


class ProcessExecutor:
    """Execute processes in E2B sandboxes with monitoring"""
    
    def __init__(self, sandbox_manager: SandboxManager):
        """Initialize process executor"""
        self.sandbox_manager = sandbox_manager
        self.active_processes: Dict[str, Dict[str, Any]] = {}
        
    async def execute_process(self, config: ProcessConfig, 
                             sandbox_id: Optional[str] = None) -> ProcessResult:
        """Execute a process in a sandbox"""
        started_at = datetime.now()
        
        try:
            # Create or use existing sandbox
            if not sandbox_id:
                sandbox_config = SandboxConfig(
                    name=f"process_{started_at.strftime('%Y%m%d_%H%M%S')}",
                    timeout=config.timeout or 300,
                    envs=config.env_vars
                )
                sandbox_id = self.sandbox_manager.create_sandbox(sandbox_config)
            
            # Get sandbox
            sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
            if not sandbox:
                raise ValueError(f"Sandbox {sandbox_id} not found")
            
            # Change to working directory
            if config.working_dir != "/tmp":
                sandbox.commands.run(f"mkdir -p {config.working_dir}")
                
            # Set environment variables
            for key, value in config.env_vars.items():
                sandbox.commands.run(f"export {key}='{value}'")
            
            # Build command with arguments
            full_command = config.command
            if config.args:
                args_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in config.args)
                full_command = f"{config.command} {args_str}"
            
            # Execute command
            if config.working_dir:
                full_command = f"cd {config.working_dir} && {full_command}"
            
            # Track process
            process_id = f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            self.active_processes[process_id] = {
                "sandbox_id": sandbox_id,
                "command": full_command,
                "started_at": started_at,
                "config": config
            }
            
            # Execute with timeout
            result = self.sandbox_manager.execute_command(
                sandbox_id, 
                full_command,
                timeout=config.timeout or 60
            )
            
            # Remove from active processes
            if process_id in self.active_processes:
                del self.active_processes[process_id]
            
            # Add process ID to result
            result.process_id = process_id
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute process: {e}")
            
            return ProcessResult(
                sandbox_id=sandbox_id or "error",
                command=config.command,
                started_at=started_at,
                completed_at=datetime.now(),
                error=str(e)
            )
    
    async def execute_batch(self, configs: List[ProcessConfig],
                          parallel: bool = True) -> List[ProcessResult]:
        """Execute multiple processes"""
        if parallel:
            # Run processes in parallel
            tasks = [self.execute_process(config) for config in configs]
            results = await asyncio.gather(*tasks)
        else:
            # Run processes sequentially
            results = []
            for config in configs:
                result = await self.execute_process(config)
                results.append(result)
        
        return results
    
    async def execute_pipeline(self, configs: List[ProcessConfig],
                              sandbox_id: Optional[str] = None) -> List[ProcessResult]:
        """Execute a pipeline of processes in the same sandbox"""
        results = []
        
        # Create sandbox if not provided
        if not sandbox_id:
            sandbox_config = SandboxConfig(
                name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timeout=1800  # 30 minutes for pipeline
            )
            sandbox_id = self.sandbox_manager.create_sandbox(sandbox_config)
        
        # Execute processes sequentially in same sandbox
        for config in configs:
            result = await self.execute_process(config, sandbox_id)
            results.append(result)
            
            # Stop pipeline if process failed
            if result.exit_code and result.exit_code != 0:
                logger.warning(f"Pipeline stopped due to process failure: {config.command}")
                break
        
        return results
    
    async def run_background_process(self, config: ProcessConfig,
                                    sandbox_id: Optional[str] = None) -> str:
        """Start a background process"""
        started_at = datetime.now()
        
        # Create or use existing sandbox
        if not sandbox_id:
            sandbox_config = SandboxConfig(
                name=f"background_{started_at.strftime('%Y%m%d_%H%M%S')}",
                timeout=3600  # 1 hour for background processes
            )
            sandbox_id = self.sandbox_manager.create_sandbox(sandbox_config)
        
        # Get sandbox
        sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        # Build background command with proper escaping
        import shlex
        
        # Build command with properly escaped arguments
        if config.args:
            # Properly escape each argument
            escaped_args = [shlex.quote(arg) for arg in config.args]
            full_command = f"{config.command} {' '.join(escaped_args)}"
        else:
            full_command = config.command
        
        # Create a script file to avoid shell escaping issues
        script_content = f"""#!/bin/bash
{full_command}
"""
        script_path = f"/tmp/bg_script_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.sh"
        sandbox.files.write(script_path, script_content)
        sandbox.commands.run(f"chmod +x {script_path}")
        
        # Run the script in background with nohup
        background_command = f"nohup {script_path} > /tmp/process.log 2>&1 &"
        
        if config.working_dir:
            background_command = f"cd {config.working_dir} && {background_command}"
        
        # Execute background command
        sandbox.commands.run(background_command)
        
        # Get process ID
        pid_result = sandbox.commands.run("echo $!")
        process_id = pid_result.stdout.strip() if hasattr(pid_result, 'stdout') else "unknown"
        
        # Track background process (store original command for display)
        original_command = config.command
        if config.args:
            original_command = f"{config.command} {' '.join(config.args)}"
            
        self.active_processes[process_id] = {
            "sandbox_id": sandbox_id,
            "command": original_command,
            "started_at": started_at,
            "config": config,
            "background": True,
            "script_path": script_path
        }
        
        logger.info(f"Started background process {process_id} in sandbox {sandbox_id}")
        
        return process_id
    
    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a process"""
        if process_id not in self.active_processes:
            return None
        
        process_info = self.active_processes[process_id]
        sandbox_id = process_info["sandbox_id"]
        
        sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
        if not sandbox:
            return {
                "process_id": process_id,
                "status": "sandbox_terminated",
                **process_info
            }
        
        # Check if process is still running (for background processes)
        if process_info.get("background"):
            check_command = f"ps -p {process_id} > /dev/null 2>&1 && echo 'running' || echo 'stopped'"
            result = sandbox.commands.run(check_command)
            status = result.stdout.strip() if hasattr(result, 'stdout') else "unknown"
            
            return {
                "process_id": process_id,
                "status": status,
                **process_info
            }
        
        return {
            "process_id": process_id,
            "status": "completed",
            **process_info
        }
    
    def kill_process(self, process_id: str) -> bool:
        """Kill a running process"""
        if process_id not in self.active_processes:
            logger.warning(f"Process {process_id} not found")
            return False
        
        process_info = self.active_processes[process_id]
        sandbox_id = process_info["sandbox_id"]
        
        sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
        if not sandbox:
            logger.warning(f"Sandbox {sandbox_id} not found")
            return False
        
        try:
            # Kill the process
            sandbox.commands.run(f"kill -9 {process_id}")
            
            # Remove from active processes
            del self.active_processes[process_id]
            
            logger.info(f"Killed process {process_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to kill process {process_id}: {e}")
            return False
    
    def list_active_processes(self) -> List[Dict[str, Any]]:
        """List all active processes"""
        processes = []
        
        for process_id, info in self.active_processes.items():
            status = self.get_process_status(process_id)
            if status:
                processes.append(status)
        
        return processes
    
    async def execute_script(self, script_content: str, 
                           language: str = "python",
                           sandbox_id: Optional[str] = None) -> ProcessResult:
        """Execute a script in a sandbox"""
        # Create temporary script file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        if language == "python":
            script_name = f"script_{timestamp}.py"
            interpreter = "python"
        elif language == "javascript":
            script_name = f"script_{timestamp}.js"
            interpreter = "node"
        elif language == "bash":
            script_name = f"script_{timestamp}.sh"
            interpreter = "bash"
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        # Create or use existing sandbox
        if not sandbox_id:
            sandbox_config = SandboxConfig(
                name=f"script_{timestamp}",
                timeout=300
            )
            sandbox_id = self.sandbox_manager.create_sandbox(sandbox_config)
        
        # Get sandbox
        sandbox = self.sandbox_manager.get_sandbox(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        # Write script to sandbox
        script_path = f"/workspace/{script_name}"
        sandbox.files.write(script_path, script_content)
        
        # Execute script
        config = ProcessConfig(
            command=interpreter,
            args=[script_path],
            working_dir="/tmp"
        )
        
        result = await self.execute_process(config, sandbox_id)
        
        return result