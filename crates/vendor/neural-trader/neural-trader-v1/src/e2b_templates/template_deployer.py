"""
Template Deployer for E2B Templates
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from ..e2b_integration.sandbox_manager import SandboxManager
from ..e2b_integration.models import SandboxConfig, ProcessConfig
from .models import TemplateConfig, TemplateType, TemplateMetadata
from .template_registry import TemplateRegistry

logger = logging.getLogger(__name__)


class TemplateDeployer:
    """Deploy templates to E2B sandboxes"""
    
    def __init__(self, sandbox_manager: SandboxManager = None):
        """Initialize template deployer"""
        self.sandbox_manager = sandbox_manager or SandboxManager()
        self.registry = TemplateRegistry()
        self.deployments: Dict[str, Dict[str, Any]] = {}
    
    async def deploy_template(self, 
                             template_id: str, 
                             config: Dict[str, Any] = None,
                             timeout: int = 300) -> Dict[str, Any]:
        """Deploy a template to E2B sandbox"""
        logger.info(f"Deploying template: {template_id}")
        
        # Get template configuration
        template = self.registry.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        # Create sandbox configuration
        sandbox_config = self._create_sandbox_config(template, config)
        
        try:
            # Create sandbox
            sandbox = await self.sandbox_manager.create_sandbox(sandbox_config)
            
            # Setup environment
            await self._setup_environment(sandbox, template)
            
            # Deploy files
            await self._deploy_files(sandbox, template)
            
            # Run hooks
            await self._run_hooks(sandbox, template, 'pre_start')
            
            # Record deployment
            deployment = {
                "template_id": template_id,
                "sandbox_id": sandbox.sandbox_id,
                "status": "deployed",
                "deployed_at": datetime.now().isoformat(),
                "config": config or {},
                "template": template.dict()
            }
            
            self.deployments[sandbox.sandbox_id] = deployment
            
            logger.info(f"Template deployed successfully: {sandbox.sandbox_id}")
            
            return {
                "status": "success",
                "sandbox_id": sandbox.sandbox_id,
                "template_id": template_id,
                "deployment": deployment
            }
            
        except Exception as e:
            logger.error(f"Failed to deploy template {template_id}: {e}")
            raise
    
    def _create_sandbox_config(self, template: TemplateConfig, config: Dict[str, Any] = None) -> SandboxConfig:
        """Create sandbox configuration from template"""
        config = config or {}
        
        return SandboxConfig(
            name=template.metadata.name,
            template="e2b-official/base",
            metadata={
                "template_id": template.template_type.value,
                "template_name": template.metadata.name,
                "created_at": datetime.now().isoformat(),
                "setup_type": "full" if template.requirements.python_packages or template.requirements.node_packages else "minimal"
            },
            env_vars={
                **template.requirements.env_vars,
                **config.get("env_vars", {})
            },
            timeout=config.get("timeout", 300)
        )
    
    async def _setup_environment(self, sandbox, template: TemplateConfig):
        """Setup sandbox environment based on template requirements"""
        logger.info("Setting up environment...")
        
        # Create directories
        await sandbox.files.write("/tmp/setup.sh", "#!/bin/bash\nset -e\n")
        
        # Install system packages
        if template.requirements.system_packages:
            packages = " ".join(template.requirements.system_packages)
            await sandbox.run_command(f"apt-get update && apt-get install -y {packages}")
        
        # Install Python packages
        if template.requirements.python_packages:
            for package in template.requirements.python_packages:
                try:
                    await sandbox.run_command(f"pip install {package}")
                except Exception as e:
                    logger.warning(f"Failed to install {package}: {e}")
        
        # Install Node.js packages
        if template.requirements.node_packages:
            # Initialize package.json if not exists
            await sandbox.files.write("/tmp/package.json", json.dumps({
                "name": "e2b-template",
                "version": "1.0.0",
                "dependencies": {}
            }))
            
            for package in template.requirements.node_packages:
                try:
                    await sandbox.run_command(f"cd /tmp && npm install {package}")
                except Exception as e:
                    logger.warning(f"Failed to install {package}: {e}")
        
        logger.info("Environment setup completed")
    
    async def _deploy_files(self, sandbox, template: TemplateConfig):
        """Deploy template files to sandbox"""
        logger.info("Deploying files...")
        
        # Deploy main script
        if template.files.main_script:
            await sandbox.files.write("/tmp/main_script.py", template.files.main_script)
            await sandbox.run_command("chmod +x /tmp/main_script.py")
        
        # Deploy modules
        if template.files.modules:
            for name, content in template.files.modules.items():
                filepath = f"/tmp/modules/{name}"
                await sandbox.files.write(filepath, content)
        
        # Deploy configs
        if template.files.configs:
            for name, content in template.files.configs.items():
                filepath = f"/tmp/config/{name}"
                await sandbox.files.write(filepath, content)
        
        # Deploy data files
        if template.files.data:
            for name, content in template.files.data.items():
                filepath = f"/tmp/data/{name}"
                await sandbox.files.write(filepath, content)
        
        # Deploy scripts
        if template.files.scripts:
            for name, content in template.files.scripts.items():
                filepath = f"/tmp/scripts/{name}"
                await sandbox.files.write(filepath, content)
                await sandbox.run_command(f"chmod +x {filepath}")
        
        logger.info("Files deployed successfully")
    
    async def _run_hooks(self, sandbox, template: TemplateConfig, hook_type: str):
        """Run template hooks"""
        if not template.hooks:
            return
        
        hook_script = getattr(template.hooks, hook_type, None)
        if hook_script:
            logger.info(f"Running {hook_type} hook...")
            try:
                await sandbox.run_command(hook_script)
            except Exception as e:
                logger.warning(f"Hook {hook_type} failed: {e}")
    
    async def execute_template(self, 
                              sandbox_id: str, 
                              args: List[str] = None,
                              input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute deployed template"""
        deployment = self.deployments.get(sandbox_id)
        if not deployment:
            raise ValueError(f"No deployment found for sandbox: {sandbox_id}")
        
        logger.info(f"Executing template in sandbox: {sandbox_id}")
        
        # Get sandbox
        sandbox = await self.sandbox_manager.get_sandbox(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox not found: {sandbox_id}")
        
        # Prepare execution command
        main_script = "/tmp/main_script.py"
        
        # Add input data as argument if provided
        cmd_args = args or []
        if input_data:
            cmd_args.append(json.dumps(input_data))
        
        # Execute template
        try:
            if main_script.endswith('.py'):
                cmd = f"python {main_script}"
            elif main_script.endswith('.js'):
                cmd = f"node {main_script}"
            else:
                cmd = main_script
            
            if cmd_args:
                cmd += " " + " ".join(f'"{{arg}}"' for arg in cmd_args).format(*cmd_args)
            
            result = await sandbox.run_command(cmd)
            
            return {
                "status": "success",
                "output": result.stdout,
                "error": result.stderr,
                "exit_code": result.exit_code
            }
            
        except Exception as e:
            logger.error(f"Template execution failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def update_template(self, 
                             sandbox_id: str, 
                             template_id: str,
                             config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Update deployed template"""
        logger.info(f"Updating template in sandbox: {sandbox_id}")
        
        # Get current deployment
        deployment = self.deployments.get(sandbox_id)
        if not deployment:
            raise ValueError(f"No deployment found for sandbox: {sandbox_id}")
        
        # Get new template
        template = self.registry.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        # Get sandbox
        sandbox = await self.sandbox_manager.get_sandbox(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox not found: {sandbox_id}")
        
        try:
            # Redeploy files
            await self._deploy_files(sandbox, template)
            
            # Run post-install hooks
            await self._run_hooks(sandbox, template, 'post_install')
            
            # Update deployment record
            deployment["template_id"] = template_id
            deployment["updated_at"] = datetime.now().isoformat()
            deployment["template"] = template.dict()
            deployment["config"].update(config or {})
            
            return {
                "status": "success",
                "message": "Template updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Template update failed: {e}")
            raise
    
    async def scale_template(self, 
                           template_id: str, 
                           instances: int,
                           config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Deploy multiple instances of a template"""
        logger.info(f"Scaling template {template_id} to {instances} instances")
        
        deployments = []
        for i in range(instances):
            instance_config = {**(config or {}), "instance_id": i}
            deployment = await self.deploy_template(template_id, instance_config)
            deployments.append(deployment)
        
        return deployments
    
    async def cleanup_deployment(self, sandbox_id: str) -> bool:
        """Clean up a template deployment"""
        logger.info(f"Cleaning up deployment: {sandbox_id}")
        
        try:
            # Get deployment
            deployment = self.deployments.get(sandbox_id)
            if deployment:
                # Run cleanup hooks if available
                template = TemplateConfig(**deployment["template"])
                sandbox = await self.sandbox_manager.get_sandbox(sandbox_id)
                if sandbox and template.hooks:
                    await self._run_hooks(sandbox, template, 'cleanup')
            
            # Terminate sandbox
            await self.sandbox_manager.terminate_sandbox(sandbox_id)
            
            # Remove deployment record
            if sandbox_id in self.deployments:
                del self.deployments[sandbox_id]
            
            logger.info(f"Deployment cleaned up: {sandbox_id}")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed for {sandbox_id}: {e}")
            return False
    
    def get_deployment_status(self, sandbox_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status"""
        return self.deployments.get(sandbox_id)
    
    def list_deployments(self, template_id: str = None) -> List[Dict[str, Any]]:
        """List all deployments, optionally filtered by template"""
        deployments = list(self.deployments.values())
        
        if template_id:
            deployments = [d for d in deployments if d["template_id"] == template_id]
        
        return deployments
    
    async def health_check(self, sandbox_id: str) -> Dict[str, Any]:
        """Run health check on deployed template"""
        deployment = self.deployments.get(sandbox_id)
        if not deployment:
            return {"status": "error", "message": "Deployment not found"}
        
        try:
            sandbox = await self.sandbox_manager.get_sandbox(sandbox_id)
            if not sandbox:
                return {"status": "error", "message": "Sandbox not found"}
            
            # Run health check hook if available
            template = TemplateConfig(**deployment["template"])
            if template.hooks and template.hooks.health_check:
                result = await sandbox.run_command(template.hooks.health_check)
                return {
                    "status": "healthy" if result.exit_code == 0 else "unhealthy",
                    "output": result.stdout,
                    "error": result.stderr
                }
            else:
                # Basic health check - just verify sandbox is responding
                result = await sandbox.run_command("echo 'healthy'")
                return {
                    "status": "healthy" if result.exit_code == 0 else "unhealthy",
                    "message": "Basic health check passed"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {e}"
            }
    
    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get deployment statistics"""
        total = len(self.deployments)
        by_template = {}
        
        for deployment in self.deployments.values():
            template_id = deployment["template_id"]
            by_template[template_id] = by_template.get(template_id, 0) + 1
        
        return {
            "total_deployments": total,
            "by_template": by_template,
            "active_sandboxes": list(self.deployments.keys())
        }
