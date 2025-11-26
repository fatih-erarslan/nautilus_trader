"""
Template Management API for E2B Templates
"""

import json
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from .template_registry import TemplateRegistry
from .template_deployer import TemplateDeployer
from .template_builder import TemplateBuilder
from .models import TemplateType
from ..e2b_integration.sandbox_manager import SandboxManager

logger = logging.getLogger(__name__)

# Initialize components
registry = TemplateRegistry()
sandbox_manager = SandboxManager()
deployer = TemplateDeployer(sandbox_manager)

# Create API router
router = APIRouter(prefix="/e2b/templates", tags=["templates"])


# Request models
class TemplateDeployRequest(BaseModel):
    template_id: str
    config: Dict[str, Any] = {}
    timeout: int = 300


class TemplateExecuteRequest(BaseModel):
    args: List[str] = []
    input_data: Dict[str, Any] = {}


class TemplateUpdateRequest(BaseModel):
    template_id: str
    config: Dict[str, Any] = {}


class TemplateScaleRequest(BaseModel):
    template_id: str
    instances: int
    config: Dict[str, Any] = {}


class CustomTemplateRequest(BaseModel):
    name: str
    description: str
    template_type: str
    runtime: str = "python_3_10"
    main_script: str
    requirements: Dict[str, Any] = {}
    config: Dict[str, Any] = {}


# Template Registry Endpoints
@router.get("/registry/list")
async def list_templates(template_type: Optional[str] = None, 
                        category: Optional[str] = None,
                        tags: Optional[str] = None):
    """List available templates"""
    try:
        tag_list = tags.split(",") if tags else None
        template_type_enum = TemplateType(template_type) if template_type else None
        
        templates = registry.list_templates(
            template_type=template_type_enum,
            category=category,
            tags=tag_list
        )
        
        return {
            "status": "success",
            "templates": templates,
            "count": len(templates)
        }
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/search")
async def search_templates(query: str):
    """Search templates by name, description, or tags"""
    try:
        results = registry.search_templates(query)
        
        return {
            "status": "success",
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Failed to search templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/template/{template_id}")
async def get_template(template_id: str):
    """Get specific template details"""
    try:
        template = registry.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "status": "success",
            "template": template.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/categories")
async def get_categories():
    """Get all available template categories"""
    try:
        categories = registry.get_categories()
        return {
            "status": "success",
            "categories": categories
        }
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/tags")
async def get_tags():
    """Get all available template tags"""
    try:
        tags = registry.get_tags()
        return {
            "status": "success",
            "tags": tags
        }
    except Exception as e:
        logger.error(f"Failed to get tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/stats")
async def get_registry_stats():
    """Get template registry statistics"""
    try:
        stats = registry.get_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get registry stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Template Builder Endpoints
@router.post("/builder/python")
async def create_python_template(request: CustomTemplateRequest):
    """Create a custom Python template"""
    try:
        builder = TemplateBuilder.create_python_template(
            name=request.name,
            description=request.description
        )
        
        # Update with custom requirements
        if "python_packages" in request.requirements:
            builder.set_requirements(
                python_packages=request.requirements["python_packages"]
            )
        
        # Set custom main script
        builder.set_main_script(request.main_script)
        
        # Build template
        template = builder.build()
        
        # Register template
        template_id = f"custom_{request.name.lower().replace(' ', '_')}"
        success = registry.register_template(template_id, template)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register template")
        
        registry.save_custom_templates()
        
        return {
            "status": "success",
            "template_id": template_id,
            "message": "Python template created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create Python template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/builder/trading")
async def create_trading_template(strategy_type: str, request: CustomTemplateRequest):
    """Create a custom trading template"""
    try:
        builder = TemplateBuilder.create_trading_template(
            name=request.name,
            strategy_type=strategy_type
        )
        
        # Update with custom configuration
        if "symbols" in request.config:
            builder.set_trading_agent_config(
                strategy_type=strategy_type,
                symbols=request.config["symbols"]
            )
        
        # Set custom main script if provided
        if request.main_script:
            builder.set_main_script(request.main_script)
        
        # Build and register template
        template = builder.build()
        template_id = f"trading_{strategy_type}_{request.name.lower().replace(' ', '_')}"
        
        success = registry.register_template(template_id, template)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register template")
        
        registry.save_custom_templates()
        
        return {
            "status": "success",
            "template_id": template_id,
            "message": "Trading template created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create trading template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/builder/claude-flow")
async def create_claude_flow_template(request: CustomTemplateRequest):
    """Create a custom Claude-Flow template"""
    try:
        agent_types = request.config.get("agent_types", ["researcher", "coder"])
        
        builder = TemplateBuilder.create_claude_flow_template(
            name=request.name,
            agent_types=agent_types
        )
        
        # Update with custom Claude-Flow config
        if "max_agents" in request.config:
            builder.set_claude_flow_config(
                agent_types=agent_types,
                max_agents=request.config["max_agents"]
            )
        
        # Build and register template
        template = builder.build()
        template_id = f"claude_flow_{request.name.lower().replace(' ', '_')}"
        
        success = registry.register_template(template_id, template)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register template")
        
        registry.save_custom_templates()
        
        return {
            "status": "success",
            "template_id": template_id,
            "message": "Claude-Flow template created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create Claude-Flow template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Template Deployment Endpoints
@router.post("/deploy")
async def deploy_template(request: TemplateDeployRequest, background_tasks: BackgroundTasks):
    """Deploy a template to E2B sandbox"""
    try:
        deployment = await deployer.deploy_template(
            template_id=request.template_id,
            config=request.config,
            timeout=request.timeout
        )
        
        return deployment
        
    except Exception as e:
        logger.error(f"Failed to deploy template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deploy/{sandbox_id}/execute")
async def execute_template(sandbox_id: str, request: TemplateExecuteRequest):
    """Execute deployed template"""
    try:
        result = await deployer.execute_template(
            sandbox_id=sandbox_id,
            args=request.args,
            input_data=request.input_data
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to execute template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/deploy/{sandbox_id}/update")
async def update_template(sandbox_id: str, request: TemplateUpdateRequest):
    """Update deployed template"""
    try:
        result = await deployer.update_template(
            sandbox_id=sandbox_id,
            template_id=request.template_id,
            config=request.config
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to update template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deploy/scale")
async def scale_template(request: TemplateScaleRequest):
    """Scale template deployment"""
    try:
        deployments = await deployer.scale_template(
            template_id=request.template_id,
            instances=request.instances,
            config=request.config
        )
        
        return {
            "status": "success",
            "deployments": deployments,
            "instances": len(deployments)
        }
        
    except Exception as e:
        logger.error(f"Failed to scale template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/deploy/{sandbox_id}")
async def cleanup_deployment(sandbox_id: str, background_tasks: BackgroundTasks):
    """Clean up template deployment"""
    try:
        # Run cleanup in background
        background_tasks.add_task(deployer.cleanup_deployment, sandbox_id)
        
        return {
            "status": "success",
            "message": "Cleanup initiated"
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deploy/{sandbox_id}/status")
async def get_deployment_status(sandbox_id: str):
    """Get deployment status"""
    try:
        status = deployer.get_deployment_status(sandbox_id)
        if not status:
            raise HTTPException(status_code=404, detail="Deployment not found")
        
        return {
            "status": "success",
            "deployment": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deploy/list")
async def list_deployments(template_id: Optional[str] = None):
    """List all deployments"""
    try:
        deployments = deployer.list_deployments(template_id)
        
        return {
            "status": "success",
            "deployments": deployments,
            "count": len(deployments)
        }
        
    except Exception as e:
        logger.error(f"Failed to list deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deploy/{sandbox_id}/health")
async def health_check_deployment(sandbox_id: str):
    """Run health check on deployment"""
    try:
        health = await deployer.health_check(sandbox_id)
        return health
        
    except Exception as e:
        logger.error(f"Failed to run health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deploy/stats")
async def get_deployment_stats():
    """Get deployment statistics"""
    try:
        stats = deployer.get_deployment_stats()
        return {
            "status": "success",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get deployment stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Template Management Endpoints
@router.delete("/registry/{template_id}")
async def delete_template(template_id: str):
    """Delete a custom template"""
    try:
        success = registry.delete_template(template_id)
        if not success:
            raise HTTPException(status_code=400, detail="Cannot delete template")
        
        return {
            "status": "success",
            "message": "Template deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/registry/export/{template_id}")
async def export_template(template_id: str, export_path: str = "/tmp/exported_template.json"):
    """Export template to file"""
    try:
        success = registry.export_template(template_id, export_path)
        if not success:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "status": "success",
            "export_path": export_path,
            "message": "Template exported successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/registry/import")
async def import_template(import_path: str):
    """Import template from file"""
    try:
        template_id = registry.import_template(import_path)
        if not template_id:
            raise HTTPException(status_code=400, detail="Failed to import template")
        
        return {
            "status": "success",
            "template_id": template_id,
            "message": "Template imported successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import template: {e}")
        raise HTTPException(status_code=500, detail=str(e))
