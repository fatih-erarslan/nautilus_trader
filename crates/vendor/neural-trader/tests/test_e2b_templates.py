#!/usr/bin/env python3
"""
Test suite for E2B Templates system
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.e2b_templates import (
    TemplateConfig,
    TemplateType,
    TemplateBuilder,
    TemplateRegistry,
    TemplateDeployer,
    BaseTemplates,
    ClaudeFlowTemplates,
    ClaudeCodeTemplates,
    RuntimeEnvironment
)


class TestTemplateModels:
    """Test template models and data structures"""
    
    def test_template_type_enum(self):
        """Test TemplateType enum"""
        assert TemplateType.PYTHON_BASE.value == "python_base"
        assert TemplateType.CLAUDE_FLOW_SWARM.value == "claude_flow_swarm"
        assert TemplateType.TRADING_AGENT.value == "trading_agent"
    
    def test_runtime_environment_enum(self):
        """Test RuntimeEnvironment enum"""
        assert RuntimeEnvironment.PYTHON_3_10.value == "python:3.10"
        assert RuntimeEnvironment.NODE_20.value == "node:20"


class TestBaseTemplates:
    """Test base template creation"""
    
    def test_python_base_template(self):
        """Test Python base template"""
        template = BaseTemplates.python_base()
        
        assert template.template_type == TemplateType.PYTHON_BASE
        assert template.metadata.name == "Python Base Environment"
        assert template.requirements.runtime == RuntimeEnvironment.PYTHON_3_10
        assert "numpy" in template.requirements.python_packages
        assert template.files.main_script is not None
        assert "utils.py" in template.files.modules
    
    def test_node_base_template(self):
        """Test Node.js base template"""
        template = BaseTemplates.node_base()
        
        assert template.template_type == TemplateType.NODE_BASE
        assert template.metadata.name == "Node.js Base Environment"
        assert template.requirements.runtime == RuntimeEnvironment.NODE_20
        assert "express" in template.requirements.node_packages
        assert template.files.main_script is not None
        assert "package.json" in template.files.configs
    
    def test_trading_agent_template(self):
        """Test trading agent base template"""
        template = BaseTemplates.trading_agent_base()
        
        assert template.template_type == TemplateType.TRADING_AGENT
        assert template.metadata.name == "Trading Agent Base"
        assert "yfinance" in template.requirements.python_packages
        assert "strategies.py" in template.files.modules
        assert "risk_manager.py" in template.files.modules


class TestClaudeFlowTemplates:
    """Test Claude-Flow templates"""
    
    def test_swarm_orchestrator_template(self):
        """Test Claude-Flow swarm orchestrator"""
        template = ClaudeFlowTemplates.swarm_orchestrator()
        
        assert template.template_type == TemplateType.CLAUDE_FLOW_ORCHESTRATOR
        assert template.claude_flow is not None
        assert template.claude_flow.swarm_topology == "mesh"
        assert template.claude_flow.max_agents == 8
        assert "researcher" in template.claude_flow.agent_types
        assert "agent_types.js" in template.files.modules
        assert "memory_manager.js" in template.files.modules
    
    def test_neural_agent_template(self):
        """Test Claude-Flow neural agent"""
        template = ClaudeFlowTemplates.neural_agent()
        
        assert template.template_type == TemplateType.CLAUDE_FLOW_AGENT
        assert template.claude_flow.enable_neural is True
        assert template.requirements.gpu_enabled is True
        assert "torch" in template.requirements.python_packages


class TestClaudeCodeTemplates:
    """Test Claude Code templates"""
    
    def test_sparc_developer_template(self):
        """Test SPARC developer template"""
        template = ClaudeCodeTemplates.sparc_developer()
        
        assert template.template_type == TemplateType.CLAUDE_CODE_SPARC
        assert template.claude_code is not None
        assert template.claude_code.sparc_enabled is True
        assert template.claude_code.tdd_mode is True
        assert template.claude_code.parallel_execution is True
        assert "sparc_phases.js" in template.files.modules
        assert "todo_manager.js" in template.files.modules
    
    def test_code_reviewer_template(self):
        """Test code reviewer template"""
        template = ClaudeCodeTemplates.code_reviewer()
        
        assert template.template_type == TemplateType.CLAUDE_CODE_REVIEWER
        assert template.claude_code.github_integration is True
        assert "pylint" in template.requirements.python_packages
        assert "mypy" in template.requirements.python_packages


class TestTemplateBuilder:
    """Test template builder"""
    
    def test_builder_creation(self):
        """Test basic builder creation"""
        builder = TemplateBuilder()
        
        # Should start empty
        errors = builder.validate()
        assert len(errors) > 0  # Should have validation errors
    
    def test_python_template_factory(self):
        """Test Python template factory method"""
        builder = TemplateBuilder.create_python_template(
            "Test Python", 
            "Test description"
        )
        
        errors = builder.validate()
        assert len(errors) == 0  # Should be valid
        
        template = builder.build()
        assert template.template_type == TemplateType.PYTHON_BASE
        assert template.metadata.name == "Test Python"
    
    def test_trading_template_factory(self):
        """Test trading template factory method"""
        builder = TemplateBuilder.create_trading_template(
            "Test Momentum", 
            "momentum"
        )
        
        template = builder.build()
        assert template.template_type == TemplateType.TRADING_AGENT
        assert template.trading_agent is not None
        assert template.trading_agent.strategy_type == "momentum"
    
    def test_claude_flow_template_factory(self):
        """Test Claude-Flow template factory method"""
        builder = TemplateBuilder.create_claude_flow_template(
            "Test Swarm",
            ["researcher", "coder"]
        )
        
        template = builder.build()
        assert template.template_type == TemplateType.CLAUDE_FLOW_SWARM
        assert template.claude_flow is not None
        assert "researcher" in template.claude_flow.agent_types
    
    def test_builder_chaining(self):
        """Test builder method chaining"""
        template = (TemplateBuilder()
                   .set_type(TemplateType.PYTHON_BASE)
                   .set_metadata("Test", "Description")
                   .set_requirements(python_packages=["numpy"])
                   .set_main_script("print('hello')")
                   .build())
        
        assert template.metadata.name == "Test"
        assert "numpy" in template.requirements.python_packages
    
    def test_from_template_factory(self):
        """Test creating builder from existing template"""
        original = BaseTemplates.python_base()
        builder = TemplateBuilder.from_template(original)
        
        # Modify and rebuild
        new_template = (builder
                       .set_metadata("Modified", "Modified description")
                       .build())
        
        assert new_template.metadata.name == "Modified"
        assert new_template.template_type == original.template_type


class TestTemplateRegistry:
    """Test template registry"""
    
    @pytest.fixture
    def registry(self, tmp_path):
        """Create test registry"""
        return TemplateRegistry(storage_path=str(tmp_path))
    
    def test_builtin_templates_loaded(self, registry):
        """Test that built-in templates are loaded"""
        templates = registry.list_templates()
        
        # Should have built-in templates
        template_ids = [t["id"] for t in templates]
        assert "python_base" in template_ids
        assert "claude_flow_swarm" in template_ids
        assert "claude_code_sparc" in template_ids
    
    def test_register_custom_template(self, registry):
        """Test registering custom template"""
        template = BaseTemplates.python_base()
        template.metadata.name = "Custom Python"
        
        success = registry.register_template("custom_python", template)
        assert success is True
        
        retrieved = registry.get_template("custom_python")
        assert retrieved is not None
        assert retrieved.metadata.name == "Custom Python"
    
    def test_search_templates(self, registry):
        """Test template search"""
        results = registry.search_templates("python")
        
        # Should find Python-related templates
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert any("Python" in name for name in names)
    
    def test_filter_templates(self, registry):
        """Test template filtering"""
        # Filter by type
        claude_flow_templates = registry.list_templates(
            template_type=TemplateType.CLAUDE_FLOW_SWARM
        )
        assert len(claude_flow_templates) > 0
        
        # Filter by category
        base_templates = registry.list_templates(category="base")
        assert len(base_templates) > 0
        
        # Filter by tags
        trading_templates = registry.list_templates(tags=["trading"])
        assert len(trading_templates) > 0
    
    def test_delete_template(self, registry):
        """Test template deletion"""
        # Cannot delete built-in templates
        success = registry.delete_template("python_base")
        assert success is False
        
        # Can delete custom templates
        template = BaseTemplates.python_base()
        registry.register_template("custom_delete_test", template)
        
        success = registry.delete_template("custom_delete_test")
        assert success is True
        
        retrieved = registry.get_template("custom_delete_test")
        assert retrieved is None
    
    def test_export_import_template(self, registry, tmp_path):
        """Test template export/import"""
        # Export template
        export_path = tmp_path / "exported_template.json"
        success = registry.export_template("python_base", str(export_path))
        assert success is True
        assert export_path.exists()
        
        # Import template
        template_id = registry.import_template(str(export_path))
        assert template_id is not None
    
    def test_registry_stats(self, registry):
        """Test registry statistics"""
        stats = registry.get_stats()
        
        assert "total_templates" in stats
        assert "by_type" in stats
        assert "by_category" in stats
        assert stats["total_templates"] > 0


class TestTemplateDeployer:
    """Test template deployment"""
    
    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create mock sandbox manager"""
        manager = Mock()
        manager.create_sandbox = AsyncMock()
        manager.get_sandbox = AsyncMock()
        manager.terminate_sandbox = AsyncMock()
        return manager
    
    @pytest.fixture
    def deployer(self, mock_sandbox_manager):
        """Create test deployer"""
        return TemplateDeployer(mock_sandbox_manager)
    
    @pytest.mark.asyncio
    async def test_deploy_template(self, deployer, mock_sandbox_manager):
        """Test template deployment"""
        # Mock sandbox creation
        mock_sandbox = Mock()
        mock_sandbox.sandbox_id = "test_sandbox_123"
        mock_sandbox.run_command = AsyncMock()
        mock_sandbox.files = Mock()
        mock_sandbox.files.write = AsyncMock()
        
        mock_sandbox_manager.create_sandbox.return_value = mock_sandbox
        
        # Deploy template
        result = await deployer.deploy_template("python_base")
        
        assert result["status"] == "success"
        assert result["sandbox_id"] == "test_sandbox_123"
        assert "test_sandbox_123" in deployer.deployments
    
    @pytest.mark.asyncio
    async def test_execute_template(self, deployer, mock_sandbox_manager):
        """Test template execution"""
        # Setup mock deployment
        sandbox_id = "test_sandbox_123"
        mock_sandbox = Mock()
        mock_sandbox.run_command = AsyncMock()
        mock_sandbox.run_command.return_value = Mock(
            stdout="{'result': 'success'}",
            stderr="",
            exit_code=0
        )
        
        mock_sandbox_manager.get_sandbox.return_value = mock_sandbox
        
        # Create fake deployment
        deployer.deployments[sandbox_id] = {
            "template_id": "python_base",
            "sandbox_id": sandbox_id,
            "status": "deployed"
        }
        
        # Execute template
        result = await deployer.execute_template(
            sandbox_id,
            input_data={"test": "data"}
        )
        
        assert result["status"] == "success"
        assert result["output"] == "{'result': 'success'}"
    
    @pytest.mark.asyncio
    async def test_scale_template(self, deployer, mock_sandbox_manager):
        """Test template scaling"""
        # Mock sandbox creation
        mock_sandbox = Mock()
        mock_sandbox.sandbox_id = "test_sandbox_123"
        mock_sandbox.run_command = AsyncMock()
        mock_sandbox.files = Mock()
        mock_sandbox.files.write = AsyncMock()
        
        mock_sandbox_manager.create_sandbox.return_value = mock_sandbox
        
        # Scale to 3 instances
        deployments = await deployer.scale_template("python_base", 3)
        
        assert len(deployments) == 3
        assert all(d["status"] == "success" for d in deployments)
    
    @pytest.mark.asyncio
    async def test_health_check(self, deployer, mock_sandbox_manager):
        """Test deployment health check"""
        sandbox_id = "test_sandbox_123"
        
        # Setup mock
        mock_sandbox = Mock()
        mock_sandbox.run_command = AsyncMock()
        mock_sandbox.run_command.return_value = Mock(
            stdout="healthy",
            stderr="",
            exit_code=0
        )
        
        mock_sandbox_manager.get_sandbox.return_value = mock_sandbox
        
        # Create deployment
        deployer.deployments[sandbox_id] = {
            "template_id": "python_base",
            "template": BaseTemplates.python_base().dict()
        }
        
        # Run health check
        health = await deployer.health_check(sandbox_id)
        
        assert health["status"] == "healthy"
    
    def test_deployment_stats(self, deployer):
        """Test deployment statistics"""
        # Add some fake deployments
        deployer.deployments["sandbox1"] = {"template_id": "python_base"}
        deployer.deployments["sandbox2"] = {"template_id": "python_base"}
        deployer.deployments["sandbox3"] = {"template_id": "trading_agent"}
        
        stats = deployer.get_deployment_stats()
        
        assert stats["total_deployments"] == 3
        assert stats["by_template"]["python_base"] == 2
        assert stats["by_template"]["trading_agent"] == 1


class TestTemplateAPI:
    """Test template API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from src.e2b_templates.template_api import router
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    def test_list_templates_endpoint(self, client):
        """Test list templates endpoint"""
        response = client.get("/e2b/templates/registry/list")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "templates" in data
        assert len(data["templates"]) > 0
    
    def test_search_templates_endpoint(self, client):
        """Test search templates endpoint"""
        response = client.get("/e2b/templates/registry/search?query=python")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
    
    def test_get_template_endpoint(self, client):
        """Test get specific template endpoint"""
        response = client.get("/e2b/templates/registry/template/python_base")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "template" in data
    
    def test_create_python_template_endpoint(self, client):
        """Test create Python template endpoint"""
        template_data = {
            "name": "Test Python Template",
            "description": "Test description",
            "template_type": "python_base",
            "main_script": "print('hello world')",
            "requirements": {
                "python_packages": ["numpy", "pandas"]
            }
        }
        
        response = client.post("/e2b/templates/builder/python", json=template_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "template_id" in data


@pytest.mark.integration
class TestTemplateIntegration:
    """Integration tests for the complete template system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, tmp_path):
        """Test complete workflow from creation to deployment"""
        # Create registry
        registry = TemplateRegistry(storage_path=str(tmp_path))
        
        # Create custom template
        builder = TemplateBuilder.create_python_template(
            "Integration Test", 
            "End-to-end test template"
        )
        template = builder.build()
        
        # Register template
        template_id = "integration_test"
        success = registry.register_template(template_id, template)
        assert success
        
        # Mock deployment (would normally deploy to E2B)
        with patch('src.e2b_templates.template_deployer.SandboxManager') as mock_manager:
            mock_sandbox = Mock()
            mock_sandbox.sandbox_id = "test_integration_sandbox"
            mock_sandbox.run_command = AsyncMock()
            mock_sandbox.files = Mock()
            mock_sandbox.files.write = AsyncMock()
            
            mock_manager.return_value.create_sandbox.return_value = mock_sandbox
            
            deployer = TemplateDeployer()
            deployment = await deployer.deploy_template(template_id)
            
            assert deployment["status"] == "success"
            assert deployment["template_id"] == template_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])