"""
E2B Templates Module

Provides comprehensive template system for E2B sandboxes including:
- Base templates for Python, Node.js, and trading agents
- Claude-Flow integration templates for swarm orchestration
- Claude Code templates for SPARC methodology
- Template builder for creating custom templates
- Template registry for managing templates
- Template deployer for E2B sandbox deployment
- Template API for web interface
"""

from .models import (
    TemplateConfig,
    TemplateType,
    TemplateMetadata,
    TemplateRequirements,
    TemplateFiles,
    TemplateHooks,
    ClaudeFlowConfig,
    ClaudeCodeConfig,
    TradingAgentConfig,
    RuntimeEnvironment
)

from .base_templates import BaseTemplates
from .claude_flow_templates import ClaudeFlowTemplates
from .claude_code_templates import ClaudeCodeTemplates
from .template_builder import TemplateBuilder
from .template_registry import TemplateRegistry
from .template_deployer import TemplateDeployer
from .template_api import router as template_router

__all__ = [
    # Models
    "TemplateConfig",
    "TemplateType",
    "TemplateMetadata",
    "TemplateRequirements",
    "TemplateFiles",
    "TemplateHooks",
    "ClaudeFlowConfig",
    "ClaudeCodeConfig",
    "TradingAgentConfig",
    "RuntimeEnvironment",
    
    # Template Collections
    "BaseTemplates",
    "ClaudeFlowTemplates",
    "ClaudeCodeTemplates",
    
    # Core Components
    "TemplateBuilder",
    "TemplateRegistry",
    "TemplateDeployer",
    
    # API
    "template_router"
]

# Version
__version__ = "1.0.0"