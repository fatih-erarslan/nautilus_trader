"""
Template Registry for E2B Templates
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .models import TemplateConfig, TemplateType, TemplateMetadata
from .base_templates import BaseTemplates
from .claude_flow_templates import ClaudeFlowTemplates
from .claude_code_templates import ClaudeCodeTemplates

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """Registry for managing E2B templates"""
    
    def __init__(self, storage_path: str = "/tmp/e2b_templates"):
        """Initialize template registry"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.templates: Dict[str, TemplateConfig] = {}
        self._load_builtin_templates()
        self._load_custom_templates()
    
    def _load_builtin_templates(self):
        """Load built-in templates"""
        logger.info("Loading built-in templates...")
        
        # Base templates
        self.register_template("python_base", BaseTemplates.python_base())
        self.register_template("node_base", BaseTemplates.node_base())
        self.register_template("trading_agent_base", BaseTemplates.trading_agent_base())
        
        # Claude-Flow templates
        self.register_template("claude_flow_swarm", ClaudeFlowTemplates.swarm_orchestrator())
        self.register_template("claude_flow_neural", ClaudeFlowTemplates.neural_agent())
        
        # Claude Code templates
        self.register_template("claude_code_sparc", ClaudeCodeTemplates.sparc_developer())
        self.register_template("claude_code_reviewer", ClaudeCodeTemplates.code_reviewer())
        
        logger.info(f"Loaded {len(self.templates)} built-in templates")
    
    def _load_custom_templates(self):
        """Load custom templates from storage"""
        templates_file = self.storage_path / "custom_templates.json"
        
        if templates_file.exists():
            try:
                with open(templates_file, 'r') as f:
                    custom_data = json.load(f)
                
                for template_id, template_data in custom_data.items():
                    template = TemplateConfig(**template_data)
                    self.templates[template_id] = template
                
                logger.info(f"Loaded {len(custom_data)} custom templates")
                
            except Exception as e:
                logger.error(f"Failed to load custom templates: {e}")
    
    def register_template(self, template_id: str, template: TemplateConfig) -> bool:
        """Register a new template"""
        try:
            # Validate template
            if not self.validate_template(template):
                logger.error(f"Template validation failed: {template_id}")
                return False
            
            # Store template
            self.templates[template_id] = template
            template.metadata.updated_at = datetime.now()
            
            logger.info(f"Registered template: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register template {template_id}: {e}")
            return False
    
    def get_template(self, template_id: str) -> Optional[TemplateConfig]:
        """Get a template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self, 
                      template_type: Optional[TemplateType] = None,
                      category: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List templates with optional filtering"""
        
        templates = []
        
        for template_id, template in self.templates.items():
            # Apply filters
            if template_type and template.template_type != template_type:
                continue
            
            if category and template.metadata.category != category:
                continue
            
            if tags:
                template_tags = set(template.metadata.tags)
                if not set(tags).intersection(template_tags):
                    continue
            
            templates.append({
                "id": template_id,
                "name": template.metadata.name,
                "description": template.metadata.description,
                "version": template.metadata.version,
                "type": template.template_type.value,
                "category": template.metadata.category,
                "tags": template.metadata.tags,
                "created_at": template.metadata.created_at.isoformat(),
                "updated_at": template.metadata.updated_at.isoformat()
            })
        
        return sorted(templates, key=lambda x: x["updated_at"], reverse=True)
    
    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """Search templates by name, description, or tags"""
        query_lower = query.lower()
        results = []
        
        for template_id, template in self.templates.items():
            # Search in name, description, and tags
            searchable_text = " ".join([
                template.metadata.name.lower(),
                template.metadata.description.lower(),
                " ".join(template.metadata.tags).lower()
            ])
            
            if query_lower in searchable_text:
                results.append({
                    "id": template_id,
                    "name": template.metadata.name,
                    "description": template.metadata.description,
                    "type": template.template_type.value,
                    "category": template.metadata.category,
                    "tags": template.metadata.tags,
                    "relevance": self._calculate_relevance(query_lower, searchable_text)
                })
        
        return sorted(results, key=lambda x: x["relevance"], reverse=True)
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate search relevance score"""
        # Simple relevance calculation
        words = query.split()
        score = 0.0
        
        for word in words:
            if word in text:
                score += 1.0
        
        return score / len(words) if words else 0.0
    
    def validate_template(self, template: TemplateConfig) -> bool:
        """Validate template configuration"""
        try:
            # Check required fields
            if not template.metadata.name:
                return False
            
            if not template.files.main_script:
                return False
            
            # Validate requirements
            if template.requirements.cpu_cores < 1 or template.requirements.cpu_cores > 8:
                return False
            
            if template.requirements.memory_mb < 256 or template.requirements.memory_mb > 8192:
                return False
            
            # Check script syntax (basic validation)
            if template.files.main_script.strip() == "":
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Template validation error: {e}")
            return False
    
    def save_custom_templates(self):
        """Save custom templates to storage"""
        custom_templates = {}
        
        for template_id, template in self.templates.items():
            # Only save non-builtin templates
            if not self._is_builtin_template(template_id):
                custom_templates[template_id] = template.dict()
        
        templates_file = self.storage_path / "custom_templates.json"
        
        try:
            with open(templates_file, 'w') as f:
                json.dump(custom_templates, f, indent=2, default=str)
            
            logger.info(f"Saved {len(custom_templates)} custom templates")
            
        except Exception as e:
            logger.error(f"Failed to save custom templates: {e}")
    
    def _is_builtin_template(self, template_id: str) -> bool:
        """Check if template is built-in"""
        builtin_templates = [
            "python_base", "node_base", "trading_agent_base",
            "claude_flow_swarm", "claude_flow_neural",
            "claude_code_sparc", "claude_code_reviewer"
        ]
        return template_id in builtin_templates
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a template"""
        if self._is_builtin_template(template_id):
            logger.error(f"Cannot delete built-in template: {template_id}")
            return False
        
        if template_id in self.templates:
            del self.templates[template_id]
            self.save_custom_templates()
            logger.info(f"Deleted template: {template_id}")
            return True
        
        return False
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        categories = set()
        for template in self.templates.values():
            categories.add(template.metadata.category)
        return sorted(list(categories))
    
    def get_tags(self) -> List[str]:
        """Get all available tags"""
        tags = set()
        for template in self.templates.values():
            tags.update(template.metadata.tags)
        return sorted(list(tags))
    
    def get_template_types(self) -> List[str]:
        """Get all available template types"""
        types = set()
        for template in self.templates.values():
            types.add(template.template_type.value)
        return sorted(list(types))
    
    def export_template(self, template_id: str, export_path: str) -> bool:
        """Export template to file"""
        template = self.get_template(template_id)
        if not template:
            return False
        
        try:
            export_data = {
                "id": template_id,
                "template": template.dict()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported template {template_id} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export template: {e}")
            return False
    
    def import_template(self, import_path: str) -> Optional[str]:
        """Import template from file"""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            template_id = import_data["id"]
            template_data = import_data["template"]
            
            template = TemplateConfig(**template_data)
            
            if self.register_template(template_id, template):
                self.save_custom_templates()
                logger.info(f"Imported template: {template_id}")
                return template_id
            
        except Exception as e:
            logger.error(f"Failed to import template: {e}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        stats = {
            "total_templates": len(self.templates),
            "by_type": {},
            "by_category": {},
            "builtin_count": 0,
            "custom_count": 0
        }
        
        for template_id, template in self.templates.items():
            # Count by type
            template_type = template.template_type.value
            stats["by_type"][template_type] = stats["by_type"].get(template_type, 0) + 1
            
            # Count by category
            category = template.metadata.category
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            # Count builtin vs custom
            if self._is_builtin_template(template_id):
                stats["builtin_count"] += 1
            else:
                stats["custom_count"] += 1
        
        return stats