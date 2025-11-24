#!/usr/bin/env python3
"""
Core Agentic Workflow Setup Script

This script sets up the agentic workflow in the core directory.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Base directory for the workflow
CORE_DIR = Path(__file__).parent
WORKFLOW_DIR = CORE_DIR / "agentic_workflow"

# Configuration for workflow phases
WORKFLOW_CONFIG = {
    "phases": [
        {"name": "problem_definition", "agent": "claude"},
        {"name": "architecture_design", "agent": "roo_code"},
        {"name": "implementation", "agent": "claude_roo"},
        {"name": "validation", "agent": "sparc2"},
        {"name": "testing", "agent": "aigi"},
        {"name": "deployment", "agent": "roo_aigi"}
    ],
    "agents": {
        "claude": {"description": "Problem definition and requirements analysis"},
        "roo_code": {"description": "Architecture and implementation"},
        "sparc2": {"description": "Verification and validation"},
        "aigi": {"description": "Testing and deployment"}
    }
}

def create_directory_structure() -> None:
    """Create the directory structure for the workflow."""
    dirs = [
        WORKFLOW_DIR / "config",
        WORKFLOW_DIR / "docs",
        WORKFLOW_DIR / "src",
        WORKFLOW_DIR / "tests" / "unit",
        WORKFLOW_DIR / "tests" / "integration",
        WORKFLOW_DIR / "tests" / "e2e",
        WORKFLOW_DIR / "deployment",
        WORKFLOW_DIR / "monitoring"
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch(exist_ok=True)

def create_config_files() -> None:
    """Create configuration files for the workflow."""
    # Main workflow config
    with open(WORKFLOW_DIR / "config" / "workflow.json", "w") as f:
        json.dump(WORKFLOW_CONFIG, f, indent=2)
    
    # Agent configurations
    agent_dir = WORKFLOW_DIR / "config" / "agents"
    agent_dir.mkdir(exist_ok=True)
    
    for agent_name, agent_config in WORKFLOW_CONFIG["agents"].items():
        with open(agent_dir / f"{agent_name}.json", "w") as f:
            json.dump(agent_config, f, indent=2)

def create_readme() -> None:
    """Create a README file for the workflow."""
    readme = """# Core Agentic Workflow System

This directory contains the agentic workflow system for the core project.

## Directory Structure

- `config/`: Configuration files
- `docs/`: Documentation
- `src/`: Source code
- `tests/`: Test files
- `deployment/`: Deployment configs
- `monitoring/`: Monitoring configs

## Getting Started

1. Review the configuration in `config/workflow.json`
2. Run the workflow: `python run_workflow.py`
3. Check `docs/` for phase documentation

## Workflow Phases
"""
    
    for phase in WORKFLOW_CONFIG["phases"]:
        readme += f"\n### {phase['name'].replace('_', ' ').title()}\n"
        readme += f"- **Agent**: {phase['agent']}\n"
    
    with open(WORKFLOW_DIR / "README.md", "w") as f:
        f.write(readme)

def main() -> None:
    """Set up the agentic workflow in the core directory."""
    print("Setting up core agentic workflow...")
    
    if WORKFLOW_DIR.exists():
        print(f"Warning: {WORKFLOW_DIR} already exists. Overwriting configuration...")
    
    print("Creating directory structure...")
    create_directory_structure()
    
    print("Creating configuration files...")
    create_config_files()
    
    print("Creating README...")
    create_readme()
    
    print("\nSetup complete!")
    print(f"Workflow directory: {WORKFLOW_DIR}")
    print("\nNext steps:")
    print(f"1. Review the configuration in {WORKFLOW_DIR}/config")
    print(f"2. Check {WORKFLOW_DIR}/README.md for more information")

if __name__ == "__main__":
    main()
