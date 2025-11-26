"""
Template Builder for creating custom E2B templates
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

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

logger = logging.getLogger(__name__)


class TemplateBuilder:
    """Builder for creating custom E2B templates"""
    
    def __init__(self):
        """Initialize template builder"""
        self.reset()
    
    def reset(self):
        """Reset builder to start fresh"""
        self._template_type = None
        self._metadata = None
        self._requirements = None
        self._files = None
        self._hooks = None
        self._claude_flow = None
        self._claude_code = None
        self._trading_agent = None
        self._custom_config = {}
    
    def set_type(self, template_type: TemplateType) -> 'TemplateBuilder':
        """Set template type"""
        self._template_type = template_type
        return self
    
    def set_metadata(self, 
                     name: str,
                     description: str,
                     version: str = "1.0.0",
                     author: str = "Custom",
                     tags: List[str] = None,
                     category: str = "custom") -> 'TemplateBuilder':
        """Set template metadata"""
        self._metadata = TemplateMetadata(
            name=name,
            description=description,
            version=version,
            author=author,
            tags=tags or [],
            category=category
        )
        return self
    
    def set_requirements(self,
                        runtime: RuntimeEnvironment = RuntimeEnvironment.PYTHON_3_10,
                        cpu_cores: int = 2,
                        memory_mb: int = 1024,
                        storage_gb: int = 1,
                        gpu_enabled: bool = False,
                        network_access: bool = True,
                        python_packages: List[str] = None,
                        node_packages: List[str] = None,
                        system_packages: List[str] = None,
                        env_vars: Dict[str, str] = None) -> 'TemplateBuilder':
        """Set template requirements"""
        self._requirements = TemplateRequirements(
            runtime=runtime,
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            storage_gb=storage_gb,
            gpu_enabled=gpu_enabled,
            network_access=network_access,
            python_packages=python_packages or [],
            node_packages=node_packages or [],
            system_packages=system_packages or [],
            env_vars=env_vars or {}
        )
        return self
    
    def set_main_script(self, script_content: str) -> 'TemplateBuilder':
        """Set main script content"""
        if not self._files:
            self._files = TemplateFiles(main_script=script_content)
        else:
            self._files.main_script = script_content
        return self
    
    def add_module(self, name: str, content: str) -> 'TemplateBuilder':
        """Add a module file"""
        if not self._files:
            self._files = TemplateFiles(main_script="", modules={name: content})
        else:
            self._files.modules[name] = content
        return self
    
    def add_config_file(self, name: str, content: str) -> 'TemplateBuilder':
        """Add a configuration file"""
        if not self._files:
            self._files = TemplateFiles(main_script="", configs={name: content})
        else:
            self._files.configs[name] = content
        return self
    
    def add_data_file(self, name: str, content: str) -> 'TemplateBuilder':
        """Add a data file"""
        if not self._files:
            self._files = TemplateFiles(main_script="", data={name: content})
        else:
            self._files.data[name] = content
        return self
    
    def add_script(self, name: str, content: str) -> 'TemplateBuilder':
        """Add a utility script"""
        if not self._files:
            self._files = TemplateFiles(main_script="", scripts={name: content})
        else:
            self._files.scripts[name] = content
        return self
    
    def set_hooks(self,
                  pre_install: Optional[str] = None,
                  post_install: Optional[str] = None,
                  pre_start: Optional[str] = None,
                  post_start: Optional[str] = None,
                  health_check: Optional[str] = None,
                  cleanup: Optional[str] = None) -> 'TemplateBuilder':
        """Set lifecycle hooks"""
        self._hooks = TemplateHooks(
            pre_install=pre_install,
            post_install=post_install,
            pre_start=pre_start,
            post_start=post_start,
            health_check=health_check,
            cleanup=cleanup
        )
        return self
    
    def set_claude_flow_config(self,
                              swarm_topology: str = "mesh",
                              max_agents: int = 5,
                              agent_types: List[str] = None,
                              memory_namespace: str = "default",
                              coordination_mode: str = "collaborative",
                              enable_neural: bool = False,
                              enable_memory: bool = True,
                              hooks: Dict[str, str] = None) -> 'TemplateBuilder':
        """Set Claude-Flow configuration"""
        self._claude_flow = ClaudeFlowConfig(
            swarm_topology=swarm_topology,
            max_agents=max_agents,
            agent_types=agent_types or [],
            memory_namespace=memory_namespace,
            coordination_mode=coordination_mode,
            enable_neural=enable_neural,
            enable_memory=enable_memory,
            hooks=hooks or {}
        )
        return self
    
    def set_claude_code_config(self,
                              sparc_enabled: bool = True,
                              tdd_mode: bool = True,
                              parallel_execution: bool = True,
                              max_todos: int = 10,
                              file_organization: Dict[str, str] = None,
                              agent_spawning: bool = True,
                              memory_persistence: bool = True,
                              github_integration: bool = False) -> 'TemplateBuilder':
        """Set Claude Code configuration"""
        self._claude_code = ClaudeCodeConfig(
            sparc_enabled=sparc_enabled,
            tdd_mode=tdd_mode,
            parallel_execution=parallel_execution,
            max_todos=max_todos,
            file_organization=file_organization or {},
            agent_spawning=agent_spawning,
            memory_persistence=memory_persistence,
            github_integration=github_integration
        )
        return self
    
    def set_trading_agent_config(self,
                                strategy_type: str,
                                symbols: List[str] = None,
                                risk_params: Dict[str, float] = None,
                                execution_mode: str = "simulation",
                                data_sources: List[str] = None,
                                indicators: List[str] = None,
                                backtest_enabled: bool = True,
                                paper_trading: bool = True,
                                live_trading: bool = False) -> 'TemplateBuilder':
        """Set trading agent configuration"""
        self._trading_agent = TradingAgentConfig(
            strategy_type=strategy_type,
            symbols=symbols or [],
            risk_params=risk_params or {},
            execution_mode=execution_mode,
            data_sources=data_sources or [],
            indicators=indicators or [],
            backtest_enabled=backtest_enabled,
            paper_trading=paper_trading,
            live_trading=live_trading
        )
        return self
    
    def set_custom_config(self, config: Dict[str, Any]) -> 'TemplateBuilder':
        """Set custom configuration"""
        self._custom_config = config
        return self
    
    def build(self) -> TemplateConfig:
        """Build the template configuration"""
        # Validate required fields
        if not self._template_type:
            raise ValueError("Template type is required")
        
        if not self._metadata:
            raise ValueError("Template metadata is required")
        
        if not self._requirements:
            raise ValueError("Template requirements are required")
        
        if not self._files or not self._files.main_script:
            raise ValueError("Main script is required")
        
        # Build template config
        template = TemplateConfig(
            template_type=self._template_type,
            metadata=self._metadata,
            requirements=self._requirements,
            files=self._files,
            hooks=self._hooks,
            claude_flow=self._claude_flow,
            claude_code=self._claude_code,
            trading_agent=self._trading_agent,
            custom_config=self._custom_config
        )
        
        return template
    
    @classmethod
    def from_template(cls, template: TemplateConfig) -> 'TemplateBuilder':
        """Create builder from existing template"""
        builder = cls()
        
        builder._template_type = template.template_type
        builder._metadata = template.metadata
        builder._requirements = template.requirements
        builder._files = template.files
        builder._hooks = template.hooks
        builder._claude_flow = template.claude_flow
        builder._claude_code = template.claude_code
        builder._trading_agent = template.trading_agent
        builder._custom_config = template.custom_config
        
        return builder
    
    @classmethod
    def create_python_template(cls, name: str, description: str) -> 'TemplateBuilder':
        """Create a Python template with common setup"""
        return (cls()
                .set_type(TemplateType.PYTHON_BASE)
                .set_metadata(name, description)
                .set_requirements(
                    runtime=RuntimeEnvironment.PYTHON_3_10,
                    python_packages=["requests", "python-dotenv"]
                )
                .set_main_script('''#!/usr/bin/env python3
import sys
import json

def main():
    config = {}
    if len(sys.argv) > 1:
        try:
            config = json.loads(sys.argv[1])
        except:
            pass
    
    print("Python template running...")
    print(f"Config: {config}")
    
    return {"status": "success", "message": "Template executed"}

if __name__ == "__main__":
    result = main()
    print(json.dumps(result))
'''))
    
    @classmethod
    def create_node_template(cls, name: str, description: str) -> 'TemplateBuilder':
        """Create a Node.js template with common setup"""
        return (cls()
                .set_type(TemplateType.NODE_BASE)
                .set_metadata(name, description)
                .set_requirements(
                    runtime=RuntimeEnvironment.NODE_20,
                    node_packages=["express", "axios"]
                )
                .set_main_script('''#!/usr/bin/env node
const process = require('process');

async function main() {
    let config = {};
    if (process.argv.length > 2) {
        try {
            config = JSON.parse(process.argv[2]);
        } catch (e) {
            // ignore
        }
    }
    
    console.log('Node.js template running...');
    console.log('Config:', config);
    
    const result = {
        status: 'success',
        message: 'Template executed'
    };
    
    console.log(JSON.stringify(result));
    return 0;
}

main().then(code => process.exit(code)).catch(error => {
    console.error('Error:', error);
    process.exit(1);
});
'''))
    
    @classmethod
    def create_trading_template(cls, name: str, strategy_type: str) -> 'TemplateBuilder':
        """Create a trading agent template"""
        return (cls()
                .set_type(TemplateType.TRADING_AGENT)
                .set_metadata(
                    name=name,
                    description=f"{strategy_type} trading strategy",
                    category="trading",
                    tags=["trading", strategy_type]
                )
                .set_requirements(
                    python_packages=[
                        "yfinance", "pandas", "numpy", "ta", "scikit-learn"
                    ]
                )
                .set_trading_agent_config(strategy_type=strategy_type)
                .set_main_script(f'''#!/usr/bin/env python3
"""
{strategy_type} Trading Strategy
"""

import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np

class {strategy_type.title().replace('_', '')}Strategy:
    def __init__(self, config):
        self.config = config
        self.symbols = config.get('symbols', ['AAPL'])
    
    def fetch_data(self, symbol, period='1mo'):
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period)
    
    def analyze(self, data):
        # Implement {strategy_type} logic here
        signals = []
        return signals
    
    def run(self):
        results = {{"status": "success", "trades": [], "analysis": {{}}}}
        
        for symbol in self.symbols:
            data = self.fetch_data(symbol)
            signals = self.analyze(data)
            results["analysis"][symbol] = signals
        
        return results

def main():
    config = {{}}
    if len(sys.argv) > 1:
        try:
            config = json.loads(sys.argv[1])
        except:
            pass
    
    strategy = {strategy_type.title().replace('_', '')}Strategy(config)
    result = strategy.run()
    
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''))
    
    @classmethod
    def create_claude_flow_template(cls, name: str, agent_types: List[str]) -> 'TemplateBuilder':
        """Create a Claude-Flow template"""
        return (cls()
                .set_type(TemplateType.CLAUDE_FLOW_SWARM)
                .set_metadata(
                    name=name,
                    description="Claude-Flow swarm orchestration",
                    category="claude-flow",
                    tags=["claude-flow", "swarm", "ai"]
                )
                .set_requirements(
                    runtime=RuntimeEnvironment.NODE_20,
                    node_packages=["claude-flow@alpha"]
                )
                .set_claude_flow_config(
                    agent_types=agent_types,
                    enable_neural=True,
                    enable_memory=True
                )
                .set_main_script('''#!/usr/bin/env node
// Claude-Flow Template
const process = require('process');

async function main() {
    const config = process.argv.length > 2 ? JSON.parse(process.argv[2]) : {};
    console.log('Claude-Flow template running...');
    console.log('Agent types:', config.agent_types || []);
    return {status: 'success'};
}

main().then(() => process.exit(0)).catch(err => {
    console.error(err);
    process.exit(1);
});
'''))
    
    def validate(self) -> List[str]:
        """Validate current configuration"""
        errors = []
        
        if not self._template_type:
            errors.append("Template type is required")
        
        if not self._metadata:
            errors.append("Template metadata is required")
        elif not self._metadata.name:
            errors.append("Template name is required")
        
        if not self._requirements:
            errors.append("Template requirements are required")
        
        if not self._files or not self._files.main_script:
            errors.append("Main script is required")
        
        return errors