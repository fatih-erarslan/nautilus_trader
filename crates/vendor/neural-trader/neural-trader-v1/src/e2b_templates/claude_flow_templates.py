"""
Claude-Flow specific templates for E2B
"""

from .models import (
    TemplateConfig,
    TemplateType,
    TemplateMetadata,
    TemplateRequirements,
    TemplateFiles,
    ClaudeFlowConfig,
    RuntimeEnvironment
)


class ClaudeFlowTemplates:
    """Claude-Flow integration templates"""
    
    @staticmethod
    def swarm_orchestrator() -> TemplateConfig:
        """Claude-Flow swarm orchestrator template"""
        return TemplateConfig(
            template_type=TemplateType.CLAUDE_FLOW_ORCHESTRATOR,
            metadata=TemplateMetadata(
                name="Claude-Flow Swarm Orchestrator",
                description="Orchestrates multi-agent swarms using Claude-Flow",
                version="2.0.0",
                tags=["claude-flow", "swarm", "orchestration", "ai"],
                category="claude-flow"
            ),
            requirements=TemplateRequirements(
                runtime=RuntimeEnvironment.NODE_20,
                cpu_cores=4,
                memory_mb=2048,
                node_packages=[
                    "claude-flow@alpha",
                    "axios",
                    "ws",
                    "express"
                ],
                system_packages=[
                    "git",
                    "curl"
                ],
                env_vars={
                    "CLAUDE_FLOW_MODE": "swarm",
                    "MAX_AGENTS": "8",
                    "TOPOLOGY": "mesh"
                }
            ),
            claude_flow=ClaudeFlowConfig(
                swarm_topology="mesh",
                max_agents=8,
                agent_types=["researcher", "coder", "tester", "reviewer", "planner"],
                coordination_mode="collaborative",
                enable_neural=True,
                enable_memory=True,
                hooks={
                    "pre-task": "npx claude-flow@alpha hooks pre-task",
                    "post-task": "npx claude-flow@alpha hooks post-task",
                    "session-restore": "npx claude-flow@alpha hooks session-restore"
                }
            ),
            files=TemplateFiles(
                main_script='''#!/usr/bin/env node
/**
 * Claude-Flow Swarm Orchestrator
 */

const { exec } = require('child_process');
const util = require('util');
const execAsync = util.promisify(exec);

class SwarmOrchestrator {
    constructor(config) {
        this.config = config;
        this.swarmId = null;
        this.agents = [];
        this.tasks = [];
    }
    
    async initialize() {
        console.log('Initializing Claude-Flow Swarm...');
        
        // Initialize swarm
        const topology = this.config.topology || 'mesh';
        const maxAgents = this.config.maxAgents || 8;
        
        const { stdout } = await execAsync(
            `npx claude-flow@alpha swarm init --topology ${topology} --max-agents ${maxAgents}`
        );
        
        const result = JSON.parse(stdout);
        this.swarmId = result.swarmId;
        console.log(`Swarm initialized: ${this.swarmId}`);
        
        return this.swarmId;
    }
    
    async spawnAgent(type, capabilities = []) {
        console.log(`Spawning ${type} agent...`);
        
        const capStr = capabilities.join(',');
        const { stdout } = await execAsync(
            `npx claude-flow@alpha agent spawn --type ${type} --capabilities "${capStr}" --swarm ${this.swarmId}`
        );
        
        const agent = JSON.parse(stdout);
        this.agents.push(agent);
        console.log(`Agent spawned: ${agent.id}`);
        
        return agent;
    }
    
    async orchestrateTask(task, strategy = 'adaptive') {
        console.log(`Orchestrating task: ${task}`);
        
        const { stdout } = await execAsync(
            `npx claude-flow@alpha task orchestrate --task "${task}" --strategy ${strategy} --swarm ${this.swarmId}`
        );
        
        const result = JSON.parse(stdout);
        this.tasks.push(result);
        console.log(`Task orchestrated: ${result.taskId}`);
        
        return result;
    }
    
    async getStatus() {
        const { stdout } = await execAsync(
            `npx claude-flow@alpha swarm status --swarm ${this.swarmId}`
        );
        
        return JSON.parse(stdout);
    }
    
    async cleanup() {
        console.log('Cleaning up swarm...');
        
        await execAsync(
            `npx claude-flow@alpha swarm destroy --swarm ${this.swarmId}`
        );
        
        console.log('Swarm destroyed');
    }
}

async function main() {
    const config = process.argv.length > 2 ? JSON.parse(process.argv[2]) : {};
    
    const orchestrator = new SwarmOrchestrator(config);
    
    try {
        // Initialize swarm
        await orchestrator.initialize();
        
        // Spawn agents based on config
        const agentTypes = config.agents || ['researcher', 'coder', 'tester'];
        for (const type of agentTypes) {
            await orchestrator.spawnAgent(type);
        }
        
        // Orchestrate tasks
        const tasks = config.tasks || ['Analyze requirements', 'Implement solution', 'Test implementation'];
        for (const task of tasks) {
            await orchestrator.orchestrateTask(task);
        }
        
        // Get final status
        const status = await orchestrator.getStatus();
        
        console.log(JSON.stringify({
            status: 'success',
            swarmId: orchestrator.swarmId,
            agents: orchestrator.agents,
            tasks: orchestrator.tasks,
            finalStatus: status
        }, null, 2));
        
        // Cleanup
        await orchestrator.cleanup();
        
    } catch (error) {
        console.error('Error:', error);
        process.exit(1);
    }
}

main();
''',
                modules={
                    "agent_types.js": '''/**
 * Claude-Flow Agent Type Definitions
 */

module.exports = {
    RESEARCHER: {
        type: 'researcher',
        capabilities: ['search', 'analyze', 'summarize'],
        tools: ['web-search', 'file-read', 'pattern-match']
    },
    CODER: {
        type: 'coder',
        capabilities: ['implement', 'refactor', 'optimize'],
        tools: ['file-write', 'code-analysis', 'test-run']
    },
    TESTER: {
        type: 'tester',
        capabilities: ['test', 'validate', 'benchmark'],
        tools: ['test-framework', 'coverage-analysis', 'performance-test']
    },
    REVIEWER: {
        type: 'reviewer',
        capabilities: ['review', 'audit', 'suggest'],
        tools: ['code-review', 'security-scan', 'quality-check']
    },
    PLANNER: {
        type: 'planner',
        capabilities: ['plan', 'coordinate', 'prioritize'],
        tools: ['task-breakdown', 'dependency-analysis', 'resource-allocation']
    }
};
''',
                    "memory_manager.js": '''/**
 * Claude-Flow Memory Management
 */

class MemoryManager {
    constructor(namespace = 'default') {
        this.namespace = namespace;
    }
    
    async store(key, value, ttl = 3600) {
        const { exec } = require('child_process');
        const util = require('util');
        const execAsync = util.promisify(exec);
        
        const { stdout } = await execAsync(
            `npx claude-flow@alpha memory store --key "${key}" --value "${value}" --ttl ${ttl} --namespace ${this.namespace}`
        );
        
        return JSON.parse(stdout);
    }
    
    async retrieve(key) {
        const { exec } = require('child_process');
        const util = require('util');
        const execAsync = util.promisify(exec);
        
        const { stdout } = await execAsync(
            `npx claude-flow@alpha memory retrieve --key "${key}" --namespace ${this.namespace}`
        );
        
        return JSON.parse(stdout);
    }
    
    async search(pattern) {
        const { exec } = require('child_process');
        const util = require('util');
        const execAsync = util.promisify(exec);
        
        const { stdout } = await execAsync(
            `npx claude-flow@alpha memory search --pattern "${pattern}" --namespace ${this.namespace}`
        );
        
        return JSON.parse(stdout);
    }
}

module.exports = MemoryManager;
'''
                }
            )
        )
    
    @staticmethod
    def neural_agent() -> TemplateConfig:
        """Claude-Flow neural agent template"""
        return TemplateConfig(
            template_type=TemplateType.CLAUDE_FLOW_AGENT,
            metadata=TemplateMetadata(
                name="Claude-Flow Neural Agent",
                description="Neural-powered agent with pattern recognition and learning",
                version="2.0.0",
                tags=["claude-flow", "neural", "ai", "learning"],
                category="claude-flow"
            ),
            requirements=TemplateRequirements(
                runtime=RuntimeEnvironment.PYTHON_3_10,
                cpu_cores=4,
                memory_mb=4096,
                gpu_enabled=True,
                python_packages=[
                    "numpy>=1.24.0",
                    "torch>=2.0.0",
                    "transformers>=4.30.0",
                    "scikit-learn>=1.3.0",
                    "pandas>=2.0.0"
                ],
                node_packages=[
                    "claude-flow@alpha"
                ],
                env_vars={
                    "CLAUDE_FLOW_NEURAL": "true",
                    "MODEL_CACHE": "/tmp/models"
                }
            ),
            claude_flow=ClaudeFlowConfig(
                enable_neural=True,
                enable_memory=True,
                coordination_mode="autonomous"
            ),
            files=TemplateFiles(
                main_script='''#!/usr/bin/env python3
"""Claude-Flow Neural Agent"""

import os
import sys
import json
import subprocess
import numpy as np
from datetime import datetime

class NeuralAgent:
    """Neural-powered Claude-Flow agent"""
    
    def __init__(self, config):
        self.config = config
        self.agent_id = None
        self.patterns = []
        self.memory = {}
        
    def initialize(self):
        """Initialize neural agent"""
        print("Initializing Claude-Flow Neural Agent...")
        
        # Initialize neural components
        result = self._run_claude_flow([
            "neural", "init",
            "--type", "pattern-recognition",
            "--model", self.config.get("model", "default")
        ])
        
        self.agent_id = result.get("agentId")
        print(f"Neural agent initialized: {self.agent_id}")
        
    def train_patterns(self, data):
        """Train on patterns"""
        print("Training neural patterns...")
        
        # Prepare training data
        training_file = "/tmp/training_data.json"
        with open(training_file, 'w') as f:
            json.dump(data, f)
        
        # Train neural model
        result = self._run_claude_flow([
            "neural", "train",
            "--data", training_file,
            "--epochs", str(self.config.get("epochs", 10)),
            "--pattern-type", "coordination"
        ])
        
        self.patterns = result.get("patterns", [])
        return result
    
    def predict(self, input_data):
        """Make predictions"""
        print("Generating predictions...")
        
        # Prepare input
        input_file = "/tmp/input_data.json"
        with open(input_file, 'w') as f:
            json.dump(input_data, f)
        
        # Run prediction
        result = self._run_claude_flow([
            "neural", "predict",
            "--input", input_file,
            "--model", self.agent_id
        ])
        
        return result
    
    def learn_from_feedback(self, feedback):
        """Adaptive learning from feedback"""
        print("Learning from feedback...")
        
        result = self._run_claude_flow([
            "neural", "adapt",
            "--agent", self.agent_id,
            "--feedback", json.dumps(feedback)
        ])
        
        return result
    
    def store_memory(self, key, value):
        """Store in persistent memory"""
        result = self._run_claude_flow([
            "memory", "store",
            "--key", key,
            "--value", json.dumps(value),
            "--namespace", f"neural_{self.agent_id}"
        ])
        
        self.memory[key] = value
        return result
    
    def retrieve_memory(self, key):
        """Retrieve from memory"""
        result = self._run_claude_flow([
            "memory", "retrieve",
            "--key", key,
            "--namespace", f"neural_{self.agent_id}"
        ])
        
        return result.get("value")
    
    def _run_claude_flow(self, args):
        """Run Claude-Flow command"""
        cmd = ["npx", "claude-flow@alpha"] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr}")
            return {}
    
    def run(self):
        """Run the neural agent"""
        self.initialize()
        
        # Example workflow
        if "training_data" in self.config:
            self.train_patterns(self.config["training_data"])
        
        if "input_data" in self.config:
            predictions = self.predict(self.config["input_data"])
            
            # Store results in memory
            self.store_memory(
                f"predictions_{datetime.now().isoformat()}",
                predictions
            )
            
            return {
                "status": "success",
                "agent_id": self.agent_id,
                "predictions": predictions,
                "patterns": self.patterns
            }
        
        return {
            "status": "initialized",
            "agent_id": self.agent_id
        }

def main():
    config = {}
    if len(sys.argv) > 1:
        try:
            config = json.loads(sys.argv[1])
        except:
            pass
    
    agent = NeuralAgent(config)
    result = agent.run()
    
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
            )
        )