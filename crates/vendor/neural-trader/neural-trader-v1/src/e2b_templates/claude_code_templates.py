"""
Claude Code specific templates for E2B
"""

from .models import (
    TemplateConfig,
    TemplateType,
    TemplateMetadata,
    TemplateRequirements,
    TemplateFiles,
    ClaudeCodeConfig,
    RuntimeEnvironment
)


class ClaudeCodeTemplates:
    """Claude Code integration templates"""
    
    @staticmethod
    def sparc_developer() -> TemplateConfig:
        """Claude Code SPARC methodology template"""
        return TemplateConfig(
            template_type=TemplateType.CLAUDE_CODE_SPARC,
            metadata=TemplateMetadata(
                name="Claude Code SPARC Developer",
                description="SPARC methodology implementation with TDD and parallel execution",
                version="1.0.0",
                tags=["claude-code", "sparc", "tdd", "development"],
                category="claude-code"
            ),
            requirements=TemplateRequirements(
                runtime=RuntimeEnvironment.NODE_20,
                cpu_cores=4,
                memory_mb=2048,
                node_packages=[
                    "claude-flow@alpha",
                    "jest",
                    "typescript",
                    "@types/node"
                ],
                python_packages=[
                    "pytest",
                    "black",
                    "mypy",
                    "ruff"
                ],
                system_packages=[
                    "git",
                    "make"
                ],
                env_vars={
                    "SPARC_MODE": "enabled",
                    "TDD_MODE": "true",
                    "PARALLEL_EXECUTION": "true"
                }
            ),
            claude_code=ClaudeCodeConfig(
                sparc_enabled=True,
                tdd_mode=True,
                parallel_execution=True,
                max_todos=10,
                file_organization={
                    "/src": "Source code",
                    "/tests": "Test files",
                    "/docs": "Documentation",
                    "/config": "Configuration"
                },
                agent_spawning=True,
                memory_persistence=True
            ),
            files=TemplateFiles(
                main_script='''#!/usr/bin/env node
/**
 * Claude Code SPARC Developer Template
 */

const { exec } = require('child_process');
const util = require('util');
const fs = require('fs').promises;
const path = require('path');
const execAsync = util.promisify(exec);

class SPARCDeveloper {
    constructor(config) {
        this.config = config;
        this.phase = 'specification';
        this.todos = [];
        this.agents = {};
    }
    
    async runSPARCPhase(phase, task) {
        console.log(`Running SPARC ${phase} phase...`);
        
        const { stdout } = await execAsync(
            `npx claude-flow@alpha sparc run ${phase} "${task}"`
        );
        
        return JSON.parse(stdout);
    }
    
    async runFullSPARC(task) {
        const phases = [
            'specification',
            'pseudocode',
            'architecture',
            'refinement',
            'completion'
        ];
        
        const results = {};
        
        for (const phase of phases) {
            console.log(`\\n=== SPARC Phase: ${phase.toUpperCase()} ===`);
            results[phase] = await this.runSPARCPhase(phase, task);
            
            // Save phase output
            await this.savePhaseOutput(phase, results[phase]);
        }
        
        return results;
    }
    
    async runTDD(feature) {
        console.log(`Running TDD for feature: ${feature}`);
        
        // Run TDD workflow
        const { stdout } = await execAsync(
            `npx claude-flow@alpha sparc tdd "${feature}"`
        );
        
        const result = JSON.parse(stdout);
        
        // Create test files
        if (result.tests) {
            await this.createTestFiles(result.tests);
        }
        
        // Create implementation files
        if (result.implementation) {
            await this.createImplementationFiles(result.implementation);
        }
        
        return result;
    }
    
    async manageTodos(todos) {
        console.log(`Managing ${todos.length} todos...`);
        
        // Batch todo creation
        const todoCommands = todos.map((todo, index) => ({
            id: `todo_${index}`,
            content: todo,
            status: 'pending',
            priority: this.calculatePriority(todo)
        }));
        
        this.todos = todoCommands;
        
        // Execute todos in parallel where possible
        const parallelTodos = this.todos.filter(t => t.priority === 'low');
        const sequentialTodos = this.todos.filter(t => t.priority !== 'low');
        
        // Run parallel todos
        if (parallelTodos.length > 0) {
            await Promise.all(
                parallelTodos.map(todo => this.executeTodo(todo))
            );
        }
        
        // Run sequential todos
        for (const todo of sequentialTodos) {
            await this.executeTodo(todo);
        }
        
        return this.todos;
    }
    
    async executeTodo(todo) {
        console.log(`Executing: ${todo.content}`);
        todo.status = 'in_progress';
        
        try {
            // Simulate todo execution
            // In real implementation, this would spawn appropriate agents
            await new Promise(resolve => setTimeout(resolve, 100));
            
            todo.status = 'completed';
            console.log(`✅ Completed: ${todo.content}`);
        } catch (error) {
            todo.status = 'failed';
            todo.error = error.message;
            console.error(`❌ Failed: ${todo.content}`);
        }
        
        return todo;
    }
    
    async spawnAgent(type, task) {
        console.log(`Spawning ${type} agent for: ${task}`);
        
        const { stdout } = await execAsync(
            `npx claude-flow@alpha agent spawn --type ${type} --task "${task}"`
        );
        
        const agent = JSON.parse(stdout);
        this.agents[agent.id] = agent;
        
        return agent;
    }
    
    async savePhaseOutput(phase, output) {
        const dir = path.join('/tmp', 'sparc_output');
        await fs.mkdir(dir, { recursive: true });
        
        const file = path.join(dir, `${phase}.json`);
        await fs.writeFile(file, JSON.stringify(output, null, 2));
        
        console.log(`Saved ${phase} output to ${file}`);
    }
    
    async createTestFiles(tests) {
        const testDir = '/tmp/tests';
        await fs.mkdir(testDir, { recursive: true });
        
        for (const [name, content] of Object.entries(tests)) {
            const file = path.join(testDir, name);
            await fs.writeFile(file, content);
            console.log(`Created test file: ${file}`);
        }
    }
    
    async createImplementationFiles(implementation) {
        const srcDir = '/tmp/src';
        await fs.mkdir(srcDir, { recursive: true });
        
        for (const [name, content] of Object.entries(implementation)) {
            const file = path.join(srcDir, name);
            await fs.writeFile(file, content);
            console.log(`Created implementation file: ${file}`);
        }
    }
    
    calculatePriority(todo) {
        if (todo.includes('critical') || todo.includes('fix')) {
            return 'high';
        } else if (todo.includes('test') || todo.includes('document')) {
            return 'low';
        }
        return 'medium';
    }
}

async function main() {
    const config = process.argv.length > 2 ? JSON.parse(process.argv[2]) : {};
    
    const developer = new SPARCDeveloper(config);
    
    try {
        let result;
        
        if (config.mode === 'sparc') {
            // Run full SPARC workflow
            result = await developer.runFullSPARC(config.task || 'Build a REST API');
            
        } else if (config.mode === 'tdd') {
            // Run TDD workflow
            result = await developer.runTDD(config.feature || 'User authentication');
            
        } else if (config.todos) {
            // Manage todos
            result = await developer.manageTodos(config.todos);
            
        } else {
            // Default: spawn agents for tasks
            const tasks = config.tasks || ['Research requirements', 'Write code', 'Test'];
            
            for (const task of tasks) {
                await developer.spawnAgent('coder', task);
            }
            
            result = {
                agents: developer.agents,
                message: 'Agents spawned successfully'
            };
        }
        
        console.log(JSON.stringify({
            status: 'success',
            result: result,
            todos: developer.todos,
            agents: developer.agents
        }, null, 2));
        
    } catch (error) {
        console.error('Error:', error);
        process.exit(1);
    }
}

main();
''',
                modules={
                    "sparc_phases.js": '''/**
 * SPARC Phase Definitions
 */

module.exports = {
    SPECIFICATION: {
        name: 'specification',
        description: 'Requirements analysis and specification',
        outputs: ['requirements.md', 'acceptance_criteria.md']
    },
    PSEUDOCODE: {
        name: 'pseudocode',
        description: 'Algorithm design and pseudocode',
        outputs: ['algorithms.md', 'data_structures.md']
    },
    ARCHITECTURE: {
        name: 'architecture',
        description: 'System design and architecture',
        outputs: ['architecture.md', 'component_diagram.md']
    },
    REFINEMENT: {
        name: 'refinement',
        description: 'TDD implementation and refinement',
        outputs: ['tests/', 'src/']
    },
    COMPLETION: {
        name: 'completion',
        description: 'Integration and finalization',
        outputs: ['README.md', 'documentation/']
    }
};
''',
                    "todo_manager.js": '''/**
 * Todo Management for Claude Code
 */

class TodoManager {
    constructor(maxTodos = 10) {
        this.maxTodos = maxTodos;
        this.todos = [];
    }
    
    addTodo(content, priority = 'medium') {
        const todo = {
            id: `todo_${Date.now()}`,
            content: content,
            priority: priority,
            status: 'pending',
            created: new Date().toISOString()
        };
        
        this.todos.push(todo);
        
        // Keep only maxTodos
        if (this.todos.length > this.maxTodos) {
            this.todos = this.todos.slice(-this.maxTodos);
        }
        
        return todo;
    }
    
    updateStatus(id, status) {
        const todo = this.todos.find(t => t.id === id);
        if (todo) {
            todo.status = status;
            todo.updated = new Date().toISOString();
        }
        return todo;
    }
    
    getPending() {
        return this.todos.filter(t => t.status === 'pending');
    }
    
    getInProgress() {
        return this.todos.filter(t => t.status === 'in_progress');
    }
    
    getCompleted() {
        return this.todos.filter(t => t.status === 'completed');
    }
}

module.exports = TodoManager;
'''
                }
            )
        )
    
    @staticmethod
    def code_reviewer() -> TemplateConfig:
        """Claude Code reviewer template"""
        return TemplateConfig(
            template_type=TemplateType.CLAUDE_CODE_REVIEWER,
            metadata=TemplateMetadata(
                name="Claude Code Reviewer",
                description="Automated code review with Claude Code standards",
                version="1.0.0",
                tags=["claude-code", "review", "quality", "testing"],
                category="claude-code"
            ),
            requirements=TemplateRequirements(
                runtime=RuntimeEnvironment.PYTHON_3_10,
                cpu_cores=2,
                memory_mb=1024,
                python_packages=[
                    "ast",
                    "pylint",
                    "black",
                    "mypy",
                    "ruff",
                    "bandit",
                    "pytest",
                    "coverage"
                ],
                env_vars={
                    "REVIEW_MODE": "comprehensive",
                    "AUTO_FIX": "false"
                }
            ),
            claude_code=ClaudeCodeConfig(
                sparc_enabled=False,
                tdd_mode=False,
                parallel_execution=True,
                github_integration=True
            ),
            files=TemplateFiles(
                main_script='''#!/usr/bin/env python3
"""Claude Code Reviewer Template"""

import os
import sys
import json
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Any

class CodeReviewer:
    """Automated code reviewer following Claude Code standards"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.issues = []
        self.suggestions = []
        self.metrics = {}
        
    def review_file(self, filepath: str) -> Dict[str, Any]:
        """Review a single file"""
        print(f"Reviewing: {filepath}")
        
        file_issues = []
        
        # Check file exists
        if not Path(filepath).exists():
            return {"error": f"File not found: {filepath}"}
        
        # Get file extension
        ext = Path(filepath).suffix
        
        if ext == '.py':
            file_issues.extend(self.review_python(filepath))
        elif ext in ['.js', '.ts']:
            file_issues.extend(self.review_javascript(filepath))
        elif ext in ['.md', '.txt']:
            file_issues.extend(self.review_documentation(filepath))
        
        return {
            "file": filepath,
            "issues": file_issues,
            "metrics": self.calculate_metrics(filepath)
        }
    
    def review_python(self, filepath: str) -> List[Dict[str, Any]]:
        """Review Python code"""
        issues = []
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # AST analysis
        try:
            tree = ast.parse(content)
            issues.extend(self.analyze_ast(tree, filepath))
        except SyntaxError as e:
            issues.append({
                "type": "syntax_error",
                "severity": "critical",
                "line": e.lineno,
                "message": str(e)
            })
        
        # Run linters
        issues.extend(self.run_pylint(filepath))
        issues.extend(self.run_mypy(filepath))
        issues.extend(self.run_ruff(filepath))
        issues.extend(self.run_bandit(filepath))
        
        # Check Claude Code standards
        issues.extend(self.check_claude_standards(content))
        
        return issues
    
    def analyze_ast(self, tree: ast.AST, filepath: str) -> List[Dict[str, Any]]:
        """Analyze Python AST"""
        issues = []
        
        # Check for complex functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self.calculate_complexity(node)
                if complexity > 10:
                    issues.append({
                        "type": "complexity",
                        "severity": "warning",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' has high complexity: {complexity}"
                    })
        
        return issues
    
    def calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
        return complexity
    
    def run_pylint(self, filepath: str) -> List[Dict[str, Any]]:
        """Run pylint"""
        try:
            result = subprocess.run(
                ["pylint", "--output-format=json", filepath],
                capture_output=True,
                text=True
            )
            if result.stdout:
                pylint_issues = json.loads(result.stdout)
                return [
                    {
                        "type": "pylint",
                        "severity": self.map_severity(issue["type"]),
                        "line": issue.get("line", 0),
                        "message": issue["message"]
                    }
                    for issue in pylint_issues
                ]
        except Exception as e:
            print(f"Pylint error: {e}")
        return []
    
    def run_mypy(self, filepath: str) -> List[Dict[str, Any]]:
        """Run mypy type checker"""
        try:
            result = subprocess.run(
                ["mypy", filepath],
                capture_output=True,
                text=True
            )
            if result.stdout:
                # Parse mypy output
                issues = []
                for line in result.stdout.split('\\n'):
                    if ':' in line and 'error:' in line:
                        parts = line.split(':')
                        if len(parts) >= 4:
                            issues.append({
                                "type": "type_error",
                                "severity": "error",
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "message": ':'.join(parts[3:]).strip()
                            })
                return issues
        except Exception as e:
            print(f"Mypy error: {e}")
        return []
    
    def run_ruff(self, filepath: str) -> List[Dict[str, Any]]:
        """Run ruff linter"""
        try:
            result = subprocess.run(
                ["ruff", "check", filepath, "--format=json"],
                capture_output=True,
                text=True
            )
            if result.stdout:
                ruff_output = json.loads(result.stdout)
                return [
                    {
                        "type": "style",
                        "severity": "info",
                        "line": issue.get("location", {}).get("row", 0),
                        "message": issue["message"]
                    }
                    for issue in ruff_output
                ]
        except Exception as e:
            print(f"Ruff error: {e}")
        return []
    
    def run_bandit(self, filepath: str) -> List[Dict[str, Any]]:
        """Run bandit security checker"""
        try:
            result = subprocess.run(
                ["bandit", "-f", "json", filepath],
                capture_output=True,
                text=True
            )
            if result.stdout:
                bandit_output = json.loads(result.stdout)
                return [
                    {
                        "type": "security",
                        "severity": issue["issue_severity"].lower(),
                        "line": issue["line_number"],
                        "message": issue["issue_text"]
                    }
                    for issue in bandit_output.get("results", [])
                ]
        except Exception as e:
            print(f"Bandit error: {e}")
        return []
    
    def check_claude_standards(self, content: str) -> List[Dict[str, Any]]:
        """Check Claude Code specific standards"""
        issues = []
        lines = content.split('\\n')
        
        # Check for comments (Claude Code prefers minimal comments)
        for i, line in enumerate(lines, 1):
            if '#' in line and not line.strip().startswith('#'):
                comment = line.split('#')[1].strip()
                if len(comment) > 0 and not comment.startswith('TODO') and not comment.startswith('FIXME'):
                    issues.append({
                        "type": "claude_standard",
                        "severity": "info",
                        "line": i,
                        "message": "Avoid unnecessary comments (Claude Code standard)"
                    })
        
        # Check file length
        if len(lines) > 500:
            issues.append({
                "type": "claude_standard",
                "severity": "warning",
                "line": 0,
                "message": f"File exceeds 500 lines ({len(lines)} lines). Consider splitting into modules."
            })
        
        return issues
    
    def review_javascript(self, filepath: str) -> List[Dict[str, Any]]:
        """Review JavaScript/TypeScript code"""
        # Simplified JS review
        return []
    
    def review_documentation(self, filepath: str) -> List[Dict[str, Any]]:
        """Review documentation files"""
        # Simplified doc review
        return []
    
    def calculate_metrics(self, filepath: str) -> Dict[str, Any]:
        """Calculate code metrics"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        return {
            "total_lines": len(lines),
            "code_lines": len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
            "comment_lines": len([l for l in lines if l.strip().startswith('#')]),
            "blank_lines": len([l for l in lines if not l.strip()])
        }
    
    def map_severity(self, pylint_type: str) -> str:
        """Map pylint message types to severity"""
        mapping = {
            'error': 'error',
            'warning': 'warning',
            'refactor': 'info',
            'convention': 'info'
        }
        return mapping.get(pylint_type.lower(), 'info')
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate review report"""
        return {
            "status": "completed",
            "total_issues": len(self.issues),
            "by_severity": {
                "critical": len([i for i in self.issues if i.get("severity") == "critical"]),
                "error": len([i for i in self.issues if i.get("severity") == "error"]),
                "warning": len([i for i in self.issues if i.get("severity") == "warning"]),
                "info": len([i for i in self.issues if i.get("severity") == "info"])
            },
            "issues": self.issues,
            "suggestions": self.suggestions,
            "metrics": self.metrics
        }

def main():
    config = {}
    if len(sys.argv) > 1:
        try:
            config = json.loads(sys.argv[1])
        except:
            pass
    
    reviewer = CodeReviewer(config)
    
    # Review files
    files = config.get("files", [])
    if not files and "directory" in config:
        # Find all Python files in directory
        directory = Path(config["directory"])
        files = list(directory.glob("**/*.py"))
    
    for filepath in files:
        result = reviewer.review_file(str(filepath))
        reviewer.issues.extend(result.get("issues", []))
        reviewer.metrics[str(filepath)] = result.get("metrics", {})
    
    # Generate report
    report = reviewer.generate_report()
    print(json.dumps(report, indent=2))
    
    return 0 if report["total_issues"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
'''
            )
        )