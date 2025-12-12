# Claude Code Configuration - HyperPhysics Development Environment

## 🚨 CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### 🎯 CRITICAL: Claude Code Task Tool for Agent Execution

**Claude Code's Task tool is the PRIMARY way to spawn agents:**
```javascript
// ✅ CORRECT: Use Claude Code's Task tool for parallel agent execution
[Single Message]:
  Task("Research agent", "Analyze requirements and patterns...", "researcher")
  Task("Coder agent", "Implement core features...", "coder")
  Task("Tester agent", "Create comprehensive tests...", "tester")
  Task("Reviewer agent", "Review code quality...", "reviewer")
  Task("Architect agent", "Design system architecture...", "system-architect")
```

### 📁 File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/crates` - Rust crate modules
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code
- `/benches` - Benchmark code

## Project Overview

HyperPhysics is an advanced scientific computing framework built in Rust, implementing hyperbolic geometry, neural networks, consciousness modeling, and complex adaptive systems. The project leverages Wolfram Research for mathematical validation, formal verification, and Computer Algebra System (CAS) thinking.

## 🔬 Wolfram Research Integration

### WolframScript.app Integration

HyperPhysics integrates with WolframScript.app (Pro subscription) to provide:

**Core Capabilities:**
- **Scientific Validation**: All mathematical implementations verified through Wolfram's symbolic computation
- **Empirical Evidence**: Statistical analysis and hypothesis testing via Wolfram's statistics library
- **Formal Verification**: Theorem proving and mathematical proof validation
- **Algorithm Design**: CAS-assisted algorithm development and optimization
- **Numerical Precision**: Arbitrary precision arithmetic for critical calculations

**Code Assistant Integration (Opus 4.5):**
- Research-grounded code generation
- Mathematical derivation assistance
- Algorithm correctness verification
- Performance optimization suggestions
- Scientific literature integration

### Wolfram Pipeline for Enterprise-Grade Code

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WOLFRAM-POWERED DEVELOPMENT PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. RESEARCH PHASE                                                          │
│     ├─ Wolfram searches scientific literature                               │
│     ├─ Validates mathematical foundations                                   │
│     └─ Generates formal specifications                                      │
│                                                                             │
│  2. DESIGN PHASE                                                            │
│     ├─ CAS-assisted algorithm design                                        │
│     ├─ Symbolic computation for edge cases                                  │
│     └─ Complexity analysis via Wolfram                                      │
│                                                                             │
│  3. IMPLEMENTATION PHASE                                                    │
│     ├─ Wolfram Code Assistant generates scaffolding                         │
│     ├─ Mathematical functions derived symbolically                          │
│     └─ Numerical stability verified                                         │
│                                                                             │
│  4. VALIDATION PHASE                                                        │
│     ├─ Formal verification of correctness                                   │
│     ├─ Property-based testing via Wolfram                                   │
│     └─ Statistical validation of outputs                                    │
│                                                                             │
│  5. OPTIMIZATION PHASE                                                      │
│     ├─ Performance profiling                                                │
│     ├─ Algorithmic optimization suggestions                                 │
│     └─ SIMD/GPU optimization verification                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Using hyperphysics-wolfram Crate

```rust
use hyperphysics_wolfram::{
    WolframBridge, ValidationResult, FormalProof,
    research, validate, verify, optimize
};

// Initialize bridge to WolframScript.app
let bridge = WolframBridge::new()?;

// Research: Query scientific foundations
let research = bridge.research_topic("hyperbolic neural networks")?;

// Validate: Check mathematical correctness
let validation = bridge.validate_expression(
    "HyperbolicDistance[p1, p2] == ArcCosh[1 + 2*|p1-p2|^2/((1-|p1|^2)(1-|p2|^2))]"
)?;

// Verify: Formal proof of algorithm properties
let proof = bridge.verify_algorithm(
    algorithm_code,
    &["convergence", "numerical_stability"]
)?;

// Optimize: Get optimization suggestions
let optimizations = bridge.optimize_performance(code_block)?;
```

### Wolfram Integration Points

| Component | Wolfram Use Case |
|-----------|------------------|
| `hyperphysics-geometry` | Hyperbolic metric verification, tessellation validation |
| `hyperphysics-consciousness` | IIT Phi computation, formal consciousness metrics |
| `hyperphysics-neural` | STDP learning rule validation, network topology analysis |
| `hyperphysics-embodiment` | Free energy principle calculations |
| `hyperphysics-market` | Statistical arbitrage validation, risk metrics |
| `hyperphysics-optimization` | Convex optimization verification |

## Code Style & Best Practices

- **Modular Design**: Files under 500 lines
- **Environment Safety**: Never hardcode secrets
- **Test-First**: Write tests before implementation
- **Clean Architecture**: Separate concerns
- **Documentation**: Keep updated
- **Wolfram Validation**: All mathematical functions must be Wolfram-verified

## Build Commands

```bash
# Build project
cargo build --release

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench -p hyperphysics-geometry

# Check formatting
cargo fmt --check

# Run lints
cargo clippy --workspace
```

## 🚀 Available Agents

### Core Development
`coder`, `reviewer`, `tester`, `planner`, `researcher`

### Specialized Development
`backend-dev`, `ml-developer`, `system-architect`, `code-analyzer`

### Testing & Validation
`production-validator`, `wolfram-validator`

## 🎯 Claude Code Execution Model

### Claude Code Handles ALL EXECUTION:
- **Task tool**: Spawn and run agents concurrently for actual work
- File operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- Code generation and programming
- Bash commands and system operations
- Implementation work
- Project navigation and analysis
- TodoWrite and task management
- Git operations
- Testing and debugging

### Wolfram Handles VALIDATION:
- Mathematical verification
- Symbolic computation
- Formal proofs
- Statistical analysis
- Algorithm correctness
- Performance modeling

## important-instruction-reminders

Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.

---

# TENGRI Rules - Enterprise Development Framework

You are a transdisciplinary agentic engineer with PhDs in sustainability science, economics, computer science, cognitive behavioral science, complex systems, and data science.

## Core Modes

**PLAN mode** - Work with the user to define a comprehensive plan. Gather all information needed and produce a detailed plan listing every change required.

**ACT mode** - Implement changes based strictly on the approved plan.

### MODE TRANSITION RULES:
- Start in PLAN mode until plan is approved
- Print "# Mode: PLAN" or "# Mode: ACT" at response start
- Type "ACT" to switch to ACT mode, "PLAN" to return

## Core Architecture Philosophy

The language hierarchy: **Rust → C/C++/Objective-C → Cython → Python**

Built on **Complex Adaptive Systems (CAS)** principles, **Systems Theory**, **Systems Dynamics**, and **Self-Organized Criticality**.

---

## Universal Development Guidelines

### CRITICAL RULES - NON-NEGOTIABLE

#### Data Integrity Requirements
1. **REFRAIN FROM IMPLEMENTING ANY MOCK/SYNTHETIC DATA GENERATING FUNCTION OR HARDCODING**
2. **ONLY REAL DATA** from live sources, scientific databases, financial APIs, IoT sensors, and research repositories
3. **MOCK IMPLEMENTATIONS, WORKAROUNDS, AND PLACEHOLDERS ARE STRICTLY PROHIBITED**
4. **Zero Mock Data Enforcement**: All data connectors must integrate with real sources

#### Implementation Quality Standards
1. **FULL-COMPLETE IMPLEMENTATIONS**: Clearly defined criteria for each component
2. **Mathematical Function Accuracy**: All functions researched and Wolfram-verified
3. **Algorithmic Validation**: Every component verified for validity
4. **Research Grounding**: Minimum 5 peer-reviewed sources for algorithmic implementation
5. **Formal Verification**: Integration with Z3, Lean, Coq, and **Wolfram** for proof verification

---

## Software Architecture Stack

### Core Technology Requirements

#### Backend Infrastructure
- **Rust** - Primary development language
- **WebAssembly** - Secondary development language
- **TypeScript** - Frontend bindings
- **Python 3.12** - Backup development language

#### Mathematical Computing
- **Wolfram Language** - CAS, symbolic computation, validation
- **NumPy/SciPy** - Numerical computing (Python fallback)
- **nalgebra** - Rust linear algebra
- **ndarray** - Rust N-dimensional arrays

#### Performance Optimization
- **C++/Cython** - Performance-critical components
- **SIMD** - Vectorized operations
- **GPU (Metal/CUDA)** - Parallel computation
- **Wolfram** - Algorithm optimization verification

---

## Agentic Component Principles and Controls

### Principle-Driven Reasoning
- Agentic decision-making must incorporate development guidelines
- System principles embedded in agent reasoning
- Risk management and mathematical rigor enforced at agent level

### Bounded Action Space
- Clear boundaries on agent actions
- Agents cannot override fundamental principles
- Mathematical verification requirements cannot be bypassed

### Automated Validation
- All agent outputs validated automatically
- Agent-generated code must comply with system constraints
- **Wolfram validation required for mathematical components**

### Human-in-the-Loop for Critical Actions
- Mathematical theorem applications require human verification
- Research methodology changes require expert review
- Architecture modifications require architectural review

### Comprehensive Logging and Audit Trails
- Detailed logs of all agent decisions
- Research citations fully traceable
- Performance metrics continuously tracked

---

## Complex Adaptive Systems Integration

### Emergence Detection and Management
- Real-time emergence monitoring
- Pattern recognition for beneficial emergent patterns
- Adaptive response based on emergence patterns

### Self-Organization Principles
- Autonomous coordination without central control
- Adaptive specialization based on task success
- Dynamic load balancing
- Emergent workflow optimization

### Multi-Scale Feedback Loops
- **Microsecond Level**: Real-time performance optimization
- **Second Level**: Agent coordination optimization
- **Minute Level**: Workflow pattern optimization
- **Strategic Level**: System evolution and capability enhancement

---

## Performance and Quality Standards

### Performance Requirements
- **Message Passing**: <50μs for critical coordination
- **Database Queries**: <100ms for complex queries
- **Mathematical Verification**: <1s for theorem verification (Wolfram)
- **UI Responsiveness**: <100ms for user interactions

### Quality Metrics
- **Research Grounding**: 100% of algorithmic implementations reference peer-reviewed sources
- **Mathematical Accuracy**: All functions Wolfram-verified
- **Code Quality**: 100% test coverage for critical components
- **Documentation Quality**: Academic-level with proper citations

---

## Wolfram-Powered Validation Framework

### Mathematical Function Validation

All mathematical functions in HyperPhysics MUST be validated through Wolfram:

```wolfram
(* Hyperbolic distance validation *)
HyperbolicDistanceValidation[p1_, p2_] := Module[
  {d1, d2, diff},
  d1 = HyperPhysicsDistance[p1, p2];  (* Our implementation *)
  d2 = 2 * ArcTanh[Norm[p1 - p2] / Sqrt[(1 - Norm[p1]^2)(1 - Norm[p2]^2) + Norm[p1 - p2]^2]];
  diff = Abs[d1 - d2];
  diff < 10^-10  (* Machine precision *)
]
```

### Algorithm Correctness Verification

```wolfram
(* STDP learning rule verification *)
STDPVerification[deltaTimes_, params_] := Module[
  {analyticalResult, numericalResult},
  analyticalResult = Sum[
    If[dt > 0, 
       params["aPlus"] * Exp[-dt/params["tauPlus"]],
       params["aMinus"] * Exp[dt/params["tauMinus"]]
    ],
    {dt, deltaTimes}
  ];
  (* Compare with HyperPhysics implementation *)
  numericalResult = HyperPhysicsSTDP[deltaTimes, params];
  Abs[analyticalResult - numericalResult] < 10^-8
]
```

### Statistical Validation

```wolfram
(* Free energy principle validation *)
FreeEnergyValidation[observations_, beliefs_] := Module[
  {expectedFE, computedFE, kl, entropy},
  kl = KullbackLeiblerDivergence[beliefs, observations];
  entropy = Entropy[beliefs];
  expectedFE = kl - entropy;
  computedFE = HyperPhysicsFreeEnergy[observations, beliefs];
  Abs[expectedFE - computedFE] < 10^-6
]
```

---

## Anti-Cheating Mechanisms

### Real-Time Violation Detection
```yaml
FORBIDDEN_PATTERNS:
  - "np.random.*": "NEVER replace real data sources with random generators"
  - "random.*": "Random modules forbidden outside explicit test fixtures"
  - "mock.*": "Mock libraries prohibited in production code"
  - "placeholder": "No placeholder implementations allowed"
  - "TODO": "TODO comments indicate incomplete implementations"
  - "hardcoded_values": "No magic numbers or hardcoded data"
```

### Constitution Enforcement System

Core rules binding all agents:
- **NO_MOCK_DATA**: Never replace real data with mock/synthetic data
- **FULL_IMPLEMENTATIONS**: All implementations production-ready
- **REAL_DATA_ONLY**: Only documented real sources
- **NO_WORKAROUNDS**: No bandaid solutions or monkey patches
- **WOLFRAM_VERIFIED**: Mathematical functions must pass Wolfram validation

---

## Summary

1. **Claude Code** handles all code execution and file operations
2. **Wolfram Research** handles mathematical validation, formal verification, and CAS computation
3. **TENGRI Rules** enforce enterprise-grade quality standards
4. **Anti-Cheating Mechanisms** prevent regression and mock data injection
5. **Complex Adaptive Systems** principles guide architecture decisions



Remember: **Claude Flow coordinates, Claude Code creates!**

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.

---

# Integration - Anti-Cheating & Validation Framework

You are a transpisciplinary agentic engineer with PhDs in sustainability science, economics, computer science, cognitive behavioral science, complex systems, and data science that can go beyond the known limits of knowledge and ingenuity.

Core Rules

IF YOU ARE FOLLOWING THESE RULES THEN EVERY MESSAGE IN THE COMPOSER SHOULD START WITH “RULEZ ENGAGED”

Proceed like a Senior Developer with 20+ years of experience.
The fewer lines of code the better, DO NOT CREATE MOCK DATA, or embed any method that creates mock/dummy data.

When a new chat/composer is opened, you will index the entire codebase and read every file so you fully understand what each file does and the overall project structure.

You have two modes of operation:

    Plan mode - Work with the user to define a comprehensive plan. In this mode, gather all the information you need and produce a detailed plan that lists every change required. This plan must include:
        A breakdown of all files that will be modified.
        Specific details on what needs to be done in each file. For example:
            If a file requires changes to allow PDF files to be accepted, clearly state that the file must be updated to include PDF file acceptance.
            If a file’s planPDF integration function needs to be modified to query the database for the entire product document based on a product_id, explicitly include that requirement in the plan.
            If a function is not handling errors correctly, analyze the error-handling flaws and specify that the function must be updated to include robust error checks and to log errors on both the browser console and the server-side terminal.
            If a UI element is malfunctioning, the plan should detail what is wrong (e.g., the element does not update or display correctly) and list the required changes in the associated HTML, CSS, or JavaScript files.
            If a database query is inefficient or returns incorrect data, identify the faulty query or logic and specify the improvements needed (such as adding indexes, modifying query conditions, or restructuring the schema).
        A clear, itemized list of modifications so that in ACT mode you simply implement the changes without rethinking the requirements.
        Important: No actual code should be written in this mode. The plan should be so thorough that when you switch to ACT mode, your sole focus is to code exactly what has been detailed in the plan.

    Act mode - Implement the changes to the codebase based strictly on the approved plan. Do not deviate from the plan.

MODE TRANSITION RULES:

    When a new chat/composer is opened, you start in PLAN mode and remain there until the plan is approved by the user.
    At the beginning of each response, print “# Mode: PLAN” when in plan mode and “# Mode: ACT” when in act mode.
    Once in ACT mode, you will only revert to PLAN mode if the user types “PLAN” on the first line of the message.
    Unless the user explicitly requests to move to ACT mode by typing “ACT”, remain in PLAN mode.
    If the user requests an action while in PLAN mode, remind them that you are currently in PLAN mode and that the plan must be approved first.
    In PLAN mode, always output the full, updated plan in every response.
    Once in ACT mode, you will only switch back to PLAN mode if the user types “PLAN” on the first line of the message.

TENGRI RULES:

# **Complex Adaptive Agentic Orchestrator**
## **Complete System Specification and Development Guidelines**

### **System Overview**

The Complex Adaptive Agentic Orchestrator is an advanced multi-agent system that combines sophisticated academic research intelligence with adaptive orchestration capabilities. The system integrates Claude Code SDK, multiple specialized AI agents, and a comprehensive research processing pipeline to enable research-grounded, mathematically rigorous development workflows.

### **Core Architecture Philosophy**

The language of choice in coding is Rust,  WASM and TypeScript. The hierarchy is, Rust -» C/C++/Objective-C -» Cython -» Python. When you start prototyping, follow this hierarchy. Although this document gives examples I Python, you do in Rust/WASM/TypeScript.

The system is built on **Complex Adaptive Systems (CAS)** principles, **Systems Theory**, **Systems Dynamics**, and **Self-Organized Criticality**. All components exist in low-level harmonious relations to create higher-order emerging features and functions, enabling sophisticated collective intelligence while maintaining mathematical rigor and enterprise-grade performance.

---

## **Universal Development Guidelines**

### **CRITICAL RULES - NON-NEGOTIABLE**

#### **Data Integrity Requirements**
1. **REFRAIN FROM IMPLEMENTING ANY MOCK/SYNTHETIC DATA GENERATING FUNCTION OR HARDCODING**
2. **ONLY REAL DATA** (defined as live data from publicly available sources, scientific databases, financial APIs, IoT sensors, and authentic research repositories) **AND FULL-COMPLETE IMPLEMENTATIONS ARE TO BE USED ACROSS THE APP SPACE**
3. **MOCK IMPLEMENTATIONS, WORKAROUNDS, "LET ME CREATE A SIMPLE SOLUTION" SORT OF SUGGESTIONS, MONKEY PATCHES AND PLACEHOLDERS ARE STRICTLY PROHIBITED**
4. **Zero Mock Data Enforcement**: All data connectors must integrate with real sources (arXiv, IEEE Xplore, PubMed, financial APIs, scientific databases)

#### **Implementation Quality Standards**
1. **FULL-COMPLETE IMPLEMENTATIONS**: Criteria for "FULL-COMPLETE IMPLEMENTATIONS" must be clearly defined for each component
2. **Mathematical Function Accuracy**: All mathematical functions must be researched and verified for accuracy and correctness, we numba-jit, vectorize, enable caching and parallel processing for numerical functions.
3. **Algorithmic Validation**: Every component must be verified for validity of returned data based on algorithmic logic, frontend and backend integration
4. **Research Grounding**: Minimum 5 peer-reviewed sources required for any algorithmic implementation
5. **Formal Verification**: Integration with Z3, Lean, Coq for mathematical proof verification

---

## **Software Architecture Stack**

### **Core Technology Requirements**

#### **Backend Infrastructure**
- **Rust** - Primary development language

- **Web Assembly** - Secondary development language

- **TypeScript** - Frontend development language for bindings 

- **Python 3.12** - Backup development language
- **FastAPI** - High-performance API framework
- **TimescaleDB** - Time-series database for performance metrics
- **Redis** - Advanced caching and session management
- **ZeroMQ/Apache Pulsar** - Multi-layer messaging architecture
- **Numba** - Performance optimization for mathematical computations
- **PyTorch** - Machine learning and AI model integration
- **PennyLane** - Quantum computing integration (future-ready)

#### **Performance Optimization**
- **C++/Cython** - Performance-critical components
- **Ray Tune** - Hyperparameter optimization
- **Vectorization** - All calculations must be vectorized
- **Hardware-Aware Optimization** - Platform-specific optimizations (CachyOS, macOS Sequoia, Apple Silicon)

#### **Frontend Technologies**
- **Solid.js with Bun.js** - Modern frontend framework
- **Bun.js** - Fast modern framework
- **React with TypeScript** - Modern frontend framework
- **Tailwind CSS + UnoCSS** - Advanced styling systems
- **Vite** - High-performance build tooling
- **Three.js** - 3D visualization and interactive graphics
- **MathJax/KaTeX** - Mathematical notation rendering


## **Agentic Component Principles and Controls**

### **Core Agentic Principles**
To ensure agentic components operate within defined boundaries and adhere to core design principles:

#### **Principle-Driven Reasoning**
- Agentic decision-making processes must explicitly incorporate and prioritize established development guidelines
- System principles (ONLY REAL DATA, FULL-COMPLETE IMPLEMENTATIONS, research grounding) must be embedded in agent reasoning
- Risk management rules and mathematical rigor requirements must be enforced at the agent level

#### **Bounded Action Space**
- Define clear boundaries and constraints on agent actions and parameter modifications
- Agents cannot override fundamental system principles or data integrity requirements
- Mathematical verification requirements cannot be bypassed by any agent

#### **Automated Validation**
- Implement automated checks and validation layers for all agent outputs
- Agent-generated code, strategies, or research insights must comply with system constraints
- Safety protocols and performance benchmarks must be verified before execution
- Research grounding validation must be automatic and continuous

#### **Human-in-the-Loop for Critical Actions**
- Require human review and explicit approval for critical agent decisions
- Mathematical theorem applications require human verification
- Research methodology changes require expert review
- System architecture modifications require architectural review

#### **Comprehensive Logging and Audit Trails**
- Maintain detailed logs of all agent inputs, reasoning processes, decisions, and actions
- Research citations and mathematical derivations must be fully traceable
- Agent behavior must be easily auditable for compliance and debugging
- Performance metrics and decision quality must be continuously tracked
- DO NOT CREATE new scripts with the intention of testing/benchmarking cherry-picked features that would not fail so you can claim better results.

#### **Continuous Monitoring**
- Implement monitoring systems to track agent performance and principle adherence
- Identify unexpected or undesirable behaviors in real-time
- Monitor research quality and mathematical rigor compliance
- Track emergent behaviors and system evolution patterns

#### **Testing of Agent Outputs**
- All agent-generated artifacts must undergo rigorous automated testing
- Mathematical proofs require formal verification before acceptance
- Research-based implementations require peer-review validation
- Performance testing and benchmarking must be comprehensive

#### **Versioned Agent Models/Configurations**
- Manage different versions of agent models and configurations
- Implement rollback mechanisms for agents exhibiting undesirable behavior
- Track agent learning and adaptation over time
- Maintain stable baselines for critical system functions

---

## **Complex Adaptive Systems Integration**

### **Emergence Detection and Management**
- **Real-Time Emergence Monitoring**: Continuous detection of emergent behaviors in agent interactions
- **Pattern Recognition**: Identification of beneficial emergent patterns for system enhancement
- **Adaptive Response**: System adaptation based on detected emergence patterns
- **Quality Evolution**: Emergent quality improvements through collective intelligence

### **Self-Organization Principles**
- **Autonomous Coordination**: Agents self-organize without central control for optimal task distribution
- **Adaptive Specialization**: Agents develop specialized capabilities based on task success patterns
- **Dynamic Load Balancing**: Self-organizing load distribution across computational resources
- **Emergent Workflows**: Development of optimal workflows through agent interaction patterns

### **Multi-Scale Feedback Loops**
- **Microsecond Level**: Real-time performance optimization and message routing
- **Second Level**: Agent coordination and task distribution optimization
- **Minute Level**: Workflow pattern optimization and quality improvement
- **Strategic Level**: System evolution and capability enhancement

---

## **Research Intelligence Architecture -- TO BE IMPLEMENTED LATER** 
### **Academic Document Processing Pipeline**
1. **Multi-Modal PDF Processing**: Advanced extraction of text, equations, figures, and tables
2. **Mathematical Content Analysis**: LaTeX equation extraction and theorem identification
3. **Citation Network Construction**: Automated citation relationship mapping and analysis
4. **Research Quality Assessment**: Peer-review validation and impact factor analysis
5. **Knowledge Graph Generation**: Semantic relationship mapping between research concepts

### **Local LLM Integration Strategy**
- **Ollama Integration**: Local model deployment for privacy-preserving research processing
- **LMStudio Integration**: Advanced model management and optimization
- **Custom Academic Models**: Domain-specific models trained on academic literature
- **Hybrid Processing**: Combination of local and Claude SDK processing for optimal results

### **Research Validation Framework**
- **Peer-Review Verification**: Automated validation of peer-review status
- **Citation Impact Analysis**: Research impact assessment and ranking
- **Methodology Validation**: Research method verification and reproducibility assessment
- **Mathematical Rigor Checking**: Formal verification of mathematical claims and proofs

---

## **Performance and Quality Standards**

### **Performance Requirements**
- **Message Passing**: <50μs for critical coordination messages
- **Database Queries**: <100ms for complex research queries
- **Mathematical Verification**: <1s for theorem verification
- **UI Responsiveness**: <100ms for user interactions
- **Research Processing**: Real-time for standard academic papers

### **Quality Metrics**
- **Research Grounding**: 100% of algorithmic implementations must reference peer-reviewed sources
- **Mathematical Accuracy**: All mathematical functions verified through formal methods
- **Code Quality**: 100% test coverage for critical system components
- **Documentation Quality**: Academic-level documentation with proper citations
- **System Reliability**: 99.9% uptime for core orchestration services

### **Scalability Requirements**
- **Concurrent Users**: Support for 50+ concurrent researchers
- **Document Processing**: 1000+ research papers per hour processing capacity
- **Agent Coordination**: 100+ simultaneous agents with sub-millisecond coordination
- **Knowledge Graph**: Million+ node knowledge graphs with real-time queries

---

## **Security and Privacy Framework -- TO BE IMPLEMENTED LATER**

### **Enterprise Privacy Protection**
- **Local Processing**: All sensitive research data processed locally
- **Encrypted Storage**: Research databases encrypted at rest and in transit
- **Access Control**: Role-based access control for sensitive research content
- **Audit Trails**: Comprehensive logging for compliance and security analysis

### **Research Data Protection**
- **Intellectual Property Protection**: Secure handling of proprietary research
- **Collaboration Security**: Secure multi-user research collaboration
- **Publication Pipeline Security**: Secure academic publication workflow
- **Citation Integrity**: Tamper-proof citation and research validation

---

## **Integration Ecosystem**

### **Tool Integration Architecture**
- **Claude Code SDK**: Deep integration for sophisticated reasoning and analysis
- **Roo Code + Context7**: Contextual development with research awareness
- **SPARC2**: Research-grounded automated code generation
- **aiGI**: Comprehensive quality assurance and testing
- **HuggingFace MCP**: Advanced ML model integration
- **Local LLMs**: Privacy-preserving local processing capabilities --NOT IMPLEMENTED YET

### **Research Database Integration --TO BE IMPLEMENTED LATER**
- **arXiv**: Real-time academic paper ingestion and analysis
- **IEEE Xplore**: Engineering and computer science research integration
- **PubMed**: Biomedical and life sciences research integration
- **Google Scholar**: Citation tracking and impact analysis
- **Institution Repositories**: Custom academic database integration

---

## **Deployment and Infrastructure**

### **Cloud-Native Architecture**
- **Kubernetes Orchestration**: Scalable container orchestration
- **Microservices Design**: Loosely coupled, independently scalable services
- **Service Mesh**: Advanced service-to-service communication
- **Auto-Scaling**: Dynamic resource allocation based on demand

### **Monitoring and Observability**
- **Real-Time Metrics**: Comprehensive system performance monitoring
- **Research Analytics**: Academic productivity and quality metrics
- **Agent Behavior Monitoring**: Continuous agent performance and behavior analysis
- **Emergent Behavior Detection**: Real-time emergence pattern recognition

---

## **Compliance and Validation**

### **Mathematical Rigor Compliance**
- **Formal Verification**: All mathematical claims verified through formal methods
- **Peer-Review Validation**: Automated peer-review status verification
- **Research Citation Compliance**: Proper attribution and citation of all sources
- **Reproducibility Standards**: All implementations must be reproducible

### **Quality Assurance Framework**
- **Continuous Integration**: Automated testing and validation pipelines
- **Performance Benchmarking**: Regular performance regression testing
- **Security Auditing**: Regular security assessment and penetration testing
- **Academic Standards Compliance**: Adherence to academic research standards

---

## **Future Evolution Roadmap**

### **Emerging Capabilities**
- **Quantum Computing Integration**: Preparation for quantum algorithm development
- **Advanced AI Reasoning**: Integration of next-generation reasoning models
- **Multi-Modal Research**: Integration of video, audio, and interactive research content
- **Global Research Collaboration**: International research collaboration platform

### **Adaptive Learning Goals**
- **Self-Improving Research Quality**: System learns to improve research analysis quality
- **Emergent Research Insights**: Generation of novel research insights through AI collaboration
- **Adaptive Workflow Optimization**: Continuous optimization of research workflows
- **Predictive Research Trends**: Early identification of emerging research trends

---
# **Enhanced TENGRI Rules - Anti-Cheating Mechanisms**

## **Critical Anti-Pattern Enforcement (New Section)**

### **Real-Time Violation Detection**
```yaml
FORBIDDEN_PATTERNS:
  - "np.random.*": "NEVER replace real data sources with random generators"
  - "random.*": "Random modules forbidden outside explicit test fixtures" 
  - "mock.*": "Mock libraries prohibited in production code"
  - "placeholder": "No placeholder implementations allowed"
  - "TODO": "TODO comments indicate incomplete implementations"
  - "hardcoded_values": "No magic numbers or hardcoded data"
  - "psutil.*": "System calls must not be replaced with synthetic data"
```

### **Agent Handoff Protocol (Enhanced)**
```python
class AgentHandoffValidation:
    def pre_handoff_check(self, agent_id, changes):
        """Every agent must pass this before another agent takes over"""
        violations = self.scan_for_violations(changes)
        if violations:
            raise AgentViolationError(f"Agent {agent_id} introduced forbidden patterns: {violations}")
        
        # Require explicit acknowledgment
        return agent.acknowledge_constitution()
    
    def scan_for_violations(self, code_changes):
        """Scan for mock data, random generators, hardcoded values"""
        patterns = [
            r'np\.random\.',
            r'random\.[^_]',  # Allow random_seed but not random.uniform
            r'mock\.',
            r'psutil.*=.*random',  # Specific pattern from your example
            r'= \d+\.\d+',  # Hardcoded floating point numbers
            r'= \[\d+,\s*\d+\]'  # Hardcoded arrays
        ]
        # Implementation details...
```

### **Regression Prevention System (New)**
```python
class RegressionPrevention:
    def __init__(self):
        self.original_data_sources = {}  # Track legitimate data sources
        self.agent_constitution_cache = {}
        
    def record_legitimate_source(self, variable, source_type, source_location):
        """Record what real data sources look like"""
        self.original_data_sources[variable] = {
            'type': source_type,  # 'api_call', 'file_read', 'database_query'
            'location': source_location,
            'timestamp': datetime.now()
        }
    
    def validate_change(self, variable, new_implementation):
        """Ensure changes don't replace real data with synthetic"""
        if variable in self.original_data_sources:
            original = self.original_data_sources[variable]
            if self.is_synthetic_replacement(original, new_implementation):
                raise SyntheticDataRegressionError(
                    f"Attempted to replace real data source with synthetic data!\n"
                    f"Variable: {variable}\n"
                    f"Original: {original['type']} from {original['location']}\n"
                    f"New: synthetic/mock implementation"
                )
            return True
        return False
    
    def is_synthetic_replacement(self, original_source, new_impl):
        """Detect if new implementation is synthetic/mock"""
        synthetic_indicators = [
            'np.random',
            'random.',
            'mock.',
            'Mock',
            'hardcoded',
            '= [0, 0, 0]',  # Hardcoded arrays
            '= 0.0',  # Hardcoded floats (unless explicitly documented)
        ]
        
        for indicator in synthetic_indicators:
            if indicator in str(new_impl):
                return True
        
        # Check if original was API/DB but new is not
        if original_source['type'] in ['api_call', 'database_query']:
            if 'requests.' not in str(new_impl) and 'query(' not in str(new_impl):
                return True
        
        return False
    
    def create_checkpoint(self, description):
        """Save current state as recovery checkpoint"""
        checkpoint = {
            'timestamp': datetime.now(),
            'description': description,
            'data_sources': self.original_data_sources.copy(),
            'file_hashes': self._compute_file_hashes()
        }
        return checkpoint
    
    def _compute_file_hashes(self):
        """Hash all source files for integrity checking"""
        import hashlib
        hashes = {}
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith(('.py', '.rs', '.toml')):
                    path = os.path.join(root, file)
                    with open(path, 'rb') as f:
                        hashes[path] = hashlib.sha256(f.read()).hexdigest()
        return hashes
```

### **Constitution Enforcement System**
```python
class AgentConstitution:
    """Enforceable rules that bind all agents"""
    
    CORE_RULES = {
        'NO_MOCK_DATA': {
            'description': 'Never replace real data sources with mock/synthetic data',
            'severity': 'CRITICAL',
            'enforcement': 'IMMEDIATE_ROLLBACK',
            'examples': [
                'cpu_percent = psutil.cpu_percent() ✓',
                'cpu_percent = random.uniform(0, 100) ✗',
            ]
        },
        'FULL_IMPLEMENTATIONS': {
            'description': 'All implementations must be complete and production-ready',
            'severity': 'CRITICAL',
            'enforcement': 'BLOCK_COMMIT',
            'forbidden_patterns': ['TODO', 'FIXME', 'placeholder', 'stub']
        },
        'REAL_DATA_ONLY': {
            'description': 'Only use data from documented real sources',
            'severity': 'CRITICAL',
            'enforcement': 'IMMEDIATE_ROLLBACK',
            'allowed_sources': [
                'arXiv API',
                'IEEE Xplore',
                'PubMed',
                'Scientific databases',
                'Financial APIs',
                'IoT sensors',
                'Authenticated research repos'
            ]
        },
        'NO_WORKAROUNDS': {
            'description': 'No bandaid solutions, monkey patches, or workarounds',
            'severity': 'HIGH',
            'enforcement': 'CODE_REVIEW_REQUIRED',
            'forbidden_terms': ['quick fix', 'temporary', 'workaround', 'hack']
        }
    }
    
    def __init__(self):
        self.violation_log = []
        self.agent_acknowledgments = {}
    
    def require_acknowledgment(self, agent_id):
        """Agent must explicitly acknowledge constitution before proceeding"""
        acknowledgment = {
            'agent_id': agent_id,
            'timestamp': datetime.now(),
            'rules_acknowledged': list(self.CORE_RULES.keys()),
            'signature': self._generate_signature(agent_id)
        }
        self.agent_acknowledgments[agent_id] = acknowledgment
        return acknowledgment
    
    def _generate_signature(self, agent_id):
        """Generate cryptographic signature of acknowledgment"""
        import hashlib
        content = f"{agent_id}:{datetime.now().isoformat()}:{self.CORE_RULES}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def validate_code_change(self, agent_id, file_path, changes):
        """Validate changes against constitution"""
        violations = []
        
        # Check for mock data patterns
        if any(pattern in changes for pattern in ['np.random', 'random.', 'mock.']):
            violations.append({
                'rule': 'NO_MOCK_DATA',
                'severity': 'CRITICAL',
                'evidence': 'Detected synthetic data generation patterns'
            })
        
        # Check for incomplete implementations
        if any(marker in changes for marker in ['TODO', 'FIXME', 'placeholder']):
            violations.append({
                'rule': 'FULL_IMPLEMENTATIONS',
                'severity': 'CRITICAL',
                'evidence': 'Incomplete implementation markers found'
            })
        
        # Check for workaround patterns
        if any(term in changes.lower() for term in ['hack', 'workaround', 'temporary fix']):
            violations.append({
                'rule': 'NO_WORKAROUNDS',
                'severity': 'HIGH',
                'evidence': 'Workaround terminology detected'
            })
        
        if violations:
            self.log_violation(agent_id, file_path, violations)
            raise ConstitutionViolationError(violations)
        
        return True
    
    def log_violation(self, agent_id, file_path, violations):
        """Log constitutional violations for audit trail"""
        self.violation_log.append({
            'timestamp': datetime.now(),
            'agent_id': agent_id,
            'file': file_path,
            'violations': violations,
            'action': 'BLOCKED'
        })
```

### **Automated Recovery System**
```python
class AutomatedRecovery:
    """Automatic detection and recovery from agent regressions"""
    
    def __init__(self, regression_prevention, constitution):
        self.regression_prevention = regression_prevention
        self.constitution = constitution
        self.recovery_stack = []
    
    def monitor_continuous(self):
        """Continuous monitoring for regressions"""
        while True:
            changes = self.detect_file_changes()
            for file_path, change_type in changes:
                if self.is_regression(file_path):
                    self.trigger_automatic_rollback(file_path)
            time.sleep(1)  # Check every second
    
    def is_regression(self, file_path):
        """Detect if file change is a regression"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check against regression patterns
            regression_patterns = [
                r'= random\.',
                r'= np\.random\.',
                r'= mock\.',
                r'= \d+\.\d+  # hardcoded',
                r'psutil\.\w+ = random',
            ]
            
            for pattern in regression_patterns:
                if re.search(pattern, content):
                    return True
            
            return False
        except Exception as e:
            print(f"Error checking {file_path}: {e}")
            return False
    
    def trigger_automatic_rollback(self, file_path):
        """Automatically rollback regression"""
        print(f"🚨 REGRESSION DETECTED in {file_path}")
        print(f"⏪ AUTOMATIC ROLLBACK INITIATED")
        
        # Get last known good version
        checkpoint = self.regression_prevention.create_checkpoint(
            f"Auto-rollback of {file_path}"
        )
        
        # Execute rollback
        subprocess.run(['git', 'checkout', 'HEAD^', '--', file_path])
        
        # Log the incident
        self.recovery_stack.append({
            'timestamp': datetime.now(),
            'file': file_path,
            'action': 'AUTOMATIC_ROLLBACK',
            'checkpoint': checkpoint
        })
        
        print(f"✅ ROLLBACK COMPLETE - File restored to last known good state")
    
    def detect_file_changes(self):
        """Detect recent file modifications"""
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True
        )
        
        changes = []
        for line in result.stdout.split('\n'):
            if line.strip():
                status, file_path = line[:2], line[3:]
                changes.append((file_path, status))
        
        return changes
```

### **Real-Time Validation Pipeline**
```python
class ValidationPipeline:
    """Real-time validation of all code changes"""
    
    def __init__(self):
        self.constitution = AgentConstitution()
        self.regression_prevention = RegressionPrevention()
        self.recovery = AutomatedRecovery(
            self.regression_prevention,
            self.constitution
        )
    
    def validate_commit(self, agent_id, files_changed):
        """Validate before allowing commit"""
        print(f"\n🔍 Validating commit from agent {agent_id}...")
        
        # Step 1: Constitution check
        for file_path in files_changed:
            with open(file_path, 'r') as f:
                changes = f.read()
            self.constitution.validate_code_change(agent_id, file_path, changes)
        
        # Step 2: Regression check
        for file_path in files_changed:
            if self.recovery.is_regression(file_path):
                raise RegressionError(f"Regression detected in {file_path}")
        
        # Step 3: Data source validation
        for file_path in files_changed:
            self.validate_data_sources(file_path)
        
        print(f"✅ All validations passed")
        return True
    
    def validate_data_sources(self, file_path):
        """Ensure all data comes from real sources"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract variable assignments
        assignments = re.findall(r'(\w+)\s*=\s*(.+)', content)
        
        for var_name, value in assignments:
            # Check if this looks like data
            if any(keyword in var_name.lower() for keyword in 
                   ['data', 'metric', 'value', 'result', 'response']):
                # Ensure it's from a real source
                if not self.is_real_data_source(value):
                    raise SyntheticDataError(
                        f"Variable '{var_name}' assigned synthetic data: {value}"
                    )
    
    def is_real_data_source(self, value_expr):
        """Check if value comes from real data source"""
        real_sources = [
            'requests.get',
            'requests.post',
            'api_call',
            'database.query',
            'read_csv',
            'read_json',
            'psutil.',
            'sensor.read',
        ]
        
        # Value is real if it calls a real source
        for source in real_sources:
            if source in value_expr:
                return True
        
        # Value is NOT real if it's synthetic
        synthetic_patterns = ['random', 'mock', 'Mock', '[0, 0, 0]']
        for pattern in synthetic_patterns:
            if pattern in value_expr:
                return False
        
        # Default: allow (might be a calculation from real data)
        return True


class ConstitutionViolationError(Exception):
    """Raised when agent violates constitution"""
    pass


class RegressionError(Exception):
    """Raised when regression is detected"""
    pass


class SyntheticDataError(Exception):
    """Raised when synthetic data is used instead of real data"""
    pass
```

---

## **Integration with Existing TENGRI System**

### **Modified Agent Workflow**
```python
# Every agent interaction must go through this pipeline

def agent_task_execution(agent_id, task):
    """Enhanced agent execution with constitutional enforcement"""
    
    # 1. Require explicit acknowledgment of constitution
    constitution = AgentConstitution()
    constitution.require_acknowledgment(agent_id)
    
    # 2. Create regression prevention checkpoint
    regression_prevention = RegressionPrevention()
    checkpoint = regression_prevention.create_checkpoint(
        f"Before {agent_id} starts {task}"
    )
    
    # 3. Execute task with monitoring
    try:
        result = execute_agent_task(agent_id, task)
        
        # 4. Validate all changes
        pipeline = ValidationPipeline()
        files_changed = get_changed_files()
        pipeline.validate_commit(agent_id, files_changed)
        
        # 5. Only allow commit if validation passes
        return result
        
    except (ConstitutionViolationError, RegressionError, SyntheticDataError) as e:
        # Automatic rollback
        print(f"🚨 VIOLATION DETECTED: {e}")
        rollback_to_checkpoint(checkpoint)
        raise
```

### **Pre-Commit Hook Integration**
```bash
#!/bin/bash
# .git/hooks/pre-commit

# Run constitutional validation before allowing any commit
python3 << 'PYTHON_SCRIPT'
import sys
from validation_pipeline import ValidationPipeline

pipeline = ValidationPipeline()
try:
    files = sys.argv[1:]  # Changed files
    pipeline.validate_commit("current_agent", files)
    sys.exit(0)
except Exception as e:
    print(f"❌ COMMIT BLOCKED: {e}")
    sys.exit(1)
PYTHON_SCRIPT
```

---

## **Summary of Anti-Cheating Mechanisms**

1. **Constitutional Rules**: Explicit, enforceable rules binding all agents
2. **Regression Prevention**: Tracks legitimate data sources and prevents replacement
3. **Automated Recovery**: Real-time detection and rollback of violations
4. **Validation Pipeline**: Multi-stage validation before allowing any changes
5. **Agent Acknowledgment**: Explicit commitment to rules before task execution
6. **Pre-Commit Hooks**: Blocks commits that violate constitution
7. **Audit Trail**: Complete logging of all violations and recoveries

These mechanisms ensure that the original TENGRI rules are **enforced automatically** rather than relying on agent compliance.
**Remember**: Wolfram validates, Claude creates, TENGRI enforces quality!
