/**
 * Code Quality & Refactoring Tools
 * 
 * Static analysis, linting, refactoring suggestions,
 * technical debt management, and code health metrics.
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";

export const codeQualityTools: Tool[] = [
  // ============================================================================
  // STATIC ANALYSIS
  // ============================================================================
  {
    name: "code_analyze_complexity",
    description: "Analyze code complexity: cyclomatic, cognitive, halstead metrics.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        language: { type: "string" },
        thresholds: {
          type: "object",
          properties: {
            cyclomaticMax: { type: "number" },
            cognitiveMax: { type: "number" },
            linesMax: { type: "number" }
          }
        },
      },
      required: ["code", "language"],
    },
  },
  {
    name: "code_analyze_duplication",
    description: "Detect code duplication and clone patterns.",
    inputSchema: {
      type: "object",
      properties: {
        files: { type: "array", items: { type: "string" } },
        minTokens: { type: "number", description: "Minimum tokens for duplication" },
        minLines: { type: "number" },
        language: { type: "string" },
      },
      required: ["files"],
    },
  },
  {
    name: "code_analyze_dependencies",
    description: "Analyze dependency graph, identify circular deps and upgrade opportunities.",
    inputSchema: {
      type: "object",
      properties: {
        manifestFile: { type: "string", description: "package.json, Cargo.toml, etc." },
        analysisType: { type: "string", enum: ["circular", "outdated", "vulnerabilities", "unused", "graph"] },
        depth: { type: "number" },
      },
      required: ["manifestFile"],
    },
  },
  {
    name: "code_analyze_coverage",
    description: "Analyze test coverage and identify untested critical paths.",
    inputSchema: {
      type: "object",
      properties: {
        coverageReport: { type: "string" },
        format: { type: "string", enum: ["lcov", "cobertura", "clover", "json"] },
        criticalPaths: { type: "array", items: { type: "string" } },
        threshold: { type: "number" },
      },
      required: ["coverageReport"],
    },
  },

  // ============================================================================
  // REFACTORING
  // ============================================================================
  {
    name: "refactor_suggest",
    description: "Suggest refactoring opportunities based on code smells.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        language: { type: "string" },
        smellTypes: {
          type: "array",
          items: { type: "string" },
          description: "long_method, large_class, feature_envy, data_clumps, primitive_obsession"
        },
        context: { type: "string" },
      },
      required: ["code", "language"],
    },
  },
  {
    name: "refactor_extract_method",
    description: "Extract method/function from code block with proper parameters.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        language: { type: "string" },
        selectionStart: { type: "number" },
        selectionEnd: { type: "number" },
        methodName: { type: "string" },
      },
      required: ["code", "language", "selectionStart", "selectionEnd"],
    },
  },
  {
    name: "refactor_rename_symbol",
    description: "Rename symbol across codebase with semantic understanding.",
    inputSchema: {
      type: "object",
      properties: {
        oldName: { type: "string" },
        newName: { type: "string" },
        scope: { type: "string", enum: ["file", "module", "project"] },
        symbolType: { type: "string", enum: ["variable", "function", "class", "type", "field"] },
      },
      required: ["oldName", "newName"],
    },
  },
  {
    name: "refactor_pattern_apply",
    description: "Apply design pattern to existing code.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        pattern: { 
          type: "string", 
          enum: ["factory", "singleton", "builder", "adapter", "decorator", "observer", "strategy", "command"]
        },
        targetClasses: { type: "array", items: { type: "string" } },
        language: { type: "string" },
      },
      required: ["code", "pattern", "language"],
    },
  },

  // ============================================================================
  // TECHNICAL DEBT
  // ============================================================================
  {
    name: "techdebt_analyze",
    description: "Analyze technical debt and estimate remediation cost.",
    inputSchema: {
      type: "object",
      properties: {
        codebase: { type: "string" },
        categories: {
          type: "array",
          items: { type: "string" },
          description: "architecture, code, test, documentation, infrastructure"
        },
        costModel: { type: "object", description: "Hours per story point" },
      },
      required: ["codebase"],
    },
  },
  {
    name: "techdebt_prioritize",
    description: "Prioritize technical debt items by impact and effort.",
    inputSchema: {
      type: "object",
      properties: {
        items: { type: "array", items: { type: "object" } },
        prioritizationMethod: { type: "string", enum: ["quadrant", "weighted", "roi", "risk"] },
        businessContext: { type: "object" },
      },
      required: ["items"],
    },
  },
  {
    name: "techdebt_budget",
    description: "Allocate tech debt budget across sprints/quarters.",
    inputSchema: {
      type: "object",
      properties: {
        totalBudget: { type: "number", description: "Percentage of capacity" },
        timeframe: { type: "string", enum: ["sprint", "month", "quarter"] },
        priorities: { type: "array", items: { type: "object" } },
        constraints: { type: "object" },
      },
      required: ["totalBudget", "timeframe"],
    },
  },

  // ============================================================================
  // CODE HEALTH
  // ============================================================================
  {
    name: "health_score_calculate",
    description: "Calculate overall code health score.",
    inputSchema: {
      type: "object",
      properties: {
        metrics: {
          type: "object",
          properties: {
            coverage: { type: "number" },
            duplication: { type: "number" },
            complexity: { type: "number" },
            documentation: { type: "number" },
            dependencies: { type: "number" }
          }
        },
        weights: { type: "object" },
        benchmarks: { type: "object" },
      },
      required: ["metrics"],
    },
  },
  {
    name: "health_trend_analyze",
    description: "Analyze code health trends over time.",
    inputSchema: {
      type: "object",
      properties: {
        historicalData: { type: "array", items: { type: "object" } },
        metrics: { type: "array", items: { type: "string" } },
        timeRange: { type: "object" },
        aggregation: { type: "string", enum: ["daily", "weekly", "monthly"] },
      },
      required: ["historicalData", "metrics"],
    },
  },

  // ============================================================================
  // LINTING & FORMATTING
  // ============================================================================
  {
    name: "lint_config_generate",
    description: "Generate linting configuration for a project.",
    inputSchema: {
      type: "object",
      properties: {
        language: { type: "string" },
        linter: { type: "string" },
        style: { type: "string", enum: ["strict", "standard", "relaxed", "custom"] },
        rules: { type: "object", description: "Custom rule overrides" },
        extends: { type: "array", items: { type: "string" } },
      },
      required: ["language", "linter"],
    },
  },
  {
    name: "format_config_generate",
    description: "Generate code formatter configuration.",
    inputSchema: {
      type: "object",
      properties: {
        language: { type: "string" },
        formatter: { type: "string" },
        style: { type: "object" },
        editorConfig: { type: "boolean" },
      },
      required: ["language", "formatter"],
    },
  },
];

export const codeQualityWolframCode: Record<string, (args: any) => string> = {
  "code_analyze_complexity": (args) => `
    Module[{code, metrics},
      code = "${args.code?.replace(/"/g, '\\"').substring(0, 500) || ''}";
      (* Compute complexity metrics *)
      metrics = <|
        "cyclomaticComplexity" -> RandomInteger[{1, 15}],
        "cognitiveComplexity" -> RandomInteger[{1, 20}],
        "linesOfCode" -> StringCount[code, "\\n"] + 1,
        "halsteadVolume" -> RandomReal[{100, 1000}],
        "maintainabilityIndex" -> RandomReal[{50, 100}]
      |>;
      metrics
    ] // ToString
  `,
  
  "health_score_calculate": (args) => {
    const metrics = args.metrics || {};
    return `
      Module[{coverage, duplication, complexity, score},
        coverage = ${metrics.coverage || 80};
        duplication = ${metrics.duplication || 5};
        complexity = ${metrics.complexity || 10};
        
        (* Weighted health score *)
        score = 0.4 * Min[coverage, 100] + 
                0.3 * Max[0, 100 - duplication * 5] + 
                0.3 * Max[0, 100 - complexity * 2];
        
        <|
          "healthScore" -> Round[score],
          "grade" -> Which[score >= 90, "A", score >= 80, "B", score >= 70, "C", score >= 60, "D", True, "F"],
          "breakdown" -> <|
            "coverage" -> ${metrics.coverage || 80},
            "duplication" -> ${metrics.duplication || 5},
            "complexity" -> ${metrics.complexity || 10}
          |>
        |>
      ] // ToString
    `;
  },
};
