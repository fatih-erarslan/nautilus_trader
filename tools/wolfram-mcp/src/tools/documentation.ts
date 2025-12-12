/**
 * Documentation & Knowledge Management Tools
 * 
 * API docs, architecture diagrams, ADRs, runbooks,
 * and knowledge base management for enterprises.
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";

export const documentationTools: Tool[] = [
  // ============================================================================
  // API DOCUMENTATION
  // ============================================================================
  {
    name: "docs_api_generate",
    description: "Generate API documentation from code or specifications.",
    inputSchema: {
      type: "object",
      properties: {
        source: { type: "string", enum: ["openapi", "graphql", "grpc", "code", "comments"] },
        inputPath: { type: "string" },
        outputFormat: { type: "string", enum: ["markdown", "html", "redoc", "swagger_ui", "docusaurus"] },
        includeExamples: { type: "boolean" },
        includeSchemas: { type: "boolean" },
      },
      required: ["source", "inputPath"],
    },
  },
  {
    name: "docs_api_openapi_generate",
    description: "Generate OpenAPI specification from API description or code.",
    inputSchema: {
      type: "object",
      properties: {
        endpoints: {
          type: "array",
          items: {
            type: "object",
            properties: {
              method: { type: "string" },
              path: { type: "string" },
              description: { type: "string" },
              requestBody: { type: "object" },
              responses: { type: "object" }
            }
          }
        },
        version: { type: "string" },
        title: { type: "string" },
        securitySchemes: { type: "array", items: { type: "string" } },
      },
      required: ["endpoints", "title"],
    },
  },

  // ============================================================================
  // ARCHITECTURE DOCUMENTATION
  // ============================================================================
  {
    name: "docs_architecture_diagram",
    description: "Generate architecture diagrams in various formats.",
    inputSchema: {
      type: "object",
      properties: {
        diagramType: { 
          type: "string", 
          enum: ["c4_context", "c4_container", "c4_component", "sequence", "flowchart", "erd", "deployment"] 
        },
        components: { type: "array", items: { type: "object" } },
        connections: { type: "array", items: { type: "object" } },
        outputFormat: { type: "string", enum: ["mermaid", "plantuml", "d2", "graphviz", "structurizr"] },
        style: { type: "string" },
      },
      required: ["diagramType", "components"],
    },
  },
  {
    name: "docs_adr_generate",
    description: "Generate Architecture Decision Record (ADR).",
    inputSchema: {
      type: "object",
      properties: {
        title: { type: "string" },
        context: { type: "string" },
        decision: { type: "string" },
        alternatives: { type: "array", items: { type: "object" } },
        consequences: { type: "array", items: { type: "string" } },
        status: { type: "string", enum: ["proposed", "accepted", "deprecated", "superseded"] },
        relatedAdrs: { type: "array", items: { type: "string" } },
      },
      required: ["title", "context", "decision"],
    },
  },
  {
    name: "docs_system_design",
    description: "Generate system design document from requirements.",
    inputSchema: {
      type: "object",
      properties: {
        requirements: { type: "array", items: { type: "string" } },
        constraints: { type: "array", items: { type: "string" } },
        qualityAttributes: { type: "array", items: { type: "string" } },
        sections: { type: "array", items: { type: "string" } },
        depth: { type: "string", enum: ["overview", "detailed", "implementation"] },
      },
      required: ["requirements"],
    },
  },

  // ============================================================================
  // RUNBOOKS & OPERATIONS
  // ============================================================================
  {
    name: "docs_runbook_generate",
    description: "Generate operational runbook for service or incident type.",
    inputSchema: {
      type: "object",
      properties: {
        service: { type: "string" },
        runbookType: { type: "string", enum: ["deployment", "rollback", "incident", "maintenance", "scaling"] },
        steps: { type: "array", items: { type: "object" } },
        alerts: { type: "array", items: { type: "string" } },
        escalation: { type: "object" },
      },
      required: ["service", "runbookType"],
    },
  },
  {
    name: "docs_postmortem_generate",
    description: "Generate incident postmortem document.",
    inputSchema: {
      type: "object",
      properties: {
        incidentId: { type: "string" },
        timeline: { type: "array", items: { type: "object" } },
        impact: { type: "object" },
        rootCause: { type: "string" },
        contributingFactors: { type: "array", items: { type: "string" } },
        actionItems: { type: "array", items: { type: "object" } },
        lessonsLearned: { type: "array", items: { type: "string" } },
      },
      required: ["incidentId", "timeline", "rootCause"],
    },
  },

  // ============================================================================
  // CODE DOCUMENTATION
  // ============================================================================
  {
    name: "docs_code_readme",
    description: "Generate README.md for a project or module.",
    inputSchema: {
      type: "object",
      properties: {
        projectName: { type: "string" },
        description: { type: "string" },
        installation: { type: "boolean" },
        usage: { type: "boolean" },
        api: { type: "boolean" },
        contributing: { type: "boolean" },
        license: { type: "string" },
        badges: { type: "array", items: { type: "string" } },
      },
      required: ["projectName", "description"],
    },
  },
  {
    name: "docs_code_comments",
    description: "Generate documentation comments for code.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        language: { type: "string" },
        style: { type: "string", enum: ["jsdoc", "rustdoc", "pydoc", "javadoc", "xmldoc"] },
        includeExamples: { type: "boolean" },
      },
      required: ["code", "language"],
    },
  },
  {
    name: "docs_changelog_generate",
    description: "Generate changelog from commits or PR descriptions.",
    inputSchema: {
      type: "object",
      properties: {
        commits: { type: "array", items: { type: "object" } },
        version: { type: "string" },
        format: { type: "string", enum: ["keep_a_changelog", "conventional", "custom"] },
        groupBy: { type: "string", enum: ["type", "scope", "breaking"] },
      },
      required: ["commits", "version"],
    },
  },

  // ============================================================================
  // KNOWLEDGE BASE
  // ============================================================================
  {
    name: "kb_search",
    description: "Search knowledge base for relevant documentation.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string" },
        filters: { type: "object" },
        limit: { type: "number" },
        includeRelated: { type: "boolean" },
      },
      required: ["query"],
    },
  },
  {
    name: "kb_index",
    description: "Index documents into knowledge base.",
    inputSchema: {
      type: "object",
      properties: {
        documents: { type: "array", items: { type: "object" } },
        extractMetadata: { type: "boolean" },
        generateEmbeddings: { type: "boolean" },
      },
      required: ["documents"],
    },
  },
  {
    name: "kb_summarize",
    description: "Summarize documentation or codebase for quick understanding.",
    inputSchema: {
      type: "object",
      properties: {
        source: { type: "string" },
        sourceType: { type: "string", enum: ["code", "docs", "repo", "api"] },
        length: { type: "string", enum: ["brief", "standard", "detailed"] },
        focus: { type: "array", items: { type: "string" } },
      },
      required: ["source", "sourceType"],
    },
  },
  {
    name: "kb_onboarding_generate",
    description: "Generate onboarding documentation for new team members.",
    inputSchema: {
      type: "object",
      properties: {
        role: { type: "string" },
        team: { type: "string" },
        projects: { type: "array", items: { type: "string" } },
        technologies: { type: "array", items: { type: "string" } },
        duration: { type: "string", enum: ["30_days", "60_days", "90_days"] },
      },
      required: ["role", "team"],
    },
  },
];

export const documentationWolframCode: Record<string, (args: any) => string> = {
  "docs_architecture_diagram": (args) => {
    const type = args.diagramType || 'flowchart';
    const format = args.outputFormat || 'mermaid';
    return `
      Module[{components, connections, diagram},
        components = ${JSON.stringify(args.components || [])};
        (* Generate ${format} diagram for ${type} *)
        diagram = "graph TD\\n" <> 
          StringJoin[Table[
            comp["id"] <> "[" <> comp["name"] <> "]\\n",
            {comp, components}
          ]];
        diagram
      ] // ToString
    `;
  },
  
  "docs_adr_generate": (args) => `
    Module[{adr},
      adr = "# ADR: ${args.title?.replace(/"/g, '\\"') || 'Decision'}

## Status
${args.status || 'proposed'}

## Context
${args.context?.replace(/"/g, '\\"') || ''}

## Decision
${args.decision?.replace(/"/g, '\\"') || ''}

## Consequences
${(args.consequences || []).map((c: string) => `- ${c}`).join('\\n')}
";
      adr
    ] // ToString
  `,
};
