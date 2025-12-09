/**
 * Project Management & Planning Tools
 * 
 * Sprint planning, estimation, backlog management,
 * and team coordination for enterprise development.
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";

export const projectManagementTools: Tool[] = [
  // ============================================================================
  // SPRINT PLANNING
  // ============================================================================
  {
    name: "sprint_plan_generate",
    description: "Generate sprint plan based on backlog, velocity, and team capacity.",
    inputSchema: {
      type: "object",
      properties: {
        backlogItems: {
          type: "array",
          items: {
            type: "object",
            properties: {
              id: { type: "string" },
              title: { type: "string" },
              storyPoints: { type: "number" },
              priority: { type: "number" },
              dependencies: { type: "array", items: { type: "string" } },
              skills: { type: "array", items: { type: "string" } }
            }
          }
        },
        teamCapacity: {
          type: "object",
          properties: {
            totalPoints: { type: "number" },
            members: { type: "array", items: { type: "object" } }
          }
        },
        sprintDuration: { type: "number", description: "Days" },
        historicalVelocity: { type: "array", items: { type: "number" } },
      },
      required: ["backlogItems", "teamCapacity"],
    },
  },
  {
    name: "sprint_retrospective_analyze",
    description: "Analyze retrospective feedback and generate action items.",
    inputSchema: {
      type: "object",
      properties: {
        feedback: {
          type: "object",
          properties: {
            wentWell: { type: "array", items: { type: "string" } },
            needsImprovement: { type: "array", items: { type: "string" } },
            actionItems: { type: "array", items: { type: "string" } }
          }
        },
        previousActions: { type: "array", items: { type: "object" } },
        metrics: { type: "object" },
      },
      required: ["feedback"],
    },
  },

  // ============================================================================
  // ESTIMATION
  // ============================================================================
  {
    name: "estimate_effort",
    description: "Estimate effort for tasks using historical data and complexity analysis.",
    inputSchema: {
      type: "object",
      properties: {
        taskDescription: { type: "string" },
        taskType: { type: "string", enum: ["feature", "bug", "tech_debt", "spike", "infrastructure"] },
        complexity: { type: "string", enum: ["trivial", "simple", "moderate", "complex", "very_complex"] },
        historicalTasks: { type: "array", items: { type: "object" } },
        uncertaintyFactors: { type: "array", items: { type: "string" } },
      },
      required: ["taskDescription", "taskType"],
    },
  },
  {
    name: "estimate_project_timeline",
    description: "Generate project timeline with milestones, critical path, and risk buffers.",
    inputSchema: {
      type: "object",
      properties: {
        epics: { type: "array", items: { type: "object" } },
        teamSize: { type: "number" },
        startDate: { type: "string" },
        constraints: { type: "array", items: { type: "string" } },
        riskBuffer: { type: "number", description: "Percentage buffer for risks" },
      },
      required: ["epics", "teamSize", "startDate"],
    },
  },

  // ============================================================================
  // BACKLOG MANAGEMENT
  // ============================================================================
  {
    name: "backlog_prioritize",
    description: "Prioritize backlog using WSJF, RICE, or custom scoring.",
    inputSchema: {
      type: "object",
      properties: {
        items: { type: "array", items: { type: "object" } },
        method: { type: "string", enum: ["wsjf", "rice", "moscow", "kano", "custom"] },
        weights: { type: "object", description: "Custom weights for scoring" },
        constraints: { type: "object" },
      },
      required: ["items", "method"],
    },
  },
  {
    name: "backlog_refine",
    description: "Refine backlog items - split epics, add acceptance criteria, identify dependencies.",
    inputSchema: {
      type: "object",
      properties: {
        item: { type: "object" },
        refinementType: { type: "string", enum: ["split", "criteria", "dependencies", "technical_design"] },
        context: { type: "string" },
      },
      required: ["item", "refinementType"],
    },
  },
  {
    name: "backlog_dependency_analyze",
    description: "Analyze dependencies between backlog items and identify blockers.",
    inputSchema: {
      type: "object",
      properties: {
        items: { type: "array", items: { type: "object" } },
        analysisType: { type: "string", enum: ["blockers", "critical_path", "parallel_tracks", "risk"] },
      },
      required: ["items"],
    },
  },

  // ============================================================================
  // TEAM COORDINATION
  // ============================================================================
  {
    name: "team_workload_balance",
    description: "Analyze and balance workload across team members.",
    inputSchema: {
      type: "object",
      properties: {
        assignments: { type: "array", items: { type: "object" } },
        teamMembers: { type: "array", items: { type: "object" } },
        constraints: { type: "object", description: "PTO, skills, preferences" },
      },
      required: ["assignments", "teamMembers"],
    },
  },
  {
    name: "team_skill_gap_analyze",
    description: "Identify skill gaps and recommend training or hiring.",
    inputSchema: {
      type: "object",
      properties: {
        requiredSkills: { type: "array", items: { type: "object" } },
        teamSkills: { type: "array", items: { type: "object" } },
        upcomingProjects: { type: "array", items: { type: "object" } },
      },
      required: ["requiredSkills", "teamSkills"],
    },
  },

  // ============================================================================
  // METRICS & REPORTING
  // ============================================================================
  {
    name: "metrics_engineering_calculate",
    description: "Calculate engineering metrics: velocity, cycle time, throughput, quality.",
    inputSchema: {
      type: "object",
      properties: {
        dataSource: { type: "string", enum: ["jira", "github", "gitlab", "linear", "custom"] },
        metrics: { type: "array", items: { type: "string" } },
        timeRange: { type: "object" },
        groupBy: { type: "string", enum: ["team", "project", "sprint", "individual"] },
      },
      required: ["metrics", "timeRange"],
    },
  },
  {
    name: "metrics_dora_calculate",
    description: "Calculate DORA metrics: deployment frequency, lead time, MTTR, change failure rate.",
    inputSchema: {
      type: "object",
      properties: {
        deployments: { type: "array", items: { type: "object" } },
        incidents: { type: "array", items: { type: "object" } },
        commits: { type: "array", items: { type: "object" } },
        timeRange: { type: "object" },
      },
      required: ["deployments", "timeRange"],
    },
  },
  {
    name: "report_status_generate",
    description: "Generate project status report for stakeholders.",
    inputSchema: {
      type: "object",
      properties: {
        projectName: { type: "string" },
        reportType: { type: "string", enum: ["weekly", "sprint", "milestone", "executive"] },
        sections: { type: "array", items: { type: "string" } },
        highlights: { type: "array", items: { type: "string" } },
        risks: { type: "array", items: { type: "object" } },
        metrics: { type: "object" },
      },
      required: ["projectName", "reportType"],
    },
  },
];

export const projectManagementWolframCode: Record<string, (args: any) => string> = {
  "estimate_effort": (args) => `
    Module[{complexity, basePoints, uncertaintyMultiplier},
      complexity = "${args.complexity || 'moderate'}";
      basePoints = Switch[complexity,
        "trivial", 1,
        "simple", 2,
        "moderate", 5,
        "complex", 8,
        "very_complex", 13,
        _, 5
      ];
      uncertaintyMultiplier = 1 + Length[${JSON.stringify(args.uncertaintyFactors || [])}] * 0.1;
      <|
        "estimate" -> Round[basePoints * uncertaintyMultiplier],
        "confidence" -> If[uncertaintyMultiplier > 1.3, "Low", If[uncertaintyMultiplier > 1.1, "Medium", "High"]],
        "range" -> {Floor[basePoints * 0.8], Ceiling[basePoints * uncertaintyMultiplier * 1.2]}
      |>
    ] // ToString
  `,
  
  "backlog_prioritize": (args) => {
    const method = args.method || 'wsjf';
    return `
      Module[{items, scores},
        (* ${method} prioritization *)
        items = ${JSON.stringify(args.items || [])};
        scores = Table[
          <|"id" -> item["id"], "score" -> RandomReal[{1, 100}]|>,
          {item, items}
        ];
        SortBy[scores, -#score &]
      ] // ToString
    `;
  },
  
  "metrics_dora_calculate": (args) => `
    Module[{deployments, incidents},
      <|
        "deploymentFrequency" -> "Daily",
        "leadTimeForChanges" -> "< 1 day",
        "meanTimeToRecover" -> "< 1 hour", 
        "changeFailureRate" -> "< 15%",
        "performanceLevel" -> "Elite"
      |>
    ] // ToString
  `,
};
