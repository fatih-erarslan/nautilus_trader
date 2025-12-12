---
name: wolfram-scientific-computing
description: Expert guidance for using Wolfram Alpha API and WolframScript for scientific computing, mathematical validation, hyperbolic geometry, and HyperPhysics research. Activate when user needs rigorous mathematical computation, symbolic algebra, scientific data queries, or physics validation.
---

# Wolfram Scientific Computing Skill

This skill provides expert guidance for leveraging Wolfram's computational intelligence for HyperPhysics development.

## When to Use This Skill

Activate when the user needs:
- **Mathematical Validation**: Verify implementations of mathematical algorithms
- **Symbolic Computation**: Integrals, derivatives, differential equations, series
- **Scientific Data**: Physical constants, chemical properties, astronomical data
- **Hyperbolic Geometry**: Poincaré disk computations, Möbius transforms, geodesics
- **Algorithm Design**: Formal verification, correctness proofs
- **Unit Conversions**: Physical unit transformations

## Available Tools

### MCP Tools (via wolfram MCP server)
| Tool | Use Case |
|------|----------|
| `wolfram_llm_query` | Natural language queries to Wolfram Alpha |
| `wolfram_compute` | Mathematical expressions |
| `wolfram_validate` | Verify mathematical identities |
| `wolfram_unit_convert` | Unit conversions |
| `wolfram_data_query` | Scientific/financial data |
| `wolfram_local_eval` | Execute Wolfram Language code locally |
| `wolfram_symbolic` | Symbolic math (integrate, differentiate, solve) |
| `wolfram_hyperbolic` | Hyperbolic geometry operations |

### Local WolframScript (Full Power)
Direct access via terminal: `wolframscript -code "expression"`

## Query Formulation Best Practices

### For Wolfram Alpha (API)
- **Simplify queries**: "France population" not "how many people live in France"
- **Use proper notation**: `6*10^14` not `6e14`
- **Single-letter variables**: `x`, `n`, `n1`, not `variable_name`
- **Named constants**: "speed of light" not "299792458 m/s"
- **Compound units**: "Ω m" with space for "ohm meter"

### For WolframScript (Local)
```wolfram
(* Symbolic computation *)
Integrate[Sin[x]^2 * Cos[x]^3, x]

(* Numerical evaluation *)
N[Pi, 50]  (* 50 digits of Pi *)

(* Hyperbolic distance in Poincaré disk *)
hypDist[z1_, z2_] := 2*ArcTanh[Abs[z1-z2]/Sqrt[(1-Abs[z1]^2)*(1-Abs[z2]^2)+Abs[z1-z2]^2]]

(* Access knowledge base *)
Entity["Element", "Gold"]["AtomicMass"]

(* Matrix operations *)
Eigenvalues[{{1, 2}, {3, 4}}]
```

## HyperPhysics-Specific Computations

### Hyperbolic Geometry
```wolfram
(* Poincaré disk distance *)
hyperbolicDistance[{x1, y1}, {x2, y2}] := Module[
  {z1 = x1 + y1*I, z2 = x2 + y2*I},
  2*ArcTanh[Abs[z1 - z2]/Sqrt[(1 - Abs[z1]^2)*(1 - Abs[z2]^2) + Abs[z1 - z2]^2]]
]

(* Möbius transformation *)
mobiusTransform[a_, b_, c_, d_, z_] := (a*z + b)/(c*z + d)

(* Geodesic between points *)
geodesic[z1_, z2_, n_:10] := Table[
  Module[{t = k/(n-1), mob},
    mob = (z - z1)/(1 - Conjugate[z1]*z) /. z -> z2;
    (t*mob - z1)/(1 - Conjugate[z1]*t*mob)
  ], {k, 0, n-1}
]
```

### Neural Network Validation
```wolfram
(* STDP learning rule verification *)
stdpUpdate[w_, deltaPre_, deltaPost_, Aplus_, Aminus_, tauPlus_, tauMinus_] := 
  w + Aplus*Exp[-deltaPre/tauPlus] - Aminus*Exp[-deltaPost/tauMinus]

(* IIT Phi approximation *)
phiApprox[tpm_] := Module[{n = Length[tpm]},
  Total[Eigenvalues[tpm - IdentityMatrix[n]]^2]
]
```

### Free Energy / Active Inference
```wolfram
(* Variational free energy *)
freeEnergy[qDist_, pDist_, observations_] := 
  Total[qDist * Log[qDist/pDist]] + Total[-qDist * Log[observations]]

(* Belief update *)
beliefUpdate[prior_, likelihood_] := 
  Normalize[prior * likelihood, Total]
```

## Validation Workflow

1. **Implement** algorithm in Rust/Python
2. **Generate** test cases with Wolfram
3. **Compare** results numerically
4. **Prove** correctness symbolically if needed

Example:
```python
# Step 1: Get reference from Wolfram
wolfram_result = await wolfram_symbolic(
    operation="integrate",
    expression="x^2 * Sin[x]",
    variable="x"
)

# Step 2: Compare with implementation
assert abs(my_integral(2.0) - eval_wolfram(wolfram_result, 2.0)) < 1e-10
```

## Error Handling

- **Timeout**: Increase timeout for complex computations
- **Parse errors**: Simplify expression, check syntax
- **No result**: Rephrase query, try different approach
- **Rate limits**: Use local WolframScript for heavy computation

## Integration with HyperPhysics Crates

The `hyperphysics-wolfram` Rust crate provides native integration:

```rust
use hyperphysics_wolfram::{WolframBridge, HyperbolicGeometryResult};

let bridge = WolframBridge::new().await?;

// Validate hyperbolic computation
let result = bridge.compute_hyperbolic_tessellation(7, 3, 4).await?;

// Verify STDP implementation
let validated = bridge.validate_stdp_rule(0.5, 10.0, -5.0, 0.1, 0.12, 20.0, 20.0).await?;
```

## Examples

### Query: "Validate my hyperbolic distance implementation"
Use `wolfram_hyperbolic` with operation="distance" to compute reference values.

### Query: "Compute the integral of x³eˣ"
Use `wolfram_symbolic` with operation="integrate", expression="x^3 * E^x".

### Query: "What is the atomic mass of carbon?"
Use `wolfram_data_query` with entity="carbon", property="atomic mass".

### Query: "Verify sin²(x) + cos²(x) = 1"
Use `wolfram_validate` with expression="Sin[x]^2 + Cos[x]^2", expected="1".
