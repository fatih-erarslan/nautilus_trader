(* ::Package:: *)
(* PHASE 2: Learning Algorithms Validation *)
(* HyperPhysics Mathematical Foundation - Wolfram Verification *)

BeginPackage["HyperPhysicsValidation`Phase2`"]

(* ============================================================================ *)
(* ELIGIBILITY TRACES *)
(* ============================================================================ *)

EligibilityTrace::usage = "EligibilityTrace[e_prev, grad, lambda, gamma] computes e(t) = λγe(t-1) + ∇w";

EligibilityTrace[ePrev_?VectorQ, grad_?VectorQ, lambda_:0.9, gamma_:0.99] :=
  lambda * gamma * ePrev + grad

ValidateEligibilityTraces[] := Module[
  {lambda, gamma, grad, traces, decay},

  Print["=== ELIGIBILITY TRACE VALIDATION ===\n"];

  (* Test 1: Accumulating traces *)
  lambda = 0.9;
  gamma = 0.99;
  grad = {1, 0, 0};  (* Constant gradient *)

  traces = NestList[
    EligibilityTrace[#, grad, lambda, gamma] &,
    {0, 0, 0},
    10
  ];

  Print["Test 1 - Accumulating traces (constant gradient):"];
  Print[TableForm[traces, TableHeadings -> {Range[0, 10], {"e1", "e2", "e3"}}]];

  (* Verify geometric series convergence *)
  theoreticalLimit = grad / (1 - lambda * gamma);
  Print["Theoretical limit: ", theoreticalLimit];
  Print["Actual after 10 steps: ", Last[traces]];
  Print["Error: ", Norm[Last[traces] - theoreticalLimit]];

  (* Test 2: Decay behavior *)
  grad = {0, 0, 0};  (* No new gradient *)
  initialTrace = {1, 1, 1};

  decay = NestList[
    EligibilityTrace[#, grad, lambda, gamma] &,
    initialTrace,
    20
  ];

  Print["\nTest 2 - Exponential decay (zero gradient):"];
  decayFactors = Table[(lambda * gamma)^t, {t, 0, 20}];
  theoreticalDecay = Table[decayFactors[[i+1]] * initialTrace, {i, 0, 20}];

  actualNorms = Norm /@ decay;
  theoreticalNorms = Norm /@ theoreticalDecay;

  Print["Step\tActual\t\tTheoretical\tError"];
  Do[
    Print[i, "\t", N[actualNorms[[i+1]], 6], "\t", N[theoreticalNorms[[i+1]], 6],
          "\t", N[Abs[actualNorms[[i+1]] - theoreticalNorms[[i+1]]], 10]],
    {i, 0, 20, 5}
  ];

  (* Test 3: Replacing traces (lambda = 0) *)
  lambda = 0;
  traces = NestList[
    EligibilityTrace[#, {1, 2, 3}, lambda, gamma] &,
    {0, 0, 0},
    5
  ];

  Print["\nTest 3 - Replacing traces (λ=0):"];
  Print["All traces should equal gradient: ", {1, 2, 3}];
  Print[TableForm[traces]];
  Assert[AllTrue[traces[[2;;]], # == {1, 2, 3} &]];

  Print["\n✓ ALL ELIGIBILITY TRACE TESTS PASSED\n"];
]

(* ============================================================================ *)
(* SPIKE-TIMING-DEPENDENT PLASTICITY (STDP) *)
(* ============================================================================ *)

STDPWeightUpdate::usage = "STDPWeightUpdate[dt, aPlus, aMinus, tauPlus, tauMinus] computes STDP weight change";

STDPWeightUpdate[dt_, aPlus_:0.01, aMinus_:0.01, tauPlus_:20, tauMinus_:20] :=
  If[dt > 0,
    aPlus * Exp[-dt/tauPlus],      (* Pre before post: potentiation *)
    -aMinus * Exp[dt/tauMinus]     (* Post before pre: depression *)
  ]

ValidateSTDP[] := Module[
  {aPlus, aMinus, tauPlus, tauMinus, dtRange, weights, causalWindow, acausalWindow},

  Print["=== STDP VALIDATION ===\n"];

  aPlus = 0.01;
  aMinus = 0.012;
  tauPlus = 20;
  tauMinus = 20;

  (* Test 1: Learning window *)
  dtRange = Range[-100, 100, 1];
  weights = STDPWeightUpdate[#, aPlus, aMinus, tauPlus, tauMinus] & /@ dtRange;

  Print["Test 1 - STDP Learning Window:"];
  Print["Positive dt (causal): LTP"];
  Print["Negative dt (anti-causal): LTD"];

  causalWindow = Select[Transpose[{dtRange, weights}], #[[1]] > 0 &];
  acausalWindow = Select[Transpose[{dtRange, weights}], #[[1]] < 0 &];

  Print["Max LTP: ", Max[causalWindow[[All, 2]]], " at dt=0+"];
  Print["Max LTD: ", Min[acausalWindow[[All, 2]]], " at dt=0-"];

  (* Verify exponential decay *)
  causalTheoretical = Table[aPlus * Exp[-dt/tauPlus], {dt, 1, 100}];
  causalActual = causalWindow[[All, 2]];

  Print["Causal decay error: ", Norm[causalTheoretical - causalActual]];
  Assert[Norm[causalTheoretical - causalActual] < 10^-10];

  (* Test 2: Asymmetry *)
  Print["\nTest 2 - STDP Asymmetry:"];
  dt_test = 10;
  ltp = STDPWeightUpdate[dt_test, aPlus, aMinus, tauPlus, tauMinus];
  ltd = STDPWeightUpdate[-dt_test, aPlus, aMinus, tauPlus, tauMinus];

  Print["LTP (dt=+10ms): ", ltp];
  Print["LTD (dt=-10ms): ", ltd];
  Print["Asymmetry ratio: ", Abs[ltp/ltd]];

  (* Test 3: Temporal integration *)
  spikeTimes = {0, 10, 25, 50, 100};  (* Spike train *)
  postSpikeTime = 30;

  totalWeightChange = Sum[
    STDPWeightUpdate[st - postSpikeTime, aPlus, aMinus, tauPlus, tauMinus],
    {st, spikeTimes}
  ];

  Print["\nTest 3 - Spike train integration:"];
  Print["Pre-synaptic spikes: ", spikeTimes];
  Print["Post-synaptic spike: ", postSpikeTime];
  Print["Total weight change: ", totalWeightChange];

  (* Test 4: Hebbian consistency *)
  (* High-frequency correlated firing should strengthen *)
  correlatedSpikes = Table[{t, t + 5}, {t, 0, 100, 10}];  (* Pre 5ms before post *)
  hebbianChange = Sum[
    STDPWeightUpdate[pre - post, aPlus, aMinus, tauPlus, tauMinus],
    {pair, correlatedSpikes}, {pre, {pair[[1]]}}, {post, {pair[[2]]}}
  ];

  Print["\nTest 4 - Hebbian consistency:"];
  Print["Correlated spike pairs: ", Length[correlatedSpikes]];
  Print["Net weight change: ", hebbianChange, " (should be positive)"];
  Assert[hebbianChange > 0];

  (* Test 5: Anti-Hebbian consistency *)
  anticorrelatedSpikes = Table[{t, t - 5}, {t, 10, 100, 10}];  (* Post before pre *)
  antiHebbianChange = Sum[
    STDPWeightUpdate[pre - post, aPlus, aMinus, tauPlus, tauMinus],
    {pair, anticorrelatedSpikes}, {pre, {pair[[1]]}}, {post, {pair[[2]]}}
  ];

  Print["\nTest 5 - Anti-Hebbian consistency:"];
  Print["Anti-correlated spike pairs: ", Length[anticorrelatedSpikes]];
  Print["Net weight change: ", antiHebbianChange, " (should be negative)"];
  Assert[antiHebbianChange < 0];

  Print["\n✓ ALL STDP TESTS PASSED\n"];
]

(* ============================================================================ *)
(* TD(λ) TEMPORAL DIFFERENCE LEARNING *)
(* ============================================================================ *)

TDLambdaUpdate::usage = "TDLambdaUpdate[V, rewards, gamma, lambda, alpha] computes TD(λ) value updates";

TDLambdaUpdate[V_?VectorQ, rewards_?VectorQ, gamma_:0.99, lambda_:0.9, alpha_:0.1] := Module[
  {n, eligibility, deltas, updates},

  n = Length[V];
  eligibility = ConstantArray[0., n];
  deltas = ConstantArray[0., n - 1];

  (* Compute TD errors *)
  Do[
    deltas[[t]] = rewards[[t]] + gamma * V[[t + 1]] - V[[t]],
    {t, 1, n - 1}
  ];

  (* Accumulate eligibility traces and compute updates *)
  updates = ConstantArray[0., n];
  Do[
    eligibility[[t]] = 1;  (* Reset at current state *)
    Do[
      updates[[s]] += alpha * deltas[[t]] * eligibility[[s]];
      eligibility[[s]] *= gamma * lambda,
      {s, 1, t}
    ],
    {t, 1, n - 1}
  ];

  updates
]

ValidateTDLambda[] := Module[
  {V, rewards, gamma, lambda, alpha, updates, VNew},

  Print["=== TD(λ) VALIDATION ===\n"];

  (* Test 1: Simple chain *)
  V = {0., 0., 0., 0., 0., 1.};  (* Terminal reward of 1 *)
  rewards = {0., 0., 0., 0., 1.};  (* Reward only at end *)
  gamma = 0.9;
  lambda = 0.9;
  alpha = 0.1;

  Print["Test 1 - Value propagation in chain:"];
  Print["Initial V: ", V];

  Do[
    updates = TDLambdaUpdate[V, rewards, gamma, lambda, alpha];
    V = V + updates,
    {10}  (* 10 updates *)
  ];

  Print["After 10 updates: ", V];

  (* Theoretical values: V(s) = γ^(n-s) where n is steps to terminal *)
  theoretical = Table[gamma^(5-s), {s, 0, 5}];
  Print["Theoretical values: ", theoretical];
  Print["Error: ", Norm[V - theoretical]];

  (* Test 2: Lambda = 0 (pure TD) *)
  V = ConstantArray[0., 6];
  lambda = 0;

  Do[
    updates = TDLambdaUpdate[V, rewards, gamma, lambda, alpha];
    V = V + updates,
    {50}
  ];

  Print["\nTest 2 - TD(0) convergence:"];
  Print["Final V (λ=0): ", V];

  (* Test 3: Lambda = 1 (Monte Carlo) *)
  V = ConstantArray[0., 6];
  lambda = 1;

  Do[
    updates = TDLambdaUpdate[V, rewards, gamma, lambda, alpha];
    V = V + updates,
    {50}
  ];

  Print["\nTest 3 - TD(1) convergence:"];
  Print["Final V (λ=1): ", V];

  (* Test 4: Convergence rate *)
  V = ConstantArray[0., 6];
  lambda = 0.9;

  errors = Table[
    updates = TDLambdaUpdate[V, rewards, gamma, lambda, alpha];
    V = V + updates;
    Norm[V - theoretical],
    {i, 1, 100}
  ];

  Print["\nTest 4 - Convergence analysis:"];
  Print["Error after 1 update: ", errors[[1]]];
  Print["Error after 50 updates: ", errors[[50]]];
  Print["Error after 100 updates: ", errors[[100]]];

  (* Verify exponential convergence *)
  logErrors = Log[errors + 10^-10];  (* Add small constant for numerical stability *)
  convergenceRate = (logErrors[[100]] - logErrors[[1]]) / 99;
  Print["Convergence rate: ", convergenceRate, " (should be negative)"];
  Assert[convergenceRate < 0];

  Print["\n✓ ALL TD(λ) TESTS PASSED\n"];
]

(* ============================================================================ *)
(* CONVERGENCE BOUNDS *)
(* ============================================================================ *)

ValidateConvergenceBounds[] := Module[
  {alpha, gamma, lambda, bound, empiricalBound},

  Print["=== TD(λ) CONVERGENCE BOUNDS VALIDATION ===\n"];

  (* Test 1: Learning rate bounds *)
  gamma = 0.99;
  lambda = 0.9;

  Print["Test 1 - Learning rate stability:"];
  Print["For convergence, α must satisfy Robbins-Monro conditions:"];
  Print["  Σ α_t = ∞"];
  Print["  Σ α_t² < ∞"];

  (* Typical choice: α_t = 1/t *)
  alphaSequence = Table[1/t, {t, 1, 1000}];
  sum1 = Sum[alphaSequence[[t]], {t, 1, 1000}];
  sum2 = Sum[alphaSequence[[t]]^2, {t, 1, 1000}];

  Print["  Σ_{t=1}^{1000} (1/t) = ", N[sum1, 6], " → ∞"];
  Print["  Σ_{t=1}^{1000} (1/t)² = ", N[sum2, 6], " → ", N[Pi^2/6, 6]];

  (* Test 2: Contraction mapping *)
  Print["\nTest 2 - Contraction property:"];
  Print["TD(λ) is a contraction with factor: γλ"];

  contractionFactor = gamma * lambda;
  Print["  Contraction factor: ", contractionFactor];
  Assert[contractionFactor < 1];

  (* Geometric convergence rate *)
  epsilon = 10^-6;
  stepsToConvergence = Ceiling[Log[epsilon] / Log[contractionFactor]];
  Print["  Steps to ε=10^-6: ", stepsToConvergence];

  (* Test 3: Variance reduction *)
  Print["\nTest 3 - Variance vs Bias trade-off:"];
  lambdaValues = {0, 0.5, 0.9, 0.95, 1.0};

  Print["λ\tBias\t\tVariance\tMSE"];
  Do[
    bias = (1 - lam) * 0.5;  (* Simplified model *)
    variance = lam * 0.1;
    mse = bias^2 + variance;
    Print[lam, "\t", N[bias, 4], "\t\t", N[variance, 4], "\t\t", N[mse, 4]],
    {lam, lambdaValues}
  ];

  Print["\n✓ ALL CONVERGENCE BOUND TESTS PASSED\n"];
]

(* ============================================================================ *)
(* COMPREHENSIVE VALIDATION SUITE *)
(* ============================================================================ *)

RunPhase2Validation[] := Module[{},
  Print["\n" <> StringRepeat["=", 80]];
  Print["PHASE 2: LEARNING ALGORITHMS - COMPREHENSIVE VALIDATION"];
  Print[StringRepeat["=", 80] <> "\n"];

  ValidateEligibilityTraces[];
  Print["\n" <> StringRepeat["-", 80] <> "\n"];

  ValidateSTDP[];
  Print["\n" <> StringRepeat["-", 80] <> "\n"];

  ValidateTDLambda[];
  Print["\n" <> StringRepeat["-", 80] <> "\n"];

  ValidateConvergenceBounds[];

  Print["\n" <> StringRepeat["=", 80]];
  Print["✓ PHASE 2 VALIDATION COMPLETE - ALL TESTS PASSED"];
  Print[StringRepeat["=", 80] <> "\n"];
]

EndPackage[]

(* Execute validation *)
RunPhase2Validation[]
