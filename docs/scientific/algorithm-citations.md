# Algorithm Citations and Scientific Foundations

## Overview

This document provides comprehensive scientific citations for all algorithms implemented in the HyperPhysics trading system. Every algorithm has been validated against peer-reviewed literature with a minimum of 3-4 citations per algorithm.

---

## Table of Contents

1. [Bayesian Methods](#bayesian-methods)
2. [Active Inference & Free Energy Principle](#active-inference)
3. [Swarm Intelligence Algorithms](#swarm-intelligence)
4. [Risk Management Algorithms](#risk-management)
5. [Meta-heuristic Optimization](#meta-heuristic)

---

## Bayesian Methods

### Bayesian Vector Autoregression (BVAR)

**Implementation**: `/crates/cwts-ultra/core/src/algorithms/bayesian_var.rs`

**Mathematical Foundation**:
- VAR(p) model: y_t = c + A_1*y_{t-1} + ... + A_p*y_{t-p} + ε_t
- Bayesian priors: θ ~ N(θ₀, Ω₀) where Ω controls shrinkage
- Minnesota prior: Litterman's data-centric prior (θ = 0.3, λ = 0.2)

**Key References**:

1. **Litterman, R. B. (1986)**. "Forecasting with Bayesian Vector Autoregressions—Five Years of Experience"
   *Journal of Business & Economic Statistics*, 4(1), 25-38.
   DOI: [10.1080/07350015.1986.10509491](https://doi.org/10.1080/07350015.1986.10509491)
   - Foundational work on BVAR for macroeconomic forecasting
   - Demonstrated accuracy comparable to commercial forecasting services

2. **Sims, C. A. (1980)**. "Macroeconomics and Reality"
   *Econometrica*, 48(1), 1-48.
   DOI: [10.2307/1912017](https://doi.org/10.2307/1912017)
   - Original vector autoregression framework
   - Theoretical foundation for multivariate time series modeling

3. **Bańbura, M., Giannone, D., & Reichlin, L. (2010)**. "Large Bayesian Vector Auto Regressions"
   *Journal of Applied Econometrics*, 25(1), 71-92.
   DOI: [10.1002/jae.1137](https://doi.org/10.1002/jae.1137)
   - Extensions to high-dimensional BVAR
   - Advanced shrinkage techniques

4. **Giannone, D., Lenza, M., & Primiceri, G. E. (2015)**. "Prior Selection for Vector Autoregressions"
   *Review of Economics and Statistics*, 97(2), 436-451.
   - Modern BVAR prior selection methodology

**Complexity**: O(p²d³ + nd²) where p = lag order, d = dimensions, n = observations

**Sources**:
- [Bayesian VAR Wikipedia](https://en.wikipedia.org/wiki/Bayesian_vector_autoregression)
- [Litterman 1986 Paper](https://www.tandfonline.com/doi/abs/10.1080/07350015.1986.10509491)

---

## Active Inference

### Free Energy Principle & Active Inference

**Implementation**: `/crates/active-inference-agent/src/lib.rs`

**Mathematical Foundation**:
- Variational Free Energy: F = E_q[ln q(φ) - ln p(φ, y)]
- Expected Free Energy: G = E[F(s', o')] = Epistemic Value + Pragmatic Value
- Action selection: π* = argmin_π G(π)

**Key References**:

1. **Friston, K. J. (2010)**. "The free-energy principle: a unified brain theory?"
   *Nature Reviews Neuroscience*, 11(2), 127-138.
   DOI: [10.1038/nrn2787](https://doi.org/10.1038/nrn2787)
   - Seminal introduction to the Free Energy Principle
   - Conceptual framework for brain function

2. **Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017)**. "Active Inference: A Process Theory"
   *Neural Computation*, 29(1), 1-49.
   DOI: [10.1162/NECO_a_00912](https://doi.org/10.1162/NECO_a_00912)
   - Process theory formulation
   - Mathematical foundations of active inference

3. **Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020)**. "Active inference on discrete state-spaces: A synthesis"
   *Journal of Mathematical Psychology*, 99, 102447.
   DOI: [10.1016/j.jmp.2020.102447](https://doi.org/10.1016/j.jmp.2020.102447)
   - Discrete state-space formulation
   - Practical implementation guide

4. **Parr, T., Pezzulo, G., & Friston, K. J. (2022)**. *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*
   MIT Press.
   - Comprehensive textbook on active inference

**Complexity**: O(|S|² × |O|) per update, where |S| = state space size, |O| = observation space

**Sources**:
- [Free Energy Principle Wikipedia](https://en.wikipedia.org/wiki/Free_energy_principle)
- [Friston 2010 Paper](https://www.uab.edu/medicine/cinl/images/KFriston_FreeEnergy_BrainTheory.pdf)
- [Active Inference GitHub Resources](https://github.com/BerenMillidge/FEP_Active_Inference_Papers)

---

## Swarm Intelligence

### Particle Swarm Optimization (PSO)

**Implementation**: `/crates/hyperphysics-optimization/src/algorithms/pso.rs`

**Mathematical Foundation**:
- Velocity update: v_i(t+1) = χ[ωv_i(t) + c₁r₁(p_i - x_i) + c₂r₂(g - x_i)]
- Position update: x_i(t+1) = x_i(t) + v_i(t+1)
- Constriction factor: χ = 2/|2 - φ - √(φ² - 4φ)| where φ = c₁ + c₂

**Key References**:

1. **Kennedy, J., & Eberhart, R. (1995)**. "Particle Swarm Optimization"
   *Proceedings of IEEE International Conference on Neural Networks*, 4, 1942-1948.
   DOI: [10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968)
   - Original PSO algorithm
   - Inspired by social behavior of bird flocking

2. **Clerc, M., & Kennedy, J. (2002)**. "The particle swarm - explosion, stability, and convergence in a multidimensional complex space"
   *IEEE Transactions on Evolutionary Computation*, 6(1), 58-73.
   DOI: [10.1109/4235.985692](https://doi.org/10.1109/4235.985692)
   - Constriction factor for guaranteed convergence
   - Stability analysis: c₁ + c₂ ≤ 4

3. **Shi, Y., & Eberhart, R. (1998)**. "A modified particle swarm optimizer"
   *IEEE International Conference on Evolutionary Computation*, 69-73.
   - Adaptive inertia weight
   - Linear decrease from ω_max to ω_min

4. **van den Bergh, F., & Engelbrecht, A. P. (2006)**. "A study of particle swarm optimization particle trajectories"
   *Information Sciences*, 176(8), 937-971.
   - Formal convergence analysis

**Convergence Conditions**:
- ω ∈ [0, 1) for convergence
- c₁ + c₂ ≤ 4 for stability (without constriction)
- With constriction: guaranteed convergence to a point

**Complexity**: O(I × N × D) where I = iterations, N = particles, D = dimensions

**Sources**:
- [Kennedy & Eberhart 1995](https://ieeexplore.ieee.org/document/488968/)
- [PSO Historical Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516836/)

---

### Grey Wolf Optimizer (GWO)

**Implementation**: `/crates/hyperphysics-optimization/src/algorithms/grey_wolf.rs`

**Mathematical Foundation**:
- Social hierarchy: α (best), β (second), δ (third), ω (rest)
- Position update: X(t+1) = (X₁ + X₂ + X₃)/3
- Where: X₁ = Xα - A₁·|C₁·Xα - X|, similar for β and δ
- Linear decrease: a = 2 - 2t/T (controls exploration/exploitation)

**Key References**:

1. **Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014)**. "Grey Wolf Optimizer"
   *Advances in Engineering Software*, 69, 46-61.
   DOI: [10.1016/j.advengsoft.2013.12.007](https://doi.org/10.1016/j.advengsoft.2013.12.007)
   - Original GWO algorithm
   - Inspired by grey wolf pack hunting behavior

2. **Faris, H., Aljarah, I., Al-Betar, M. A., & Mirjalili, S. (2018)**. "Grey wolf optimizer: a review of recent variants and applications"
   *Neural Computing and Applications*, 30(2), 413-435.
   - Comprehensive review of GWO variants
   - Applications and performance analysis

3. **Heidari, A. A., & Pahlavani, P. (2017)**. "An efficient modified grey wolf optimizer with Lévy flight for optimization tasks"
   *Applied Soft Computing*, 60, 115-134.
   - Enhanced GWO with Lévy flights

**Complexity**: O(I × N × D × 3) where I = iterations, N = population, D = dimensions

**Sources**:
- [Mirjalili 2014 Paper](https://www.sciencedirect.com/science/article/abs/pii/S0965997813001853)
- [GWO Comprehensive Review](https://airccse.com/jares/papers/1620jares03.pdf)

---

### Differential Evolution (DE)

**Implementation**: `/crates/hyperphysics-optimization/src/algorithms/differential_evolution.rs`

**Mathematical Foundation**:
- Mutation: v_i = x_r1 + F(x_r2 - x_r3)
- Crossover: u_i,j = { v_i,j if rand() < CR or j = j_rand; x_i,j otherwise }
- Selection: x_i(t+1) = { u_i if f(u_i) < f(x_i); x_i otherwise }

**Key References**:

1. **Storn, R., & Price, K. (1997)**. "Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces"
   *Journal of Global Optimization*, 11(4), 341-359.
   DOI: [10.1023/A:1008202821328](https://doi.org/10.1023/A:1008202821328)
   - Original DE algorithm
   - Note: No formal convergence proof in original paper

2. **Zaharie, D. (2009)**. "Influence of crossover on the behavior of Differential Evolution Algorithms"
   *Applied Soft Computing*, 9(3), 1126-1138.
   - Theoretical analysis of convergence
   - Parameter selection guidelines

3. **Das, S., & Suganthan, P. N. (2011)**. "Differential Evolution: A Survey of the State-of-the-Art"
   *IEEE Transactions on Evolutionary Computation*, 15(1), 4-31.
   - Comprehensive survey
   - Variants and applications

**Note**: Original 1997 paper acknowledged lack of formal convergence proof. Later work by Zaharie provided parameter selection analysis.

**Complexity**: O(I × N × D) where I = iterations, N = population, D = dimensions

**Sources**:
- [Storn & Price 1997](https://link.springer.com/article/10.1023/A:1008202821328)
- [DE Wikipedia](https://en.wikipedia.org/wiki/Differential_evolution)

---

### Cuckoo Search (CS)

**Implementation**: `/crates/hyperphysics-optimization/src/algorithms/cuckoo.rs`

**Mathematical Foundation**:
- Lévy flight: x_i(t+1) = x_i(t) + α ⊕ Lévy(λ)
- Lévy distribution: Lévy ~ u = t^(-λ), 1 < λ ≤ 3
- Discovery probability: pa ∈ [0, 1]

**Key References**:

1. **Yang, X.-S., & Deb, S. (2009)**. "Cuckoo Search via Lévy flights"
   *World Congress on Nature & Biologically Inspired Computing (NaBIC 2009)*, IEEE, 210-214.
   DOI: [10.1109/NABIC.2009.5393690](https://doi.org/10.1109/NABIC.2009.5393690)
   - Original cuckoo search algorithm
   - Lévy flight random walk

2. **Yang, X.-S., & Deb, S. (2014)**. "Cuckoo search: recent advances and applications"
   *Neural Computing and Applications*, 24(1), 169-174.
   - Algorithm improvements
   - Application survey

3. **Mantegna, R. N. (1994)**. "Fast, accurate algorithm for numerical simulation of Lévy stable stochastic processes"
   *Physical Review E*, 49(5), 4677.
   - Lévy flight generation algorithm

**Complexity**: O(I × N × D) where I = iterations, N = nests, D = dimensions

**Sources**:
- [Yang & Deb 2009](https://ieeexplore.ieee.org/document/5393690/)
- [Cuckoo Search Wikipedia](https://en.wikipedia.org/wiki/Cuckoo_search)

---

### Bat Algorithm (BA)

**Implementation**: `/crates/hyperphysics-optimization/src/algorithms/bat.rs`

**Mathematical Foundation**:
- Frequency: f_i = f_min + (f_max - f_min)β
- Velocity: v_i(t+1) = v_i(t) + (x_i(t) - x*)f_i
- Position: x_i(t+1) = x_i(t) + v_i(t+1)
- Loudness decay: A_i(t+1) = αA_i(t)

**Key References**:

1. **Yang, X.-S. (2010)**. "A New Metaheuristic Bat-Inspired Algorithm"
   *Nature Inspired Cooperative Strategies for Optimization (NICSO 2010)*, Springer, 65-74.
   DOI: [10.1007/978-3-642-12538-6_6](https://doi.org/10.1007/978-3-642-12538-6_6)
   - Original bat algorithm
   - Inspired by echolocation behavior

2. **Yang, X.-S., & He, X. (2013)**. "Bat algorithm: literature review and applications"
   *International Journal of Bio-Inspired Computation*, 5(3), 141-149.
   - Literature review
   - Applications in engineering

3. **Yilmaz, S., & Küçüksille, E. U. (2015)**. "A new modification approach on bat algorithm for solving optimization problems"
   *Applied Soft Computing*, 28, 259-275.
   - Algorithm improvements

**Complexity**: O(I × N × D) where I = iterations, N = bats, D = dimensions

**Sources**:
- [Yang 2010 Paper](https://link.springer.com/chapter/10.1007/978-3-642-12538-6_6)
- [Bat Algorithm Wikipedia](https://en.wikipedia.org/wiki/Bat_algorithm)

---

### Firefly Algorithm (FA)

**Implementation**: `/crates/hyperphysics-optimization/src/algorithms/firefly.rs`

**Mathematical Foundation**:
- Light intensity: I(r) = I₀ e^(-γr²)
- Attractiveness: β(r) = β₀ e^(-γr²)
- Movement: x_i = x_i + β₀ e^(-γr²ij)(x_j - x_i) + α(rand - 0.5)

**Key References**:

1. **Yang, X.-S. (2008)**. "Firefly Algorithms for Multimodal Optimization"
   *Stochastic Algorithms: Foundations and Applications (SAGA 2009)*, Springer, 169-178.
   DOI: [10.1007/978-3-642-04944-6_14](https://doi.org/10.1007/978-3-642-04944-6_14)
   - Original firefly algorithm
   - Multimodal optimization

2. **Fister, I., Fister Jr, I., Yang, X. S., & Brest, J. (2013)**. "A comprehensive review of firefly algorithms"
   *Swarm and Evolutionary Computation*, 13, 34-46.
   - Comprehensive review
   - Variants and applications

3. **Yang, X.-S. (2010)**. "Firefly algorithm, stochastic test functions and design optimisation"
   *International Journal of Bio-Inspired Computation*, 2(2), 78-84.
   - Test functions
   - Engineering applications

**Complexity**: O(I × N² × D) where I = iterations, N = fireflies, D = dimensions

**Sources**:
- [Firefly Algorithm Wikipedia](https://en.wikipedia.org/wiki/Firefly_algorithm)
- [Yang 2008 Paper](https://link.springer.com/chapter/10.1007/978-3-642-04944-6_14)

---

### Artificial Bee Colony (ABC)

**Implementation**: `/crates/hyperphysics-optimization/src/algorithms/bee_colony.rs`

**Mathematical Foundation**:
- Employed bee: v_ij = x_ij + φ_ij(x_ij - x_kj)
- Onlooker selection: p_i = fitness_i / Σfitness
- Scout bee: random search when limit exceeded

**Key References**:

1. **Karaboga, D. (2005)**. "An idea based on honey bee swarm for numerical optimization"
   *Technical Report TR06, Erciyes University*
   - Original ABC algorithm

2. **Karaboga, D., & Basturk, B. (2007)**. "A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm"
   *Journal of Global Optimization*, 39(3), 459-471.
   DOI: [10.1007/s10898-007-9149-x](https://doi.org/10.1007/s10898-007-9149-x)
   - Journal publication
   - Performance comparison with GA, PSO, DE

3. **Karaboga, D., & Akay, B. (2009)**. "A comparative study of Artificial Bee Colony algorithm"
   *Applied Mathematics and Computation*, 214(1), 108-132.
   - Comprehensive comparison study

**Complexity**: O(I × N × D) where I = iterations, N = food sources, D = dimensions

**Sources**:
- [ABC Scholarpedia](http://www.scholarpedia.org/article/Artificial_bee_colony_algorithm)
- [Karaboga 2007 Paper](https://link.springer.com/article/10.1007/s10898-007-9149-x)

---

### Ant Colony Optimization (ACO)

**Implementation**: `/crates/hyperphysics-optimization/src/algorithms/ant_colony.rs`

**Mathematical Foundation**:
- Pheromone update: τ_ij(t+1) = (1-ρ)τ_ij(t) + Δτ_ij
- Probability: p_ij = [τ_ij^α][η_ij^β] / Σ[τ_ik^α][η_ik^β]
- Evaporation: ρ ∈ [0, 1]

**Key References**:

1. **Dorigo, M. (1992)**. "Optimization, Learning and Natural Algorithms"
   *PhD Thesis, Politecnico di Milano, Italy*
   - Original ant system

2. **Dorigo, M., & Gambardella, L. M. (1997)**. "Ant colony system: a cooperative learning approach to the traveling salesman problem"
   *IEEE Transactions on Evolutionary Computation*, 1(1), 53-66.
   DOI: [10.1109/4235.585892](https://doi.org/10.1109/4235.585892)
   - Ant colony system variant

3. **Stützle, T., & Dorigo, M. (2002)**. "A short convergence proof for a class of ant colony optimization algorithms"
   *IEEE Transactions on Evolutionary Computation*, 6(4), 358-365.
   DOI: [10.1109/TEVC.2002.802444](https://doi.org/10.1109/TEVC.2002.802444)
   - Formal convergence proof
   - Proves P*(t) ≥ 1 - ε for large t

4. **Gutjahr, W. J. (2000)**. "A graph-based ant system and its convergence"
   *Future Generation Computer Systems*, 16(9), 873-888.
   - First convergence proof for ACO

**Note**: Original 1992 work established framework; formal convergence proofs came in 2000-2002.

**Complexity**: O(I × M × N²) where I = iterations, M = ants, N = nodes

**Sources**:
- [ACO Scholarpedia](http://www.scholarpedia.org/article/Ant_colony_optimization)
- [Convergence Proof 2002](https://ieeexplore.ieee.org/document/1027747/)

---

### Whale Optimization Algorithm (WOA)

**Implementation**: `/crates/hyperphysics-optimization/src/algorithms/whale.rs`

**Mathematical Foundation**:
- Encircling prey: D = |C·X*(t) - X(t)|, X(t+1) = X*(t) - A·D
- Spiral update: X(t+1) = D'·e^(bl)·cos(2πl) + X*(t)
- Search: X(t+1) = X_rand - A·D (when |A| > 1)

**Key References**:

1. **Mirjalili, S., & Lewis, A. (2016)**. "The Whale Optimization Algorithm"
   *Advances in Engineering Software*, 95, 51-67.
   DOI: [10.1016/j.advengsoft.2016.01.008](https://doi.org/10.1016/j.advengsoft.2016.01.008)
   - Original WOA algorithm
   - Bubble-net hunting strategy

2. **Aljarah, I., Faris, H., & Mirjalili, S. (2018)**. "Optimizing connection weights in neural networks using the whale optimization algorithm"
   *Soft Computing*, 22(1), 1-15.
   - Neural network training applications

3. **Mafarja, M. M., & Mirjalili, S. (2017)**. "Hybrid Whale Optimization Algorithm with simulated annealing for feature selection"
   *Neurocomputing*, 260, 302-312.
   - Hybrid approaches

**Complexity**: O(I × N × D) where I = iterations, N = whales, D = dimensions

**Sources**:
- [Mirjalili & Lewis 2016](https://www.sciencedirect.com/science/article/abs/pii/S0965997816300163)
- [WOA Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC6512044/)

---

## Risk Management

### Kelly Criterion

**Implementation**: `/crates/cwts-ultra/core/src/algorithms/risk_management.rs`

**Mathematical Foundation**:
- Formula: f* = (bp - q) / b
- Where: b = odds, p = win probability, q = 1-p
- For trading: f* = (μ - r) / σ² (continuous-time version)

**Key References**:

1. **Kelly, J. L. (1956)**. "A New Interpretation of Information Rate"
   *Bell System Technical Journal*, 35(4), 917-926.
   DOI: [10.1002/j.1538-7305.1956.tb03809.x](https://doi.org/10.1002/j.1538-7305.1956.tb03809.x)
   - Original Kelly criterion
   - Information theory foundations

2. **Thorp, E. O. (2006)**. "The Kelly Criterion in Blackjack Sports Betting, and the Stock Market"
   *Handbook of Asset and Liability Management*, 385-428.
   - Application to financial markets
   - Practical implementation

3. **MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2011)**. "The Kelly Capital Growth Investment Criterion"
   World Scientific.
   - Comprehensive treatment
   - Mathematical proofs

4. **Hakansson, N. H. (1971)**. "Capital growth and the mean-variance approach to portfolio selection"
   *Journal of Financial and Quantitative Analysis*, 6(1), 517-557.
   - Relationship to portfolio theory

**Note**: Kelly criterion is the ONLY position sizing formula with mathematical proof for long-term wealth maximization.

**Complexity**: O(1) for calculation

**Sources**:
- [Kelly Criterion Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Mathematical Framework](https://medium.com/@filipstcu/the-kelly-criterion-a-mathematical-framework-for-optimal-position-sizing-edf528eb8ee2)

---

### Value at Risk (VaR)

**Implementation**: Multiple locations in risk management modules

**Mathematical Foundation**:
- Definition: VaR_α = inf{x : P(L > x) ≤ α}
- Parametric: VaR = μ + σ·Φ^(-1)(α)
- Historical: VaR = q_α of empirical distribution
- Monte Carlo: VaR = q_α of simulated distribution

**Key References**:

1. **Jorion, P. (2006)**. *Value at Risk: The New Benchmark for Managing Financial Risk*, 3rd Ed.
   McGraw-Hill.
   - Comprehensive VaR textbook
   - All three calculation methods

2. **Duffie, D., & Pan, J. (1997)**. "An Overview of Value at Risk"
   *Journal of Derivatives*, 4(3), 7-49.
   - Theoretical overview
   - Mathematical foundations

3. **Glasserman, P., Heidelberger, P., & Shahabuddin, P. (2000)**. "Efficient Monte Carlo methods for value-at-risk"
   *IBM Research Report RC 21666*
   - Monte Carlo VaR methods
   - Variance reduction techniques

4. **Basel Committee on Banking Supervision (1996)**. "Amendment to the capital accord to incorporate market risks"
   - Regulatory framework
   - Industry standards

**Methods Comparison**:
- Parametric: Fast, assumes normality, O(1)
- Historical: No distribution assumption, O(n log n)
- Monte Carlo: Most flexible, computationally expensive, O(m·n)

**Complexity**:
- Parametric: O(1)
- Historical: O(n log n) for sorting
- Monte Carlo: O(m × n) where m = simulations, n = assets

**Sources**:
- [VaR Methods Comparison](https://elischolar.library.yale.edu/cgi/viewcontent.cgi?article=1605&context=ypfs-documents)
- [Monte Carlo VaR](https://www.pyquantnews.com/the-pyquant-newsletter/quickly-compute-value-at-risk-with-monte-carlo)

---

### Sharpe Ratio

**Implementation**: `/crates/cwts-ultra/core/src/algorithms/risk_management.rs`

**Mathematical Foundation**:
- Formula: SR = (R_p - R_f) / σ_p
- Where: R_p = portfolio return, R_f = risk-free rate, σ_p = portfolio std dev
- Annualized: SR_annual = SR_daily × √252

**Key References**:

1. **Sharpe, W. F. (1966)**. "Mutual Fund Performance"
   *Journal of Business*, 39(1), 119-138.
   - Original Sharpe ratio
   - Performance measurement

2. **Sharpe, W. F. (1994)**. "The Sharpe Ratio"
   *Journal of Portfolio Management*, 21(1), 49-58.
   DOI: [10.3905/jpm.1994.409501](https://doi.org/10.3905/jpm.1994.409501)
   - Updated formulation
   - Interpretation guidelines

3. **Lo, A. W. (2002)**. "The Statistics of Sharpe Ratios"
   *Financial Analysts Journal*, 58(4), 36-52.
   - Statistical properties
   - Bias corrections

**Limitations**: Penalizes both upside and downside volatility equally

**Complexity**: O(n) where n = number of returns

**Sources**:
- [Sharpe Ratio Formula](https://www.wallstreetmojo.com/sharpe-ratio/)

---

### Sortino Ratio

**Implementation**: `/crates/cwts-ultra/core/src/algorithms/risk_management.rs`

**Mathematical Foundation**:
- Formula: Sortino = (R_p - T) / σ_d
- Where: T = target return, σ_d = downside deviation
- Downside deviation: σ_d = √(Σ min(0, R_i - T)² / n)

**Key References**:

1. **Sortino, F. A., & Price, L. N. (1994)**. "Performance measurement in a downside risk framework"
   *Journal of Investing*, 3(3), 59-64.
   - Original Sortino ratio
   - Downside risk focus

2. **Sortino, F. A., & van der Meer, R. (1991)**. "Downside risk"
   *Journal of Portfolio Management*, 17(4), 27-31.
   - Downside deviation methodology

3. **Rollinger, T. N., & Hoffman, S. T. (2013)**. "Sortino: A 'Sharper' Ratio"
   *CME Group Research*
   - Practical comparison with Sharpe
   - Implementation guidelines

**Advantages**: Only penalizes downside volatility, more appropriate for asymmetric returns

**Complexity**: O(n) where n = number of returns

**Sources**:
- [Sortino Ratio Wikipedia](https://en.wikipedia.org/wiki/Sortino_ratio)
- [CME Sortino Paper](https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf)

---

## Computational Complexity Summary

| Algorithm | Time Complexity | Space Complexity | Convergence |
|-----------|----------------|------------------|-------------|
| PSO | O(I × N × D) | O(N × D) | Guaranteed (with constriction) |
| GWO | O(I × N × D) | O(N × D) | Heuristic |
| DE | O(I × N × D) | O(N × D) | Heuristic |
| Cuckoo | O(I × N × D) | O(N × D) | Heuristic |
| Bat | O(I × N × D) | O(N × D) | Heuristic |
| Firefly | O(I × N² × D) | O(N × D) | Heuristic |
| ABC | O(I × N × D) | O(N × D) | Heuristic |
| ACO | O(I × M × N²) | O(N²) | Proven (Stützle 2002) |
| WOA | O(I × N × D) | O(N × D) | Heuristic |
| BVAR | O(p²d³ + nd²) | O(pd²) | Bayesian convergence |

Where: I = iterations, N = population size, D = dimensions, p = lag order, d = variables, M = ants, n = observations

---

## Implementation Validation Checklist

For each algorithm implementation:

- [ ] Peer-reviewed citations (minimum 3-4)
- [ ] Mathematical correctness verification
- [ ] Computational complexity documented
- [ ] Error bounds specified (where applicable)
- [ ] Numerical stability checks
- [ ] Unit tests for mathematical properties
- [ ] Convergence tests (where proofs exist)
- [ ] Parameter validation
- [ ] Edge case handling

---

## References Format

All references follow this format:
```
Author(s). (Year). "Title"
Journal/Conference, Volume(Issue), Pages.
DOI: [link]
```

---

## Maintenance Notes

- **Last Updated**: 2025-01-24
- **Review Cycle**: Quarterly
- **Citation Standard**: APA 7th Edition
- **Verification**: All DOIs and URLs checked

---

## Contributing

When adding new algorithms:

1. Search peer-reviewed literature (minimum 3-4 papers)
2. Document mathematical formulation
3. Specify computational complexity
4. Add convergence conditions (if proven)
5. Update this document with full citations
6. Add inline code comments with references

---

## License

Scientific citations are factual information and cannot be copyrighted.
This compilation is provided for academic and research purposes.
