# Part 3: Systemic Risk with PageRank

## Portfolio Risk Solution Results

### Forward-Push Algorithm Performance
- **Converged**: ✅ Yes (in 37 iterations)
- **Compute Time**: 3ms
- **Residual Error**: 0.000122 (excellent precision)
- **Memory Usage**: Minimal (0 reported)

### Optimal Hedge Ratios
Based on our $2.1M portfolio:
- **Equities**: 1.137 hedge ratio (114% hedge)
- **Bonds**: -0.070 hedge ratio (7% counter-hedge)
- **Commodities**: -0.030 hedge ratio (3% counter-hedge)
- **Real Estate**: 0.563 hedge ratio (56% hedge)
- **Currencies**: -0.011 hedge ratio (1% counter-hedge)

## Systemic Risk Analysis with PageRank

Now let's analyze systemic risk using PageRank to identify which assets pose the highest contagion risk in our portfolio network.

### Risk Network Structure
Each asset is a node, correlations are edges. Higher correlations = stronger connections = higher systemic risk transmission.