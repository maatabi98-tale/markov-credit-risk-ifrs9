# Stochastic Credit Risk Modeling & IFRS 9 Implementation

## Repository Architecture
This repository implements a continuous-time Markov projection engine engineered to evaluate Lifetime Probability of Default (PD) curves under the IFRS 9 accounting standard. The framework resolves the structural Matrix Embeddability problem and integrates macroeconomic shocks via non-homogeneous Markov chains (Stepwise and Scalar Stress regimes).

## Mathematical Framework

### 1. Matrix Embeddability and the Infinitesimal Generator
Extracting continuous dynamics from a discrete-time empirical transition matrix $P$ requires computing its principal matrix logarithm $Q_{raw} = \log(P)$. Due to empirical noise and sampling constraints, $Q_{raw}$ frequently violates the structural conditions of a valid Markov generator (exhibiting negative off-diagonal intensities). We enforce a regularization manifold:
$$ \tilde{q}_{ij} = \max(q_{ij}, 0) \quad \forall i \neq j $$
$$ \tilde{q}_{D, j} = 0 \quad (\text{Absorbing state } D) $$
$$ \tilde{q}_{ii} = -\sum_{j \neq i} \tilde{q}_{ij} $$
The projected cumulative probability matrix at continuous horizon $t$ is evaluated via matrix exponentiation: $P(t) = \exp(\tilde{Q} t)$.

### 2. Vasicek Single Factor Model (Point-In-Time Adjustment)
The conversion of historical Through-The-Cycle (TTC) probabilities into Point-In-Time (PIT) measures conditioned by macroeconomic stress utilizes the Vasicek Gaussian copula function:
$$ PD_{PIT}(t) = \Phi\left( \frac{\Phi^{-1}(PD_{TTC}(t)) - \sqrt{\rho}Z}{\sqrt{1-\rho}} \right) $$
Where $Z \sim \mathcal{N}(0,1)$ quantifies the latent systemic macroeconomic shock and $\rho$ represents the intra-sectoral asset correlation coefficient.

### 3. Non-Homogeneous Dynamics (Stepwise Trajectories)
For a macroeconomic trajectory modeled by piecewise constant regimes over intervals $[t_{k-1}, t_k]$, the forward transition probability matrix is derived from the non-commutative multiplicative product of the resolvents:
$$ P(0, T) = \prod_{k=1}^{n} \exp\left( Q_k \Delta t_k \right) $$

---

## Production Codebase

```text
├── README.md
├── src/
│   ├── __init__.py
│   ├── markov_embedder.py      # Spectral regularization and Q-generator extraction
│   ├── ifrs9_extrapolator.py   # Matrix exponentiation and Vasicek stress-testing
├── scripts/
│   ├── risk_visualization.py   # 3D topology and PD curve generation
