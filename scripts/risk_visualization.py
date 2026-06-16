import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.markov_embedder import MarkovMatrixEmbedder
from src.ifrs9_extrapolator import IFRS9CreditPricer

# Quantitative aesthetics configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'axes.edgecolor': '#333333'
})

def plot_3d_risk_topology(pricer: IFRS9CreditPricer, state_labels: list, max_years: float = 10.0) -> None:
    """
    Renders a 3D topology of continuous credit risk across the rating spectrum.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    time_steps = 50
    t_grid = np.linspace(0, max_years, time_steps)
    rating_indices = np.arange(len(state_labels) - 1)
    
    X, Y = np.meshgrid(t_grid, rating_indices)
    Z = np.zeros_like(X)
    
    for i in rating_indices:
        _, pd_val = pricer.compute_continuous_pd_curve(i, max_years, time_steps)
        Z[i, :] = pd_val * 100.0 
        
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.85)
    
    ax.set_yticks(rating_indices)
    ax.set_yticklabels(state_labels[:-1])
    ax.set_xlabel('\nTime Horizon (Years)', labelpad=10)
    ax.set_ylabel('\nInitial Rating', labelpad=10)
    ax.set_zlabel('\nCumulative PD (%)', labelpad=10)
    ax.set_title("Continuous Default Risk Topology (Markovian Framework)", pad=20)
    
    fig.colorbar(surf, shrink=0.5, aspect=10, label="Risk Intensity (%)")
    plt.tight_layout()
    plt.show()

def plot_stepwise_vs_baseline(pricer: IFRS9CreditPricer, rating_idx: int, label: str) -> None:
    """
    Deviation analysis between a Baseline trajectory and a non-homogeneous Stepwise trajectory.
    """
    t_base, pd_base = pricer.compute_continuous_pd_curve(rating_idx, max_years=5, steps=100)
    t_step, pd_step = pricer.compute_stepwise_stress_pd(rating_idx, stress_factor=2.5, stress_duration=2.0, max_years=5, steps=100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_base, pd_base * 100, label="Baseline (Neutral Environment)", color="#2c3e50", lw=2.5)
    plt.plot(t_step, pd_step * 100, label="Stepwise (2Y Crisis + Reversion)", color="#c0392b", lw=2.5, linestyle="--")
    
    plt.axvline(x=2.0, color="#7f8c8d", linestyle=":", label="Regime Shift ($t=2$)", lw=2)
    
    plt.title(f"Non-Homogeneous Dynamics: Macroeconomic Shock (Rating {label})")
    plt.xlabel("Horizon (Years)")
    plt.ylabel("Cumulative Probability of Default (%)")
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    labels = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC/C', 'D', 'NR']
    
    # S&P Global 2024 Transition Matrix (Expressed in percentages)
    raw_P = np.array([
        [87.28, 8.92, 0.51, 0.03, 0.10, 0.03, 0.05, 0.00, 3.08],
        [0.45, 87.74, 7.50, 0.44, 0.05, 0.06, 0.02, 0.02, 3.74],
        [0.02, 1.48, 89.42, 4.64, 0.23, 0.10, 0.01, 0.05, 4.04],
        [0.00, 0.07, 3.05, 87.33, 3.21, 0.40, 0.09, 0.14, 5.71],
        [0.01, 0.02, 0.10, 4.44, 78.89, 6.25, 0.50, 0.56, 9.23],
        [0.00, 0.02, 0.06, 0.14, 4.47, 75.18, 4.79, 2.93, 12.41],
        [0.00, 0.00, 0.07, 0.13, 0.40, 13.18, 45.07, 26.12, 15.03]
    ]) / 100.0

    embedder = MarkovMatrixEmbedder(raw_P, labels)
    Q_generator = embedder.get_generator()
    
    pricer = IFRS9CreditPricer(Q_generator)
    
    plot_3d_risk_topology(pricer, labels)
    plot_stepwise_vs_baseline(pricer, rating_idx=3, label='BBB')
