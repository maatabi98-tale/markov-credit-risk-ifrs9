import numpy as np
import scipy.linalg as la
import scipy.stats as stats
from typing import Tuple

class IFRS9CreditPricer:
    """
    Extrapolates Lifetime PDs and applies macroeconomic structural shocks 
    using the Vasicek Single Factor framework and Non-Homogeneous Markov Chains.
    """
    def __init__(self, base_generator: np.ndarray):
        self.Q = base_generator
        self.num_states = base_generator.shape[0]

    def compute_continuous_pd_curve(self, rating_idx: int, max_years: float, steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the exact continuous-time cumulative PD curve using matrix exponentiation.
        """
        t_array = np.linspace(0, max_years, steps)
        pd_array = np.zeros(steps)
        
        for idx, t in enumerate(t_array):
            P_t = la.expm(self.Q * t)
            pd_array[idx] = P_t[rating_idx, -1] # Probability of transition to the absorbing state
            
        return t_array, pd_array

    @staticmethod
    def apply_vasicek_adjustment(pd_ttc: np.ndarray, rho: float, z_score: float) -> np.ndarray:
        """
        Transforms Through-The-Cycle (TTC) PD into Point-In-Time (PIT) PD based on a systemic factor.
        """
        pd_clipped = np.clip(pd_ttc, 1e-9, 1 - 1e-9)
        numerator = stats.norm.ppf(pd_clipped) - np.sqrt(rho) * z_score
        denominator = np.sqrt(1 - rho)
        return stats.norm.cdf(numerator / denominator)

    def compute_stepwise_stress_pd(self, rating_idx: int, stress_factor: float, 
                                   stress_duration: float, max_years: float, steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates a non-homogeneous Markov trajectory: an initial high-stress regime 
        followed by a reversion to the baseline transition intensity.
        """
        t_array = np.linspace(0, max_years, steps)
        pd_array = np.zeros(steps)
        
        Q_stress = self.Q * stress_factor
        
        for idx, t in enumerate(t_array):
            if t <= stress_duration:
                P_t = la.expm(Q_stress * t)
            else:
                # Multiplicative phase separation preserving the Markov property
                P_t = la.expm(Q_stress * stress_duration) @ la.expm(self.Q * (t - stress_duration))
                
            pd_array[idx] = P_t[rating_idx, -1]
            
        return t_array, pd_array
