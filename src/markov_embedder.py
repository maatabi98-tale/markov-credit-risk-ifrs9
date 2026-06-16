import numpy as np
import scipy.linalg as la
from typing import Tuple, List

class MarkovMatrixEmbedder:
    """
    Handles the transformation of empirical discrete-time transition matrices 
    into valid continuous-time infinitesimal generators.
    """
    def __init__(self, raw_matrix: np.ndarray, state_labels: List[str]):
        self.raw_matrix = raw_matrix
        self.state_labels = state_labels
        self.num_states = len(state_labels)
        self._P_cleaned = self._preprocess_empirical_matrix()
        self.Q_generator = self._compute_and_regularize_generator()

    def _preprocess_empirical_matrix(self) -> np.ndarray:
        """
        Redistributes the 'Not Rated' (NR) sink state proportionally across active states 
        and enforces the strict absorbing nature of the Default (D) state.
        """
        # Isolate the NR column (assumed to be the last column)
        P_no_nr = self.raw_matrix[:, :-1]
        row_sums = P_no_nr.sum(axis=1, keepdims=True)
        
        # Prevent zero-division for terminal states
        row_sums[row_sums == 0] = 1.0 
        P_cleaned = P_no_nr / row_sums
        
        # Enforce strict absorbing constraints for the Default row
        default_idx = self.num_states - 1
        absorbing_row = np.zeros(self.num_states)
        absorbing_row[default_idx] = 1.0
        
        # Replace the empirical default row with the theoretical absorbing row
        P_cleaned = np.vstack([P_cleaned[:-1, :], absorbing_row])
        return P_cleaned

    def _compute_and_regularize_generator(self) -> np.ndarray:
        """
        Executes the matrix logarithm and regularizes the spectrum to resolve 
        the embeddability problem (eliminating negative off-diagonal intensities).
        """
        Q_raw = la.logm(self._P_cleaned)
        
        # Suppress floating point complex artifacts
        Q_real = np.real(Q_raw)
        
        Q_opt = np.copy(Q_real)
        np.fill_diagonal(Q_opt, 0.0)
        
        # Non-negativity constraint for off-diagonals (Capping mechanism)
        Q_opt = np.maximum(Q_opt, 0.0)
        
        # Enforce Default state as strictly absorbing in continuous time
        Q_opt[-1, :] = 0.0
        
        # Conservation of probability mass (Row sums must strictly equal 0)
        row_sums = np.sum(Q_opt, axis=1)
        np.fill_diagonal(Q_opt, -row_sums)
        
        return Q_opt

    def get_discrete_matrix(self) -> np.ndarray:
        return self._P_cleaned
        
    def get_generator(self) -> np.ndarray:
        return self.Q_generator
