import numpy as np
from tqdm import tqdm
import pywt


class CMAESSearch:
    def __init__(
        self,
        fit_fun,
        K=10,
        C=3,
        T=75,
        limit=0.01,
        pop_size=None,
        max_iter=1000,
        sigma0=0.3,
        use_warm_start=False,
        scale_warm=1.0,
        use_wavelet=False,
        wavelet='db4',
        compression_ratio=0.5,
        seed=42,
    ):
        # Convert all parameters to proper scalars
        self.K = int(np.asarray(K).item())
        self.C = int(np.asarray(C).item())
        self.T = int(np.asarray(T).item())
        self.num_vertices = self.K + 1
        self.limit = float(np.asarray(limit).item())
        self.max_iter = int(np.asarray(max_iter).item())
        self.sigma0 = float(np.asarray(sigma0).item())
        self.use_warm_start = bool(use_warm_start)
        self.scale_warm = float(np.asarray(scale_warm).item())
        self.fit_fun = fit_fun
        self.rng = np.random.default_rng(seed)
        
        # Problem dimension
        self.num_channels = int(np.asarray(C).item())
        
        # Wavelet parameters
        self.use_wavelet = bool(use_wavelet)
        self.wavelet = str(wavelet)
        self.compression_ratio = float(np.asarray(compression_ratio).item())
        
        # Set dimension based on representation
        if self.use_wavelet:
            # Calculate wavelet dimension with proper length handling
            dummy_signal = np.zeros(self.T)
            coeffs = pywt.wavedec(dummy_signal, self.wavelet, mode='periodization')
            total_coeffs = sum(len(c) for c in coeffs)
            
            # Store coefficient structure for reconstruction
            self.coeffs_structure = [len(c) for c in coeffs]
            
            if self.compression_ratio < 1.0:
                # Keep most important coefficients (low-frequency first)
                n_keep = int(total_coeffs * self.compression_ratio)
                self.wavelet_indices = self._get_low_freq_indices(coeffs, n_keep)
                self.dim = len(self.wavelet_indices) * self.num_channels
            else:
                self.wavelet_indices = None
                self.dim = total_coeffs * self.num_channels
        else:
            self.dim = self.num_channels * self.num_vertices
        
        # Population size (lambda)
        if pop_size is None:
            self.pop_size = int(4 + 3 * np.log(self.dim))
        else:
            self.pop_size = int(np.asarray(pop_size).item())
            
        # Parent population size (mu)
        self.mu = self.pop_size // 2
        
        # Initialize CMA-ES parameters
        self._init_cmaes_params()

    def _get_low_freq_indices(self, coeffs, n_keep):
        """Get indices for keeping low-frequency wavelet coefficients"""
        indices = []
        start_idx = 0
        remaining = n_keep
        
        # Prioritize approximation coefficients (lowest frequency)
        for level, coeff in enumerate(coeffs):
            if remaining <= 0:
                break
            take = min(len(coeff), remaining)
            indices.extend(range(start_idx, start_idx + take))
            start_idx += len(coeff)
            remaining -= take
        
        return indices

    def _trigger_to_wavelet(self, trigger):
        """Convert trigger to wavelet coefficient representation"""
        params = []
        for c in range(self.num_channels):
            coeffs = pywt.wavedec(trigger[:, c], self.wavelet, mode='periodization')
            coeffs_flat = np.concatenate(coeffs)
            
            if self.wavelet_indices is not None:
                coeffs_flat = coeffs_flat[self.wavelet_indices]
            
            params.extend(coeffs_flat)
        
        return np.array(params)

    def _wavelet_to_trigger(self, params):
        """Convert wavelet coefficients back to trigger"""
        trigger = np.zeros((self.T, self.num_channels))
        coeffs_per_channel = len(params) // self.num_channels
        
        for c in range(self.num_channels):
            start_idx = c * coeffs_per_channel
            channel_coeffs = params[start_idx:start_idx + coeffs_per_channel]
            
            # Reconstruct full coefficient array if using compression
            if self.wavelet_indices is not None:
                dummy_signal = np.zeros(self.T)
                full_coeffs = pywt.wavedec(dummy_signal, self.wavelet, mode='periodization')
                full_coeffs_flat = np.concatenate(full_coeffs)
                full_coeffs_flat[self.wavelet_indices] = channel_coeffs
                
                # Reconstruct coeffs structure
                coeffs_reconstructed = []
                start = 0
                for coeff_len in self.coeffs_structure:
                    end = start + coeff_len
                    coeffs_reconstructed.append(full_coeffs_flat[start:end])
                    start = end
                
                # Fix the dimension mismatch by ensuring proper reconstruction length
                reconstructed = pywt.waverec(coeffs_reconstructed, self.wavelet, mode='periodization')
                trigger[:, c] = reconstructed[:self.T]  # Truncate to exact length
            else:
                # Direct reconstruction
                coeffs_reconstructed = []
                start = 0
                for coeff_len in self.coeffs_structure:
                    end = start + coeff_len
                    coeffs_reconstructed.append(channel_coeffs[start:end])
                    start = end
                
                reconstructed = pywt.waverec(coeffs_reconstructed, self.wavelet, mode='periodization')
                trigger[:, c] = reconstructed[:self.T]  # Truncate to exact length
        
        return np.clip(trigger, -self.limit, self.limit)

    def _init_cmaes_params(self):
        """Initialize CMA-ES strategy parameters"""
        # Selection weights
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        
        # Step size control parameters
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / 
                      ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        
        # Initialize evolution paths and covariance matrix
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.eye(self.dim)  # eigenvectors
        self.D = np.ones(self.dim)  # eigenvalues^0.5
        self.cov_matrix = np.eye(self.dim)  # covariance matrix
        self.invsqrtC = np.eye(self.dim)  # C^(-1/2)
        
        # Step size
        self.sigma = self.sigma0
        
        # Expected length of random vector
        self.chiN = np.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 
                                        1.0 / (21 * self.dim ** 2))
        
        # Generation counter
        self.counteval = 0
        self.eigeneval = 0

    def _make_trigger(self, params):
        """Convert parameters to trigger format"""
        if self.use_wavelet:
            return self._wavelet_to_trigger(params)
        else:
            # Original vertex interpolation method
            params = params.reshape(self.num_channels, self.num_vertices)
            trigger = np.zeros((self.T, self.num_channels))
            segment_len = self.T / self.K
            for c in range(self.num_channels):
                for k in range(self.K):
                    start_val = params[c, k]
                    end_val = params[c, k + 1]
                    start_idx = int(round(k * segment_len))
                    end_idx = int(round((k + 1) * segment_len))
                    end_idx = min(end_idx, self.T)
                    if end_idx > start_idx:
                        interp = np.linspace(start_val, end_val, end_idx - start_idx, endpoint=False)
                        trigger[start_idx:end_idx, c] = interp
            
            return np.clip(trigger, -self.limit, self.limit)

    def warm_start(self, candidates):
        """Initialize mean from best candidate trigger"""
        best_trigger, best_score = None, -np.inf
        for t in candidates:
            scaled = np.clip(t * self.scale_warm, -self.limit, self.limit)
            score = self.fit_fun(scaled)
            if score > best_score:
                best_score = score
                best_trigger = scaled
    
        if best_trigger is not None:
            if self.use_wavelet:
                # Convert trigger to wavelet parameters
                params = self._trigger_to_wavelet(best_trigger)
            else:
                # Original vertex method
                C_val = self.num_channels
                vertices_val = int(self.num_vertices)
                params = np.zeros((C_val, vertices_val))
                for c in range(C_val):
                    for v in range(vertices_val):
                        T_val = int(np.asarray(self.T).item())
                        t_idx = int(round(v * (T_val - 1) / (vertices_val - 1)))
                        params[c, v] = best_trigger[t_idx, c]
                params = params.flatten()
                
            return params, best_trigger, best_score
        
        return None, None, -np.inf
        
    def _fitness(self, params):
        """Evaluate fitness of parameter vector"""
        # Clip parameters to bounds
        params_clipped = np.clip(params, -self.limit, self.limit)
        trigger = self._make_trigger(params_clipped)
        return self.fit_fun(trigger)

    def _update_covariance_matrix(self):
        """Update covariance matrix and related matrices"""
        if self.counteval - self.eigeneval > self.pop_size / (self.c1 + self.cmu) / self.dim / 10:
            self.eigeneval = self.counteval
            
            # Eigendecomposition
            D2, self.B = np.linalg.eigh(self.cov_matrix)
            self.D = np.sqrt(np.maximum(D2, 0))
            
            # Update invsqrtC
            self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T

    def search_trigger(self, candidates=None, patience=50):
        """Main CMA-ES optimization loop"""
        # Initialize mean
        if self.use_warm_start and candidates is not None:
            warm_params, warm_trigger, warm_score = self.warm_start(candidates)
            if warm_params is not None:
                self.mean = warm_params.copy()
            else:
                self.mean = self.rng.uniform(-self.limit, self.limit, self.dim)
        else:
            self.mean = self.rng.uniform(-self.limit, self.limit, self.dim)
        
        best_params = self.mean.copy()
        best_score = self._fitness(best_params)
        
        pbar = tqdm(range(self.max_iter), desc="CMA-ES")
        no_improve_counter = 0
        
        for generation in pbar:
            if no_improve_counter >= patience:
                print(f"Early stopping: no improvement for {patience} generations.")
                break
            
            # Generate population
            population = []
            fitness_scores = []
            
            for _ in range(self.pop_size):
                # Sample from multivariate normal
                z = self.rng.standard_normal(self.dim)
                y = self.B @ (self.D * z)  # y ~ N(0, C)
                x = self.mean + self.sigma * y
                
                population.append(x)
                fitness_scores.append(self._fitness(x))
            
            population = np.array(population)
            fitness_scores = np.array(fitness_scores)
            self.counteval += self.pop_size
            
            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = population[sorted_indices]
            fitness_scores = fitness_scores[sorted_indices]
            
            # Update best
            if fitness_scores[0] > best_score:
                best_score = fitness_scores[0]
                best_params = population[0].copy()
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            
            # Selection and recombination
            selected = population[:self.mu]
            old_mean = self.mean.copy()
            self.mean = np.sum(self.weights[:, np.newaxis] * selected, axis=0)
            
            # Update evolution paths
            y_mean = (self.mean - old_mean) / self.sigma
            z_mean = self.invsqrtC @ y_mean
            
            # Update ps (step size path)
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_mean
            
            # Update pc (covariance path)
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.pop_size)) < 1.4 + 2 / (self.dim + 1)
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_mean
            
            # Update covariance matrix
            artmp = (selected - old_mean) / self.sigma
            self.cov_matrix = ((1 - self.c1 - self.cmu) * self.cov_matrix + 
                     self.c1 * (self.pc[:, np.newaxis] @ self.pc[np.newaxis, :] + 
                               (1 - hsig) * self.cc * (2 - self.cc) * self.cov_matrix) +
                     self.cmu * artmp.T @ np.diag(self.weights) @ artmp)
            
            # Update step size
            self.sigma = self.sigma * np.exp((self.cs / self.damps) * 
                                           (np.linalg.norm(self.ps) / self.chiN - 1))
            
            # Update eigendecomposition
            self._update_covariance_matrix()
            
            pbar.set_postfix(score=f"{best_score:.6f}", no_improve=f"{no_improve_counter}", sigma=f"{self.sigma:.4f}")
        
        # Convert best parameters to trigger
        best_trigger = self._make_trigger(best_params)
        
        # Try rolling optimization like in genetic search
        final_trigger = best_trigger
        final_score = best_score
        best_shift = 0
        
        for shift in range(-74, 75):
            shifted = np.roll(best_trigger, shift=shift, axis=0)
            score = self.fit_fun(shifted)
            if score > final_score:
                final_score = score
                final_trigger = shifted
                best_shift = shift
        
        if best_shift != 0:
            print(f"Rolled final trigger by {best_shift} → improved score to {final_score:.6f}")
        else:
            print("No rolling improvement found.")
        
        return final_trigger, final_score