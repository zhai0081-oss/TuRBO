###############################################################################
# Pure Bayesian Optimization (PureBO) Class                                   #
# This class disables the Trust Region (TR) logic of Turbo1.                  #
###############################################################################

import math
from copy import deepcopy
import sys
import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from .gp import train_gp 
from .turbo_1 import Turbo1 
from .utils import from_unit_cube, latin_hypercube, to_unit_cube


class PureBO(Turbo1):
    """
    Pure Bayesian Optimization (PureBO) baseline. 
    
    This class inherits from Turbo1 but disables all Trust Region (TR) 
    mechanisms, forcing optimization to be done on the full search space 
    using global Thompson Sampling.
    """

    def __init__(self, *args, **kwargs):
        # Inherit all initialization and settings from Turbo1
        super().__init__(*args, **kwargs)
        
        # Disable all TR-related counters and lengths
        self.length = self.length_max  # Fix length to max value
        self.length_init = self.length_max
        self.length_min = 0.0          # Disable length contraction check
        self.failcount = 0
        self.succcount = 0

    def _adjust_length(self, fX_next):
        """Pure BO: Performs no trust region adjustment."""
        pass

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        """Generates candidates in the WHOLE [0,1]^d space, ignoring the 'length' parameter."""
        
        # GP Training (Same as Turbo1/TurboM)
        assert X.min() >= 0.0 and X.max() <= 1.0

        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )
            hypers = gp.state_dict()
        
        # Candidate Generation (Core change: Global Sobol Sampling)
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        
        # Generate Sobol sequence over the entire [0, 1]^d space
        X_cand = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        
        # Thompson Sampling (Same as Turbo1/TurboM)
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        gp = gp.to(dtype=dtype, device=device)

        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        del X_torch, y_torch, X_cand_torch, gp
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def optimize(self):
        """Runs Pure BO (Global Thompson Sampling/Acquisition) without restarts or TR checks."""
        
        # Initialization
        self._restart()
        X_init = latin_hypercube(self.n_init, self.dim)
        X_init = from_unit_cube(X_init, self.lb, self.ub)
        fX_init = np.array([[self.f(x)] for x in X_init])

        self.n_evals += self.n_init
        self._X = deepcopy(X_init)
        self._fX = deepcopy(fX_init)
        self.X = np.vstack((self.X, deepcopy(X_init)))
        self.fX = np.vstack((self.fX, deepcopy(fX_init)))

        if self.verbose:
            fbest = self._fX.min()
            print(f"Pure BO Starting from fbest = {fbest:.4}")
            sys.stdout.flush()

        # Main loop: run until max_evals is reached
        while self.n_evals < self.max_evals:
            # Data preparation
            X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)
            fX = deepcopy(self._fX).ravel()

            # Generate global candidates
            X_cand, y_cand, _ = self._create_candidates(
                X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
            )
            X_next = self._select_candidates(X_cand, y_cand)

            # Evaluate
            X_next = from_unit_cube(X_next, self.lb, self.ub)
            fX_next = np.array([[self.f(x)] for x in X_next])

            # No self._adjust_length(fX_next)

            # Update history
            self.n_evals += self.batch_size
            self._X = np.vstack((self._X, X_next))
            self._fX = np.vstack((self._fX, fX_next))
            
            if self.verbose and fX_next.min() < self.fX.min():
                n_evals, fbest = self.n_evals, fX_next.min()
                print(f"{n_evals}) New best: {fbest:.4}")
                sys.stdout.flush()

            self.X = np.vstack((self.X, deepcopy(X_next)))
            self.fX = np.vstack((self.fX, deepcopy(fX_next)))
