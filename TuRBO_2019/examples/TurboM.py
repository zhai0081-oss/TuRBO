###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

# ==============================================================================
# NOTICE OF MODIFICATION:
# This file has been modified from the original source code.
#
# Original Work: https://github.com/uber-research/TuRBO
# Modifications made by: Ziyi
# Date of modifications: 2025-10-1
# ==============================================================================

# Simple example of TuRBO-m

from matplotlib import cm # tur visualization
from mpl_toolkits.mplot3d import Axes3D # tur visualization
from turbo import TurboM
from turbo import Turbo1
from turbo import PureBO

import wandb
import numpy as np
import torch
import math
import matplotlib
import matplotlib.pyplot as plt

DIM = 40

wandb.init(
    project="turbo",
    name=f"levy-{DIM}d-comparison",
    config={      
        "dim": DIM,
        "n_init": 10 * DIM,
        "max_evals": 100 * DIM,
        "n_trust_regions": 5,
        "batch_size": 10,
        "device": "cpu"  # record only, ineffective in practice, as device is hardcoded in TurboM
    }
)

# Get config values

N_INIT = wandb.config["n_init"]
MAX_EVALS = wandb.config["max_evals"]
BATCH_SIZE = wandb.config["batch_size"]


# Set up an optimization problem class

class Levy:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
            (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
        return val

f = Levy(DIM)


# Enhanced TurboM class for history logging (Multi-TR)

class EnhancedTurboM(TurboM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trust_region_history = []

    def _adjust_length(self, fX_next, i):
        super()._adjust_length(fX_next, i)

        # record current trust region state
        tr_info = []
        for tr_idx in range(self.n_trust_regions):
            idx = np.where(self._idx[:, 0] == tr_idx)[0]
            if len(idx) == 0:
                tr_info.append({
                    "length": self.length[tr_idx],
                    "center": None,
                    "best_value": None,
                    "success_counter": self.succcount[tr_idx],
                    "failure_counter": self.failcount[tr_idx],
                })
            else:
                center = np.mean(self.X[idx], axis=0)
                best_value = np.min(self.fX[idx])
                tr_info.append({
                    "length": self.length[tr_idx],
                    "center": center,
                    "best_value": best_value,
                    "success_counter": self.succcount[tr_idx],
                    "failure_counter": self.failcount[tr_idx],
                })
        self.trust_region_history.append(tr_info)


# Enhanced Turbo1 class for history logging (Single-TR)

class EnhancedTurbo1(Turbo1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length_history = [] 
        self.success_history = []
        self.failure_history = []

    def _adjust_length(self, fX_next):
        # Call parent Turbo1's length adjustment logic
        super()._adjust_length(fX_next)

        # Record the current length
        current_length = self.length[0] if isinstance(self.length, np.ndarray) else self.length
        self.length_history.append(current_length)

        # Record the current success/failure counts
        self.success_history.append(self.succcount)
        self.failure_history.append(self.failcount)


# Turbo_M instance

turbo_m = EnhancedTurboM(
    f=f,
    lb=f.lb,
    ub=f.ub,
    n_init=N_INIT,
    max_evals=MAX_EVALS,
    n_trust_regions=wandb.config["n_trust_regions"],
    batch_size=BATCH_SIZE,
    verbose=True,
    use_ard=True,
    max_cholesky_size=2000,
    n_training_steps=50,
    min_cuda=1024,
    device="cpu",
    dtype="float64",
)


# Baseline instance - Turbo1

turbo_baseline = EnhancedTurbo1(
    f=f,
    lb=f.lb,
    ub=f.ub,
    n_init=N_INIT,
    max_evals=MAX_EVALS,
    batch_size=BATCH_SIZE,
    verbose=True,
    use_ard=True,
    max_cholesky_size=2000,
    n_training_steps=50,
    min_cuda=1024,
    device="cpu",
    dtype="float64",
)


# NEW Baseline instance - PureBO

pure_bo_baseline = PureBO(
    f=f,
    lb=f.lb,
    ub=f.ub,
    n_init=N_INIT,
    max_evals=MAX_EVALS,
    batch_size=BATCH_SIZE,
    verbose=True,
    use_ard=True,
    max_cholesky_size=2000,
    n_training_steps=50,
    min_cuda=1024,
    device="cpu",
    dtype="float64",
)


# Run the optimization process

print("--- Running TuRBO-m Optimization ---")
turbo_m.optimize()
for i, fx in enumerate(turbo_m.fX):
    wandb.log({"f(x)/TuRBO-m": fx, "best_so_far/TuRBO-m": np.min(turbo_m.fX[:i+1]), "X-Axis/TuRBO-m_Iteration": i })

X = turbo_m.X
fX = turbo_m.fX
ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best].item(), X[ind_best, :]
print("TuRBO-m Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))


print("\n--- Running Baseline (Turbo1) Optimization ---")
turbo_baseline.optimize()
for i, fx in enumerate(turbo_baseline.fX):
    wandb.log({"f(x)/TuRBO1": fx, "best_so_far/TuRBO1": np.min(turbo_baseline.fX[:i+1]), "X-Axis/Turbo1_Iteration": i})

X_baseline = turbo_baseline.X
fX_baseline = turbo_baseline.fX
ind_best_baseline = np.argmin(fX_baseline)
f_best_baseline, x_best_baseline = fX_baseline[ind_best_baseline].item(), X_baseline[ind_best_baseline, :]
print("Baseline Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best_baseline, np.around(x_best_baseline, 3)))


print("\n--- Running Baseline (Pure BO) Optimization ---")
pure_bo_baseline.optimize() 

# Logging with new unique keys
for i, fx in enumerate(pure_bo_baseline.fX):
    wandb.log({
        "f(x)/PureBO": fx, 
        "best_so_far/PureBO": np.min(pure_bo_baseline.fX[:i+1]), 
        "X-Axis/PureBO_Iteration": i
    })

X_pure_bo = pure_bo_baseline.X
fX_pure_bo = pure_bo_baseline.fX
ind_best_pure_bo = np.argmin(fX_pure_bo)
f_best_pure_bo, x_best_pure_bo = fX_pure_bo[ind_best_pure_bo].item(), X_pure_bo[ind_best_pure_bo, :]
print("Pure BO Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best_pure_bo, np.around(x_best_pure_bo, 3)))


wandb.log({
    "turbo_m_final_fx": f_best,
    "turbo_m_final_x": wandb.Table(data=[[i, x] for i, x in enumerate(x_best)], columns=["index", "x_value"]),
    "baseline_turbo1_final_fx": f_best_baseline, 
    "baseline_turbo1_final_x": wandb.Table(data=[[i, x] for i, x in enumerate(x_best_baseline)], columns=["index", "x_value"]),
    "baseline_purebo_final_fx": f_best_pure_bo,
    "baseline_purebo_final_x": wandb.Table(data=[[i, x] for i, x in enumerate(x_best_pure_bo)], columns=["index", "x_value"]),
})


# Plot the progress and compare

# 1. Optimization Progress Comparison Plot
max_len = max(len(fX), len(fX_baseline), len(fX_pure_bo))

fig1 = plt.figure(figsize=(12, 6))
plt.plot(np.minimum.accumulate(fX), 'r', lw=3, label="TuRBO-m Cumulative Best")
plt.plot(np.minimum.accumulate(fX_baseline), 'g', lw=3, label="Turbo1 Cumulative Best")
plt.plot(np.minimum.accumulate(fX_pure_bo), 'b', lw=3, label="Pure BO Cumulative Best")

plt.xlim([0, max_len])
plt.ylim([0, 300])
plt.title(f"{DIM}D Levy Function Optimization Progress (TR_m vs. TR_1 vs. NoTR)")
plt.xlabel("Evaluation")
plt.ylabel("Function Value")
plt.grid(True)
plt.legend()
fig1.tight_layout()
wandb.log({"optimization_progress_comparison": wandb.Image(fig1)})
plt.close(fig1)


# 2. Trust Region Length Evolution Plot
fig2, ax2 = plt.subplots(figsize=(10, 5))
# Plot TuRBO-m Multi-TR lengths
for tr_idx in range(turbo_m.n_trust_regions):
    lengths = [epoch[tr_idx]['length'] for epoch in turbo_m.trust_region_history if epoch[tr_idx]['length'] is not None]
    ax2.plot(lengths, label=f'TR {tr_idx+1}')

# Plot EnhancedTurbo1 (Baseline) Single-TR length evolution
lengths_baseline = turbo_baseline.length_history
ax2.plot(lengths_baseline, color='black', linestyle='--', label='Baseline (Turbo1) Length')

ax2.set_title("Trust Region Lengths Evolution (TuRBO-m vs. Turbo1 Baseline)")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Length")
ax2.legend()
ax2.grid(True)
fig2.tight_layout()
wandb.log({"trust_region_lengths_comparison": wandb.Image(fig2)})
plt.close(fig2)


# 3. Success/Failure Counts per Trust Region (TuRBO-m only)
fig3, ax3 = plt.subplots(figsize=(10, 5))
bar_width = 0.35
last_epoch = turbo_m.trust_region_history[-1]
successes = [tr['success_counter'] for tr in last_epoch]
failures = [tr['failure_counter'] for tr in last_epoch]
for tr_idx in range(turbo_m.n_trust_regions):
    ax3.bar(tr_idx - bar_width/2, successes[tr_idx], bar_width, label='Successes' if tr_idx == 0 else "", color='g')
    ax3.bar(tr_idx + bar_width/2, failures[tr_idx], bar_width, label='Failures' if tr_idx == 0 else "", color='r')
ax3.set_title("Success/Failure Counts per Trust Region (Last Iteration - TuRBO-m)")
ax3.set_xlabel("Trust Region Index")
ax3.set_ylabel("Count")
ax3.set_xticks(range(turbo_m.n_trust_regions))
ax3.set_xticklabels([f'TR {i+1}' for i in range(turbo_m.n_trust_regions)])
ax3.legend()
ax3.grid(True)
fig3.tight_layout()
wandb.log({"trust_region_counts": wandb.Image(fig3)})
plt.close(fig3)


# 4. Cumulative Success/Failure Counts Comparison Plot (TuRBO-m vs. Turbo1)
fig4, ax4 = plt.subplots(figsize=(10, 5))
N_TR_M = turbo_m.n_trust_regions # Should be 5

# TuRBO-m Cumulative Totals: Summing up counts across all TRs
turbo_m_success_total = np.array([sum(tr['success_counter'] for tr in epoch) 
                                  for epoch in turbo_m.trust_region_history])
turbo_m_failure_total = np.array([sum(tr['failure_counter'] for tr in epoch) 
                                  for epoch in turbo_m.trust_region_history])
turbo_m_success_avg_per_tr = turbo_m_success_total / N_TR_M
turbo_m_failure_avg_per_tr = turbo_m_failure_total / N_TR_M

# Turbo1 Cumulative Totals: Directly from EnhancedTurbo1 history
turbo_baseline_success_total = np.array(turbo_baseline.success_history)
turbo_baseline_failure_total = np.array(turbo_baseline.failure_history)

# Plotting
ax4.plot(np.cumsum(turbo_m_success_avg_per_tr), 'g-', label='TuRBO-m Avg. Cumulative Success')
ax4.plot(np.cumsum(turbo_baseline_success_total), 'g--', label='Turbo1 Cumulative Success')

ax4.plot(np.cumsum(turbo_m_failure_avg_per_tr), 'r-', label='TuRBO-m Avg. Cumulative Failure')
ax4.plot(np.cumsum(turbo_baseline_failure_total), 'r--', label='Turbo1 Cumulative Failure')

ax4.set_title("Standardized Cumulative Success/Failure Comparison (Avg. per TR)")
ax4.set_xlabel("Optimization Iteration")
ax4.set_ylabel("Count")
ax4.legend()
ax4.grid(True)
fig4.tight_layout()
wandb.log({"standardized_cumulative_sf_comparison": wandb.Image(fig4)})
plt.close(fig4)


wandb.finish()

# sbatch -p msismall run_turboM.sh
