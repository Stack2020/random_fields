# KR-Flow: Kriging-Residual Flow Matching for Random-Field UQ

Complete reproduction package for *Uncertainty Quantification of Random Fields
from Sparse Observations via Kriging-Residual Flow Matching*. KR-Flow uses the
kriging posterior mean as an observation-dependent reference and trains a
conditional flow-matching model on the residual; posterior field samples are
formed by adding sampled residuals back to the kriging mean.

## Contents
```
*.py                          source (figure + experiment scripts)
weights/*.pt                  trained KR-Flow weights (one per benchmark) + README
data/wind_data_generation/    wind-field generator (regenerates the wind data)
```

## Requirements
Python 3.10, PyTorch 2.5.1 (CUDA 12.4), NumPy, SciPy, matplotlib, h5py
(`pip install -r requirements.txt`).
One GPU suffices (all results on a single NVIDIA RTX A6000, 48 GB). Figure
scripts write to `./results_pretty/`; the experiment scripts write metrics to
their own `./results_*` directories. Data is read from `./data/`.

## Data
- **Gaussian, channelized, bimodal** fields are generated **inline** by the
  scripts — no data files needed.
- **Wind** fields are synthetic samples from the constrained latent flow-matching
  generator (physics-informed, *not* physical measurements). Regenerate with:
  ```bash
  python gen_wind_both.py     # writes data/wind/wind_{train,test}_data.hdf5
  ```
  The raw wind HDF5 (~1.1 GB) is **not bundled** because of its size; the
  generator above reproduces it. Regenerating requires the extra package
  **UQpy** (`pip install UQpy`); the experiment scripts themselves do not.
  Scripts use the first 8000 train / 300 test snapshots.

## Reproducing the paper
| Script | Output |
|---|---|
| `pretty_1d.py` | Gaussian posterior-recovery figure |
| `channel_recon.py` | Channelized: reconstruction, calibration, sharpness–coverage |
| `exp_B_stats.py --krig_emp` | Channelized main table + module-ablation numbers (6 seeds) |
| `exp_dataeff.py --krig_emp --seeds 0 1 2 3 4 --outdir results_emp_dataeff_mpiw` | Data-efficiency JSON |
| `exp_sparsity.py --krig_emp --seeds 0 1 2 3 4 --outdir results_emp_sparsity_mpiw` | Sensor-sparsity JSON |
| `ablation_figs.py` | Data-efficiency + sparsity figures (reads the two JSONs above) |
| `bimodal_eval.py` | Bimodal metric bar chart + posterior-fidelity figure |
| `wind_eval.py` | Wind calibration + interval-width-reduction figure, and the reconstruction |
| `wind_kriging_residual.py`, `wind_conditional.py` | Wind data-efficiency table |

Ablation figures are two-step: run `exp_dataeff.py` / `exp_sparsity.py` to produce
the multi-seed JSONs, then `ablation_figs.py`.

## Trained weights
`weights/` holds one KR-Flow model per benchmark (`gaussian`, `channel`,
`bimodal`, `wind`): the DeepSets encoder + velocity-MLP `state_dict`, the
normalization statistics, and the kriging-reference pieces. See
`weights/README.md` for keys and a loading example. Scripts also retrain from
scratch in a few minutes.

## Method notes
- **Kriging reference (per benchmark):** Gaussian — well-specified SE kernel
  (`ℓ=0.1, σ²=0.5`); channelized — empirical mean + shrinkage-regularized
  empirical covariance (`torch.cov`, `--krig_emp`); wind — empirical mean + an
  SE covariance (`ℓ=0.3`) with an empirically set variance (not a full empirical
  covariance).
- **Calibration:** reliability curves use raw ensemble quantiles (no Gaussian
  assumption); matched-coverage MPIW uses empirical central-quantile intervals
  whose nominal level is calibrated on validation and applied to test.
- **Metrics:** ES = energy score (multivariate CRPS); the bimodal benchmark
  reports energy distance, best-of-K RMSE, and mode coverage.

Paths at the top of each script are relative (`./results_pretty`, `./data`).

## Reference
Jiankun Lei and Jian Xu, *Uncertainty Quantification of Random Fields from
Sparse Observations via Kriging-Residual Flow Matching*, International Journal
of Modeling, Simulation, and Scientific Computing.
