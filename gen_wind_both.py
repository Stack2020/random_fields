"""Generate BOTH wind train (seed 0, 5000) and test (seed 1, 1000) hdf5 datasets."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, "data/wind_data_generation")
from wind_sampler import BasicWindSampler
from hdf5_sample_data import write_sample_data_to_hdf5

ROOT = Path("data/wind"); ROOT.mkdir(parents=True, exist_ok=True)
Ny = Nz = 10; H = W = 100; ZMIN = 1
x2 = np.linspace(0, W, Ny); x3 = np.linspace(ZMIN, H, Nz)
x_coords = np.zeros((Ny * Nz, 3)); x_coords[:, 1] = np.repeat(x2, Nz); x_coords[:, 2] = np.tile(x3, Ny)
F = 3.0; N_w = 2 ** 7; delta_w = F / N_w; T = 1 / delta_w; N_t = 2 * N_w; delta_t = T / N_t
time_steps = np.linspace(0, T - delta_t, N_t); freqs = np.linspace(delta_w, F, N_w)
shear_velocity = 1.9

for seed, fname, n in [(0, "wind_train_data.hdf5", 5000), (1, "wind_test_data.hdf5", 1000)]:
    np.random.seed(seed)
    sampler = BasicWindSampler(x_coords, freqs, shear_velocity)
    samples = sampler.sample(n, delta_t, N_t)
    write_sample_data_to_hdf5(ROOT / fname, samples, x_coords, time_steps,
                              input_params={"freq_grid": freqs, "shear_velocity": shear_velocity},
                              vector_field=False)
    print(f"wrote {fname}  n={n}", flush=True)
print("WIND_DATA_DONE", flush=True)
