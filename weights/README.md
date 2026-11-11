# KR-Flow trained weights

Each `*.pt` is the trained KR-Flow model for one benchmark (saved via
`torch.save` at the end of the corresponding figure script). Contents:

| File | From script | Keys |
|---|---|---|
| `gaussian_krflow.pt` | `pretty_1d.py` | `enc`, `vel`, `rmu`, `rsd`, `mu`, `sd` |
| `channel_krflow.pt` | `channel_recon.py` | `enc`, `vel`, `rmu`, `rsd`, `mu_g`, `sd_g`, `W`, `mu_k`, `kstd` |
| `bimodal_krflow.pt` | `bimodal_eval.py` | `enc`, `vel`, `rmu`, `rsd`, `mu`, `sd` |
| `wind_krflow.pt` | `wind_eval.py` | `enc`, `vel` (KR-Flow), `encd`, `veld` (direct flow), `uvel` (DPS), `rmu`, `rsd`, `mu`, `sd`, `W`, `mprof` |

- `enc` = DeepSets sensor encoder `state_dict`; `vel` = velocity-field MLP `state_dict`.
- `rmu/rsd` = residual mean/std (standardization of `r = u - u_K`).
- `mu/sd` (or `mu_g/sd_g`) = field mean/std used to normalize the network I/O.
- `W`, `mu_k`, `mprof`, `kstd` = kriging-reference pieces (BLUP weight matrix,
  prior mean, kriging marginal std) for reconstructing `u_K` from observations.

## Loading example (channel)
```python
import torch
from channel_recon import DeepSets, Vel   # same class defs as the script
D = 100
ck = torch.load("weights/channel_krflow.pt", map_location="cpu")
enc = DeepSets();  enc.load_state_dict(ck["enc"]);  enc.eval()
vel = Vel(D, 64 + D); vel.load_state_dict(ck["vel"]); vel.eval()
rmu, rsd, mu_g, sd_g = ck["rmu"], ck["rsd"], ck["mu_g"], ck["sd_g"]
W, mu_k = ck["W"], ck["mu_k"]
# u_K = mu_k + W @ (y_S - mu_k[sensors]);  sample residual r via the RK4 flow of vel
# conditioned on [enc(sensors, y_S), (u_K-mu_g)/sd_g];  u = u_K + (r*rsd + rmu)
```
The network architectures (`DeepSets`, `Vel`, `UVel`) are defined at the top of
each figure script, so import them from there before `load_state_dict`.

Note: the figure scripts also **retrain from scratch** in a few minutes, so these
weights are a convenience for skipping training, not a hard dependency.
