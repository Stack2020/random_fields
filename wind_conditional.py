"""
Step (1) on the REAL 2D wind data: conditional-flow reconstruction of the 10x10 spatial wind
velocity field from sparse vertical sensor strips (sodar-like), using the C-LFM-generated dataset.
Demonstrates the sparse-sensor conditional flow scales to the real wind application.
Compares conditional flow vs prior-mean baseline; reports reconstruction RMSE (physical units) + coverage.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
P = argparse.ArgumentParser()
P.add_argument("--snaps", type=int, default=4); P.add_argument("--epochs", type=int, default=2000)
P.add_argument("--bs", type=int, default=256); P.add_argument("--cond_dim", type=int, default=96); P.add_argument("--hidden", type=int, default=384)
P.add_argument("--K", type=int, default=64); P.add_argument("--steps", type=int, default=100); P.add_argument("--device", type=str, default="cuda")
P.add_argument("--strips", type=int, nargs="+", default=[1, 5, 8]); P.add_argument("--outdir", type=str, default="results_wind")
a = P.parse_args(); dev = a.device; Path(a.outdir).mkdir(parents=True, exist_ok=True)
V1MIN, V1MAX = -2.0, 47.0; SPAN = V1MAX - V1MIN; NX = 10; D = NX * NX

def load(fn, snaps):
    f = h5py.File(fn, "r"); g = f["wind_samples"]; keys = sorted(g.keys(), key=int)
    T = g[keys[0]]["v1"].shape[1]; tidx = np.linspace(T // 6, T - T // 6, snaps).astype(int)
    out = []
    for k in keys:
        v = g[k]["v1"][:]                       # (100,256)
        out.append(v[:, tidx].T)                # (snaps,100)
    f.close()
    return torch.tensor(np.concatenate(out, 0), dtype=torch.float32)   # (Nsnap,100)

print("loading wind hdf5...", flush=True)
F_tr = (load("data/wind/wind_train_data.hdf5", a.snaps) - V1MIN) / SPAN
F_te = (load("data/wind/wind_test_data.hdf5", 2) - V1MIN) / SPAN
F_tr, F_te = F_tr.to(dev), F_te.to(dev)
print(f"train fields {F_tr.shape}, test {F_te.shape}", flush=True)

# spatial coords (10x10): point i -> (x2 = i//10, x3 = i%10), normalized to [0,1]
ii = torch.arange(D, device=dev); cx = (ii // NX).float() / (NX - 1); cy = (ii % NX).float() / (NX - 1)
coords = torch.stack([cx, cy], -1)                                     # (100,2)
# sparse vertical strips: columns (fixed x2 index) observed at all heights
sens = torch.cat([torch.where(ii // NX == c)[0] for c in a.strips]).to(dev)
S = sens.shape[0]; sxy = coords[sens]; print(f"{S} sensors (strips {a.strips})", flush=True)

mu_g = F_tr.mean(0); std_g = F_tr.std(0) + 1e-6
norm = lambda f: (f - mu_g) / std_g; denorm = lambda f: f * std_g + mu_g; Fn_tr = norm(F_tr)
ys_tr = F_tr[:, sens]; ysn_tr = (ys_tr - mu_g[sens]) / std_g[sens]
ys_te = F_te[:, sens]; ysn_te = (ys_te - mu_g[sens]) / std_g[sens]; sxy_b = sxy.unsqueeze(0)

class DeepSets(nn.Module):
    def __init__(s, cd, h):
        super().__init__(); s.phi = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cd))
        s.rho = nn.Sequential(nn.Linear(cd, h), nn.SiLU(), nn.Linear(h, cd))
    def forward(s, xy, y): return s.rho(s.phi(torch.cat([xy, y.unsqueeze(-1)], -1)).mean(1))
class CondVel(nn.Module):
    def __init__(s, dim, cd, h):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim + 1 + cd, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                                  nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, ft, t, c): return s.net(torch.cat([ft, t, c], -1))
enc = DeepSets(a.cond_dim, a.hidden).to(dev); vel = CondVel(D, a.cond_dim, a.hidden).to(dev)
opt = torch.optim.Adam(list(enc.parameters()) + list(vel.parameters()), 1e-3); N = Fn_tr.shape[0]
for ep in range(a.epochs):
    perm = torch.randperm(N, device=dev); el = []
    for i in range(0, N, a.bs):
        b = perm[i:i + a.bs]; f1 = Fn_tr[b]; B = f1.shape[0]; c = enc(sxy_b.expand(B, -1, -1), ysn_tr[b])
        z = torch.randn_like(f1); t = torch.rand(B, 1, device=dev); ft = (1 - t) * z + t * f1
        loss = ((vel(ft, t, c) - (f1 - z)) ** 2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(vel.parameters()), 1.0); opt.step(); el.append(loss.item())
    if (ep + 1) % 400 == 0: print(f"epoch {ep+1} loss {np.mean(el):.4f}", flush=True)

@torch.no_grad()
def sample(c, n, steps):
    f = torch.randn(n, D, device=dev); ts = torch.linspace(0, 1, steps + 1, device=dev)
    for i in range(steps):
        t0 = ts[i]; dt = ts[i + 1] - ts[i]; col = lambda tt: tt.repeat(n).unsqueeze(-1)
        k1 = vel(f, col(t0), c); k2 = vel(f + .5 * dt * k1, col(t0 + .5 * dt), c)
        k3 = vel(f + .5 * dt * k2, col(t0 + .5 * dt), c); k4 = vel(f + dt * k3, col(t0 + dt), c)
        f = f + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return f

flow_mean = torch.zeros_like(F_te); flow_std = torch.zeros_like(F_te)
with torch.no_grad():
    for j in range(F_te.shape[0]):
        c = enc(sxy_b, ysn_te[j:j+1]).expand(a.K, -1); fs = denorm(sample(c, a.K, a.steps))
        flow_mean[j] = fs.mean(0); flow_std[j] = fs.std(0) + 1e-6
prior = mu_g.unsqueeze(0).expand_as(F_te)
phys = lambda f: f * SPAN + V1MIN
rmse = lambda u, v: torch.sqrt(((phys(u) - phys(v)) ** 2).mean()).item()   # physical m/s
m = {
  "reconstruction_RMSE_physical(m/s)": {"conditional_flow": rmse(flow_mean, F_te), "prior_mean": rmse(prior, F_te)},
  "sensor_consistency_RMSE_physical": rmse(flow_mean[:, sens], F_te[:, sens]),
  "coverage_2sigma": ((F_te - flow_mean).abs() <= 2 * flow_std + 1e-6).float().mean().item(),
  "num_sensors": S, "grid": [NX, NX], "n_train_fields": int(N),
}
print("\n================ WIND 2D conditional-flow reconstruction ================")
print(json.dumps(m, indent=2)); Path(a.outdir, "metrics.json").write_text(json.dumps(m, indent=2))

fig, ax = plt.subplots(3, 4, figsize=(12, 9))
def img(f): return phys(f).reshape(NX, NX).cpu()
vmin, vmax = phys(F_te[:3]).min().item(), phys(F_te[:3]).max().item()
for k in range(3):
    for col, (f, ttl) in enumerate([(F_te[k], "true"), (flow_mean[k], "flow recon"),
                                     (flow_std[k] * SPAN, "flow std"), (F_te[k] - flow_mean[k], "error")]):
        vm = None if col >= 2 else vmax; vn = None if col >= 2 else vmin
        im = ax[k, col].imshow(img(f) if col < 2 else (f.reshape(NX, NX).cpu() * (SPAN if col == 2 else 1)),
                               origin="lower", vmin=vn, vmax=vm, cmap="viridis" if col < 3 else "RdBu_r")
        if k == 0: ax[k, col].set_title(ttl)
        ax[k, col].set_xticks([]); ax[k, col].set_yticks([]); plt.colorbar(im, ax=ax[k, col], fraction=.046)
    sxa = (sens // NX).cpu(); sya = (sens % NX).cpu(); ax[k, 0].scatter(sya, sxa, c="r", s=10)
fig.tight_layout(); fig.savefig(Path(a.outdir, "wind_recon.png"), dpi=110)
print(f"saved -> {a.outdir}/")
