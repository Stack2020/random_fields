"""
Real-data application: kriging-guided generative residual on the wind-velocity field
(C-LFM-generated data), method-consistent with the paper. Reconstruct 10x10 spatial wind
snapshots from 30 sparse sodar-like vertical sensor strips. Compares, in physical units (m/s),
Prior mean / Kriging / Direct conditional flow / Kriging+generative residual (ours), with
matched-coverage MPIW.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = argparse.ArgumentParser()
P.add_argument("--snaps", type=int, default=4); P.add_argument("--epochs", type=int, default=1200); P.add_argument("--bs", type=int, default=256)
P.add_argument("--cond_dim", type=int, default=96); P.add_argument("--hidden", type=int, default=384); P.add_argument("--lam_s", type=float, default=1e-4)
P.add_argument("--krig_len", type=float, default=0.3); P.add_argument("--obs_noise_n", type=float, default=0.02)
P.add_argument("--K", type=int, default=64); P.add_argument("--steps", type=int, default=80); P.add_argument("--N_test", type=int, default=1200)
P.add_argument("--strips", type=int, nargs="+", default=[1, 5, 8]); P.add_argument("--device", type=str, default="cuda"); P.add_argument("--outdir", type=str, default="results_wind_kr")
P.add_argument("--n_train", type=int, default=0)  # 0 = use all; >0 = data-scarce subsample
a = P.parse_args(); dev = a.device; Path(a.outdir).mkdir(parents=True, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
V1MIN, V1MAX = -2.0, 47.0; SPAN = V1MAX - V1MIN; NX = 10; D = NX * NX

def load(fn, snaps):
    f = h5py.File(fn, "r"); g = f["wind_samples"]; keys = sorted(g.keys(), key=int)
    T = g[keys[0]]["v1"].shape[1]; tidx = np.linspace(T // 6, T - T // 6, snaps).astype(int); out = []
    for k in keys: out.append(g[k]["v1"][:][:, tidx].T)
    f.close(); return torch.tensor(np.concatenate(out, 0), dtype=torch.float32)
print("loading wind...", flush=True)
F_all = (load("data/wind/wind_train_data.hdf5", a.snaps) - V1MIN) / SPAN
F_te = (load("data/wind/wind_test_data.hdf5", 2) - V1MIN) / SPAN
F_all, F_te = F_all.to(dev), F_te.to(dev)[:a.N_test]
nval = 800; F_va = F_all[-nval:]; F_tr = F_all[:-nval]
if a.n_train > 0 and a.n_train < F_tr.shape[0]:
    torch.manual_seed(0); F_tr = F_tr[torch.randperm(F_tr.shape[0])[:a.n_train]]   # data-scarce regime
print(f"train {F_tr.shape} val {F_va.shape} test {F_te.shape}", flush=True)

ii = torch.arange(D, device=dev); cx = (ii // NX).float() / (NX - 1); cy = (ii % NX).float() / (NX - 1); coords = torch.stack([cx, cy], -1)
sens = torch.cat([torch.where(ii // NX == c)[0] for c in a.strips]).to(dev); S = sens.shape[0]; sxy = coords[sens]
print(f"{S} sensors", flush=True)
mu_g, std_g = F_tr.mean(0), F_tr.std(0) + 1e-6; norm = lambda f: (f - mu_g) / std_g; denorm = lambda f: f * std_g + mu_g
phys = lambda f: f * SPAN + V1MIN; rmse = lambda u, v: torch.sqrt(((phys(u) - phys(v)) ** 2).mean()).item()

# ---- 2D kriging (mean = empirical profile, SE kernel on grid coords) ----
m_prof = F_tr.mean(0)                                          # normalized mean profile (shear)
def se2d(A, B): d2 = ((A.unsqueeze(1) - B.unsqueeze(0)) ** 2).sum(-1); return torch.exp(-d2 / (2 * a.krig_len ** 2))
kvar = ((F_tr - m_prof) ** 2).mean()
Kss = kvar * se2d(sxy, sxy) + (a.obs_noise_n ** 2 + 1e-5) * torch.eye(S, device=dev)
Kxs = kvar * se2d(coords, sxy); Kff = kvar * se2d(coords, coords) + 1e-5 * torch.eye(D, device=dev)
W = Kxs @ torch.linalg.inv(Kss); SigK = Kff - Kxs @ torch.linalg.inv(Kss) @ Kxs.T
kstd = torch.sqrt(torch.clamp(torch.diag(SigK), min=1e-8))
obsv = lambda F: F[:, sens]                                    # noiseless sensor reads (data already smooth)
def krige(F): return m_prof.unsqueeze(0) + (W @ (obsv(F) - m_prof[sens]).T).T
f0_tr, f0_va, f0_te = krige(F_tr), krige(F_va), krige(F_te)
ysn = lambda F: (obsv(F) - mu_g[sens]) / std_g[sens]; sxy_b = sxy.unsqueeze(0)

class DeepSets(nn.Module):
    def __init__(s, cd, h):
        super().__init__(); s.phi = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cd))
        s.rho = nn.Sequential(nn.Linear(cd, h), nn.SiLU(), nn.Linear(h, cd))
    def forward(s, xy, y): return s.rho(s.phi(torch.cat([xy, y.unsqueeze(-1)], -1)).mean(1))
class Vel(nn.Module):
    def __init__(s, dim, cd, h):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim + 1 + cd, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                                  nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, ft, t, c): return s.net(torch.cat([ft, t, c], -1))
@torch.no_grad()
def sample(vel, c, n):
    f = torch.randn(n, D, device=dev); ts = torch.linspace(0, 1, a.steps + 1, device=dev)
    for i in range(a.steps):
        t0 = ts[i]; dt = ts[i + 1] - ts[i]; col = lambda tt: tt.repeat(n).unsqueeze(-1)
        k1 = vel(f, col(t0), c); k2 = vel(f + .5 * dt * k1, col(t0 + .5 * dt), c)
        k3 = vel(f + .5 * dt * k2, col(t0 + .5 * dt), c); k4 = vel(f + dt * k3, col(t0 + dt), c)
        f = f + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return f
def train_flow(targets_n, cond_fn, cdim, tag, resid):
    enc = DeepSets(a.cond_dim, a.hidden).to(dev); vel = Vel(D, cdim, a.hidden).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(vel.parameters()), 1e-3); N = targets_n.shape[0]
    for ep in range(a.epochs):
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, a.bs):
            b = perm[i:i + a.bs]; f1 = targets_n[b]; B = f1.shape[0]; c = cond_fn(enc, b)
            z = torch.randn_like(f1); t = torch.rand(B, 1, device=dev); ft = (1 - t) * z + t * f1
            loss = ((vel(ft, t, c) - (f1 - z)) ** 2).mean()
            if resid: f1h = ft + (1 - t) * vel(ft, t, c); loss = loss + a.lam_s * (f1h ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(vel.parameters()), 1.0); opt.step()
        if (ep + 1) % 400 == 0: print(f"[{tag}] {ep+1} loss {loss.item():.4f}", flush=True)
    return enc, vel

Fn_tr = norm(F_tr)
# direct flow (the conditioning encoder IS the one being optimized -- passed in as `e`)
encd, veld = train_flow(Fn_tr, lambda e, b: e(sxy_b.expand(len(b), -1, -1), ysn(F_tr[b])), a.cond_dim, "DIRECT", False)
# kriging + generative residual
R = F_tr - f0_tr; rmu, rsd = R.mean(0), R.std(0) + 1e-6; Rn = (R - rmu) / rsd; f0n_tr = norm(f0_tr)
encr, velr = train_flow(Rn, lambda e, b: torch.cat([e(sxy_b.expand(len(b), -1, -1), ysn(F_tr[b])), f0n_tr[b]], -1), a.cond_dim + D, "RESID", True)

def coverage(F, m, sd, al): return ((F - m).abs() <= 2 * al * sd).float().mean().item()
def calibrate(F, m, sd):
    al = 1.0
    for _ in range(60):
        c = coverage(F, m, sd, al)
        if abs(c - 0.95) < 0.004: break
        al *= 1.05 if c < 0.95 else 0.96
    return al
def pred_direct(F):
    m = torch.zeros(F.shape[0], D, device=dev); sd = torch.zeros_like(m)
    with torch.no_grad():
        for j in range(F.shape[0]):
            c = encd(sxy_b, ysn(F[j:j+1])).expand(a.K, -1); fs = denorm(sample(veld, c, a.K)); m[j] = fs.mean(0); sd[j] = fs.std(0) + 1e-6
    return m, sd
def pred_resid(F, f0):
    m = torch.zeros(F.shape[0], D, device=dev); sd = torch.zeros_like(m)
    with torch.no_grad():
        for j in range(F.shape[0]):
            c = torch.cat([encr(sxy_b, ysn(F[j:j+1])), norm(f0[j:j+1])], -1).expand(a.K, -1)
            fs = f0[j:j+1] + (sample(velr, c, a.K) * rsd + rmu); m[j] = fs.mean(0); sd[j] = fs.std(0) + 1e-6
    return m, sd

# kriging UQ
kstd_te = kstd.unsqueeze(0).expand(F_te.shape[0], -1).clone(); kstd_va = kstd.unsqueeze(0).expand(F_va.shape[0], -1).clone()
al_k = calibrate(F_va, f0_va, kstd_va)
md_va, sdd_va = pred_direct(F_va); al_d = calibrate(F_va, md_va, sdd_va)
mr_va, sdr_va = pred_resid(F_va, f0_va); al_r = calibrate(F_va, mr_va, sdr_va)
md_te, sdd_te = pred_direct(F_te); mr_te, sdr_te = pred_resid(F_te, f0_te)
prior = mu_g.unsqueeze(0).expand_as(F_te)
m = {
  "RMSE_physical_m/s": {"prior_mean": rmse(prior, F_te), "kriging": rmse(f0_te, F_te),
                        "direct_flow": rmse(md_te, F_te), "kriging+gen_residual(ours)": rmse(mr_te, F_te)},
  "coverage_2sigma_calibrated": {"kriging": coverage(F_te, f0_te, kstd_te, al_k),
                                 "direct_flow": coverage(F_te, md_te, sdd_te, al_d),
                                 "kriging+gen_residual(ours)": coverage(F_te, mr_te, sdr_te, al_r)},
  "MPIW_matched_physical_m/s": {"kriging": (4 * al_k * kstd_te).mean().item() * SPAN,
                                "direct_flow": (4 * al_d * sdd_te).mean().item() * SPAN,
                                "kriging+gen_residual(ours)": (4 * al_r * sdr_te).mean().item() * SPAN},
  "num_sensors": S, "grid": [NX, NX], "n_train": int(F_tr.shape[0]),
}
print("\n================ WIND real-data: kriging-guided residual ================")
print(json.dumps(m, indent=2)); Path(a.outdir, "metrics.json").write_text(json.dumps(m, indent=2))

# figure: true / kriging / ours mean / ours std for 3 test snapshots
fig, ax = plt.subplots(3, 4, figsize=(12, 9)); img = lambda f: phys(f).reshape(NX, NX).cpu()
vmin, vmax = phys(F_te[:3]).min().item(), phys(F_te[:3]).max().item()
for k in range(3):
    panels = [(F_te[k], "true", vmin, vmax), (f0_te[k], "kriging", vmin, vmax),
              (mr_te[k], "ours mean", vmin, vmax), (sdr_te[k] * SPAN, "ours std", None, None)]
    for col, (f, ttl, vn, vx) in enumerate(panels):
        d = (f.reshape(NX, NX).cpu() if col == 3 else img(f))
        im = ax[k, col].imshow(d, origin="lower", vmin=vn, vmax=vx, cmap="viridis")
        if k == 0: ax[k, col].set_title(ttl)
        ax[k, col].set_xticks([]); ax[k, col].set_yticks([]); plt.colorbar(im, ax=ax[k, col], fraction=.046)
    ax[k, 0].scatter((sens % NX).cpu(), (sens // NX).cpu(), c="r", s=10)
fig.tight_layout(); fig.savefig(Path(a.outdir, "wind_kr.png"), dpi=110); print("done")
