"""
Deliverable 2: data-efficiency sweep on the 1D channel field.
Hypothesis: the kriging prior helps most when training data is scarce, so
Kriging+GenResidual should separate from Direct flow at small N_train and converge as N grows.
Reports RMSE vs N_train for M2 (Direct flow) and M4 (Kriging+GenResidual), 2 seeds.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn

P = argparse.ArgumentParser()
P.add_argument("--Ns", type=int, nargs="+", default=[500, 2000, 8000]); P.add_argument("--N_test", type=int, default=600)
P.add_argument("--disc", type=int, default=100); P.add_argument("--num_sensors", type=int, default=10)
P.add_argument("--width", type=float, default=0.02); P.add_argument("--amp", type=float, default=1.5); P.add_argument("--obs_noise", type=float, default=0.02)
P.add_argument("--krig_len", type=float, default=0.1); P.add_argument("--krig_var", type=float, default=1.0)
P.add_argument("--krig_emp", action="store_true"); P.add_argument("--krig_shrink", type=float, default=0.05); P.add_argument("--krig_ref_N", type=int, default=6000)
P.add_argument("--epochs", type=int, default=1500); P.add_argument("--bs", type=int, default=256)
P.add_argument("--cond_dim", type=int, default=64); P.add_argument("--hidden", type=int, default=256)
P.add_argument("--lam_s", type=float, default=1e-4); P.add_argument("--K", type=int, default=48); P.add_argument("--steps", type=int, default=80)
P.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); P.add_argument("--device", type=str, default="cuda")
P.add_argument("--outdir", type=str, default="results_dataeff")
a = P.parse_args(); dev = a.device; Path(a.outdir).mkdir(parents=True, exist_ok=True)
D = a.disc

def gen(N, x):
    xi = 0.25 + 0.5 * torch.rand(N, 1, device=dev); sign = torch.where(torch.rand(N, 1, device=dev) < .5, -1., 1.)
    base = 0.2 * torch.randn(N, 1, device=dev); step = torch.sigmoid((x.unsqueeze(0) - xi) / a.width)
    wig = torch.zeros(N, D, device=dev)
    for k in range(1, 4): wig += (0.05 / k) * torch.randn(N, 1, device=dev) * torch.sin(k * np.pi * x.unsqueeze(0) * 2)
    return base + sign * a.amp * step + wig

class DeepSets(nn.Module):
    def __init__(s, cd, h):
        super().__init__(); s.phi = nn.Sequential(nn.Linear(2, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cd))
        s.rho = nn.Sequential(nn.Linear(cd, h), nn.SiLU(), nn.Linear(h, cd))
    def forward(s, xs, ys): return s.rho(s.phi(torch.stack([xs, ys], -1)).mean(1))
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

def train_flow(targets_n, cond_fn, cdim, Ntr):
    enc = DeepSets(a.cond_dim, a.hidden).to(dev); vel = Vel(D, cdim, a.hidden).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(vel.parameters()), 1e-3)
    for ep in range(a.epochs):
        perm = torch.randperm(Ntr, device=dev)
        for i in range(0, Ntr, a.bs):
            b = perm[i:i + a.bs]; f1 = targets_n[b]; B = f1.shape[0]; c = cond_fn(enc, b)
            z = torch.randn_like(f1); t = torch.rand(B, 1, device=dev); ft = (1 - t) * z + t * f1
            loss = ((vel(ft, t, c) - (f1 - z)) ** 2).mean()
            if a.lam_s > 0 and cdim > a.cond_dim:
                f1h = ft + (1 - t) * vel(ft, t, c); loss = loss + a.lam_s * (f1h ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(vel.parameters()), 1.0); opt.step()
    return enc, vel

rmse = lambda u, v: torch.sqrt(((u - v) ** 2).mean()).item()
def es_metric(E, F): return ((E - F.unsqueeze(1)).norm(dim=2).mean() - 0.5 * torch.cdist(E, E).mean((1, 2)).mean()).item()
def gamma_cal(M, SD, Y):  # scale factor making validation coverage 0.95
    g = 1.0
    for _ in range(80):
        c = ((Y - M).abs() <= 1.959964 * g * SD).float().mean().item()
        if abs(c - 0.95) < 0.003: break
        g *= 1.05 if c < 0.95 else 0.96
    return g
MK = ["direct", "genres", "kriging"]
res = {str(N): {m: {"RMSE": [], "ES": [], "MPIW": []} for m in MK} for N in a.Ns}
for seed in a.seeds:
    torch.manual_seed(seed); np.random.seed(seed)
    x = torch.linspace(0, 1, D, device=dev); sidx = torch.linspace(5, D - 6, a.num_sensors, dtype=torch.long, device=dev); xs = x[sidx]; S = a.num_sensors
    F_te = gen(a.N_test, x); F_va = gen(max(300, a.N_test // 2), x)
    def se(u, v): d = u.reshape(-1, 1) - v.reshape(1, -1); return a.krig_var * torch.exp(-(d ** 2) / (2 * a.krig_len ** 2))
    if a.krig_emp:
        # empirical mean + shrinkage covariance from a fixed reference pool (moment-estimated prior)
        F_ref = gen(a.krig_ref_N, x); mu_k = F_ref.mean(0); Cd = torch.cov(F_ref.T); Cd = (1 - a.krig_shrink) * Cd + a.krig_shrink * torch.diag(torch.diag(Cd))
        Kss = Cd[sidx][:, sidx] + (a.obs_noise ** 2 + 1e-6) * torch.eye(S, device=dev); Kxs = Cd[:, sidx]; Kff = Cd + 1e-4 * torch.eye(D, device=dev)
    else:
        mu_k = torch.zeros(D, device=dev)
        Kss = se(xs, xs) + (a.obs_noise ** 2 + 1e-6) * torch.eye(S, device=dev); Kxs = se(x, xs); Kff = se(x, x) + 1e-4 * torch.eye(D, device=dev)
    Kinv = torch.linalg.inv(Kss); W = Kxs @ Kinv; SigK = Kff - Kxs @ Kinv @ Kxs.T
    kstd = torch.sqrt(torch.clamp(torch.diag(SigK), min=1e-8))
    _Sk = (SigK + SigK.T) / 2 + 1e-5 * torch.eye(D, device=dev); _wk, _Vk = torch.linalg.eigh(_Sk); Lk = _Vk @ torch.diag(torch.sqrt(torch.clamp(_wk, min=1e-8)))
    krg = lambda ys: mu_k.unsqueeze(0) + (W @ (ys - mu_k[sidx]).T).T
    obs = lambda F: F[:, sidx] + a.obs_noise * torch.randn(F.shape[0], S, device=dev)
    ys_te = obs(F_te); f0_te = krg(ys_te); ys_va = obs(F_va); f0_va = krg(ys_va); xs_b = xs.unsqueeze(0)
    # kriging is N-independent: ensemble via full posterior covariance; gamma on validation
    kr_te = f0_te.unsqueeze(1) + torch.randn(f0_te.shape[0], a.K, D, device=dev) @ Lk.T
    gk = gamma_cal(f0_va, kstd.unsqueeze(0).expand(f0_va.shape[0], D), F_va)
    kr_RMSE = rmse(f0_te, F_te); kr_ES = es_metric(kr_te, F_te); kr_MPIW = (2 * 1.959964 * gk * kstd).mean().item()
    for N in a.Ns:
        F_tr = gen(N, x); mu_g, std_g = F_tr.mean(0), F_tr.std(0) + 1e-6
        norm = lambda f: (f - mu_g) / std_g; denorm = lambda f: f * std_g + mu_g; ysn = lambda ys: (ys - mu_g[sidx]) / std_g[sidx]
        ys_tr = obs(F_tr); f0_tr = krg(ys_tr)
        e2, v2 = train_flow(norm(F_tr), lambda e, b: e(xs_b.expand(len(b), -1), ysn(ys_tr[b])), a.cond_dim, N)
        R = F_tr - f0_tr; rmu, rsd = R.mean(0), R.std(0) + 1e-6; f0n = norm(f0_tr)
        e4, v4 = train_flow((R - rmu) / rsd, lambda e, b: torch.cat([e(xs_b.expand(len(b), -1), ysn(ys_tr[b])), f0n[b]], -1), a.cond_dim + D, N)
        def dens(ys, n):
            E = torch.zeros(ys.shape[0], n, D, device=dev)
            for j in range(ys.shape[0]): E[j] = denorm(sample(v2, e2(xs_b, ysn(ys[j:j+1])).expand(n, -1), n))
            return E
        def gens(f0, ys, n):
            E = torch.zeros(ys.shape[0], n, D, device=dev)
            for j in range(ys.shape[0]):
                c = torch.cat([e4(xs_b, ysn(ys[j:j+1])), norm(f0[j:j+1])], -1).expand(n, -1); E[j] = f0[j:j+1] + (sample(v4, c, n) * rsd + rmu)
            return E
        with torch.no_grad():
            Ed_te, Ed_va = dens(ys_te, a.K), dens(ys_va, a.K)
            Eg_te, Eg_va = gens(f0_te, ys_te, a.K), gens(f0_va, ys_va, a.K)
        for nm, E_te, E_va in [("direct", Ed_te, Ed_va), ("genres", Eg_te, Eg_va)]:
            g = gamma_cal(E_va.mean(1), E_va.std(1) + 1e-6, F_va)
            res[str(N)][nm]["RMSE"].append(rmse(E_te.mean(1), F_te))
            res[str(N)][nm]["ES"].append(es_metric(E_te, F_te))
            res[str(N)][nm]["MPIW"].append((2 * 1.959964 * g * (E_te.std(1) + 1e-6)).mean().item())
        for k, v in [("RMSE", kr_RMSE), ("ES", kr_ES), ("MPIW", kr_MPIW)]: res[str(N)]["kriging"][k].append(v)
        print(f"seed {seed} N={N}: dir(RMSE {res[str(N)]['direct']['RMSE'][-1]:.3f} ES {res[str(N)]['direct']['ES'][-1]:.3f} MPIW {res[str(N)]['direct']['MPIW'][-1]:.3f}) gen(RMSE {res[str(N)]['genres']['RMSE'][-1]:.3f} ES {res[str(N)]['genres']['ES'][-1]:.3f} MPIW {res[str(N)]['genres']['MPIW'][-1]:.3f})", flush=True)

summary = {N: {m: {k: f"{np.mean(v):.4f}±{np.std(v):.4f}" for k, v in d.items()} for m, d in md.items()} for N, md in res.items()}
print("\n================ Data-efficiency sweep (RMSE/ES/MPIW@0.95, mean±std) ================")
print(json.dumps(summary, indent=2)); Path(a.outdir, "metrics.json").write_text(json.dumps(summary, indent=2)); print("done")
