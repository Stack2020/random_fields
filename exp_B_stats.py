"""
Multi-seed channel-field comparison with mean+/-std and paired significance tests.
Methods: Kriging, PCA/KLE+reg, Direct flow, DPS-guidance, Kriging+Det-residual, KR-Flow.
Metrics (all calibrated to 0.95 coverage): RMSE, matched MPIW, TV error.
Significance: paired Wilcoxon signed-rank + paired t-test, KR-Flow vs each baseline, Holm-corrected.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy import stats

P = argparse.ArgumentParser()
P.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
P.add_argument("--N_train", type=int, default=6000); P.add_argument("--N_val", type=int, default=300); P.add_argument("--N_test", type=int, default=400)
P.add_argument("--disc", type=int, default=100); P.add_argument("--num_sensors", type=int, default=10)
P.add_argument("--width", type=float, default=0.02); P.add_argument("--amp", type=float, default=1.5); P.add_argument("--obs_noise", type=float, default=0.02)
P.add_argument("--krig_len", type=float, default=0.1); P.add_argument("--krig_var", type=float, default=1.0)
P.add_argument("--krig_emp", action="store_true"); P.add_argument("--krig_shrink", type=float, default=0.05)
P.add_argument("--pca_r", type=int, default=20); P.add_argument("--ridge", type=float, default=1e-3)
P.add_argument("--epochs", type=int, default=1000); P.add_argument("--bs", type=int, default=256)
P.add_argument("--cond_dim", type=int, default=64); P.add_argument("--hidden", type=int, default=256); P.add_argument("--lam_s", type=float, default=1e-4)
P.add_argument("--dps_zeta", type=float, default=0.2); P.add_argument("--K", type=int, default=48); P.add_argument("--steps", type=int, default=60)
P.add_argument("--device", type=str, default="cuda"); P.add_argument("--outdir", type=str, default="results_stats_B")
a = P.parse_args(); dev = a.device; Path(a.outdir).mkdir(parents=True, exist_ok=True); D = a.disc

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
class UVel(nn.Module):
    def __init__(s, dim, h):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim + 1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                                  nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, ft, t): return s.net(torch.cat([ft, t], -1))
class DetNet(nn.Module):
    def __init__(s, dim, cd, h):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim + cd, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                                  nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, fK, c): return s.net(torch.cat([fK, c], -1))

@torch.no_grad()
def sample(vel, c, n):
    f = torch.randn(n, D, device=dev); ts = torch.linspace(0, 1, a.steps + 1, device=dev)
    for i in range(a.steps):
        t0 = ts[i]; dt = ts[i + 1] - ts[i]; col = lambda tt: tt.repeat(n).unsqueeze(-1)
        k1 = vel(f, col(t0), c); k2 = vel(f + .5 * dt * k1, col(t0 + .5 * dt), c)
        k3 = vel(f + .5 * dt * k2, col(t0 + .5 * dt), c); k4 = vel(f + dt * k3, col(t0 + dt), c)
        f = f + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return f

def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    x = torch.linspace(0, 1, D, device=dev); sidx = torch.linspace(5, D - 6, a.num_sensors, dtype=torch.long, device=dev); xs = x[sidx]; S = a.num_sensors
    F_tr, F_va, F_te = gen(a.N_train, x), gen(a.N_val, x), gen(a.N_test, x)
    mu_g, std_g = F_tr.mean(0), F_tr.std(0) + 1e-6; norm = lambda f: (f - mu_g) / std_g; denorm = lambda f: f * std_g + mu_g
    obs = lambda F: F[:, sidx] + a.obs_noise * torch.randn(F.shape[0], S, device=dev)
    ys_tr, ys_va, ys_te = obs(F_tr), obs(F_va), obs(F_te); ysn = lambda ys: (ys - mu_g[sidx]) / std_g[sidx]; xs_b = xs.unsqueeze(0)
    def se(u, v): d = u.reshape(-1, 1) - v.reshape(1, -1); return a.krig_var * torch.exp(-(d ** 2) / (2 * a.krig_len ** 2))
    if a.krig_emp:
        # empirical mean + shrinkage covariance from the training fields (moment estimation)
        mu_k = F_tr.mean(0); Cd = torch.cov(F_tr.T)
        Cd = (1 - a.krig_shrink) * Cd + a.krig_shrink * torch.diag(torch.diag(Cd))
        Kss = Cd[sidx][:, sidx] + (a.obs_noise ** 2 + 1e-6) * torch.eye(S, device=dev); Kxs = Cd[:, sidx]; Kff = Cd + 1e-4 * torch.eye(D, device=dev)
    else:
        mu_k = torch.zeros(D, device=dev)
        Kss = se(xs, xs) + (a.obs_noise ** 2 + 1e-6) * torch.eye(S, device=dev); Kxs = se(x, xs); Kff = se(x, x) + 1e-4 * torch.eye(D, device=dev)
    Kss_inv = torch.linalg.inv(Kss)
    W = Kxs @ Kss_inv; SigK = Kff - Kxs @ Kss_inv @ Kxs.T; kstd = torch.sqrt(torch.clamp(torch.diag(SigK), min=1e-8))
    krige = lambda ys: mu_k.unsqueeze(0) + (W @ (ys - mu_k[sidx]).T).T
    f0_tr, f0_va, f0_te = krige(ys_tr), krige(ys_va), krige(ys_te)

    def train_flow(targets_n, cond_fn, cdim, resid):
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
        return enc, vel
    # direct, gen-residual, det-residual
    enc2, vel2 = train_flow(norm(F_tr), lambda e, b: e(xs_b.expand(len(b), -1), ysn(ys_tr[b])), a.cond_dim, False)
    R = F_tr - f0_tr; rmu, rsd = R.mean(0), R.std(0) + 1e-6; Rn = (R - rmu) / rsd; f0n_tr = norm(f0_tr)
    enc4, vel4 = train_flow(Rn, lambda e, b: torch.cat([e(xs_b.expand(len(b), -1), ysn(ys_tr[b])), f0n_tr[b]], -1), a.cond_dim + D, True)
    enc3 = DeepSets(a.cond_dim, a.hidden).to(dev); det = DetNet(D, a.cond_dim + D, a.hidden).to(dev)
    o3 = torch.optim.Adam(list(enc3.parameters()) + list(det.parameters()), 1e-3)
    for ep in range(a.epochs):
        perm = torch.randperm(a.N_train, device=dev)
        for i in range(0, a.N_train, a.bs):
            b = perm[i:i + a.bs]; c = torch.cat([enc3(xs_b.expand(len(b), -1), ysn(ys_tr[b])), f0n_tr[b]], -1)
            l = ((det(f0n_tr[b], c) - (F_tr[b] - f0_tr[b])) ** 2).mean(); o3.zero_grad(); l.backward(); o3.step()
    # PCA + ridge
    mu_pca = F_tr.mean(0); U, Sg, V = torch.linalg.svd(F_tr - mu_pca, full_matrices=False); Vr = V[:a.pca_r].T
    C = (F_tr - mu_pca) @ Vr; Xa = torch.cat([ys_tr, torch.ones(a.N_train, 1, device=dev)], 1)
    B_ = torch.linalg.solve(Xa.T @ Xa + a.ridge * torch.eye(S + 1, device=dev), Xa.T @ C)
    pca_pred = lambda ys: mu_pca.unsqueeze(0) + (torch.cat([ys, torch.ones(ys.shape[0], 1, device=dev)], 1) @ B_) @ Vr.T
    pca_sd = (F_va - pca_pred(ys_va)).std(0) + 1e-6
    # DPS uncond flow
    uvel = UVel(D, a.hidden).to(dev); ou = torch.optim.Adam(uvel.parameters(), 1e-3); Fn = norm(F_tr)
    for ep in range(a.epochs):
        perm = torch.randperm(a.N_train, device=dev)
        for i in range(0, a.N_train, a.bs):
            b = perm[i:i + a.bs]; f1 = Fn[b]; z = torch.randn_like(f1); t = torch.rand(f1.shape[0], 1, device=dev)
            ft = (1 - t) * z + t * f1; l = ((uvel(ft, t) - (f1 - z)) ** 2).mean(); ou.zero_grad(); l.backward(); ou.step()
    def dps(yn, n):
        f = torch.randn(n, D, device=dev); ts = torch.linspace(0, 1, a.steps + 1, device=dev)
        for i in range(a.steps):
            t0 = ts[i]; dt = ts[i + 1] - ts[i]
            with torch.no_grad(): f = f + dt * uvel(f, t0.repeat(n).unsqueeze(-1))
            f = f.detach().requires_grad_(True); v2 = uvel(f, (t0 + dt).repeat(n).unsqueeze(-1)); f1h = f + (1 - (t0 + dt)) * v2
            g = torch.autograd.grad(((f1h[:, sidx] - yn) ** 2).sum(), f)[0]; f = (f - a.dps_zeta * g).detach()
        return f

    def coverage(F, m, sd, al): return ((F - m).abs() <= 2 * al * sd).float().mean().item()
    def calibrate(F, m, sd):
        al = 1.0
        for _ in range(60):
            c = coverage(F, m, sd, al)
            if abs(c - 0.95) < 0.003: break
            al *= 1.05 if c < 0.95 else 0.96
        return al
    rmse = lambda u, v: torch.sqrt(((u - v) ** 2).mean()).item()
    Strue = (F_te[:, 1:] - F_te[:, :-1]).abs().max(1).values.mean().item(); TVtrue = (F_te[:, 1:] - F_te[:, :-1]).abs().sum(1).mean().item()

    def pred(method, ys, F):
        if method == "kriging": m = krige(ys); return m, kstd.unsqueeze(0).expand(F.shape[0], -1).clone()
        if method == "pca": return pca_pred(ys), pca_sd.unsqueeze(0).expand(F.shape[0], -1).clone()
        if method == "direct":
            m = torch.zeros_like(F); sd = torch.zeros_like(F)
            with torch.no_grad():
                for j in range(F.shape[0]): c = enc2(xs_b, ysn(ys[j:j+1])).expand(a.K, -1); fs = denorm(sample(vel2, c, a.K)); m[j] = fs.mean(0); sd[j] = fs.std(0) + 1e-6
            return m, sd
        if method == "dps":
            m = torch.zeros_like(F); sd = torch.zeros_like(F)
            for j in range(F.shape[0]): fs = denorm(dps(ysn(ys[j:j+1])[0], a.K)); m[j] = fs.mean(0); sd[j] = fs.std(0) + 1e-6
            return m, sd
        if method == "detres":
            m = torch.zeros_like(F)
            with torch.no_grad():
                for j in range(F.shape[0]): c = torch.cat([enc3(xs_b, ysn(ys[j:j+1])), norm(f0(ys[j:j+1]) if False else krige(ys[j:j+1]))], -1); m[j] = krige(ys[j:j+1])[0] + det(norm(krige(ys[j:j+1])), c).squeeze(0)
            return m, kstd.unsqueeze(0).expand(F.shape[0], -1).clone()
        if method == "genres":
            m = torch.zeros_like(F); sd = torch.zeros_like(F)
            with torch.no_grad():
                for j in range(F.shape[0]):
                    fk = krige(ys[j:j+1]); c = torch.cat([enc4(xs_b, ysn(ys[j:j+1])), norm(fk)], -1).expand(a.K, -1)
                    fs = fk + (sample(vel4, c, a.K) * rsd + rmu); m[j] = fs.mean(0); sd[j] = fs.std(0) + 1e-6
            return m, sd
    # ---- ensembles for the (multivariate) Energy Score; kriging uses the full posterior covariance ----
    _Sk = (SigK + SigK.T) / 2 + 1e-5 * torch.eye(D, device=dev)
    _wk, _Vk = torch.linalg.eigh(_Sk); Lk = _Vk @ torch.diag(torch.sqrt(torch.clamp(_wk, min=1e-8)))
    def ens_build(method, ys, F):
        Nf = F.shape[0]
        if method == "dps":
            E = torch.zeros(Nf, a.K, D, device=dev)
            for j in range(Nf): E[j] = denorm(dps(ysn(ys[j:j+1])[0], a.K))
            return E
        with torch.no_grad():
            if method == "kriging":
                m = krige(ys); return m.unsqueeze(1) + torch.randn(Nf, a.K, D, device=dev) @ Lk.T
            if method == "pca":
                m = pca_pred(ys); return m.unsqueeze(1) + pca_sd.view(1, 1, D) * torch.randn(Nf, a.K, D, device=dev)
            if method == "direct":
                E = torch.zeros(Nf, a.K, D, device=dev)
                for j in range(Nf): E[j] = denorm(sample(vel2, enc2(xs_b, ysn(ys[j:j+1])).expand(a.K, -1), a.K))
                return E
            if method == "detres":
                m = torch.zeros(Nf, D, device=dev)
                for j in range(Nf):
                    fk = krige(ys[j:j+1]); c = torch.cat([enc3(xs_b, ysn(ys[j:j+1])), norm(fk)], -1); m[j] = fk[0] + det(norm(fk), c).squeeze(0)
                return m.unsqueeze(1) + torch.randn(Nf, a.K, D, device=dev) @ Lk.T  # det. mean + kriging Gaussian spread
            if method == "genres":
                E = torch.zeros(Nf, a.K, D, device=dev)
                for j in range(Nf):
                    fk = krige(ys[j:j+1]); c = torch.cat([enc4(xs_b, ysn(ys[j:j+1])), norm(fk)], -1).expand(a.K, -1)
                    E[j] = fk + (sample(vel4, c, a.K) * rsd + rmu)
                return E
    def es_metric(E, F): return ((E - F.unsqueeze(1)).norm(dim=2).mean() - 0.5 * torch.cdist(E, E).mean((1, 2)).mean()).item()
    # ---- matched-coverage MPIW via empirical central-quantile intervals (shape-agnostic; no Gaussian assumption) ----
    def _emp_cov(E, F, tau):
        lo = torch.quantile(E, (1 - tau) / 2, dim=1); hi = torch.quantile(E, 1 - (1 - tau) / 2, dim=1)
        return ((F >= lo) & (F <= hi)).float().mean().item()
    def _emp_width(E, tau):
        lo = torch.quantile(E, (1 - tau) / 2, dim=1); hi = torch.quantile(E, 1 - (1 - tau) / 2, dim=1)
        return (hi - lo).mean().item()
    def _calib_tau(Ev, Fv):  # bisect the nominal central level to reach 0.95 empirical coverage on validation
        lo_t, hi_t = 0.50, 0.9995
        for _ in range(40):
            mid = 0.5 * (lo_t + hi_t)
            if _emp_cov(Ev, Fv, mid) < 0.95: lo_t = mid
            else: hi_t = mid
        return 0.5 * (lo_t + hi_t)
    out = {}
    for method in ["kriging", "pca", "direct", "dps", "detres", "genres"]:
        mt = pred(method, ys_te, F_te)[0]
        Ev = ens_build(method, ys_va, F_va); Et = ens_build(method, ys_te, F_te)
        tau = _calib_tau(Ev, F_va)
        TVs = (mt[:, 1:] - mt[:, :-1]).abs().sum(1).mean().item()
        out[method] = {"RMSE": rmse(mt, F_te), "ES": es_metric(Et, F_te), "MPIW": _emp_width(Et, tau), "TVerr": abs(TVs - TVtrue)}
    return out

allres = {m: {"RMSE": [], "ES": [], "MPIW": [], "TVerr": []} for m in ["kriging", "pca", "direct", "dps", "detres", "genres"]}
for s in a.seeds:
    print(f"=== seed {s} ===", flush=True); r = run_seed(s)
    for m in allres:
        for k in ["RMSE", "ES", "MPIW", "TVerr"]: allres[m][k].append(r[m][k])

# aggregate + significance (genres vs each baseline)
summary = {}
for m in allres:
    summary[m] = {k: f"{np.mean(v):.4f}+/-{np.std(v):.4f}" for k, v in allres[m].items()}
sig = {}
for k in ["RMSE", "ES", "MPIW", "TVerr"]:
    g = np.array(allres["genres"][k]); sig[k] = {}
    pvals = {}
    for m in ["kriging", "pca", "direct", "dps", "detres"]:
        b = np.array(allres[m][k])
        try: w = stats.wilcoxon(g, b).pvalue
        except Exception: w = float("nan")
        t = stats.ttest_rel(g, b).pvalue
        pvals[m] = {"wilcoxon_p": float(w), "ttest_p": float(t), "mean_diff(base-ours)": float(b.mean() - g.mean())}
    # Holm correction on ttest p across the 5 comparisons
    items = sorted(pvals.items(), key=lambda kv: kv[1]["ttest_p"]); msal = len(items)
    for rank, (m, d) in enumerate(items):
        d["ttest_p_holm"] = float(min(1.0, d["ttest_p"] * (msal - rank)))
    sig[k] = pvals
res = {"n_seeds": len(a.seeds), "summary_mean_std": summary, "significance_KRFlow_vs_baseline": sig}
print("\n================ Channel: multi-seed stats ================")
print(json.dumps(res, indent=2)); Path(a.outdir, "metrics.json").write_text(json.dumps(res, indent=2)); print("done")
