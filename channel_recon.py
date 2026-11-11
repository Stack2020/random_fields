"""Channel (sharp-interface) reconstruction visualization: mean +/- 95% band for
all Table-1 methods on ONE test field, as small multiples for easy comparison."""
import numpy as np, torch, torch.nn as nn, os
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 8.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 300})
def panel(ax, lab): ax.text(-0.06, 1.06, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
TRUE = "#222222"; OURS = "#1f77b4"; SENS = "#d62728"; BAND = "#1f77b4"
dev = "cuda"; D = 100; WIDTH = .02; AMP = 1.5; NOISE = .02; KLEN = .1; KVAR = 1.0; KEMP = True; KSHRINK = 0.05
NSENS = 10; NTR = 6000; NVAL = 300; NTE = 400; K = 200; EP = 1000; BS = 256; STEPS = 60; ZETA = .2
out = "./results_pretty"; os.makedirs(out, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
x = torch.linspace(0, 1, D, device=dev); xnp = x.cpu().numpy()
sidx = torch.linspace(5, D-6, NSENS, dtype=torch.long, device=dev); xs = x[sidx]; xsn = xs.cpu().numpy()
def gen(N):
    xi = .25+.5*torch.rand(N, 1, device=dev); sgn = torch.where(torch.rand(N, 1, device=dev) < .5, -1., 1.)
    base = .2*torch.randn(N, 1, device=dev); step = torch.sigmoid((x.unsqueeze(0)-xi)/WIDTH); wig = torch.zeros(N, D, device=dev)
    for k in range(1, 4): wig += (.05/k)*torch.randn(N, 1, device=dev)*torch.sin(k*np.pi*x.unsqueeze(0)*2)
    return base+sgn*AMP*step+wig
class DeepSets(nn.Module):
    def __init__(s, cd=64, h=256):
        super().__init__(); s.phi = nn.Sequential(nn.Linear(2, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cd))
        s.rho = nn.Sequential(nn.Linear(cd, h), nn.SiLU(), nn.Linear(h, cd))
    def forward(s, xa, ya): return s.rho(s.phi(torch.stack([xa, ya], -1)).mean(1))
class Vel(nn.Module):
    def __init__(s, dim, cd, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim+1+cd, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, ft, t, c): return s.net(torch.cat([ft, t, c], -1))
class UVel(nn.Module):
    def __init__(s, dim, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim+1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, ft, t): return s.net(torch.cat([ft, t], -1))
class DetNet(nn.Module):
    def __init__(s, dim, cd, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim+cd, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, fK, c): return s.net(torch.cat([fK, c], -1))
@torch.no_grad()
def sample(vel, c, n):
    f = torch.randn(n, D, device=dev); ts = torch.linspace(0, 1, STEPS+1, device=dev)
    for i in range(STEPS):
        t0 = ts[i]; dt = ts[i+1]-ts[i]; col = lambda tt: tt.repeat(n).unsqueeze(-1)
        k1 = vel(f, col(t0), c); k2 = vel(f+.5*dt*k1, col(t0+.5*dt), c); k3 = vel(f+.5*dt*k2, col(t0+.5*dt), c); k4 = vel(f+dt*k3, col(t0+dt), c)
        f = f+dt/6*(k1+2*k2+2*k3+k4)
    return f
def se(u, v): d = u.reshape(-1, 1)-v.reshape(1, -1); return KVAR*torch.exp(-(d**2)/(2*KLEN**2))
F_tr, F_va, F_te = gen(NTR), gen(NVAL), gen(NTE)
mu_g, sd_g = F_tr.mean(0), F_tr.std(0)+1e-6; nrm = lambda f:(f-mu_g)/sd_g; den = lambda f: f*sd_g+mu_g
obs = lambda F: F[:, sidx]+NOISE*torch.randn(F.shape[0], NSENS, device=dev)
ys_tr, ys_te = obs(F_tr), obs(F_te); ysn = lambda y:(y-mu_g[sidx])/sd_g[sidx]; xb = xs.unsqueeze(0)
if KEMP:
    # empirical mean + shrinkage covariance from the training fields (moment estimation)
    mu_k = F_tr.mean(0); Cd = torch.cov(F_tr.T); Cd = (1-KSHRINK)*Cd + KSHRINK*torch.diag(torch.diag(Cd))
    Kss = Cd[sidx][:, sidx]+(NOISE**2+1e-6)*torch.eye(NSENS, device=dev); Kxs = Cd[:, sidx]; Kff = Cd+1e-4*torch.eye(D, device=dev)
else:
    mu_k = torch.zeros(D, device=dev)
    Kss = se(xs, xs)+(NOISE**2+1e-6)*torch.eye(NSENS, device=dev); Kxs = se(x, xs); Kff = se(x, x)+1e-4*torch.eye(D, device=dev)
Kinv = torch.linalg.inv(Kss); W = Kxs @ Kinv; SigK = Kff-Kxs @ Kinv @ Kxs.T
kstd = torch.sqrt(torch.clamp(torch.diag(SigK), min=1e-8))
_Sk = (SigK+SigK.T)/2 + 1e-5*torch.eye(D, device=dev); _wk, _Vk = torch.linalg.eigh(_Sk); Lk = _Vk @ torch.diag(torch.sqrt(torch.clamp(_wk, min=1e-8)))
krige = lambda ys: mu_k.unsqueeze(0) + (W @ (ys-mu_k[sidx]).T).T; f0_tr, f0_te = krige(ys_tr), krige(ys_te)
# KR-Flow (gen residual)
R = F_tr-f0_tr; rmu, rsd = R.mean(0), R.std(0)+1e-6; Rn = (R-rmu)/rsd; f0n = nrm(f0_tr)
enc = DeepSets().to(dev); vel = Vel(D, 64+D).to(dev); op = torch.optim.Adam(list(enc.parameters())+list(vel.parameters()), 1e-3)
# Direct flow
encd = DeepSets().to(dev); veld = Vel(D, 64).to(dev); opd = torch.optim.Adam(list(encd.parameters())+list(veld.parameters()), 1e-3)
# det residual
enc3 = DeepSets().to(dev); det = DetNet(D, 64+D).to(dev); o3 = torch.optim.Adam(list(enc3.parameters())+list(det.parameters()), 1e-3)
# DPS uncond
uvel = UVel(D).to(dev); ou = torch.optim.Adam(uvel.parameters(), 1e-3); Fn = nrm(F_tr)
for e in range(EP):
    pm = torch.randperm(NTR, device=dev)
    for i in range(0, NTR, BS):
        b = pm[i:i+BS]; B = b.shape[0]; z = torch.randn(B, D, device=dev); t = torch.rand(B, 1, device=dev)
        # gen residual
        f1 = Rn[b]; ft = (1-t)*z+t*f1; c = torch.cat([enc(xb.expand(B, -1), ysn(ys_tr[b])), f0n[b]], -1)
        l = ((vel(ft, t, c)-(f1-z))**2).mean()+1e-4*((ft+(1-t)*vel(ft, t, c))**2).mean(); op.zero_grad(); l.backward(); op.step()
        # direct
        f1 = Fn[b]; z2 = torch.randn_like(f1); ft2 = (1-t)*z2+t*f1; cd = encd(xb.expand(B, -1), ysn(ys_tr[b]))
        ld = ((veld(ft2, t, cd)-(f1-z2))**2).mean(); opd.zero_grad(); ld.backward(); opd.step()
        # det
        cc = torch.cat([enc3(xb.expand(B, -1), ysn(ys_tr[b])), f0n[b]], -1); ldet = ((det(f0n[b], cc)-(F_tr[b]-f0_tr[b]))**2).mean(); o3.zero_grad(); ldet.backward(); o3.step()
        # dps uncond
        z3 = torch.randn_like(f1); ft3 = (1-t)*z3+t*f1; lu = ((uvel(ft3, t)-(f1-z3))**2).mean(); ou.zero_grad(); lu.backward(); ou.step()
    if (e+1) % 250 == 0: print(f"ep {e+1}", flush=True)
# PCA + ridge
mu_pca = F_tr.mean(0); _U, _S, _Vh = torch.linalg.svd(F_tr-mu_pca, full_matrices=False); Vr = _Vh[:20].T
Ctr = (F_tr-mu_pca) @ Vr; Xa = torch.cat([ys_tr, torch.ones(NTR, 1, device=dev)], 1)
Bp = torch.linalg.solve(Xa.T @ Xa + 1e-3*torch.eye(NSENS+1, device=dev), Xa.T @ Ctr)
pca_pred = lambda ys: mu_pca.unsqueeze(0)+(torch.cat([ys, torch.ones(ys.shape[0], 1, device=dev)], 1) @ Bp) @ Vr.T
pca_sd = (F_va-pca_pred(obs(F_va))).std(0)+1e-6
def dps(yn, n):
    f = torch.randn(n, D, device=dev); ts = torch.linspace(0, 1, STEPS+1, device=dev)
    for i in range(STEPS):
        t0 = ts[i]; dt = ts[i+1]-ts[i]
        with torch.no_grad(): f = f+dt*uvel(f, t0.repeat(n).unsqueeze(-1))
        f = f.detach().requires_grad_(True); v2 = uvel(f, (t0+dt).repeat(n).unsqueeze(-1)); f1h = f+(1-(t0+dt))*v2
        g = torch.autograd.grad(((f1h[:, sidx]-yn)**2).sum(), f)[0]; f = (f-ZETA*g).detach()
    return f
# pick a test field with interface near center
xi_est = x[(F_te[:, 1:]-F_te[:, :-1]).abs().argmax(1)]; j = int((xi_est-0.5).abs().argmin())
means = {}
with torch.no_grad():
    means["Kriging / CGS"] = f0_te[j]
    means["PCA/KLE+reg"] = pca_pred(ys_te[j:j+1])[0]
    means["Direct flow"] = den(sample(veld, encd(xb, ysn(ys_te[j:j+1])).expand(K, -1), K)).mean(0)
    cc = torch.cat([enc3(xb, ysn(ys_te[j:j+1])), nrm(f0_te[j:j+1])], -1)
    means["Kriging+det res."] = f0_te[j]+det(nrm(f0_te[j:j+1]), cc).squeeze(0)
    means["KR-Flow (ours)"] = (f0_te[j:j+1]+(sample(vel, torch.cat([enc(xb, ysn(ys_te[j:j+1])), nrm(f0_te[j:j+1])], -1).expand(K, -1), K)*rsd+rmu)).mean(0)
means["DPS-guidance"] = den(dps(ysn(ys_te[j:j+1])[0], K)).mean(0)
ft = F_te[j].cpu().numpy(); yobs = ys_te[j].cpu().numpy()
styles = {"Kriging / CGS": ("#7f7f7f", "--", 1.7), "PCA/KLE+reg": ("#9467bd", ":", 1.7),
          "Direct flow": ("#8c564b", (0, (5, 1)), 1.7), "DPS-guidance": ("#e377c2", (0, (3, 1, 1, 1)), 1.7),
          "Kriging+det res.": ("#2ca02c", "-.", 1.7), "KR-Flow (ours)": (OURS, "-", 2.6)}
# ---- test-set ensembles for KR-Flow (calibration + sharpness vs kriging) ----
Kc = 80; Nc = min(150, NTE)
ens = torch.zeros(Nc, Kc, D, device=dev); ens_dir = torch.zeros(Nc, Kc, D, device=dev); ens_dps = torch.zeros(Nc, Kc, D, device=dev)
with torch.no_grad():
    for jj in range(Nc):
        c = torch.cat([enc(xb, ysn(ys_te[jj:jj+1])), nrm(f0_te[jj:jj+1])], -1).expand(Kc, -1)
        ens[jj] = f0_te[jj:jj+1]+(sample(vel, c, Kc)*rsd+rmu)
        ens_dir[jj] = den(sample(veld, encd(xb, ysn(ys_te[jj:jj+1])).expand(Kc, -1), Kc))
for jj in range(Nc): ens_dps[jj] = den(dps(ysn(ys_te[jj:jj+1])[0], Kc))   # DPS uses autograd
Yc = F_te[:Nc]; f0c = f0_te[:Nc]; skb = kstd.unsqueeze(0).expand(Nc, D)
zq = lambda p: float(torch.distributions.Normal(0., 1.).icdf(torch.tensor(0.5+p/2)))
def cov_q(E, p): lo = torch.quantile(E, (1-p)/2, 1); hi = torch.quantile(E, 1-(1-p)/2, 1); return (((Yc >= lo) & (Yc <= hi)).float().mean()).item()
def wid_q(E, p): lo = torch.quantile(E, (1-p)/2, 1); hi = torch.quantile(E, 1-(1-p)/2, 1); return (hi-lo).mean().item()
cov_gauss = lambda p: (((Yc-f0c).abs() <= zq(p)*skb).float().mean()).item()
levels = np.array([.5, .6, .7, .8, .9, .95])
cov_k = np.array([cov_gauss(p) for p in levels]); cov_o = np.array([cov_q(ens, p) for p in levels])
cov_di = np.array([cov_q(ens_dir, p) for p in levels]); cov_dp = np.array([cov_q(ens_dps, p) for p in levels])
cal_k = float(np.abs(cov_k-levels).mean()); cal_o = float(np.abs(cov_o-levels).mean())
cal_di = float(np.abs(cov_di-levels).mean()); cal_dp = float(np.abs(cov_dp-levels).mean())
ps = np.linspace(0.5, 0.99, 14)
ck_c = [cov_gauss(p) for p in ps]; wk_c = [(2*zq(p)*skb).mean().item() for p in ps]
co_c = [cov_q(ens, p) for p in ps]; wo_c = [wid_q(ens, p) for p in ps]
cdi_c = [cov_q(ens_dir, p) for p in ps]; wdi_c = [wid_q(ens_dir, p) for p in ps]
cdp_c = [cov_q(ens_dps, p) for p in ps]; wdp_c = [wid_q(ens_dps, p) for p in ps]
# ---- Fig 3: (a) single-field reconstruction, (b) calibration, (c) sharpness-coverage ----
fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.5))
KRIG = "#ff7f0e"
with torch.no_grad():
    ensj = f0_te[j:j+1]+(sample(vel, torch.cat([enc(xb, ysn(ys_te[j:j+1])), nrm(f0_te[j:j+1])], -1).expand(120, -1), 120)*rsd+rmu)
mo = ensj.mean(0).cpu().numpy(); loj = torch.quantile(ensj, .025, 0).cpu().numpy(); hij = torch.quantile(ensj, .975, 0).cpu().numpy()
ax[0].fill_between(xnp, loj, hij, color=OURS, alpha=.20, lw=0, label="KR-Flow 95% band")
ax[0].plot(xnp, mo, color=OURS, lw=2.2, label="KR-Flow mean")
ax[0].plot(xnp, f0_te[j].cpu().numpy(), color=KRIG, ls="--", lw=1.9, label="kriging (oversmoothed)")
ax[0].plot(xnp, ft, color=TRUE, lw=2.0, label="true field")
ax[0].scatter(xsn, yobs, s=40, facecolor=SENS, edgecolor="white", linewidth=.8, zorder=6, label="sensors")
ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$u(x)$"); ax[0].set_title("Reconstruction on one test field"); panel(ax[0], "(a)")
DIRC = "#8c564b"; DPSC = "#e377c2"
ax[1].plot([0, 1], [0, 1], color="0.6", ls=":", lw=1.5, label="ideal")
ax[1].plot(levels, cov_k, "-o", color=KRIG, lw=1.8, ms=5, label="kriging")
ax[1].plot(levels, cov_di, "-^", color=DIRC, lw=1.8, ms=5, label="direct flow")
ax[1].plot(levels, cov_dp, "-v", color=DPSC, lw=1.8, ms=5, label="DPS")
ax[1].plot(levels, cov_o, "-s", color=OURS, lw=2, ms=6, label="KR-Flow")
ax[1].set_xlabel("Nominal coverage"); ax[1].set_ylabel("Empirical coverage"); ax[1].set_title("Calibration"); panel(ax[1], "(b)")
ax[1].legend(frameon=True, framealpha=.9, edgecolor="0.85", loc="upper left"); ax[1].grid(True, alpha=.25, lw=.5)
ax[2].plot(ck_c, wk_c, "-o", color=KRIG, lw=1.8, ms=4, label="kriging")
ax[2].plot(cdi_c, wdi_c, "-^", color=DIRC, lw=1.8, ms=4, label="direct flow")
ax[2].plot(cdp_c, wdp_c, "-v", color=DPSC, lw=1.8, ms=4, label="DPS")
ax[2].plot(co_c, wo_c, "-s", color=OURS, lw=2, ms=5, label="KR-Flow")
ax[2].set_xlabel("Empirical coverage"); ax[2].set_ylabel("Mean interval width"); ax[2].set_title("Sharpness–coverage (lower = sharper)"); panel(ax[2], "(c)")
ax[2].legend(frameon=True, framealpha=.92, edgecolor="0.85", loc="upper left"); ax[2].grid(True, alpha=.25, lw=.5)
ax[0].legend(loc="lower left", frameon=True, framealpha=.9, edgecolor="0.85", fontsize=7.6)
fig.tight_layout(); fig.subplots_adjust(wspace=0.28)
import os as _os; _os.makedirs(f"{out}/weights", exist_ok=True)
torch.save({"enc": enc.state_dict(), "vel": vel.state_dict(), "rmu": rmu.detach().cpu(), "rsd": rsd.detach().cpu(), "mu_g": mu_g.detach().cpu(), "sd_g": sd_g.detach().cpu(), "W": W.detach().cpu(), "mu_k": mu_k.detach().cpu(), "kstd": kstd.detach().cpu()}, f"{out}/weights/channel_krflow.pt"); print("[weights] channel_krflow.pt saved", flush=True)
fig.savefig(f"{out}/idea8_1d.pdf", bbox_inches="tight"); fig.savefig(f"{out}/idea8_1d.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("channel recon done", flush=True)
