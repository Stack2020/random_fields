"""Fig 6 redo: show KR-Flow's UQ advantage over kriging on real 2D wind.
Computes calibration / CRPS / coverage / sharpness over the full test set and
draws an advantage-focused figure (reliability + CRPS/MPIW + spatial CRPS gain)."""
import numpy as np, torch, torch.nn as nn, h5py, os, math
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 8.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 300})
def panel(ax, lab): ax.text(-0.06, 1.06, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
dev = "cuda"; V1MIN, V1MAX = -2.0, 47.0; SPAN = V1MAX - V1MIN; NX = 10; D = NX*NX
out = "./results_pretty"; os.makedirs(out, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
OURS = "#1f77b4"; KRIG = "#ff7f0e"; SENS = "#d62728"
def load(fn, snaps):
    f = h5py.File(fn, "r"); g = f["wind_samples"]; keys = sorted(g.keys(), key=int)
    T = g[keys[0]]["v1"].shape[1]; ti = np.linspace(T//6, T-T//6, snaps).astype(int); o = []
    for k in keys: o.append(g[k]["v1"][:][:, ti].T)
    f.close(); return torch.tensor(np.concatenate(o, 0), dtype=torch.float32)
base = "./"
F_tr = ((load(base+"data/wind/wind_train_data.hdf5", 2) - V1MIN)/SPAN).to(dev)[:8000]
F_te = ((load(base+"data/wind/wind_test_data.hdf5", 2) - V1MIN)/SPAN).to(dev)[:300]
ii = torch.arange(D, device=dev); cx = (ii//NX).float()/(NX-1); cy = (ii % NX).float()/(NX-1); coords = torch.stack([cx, cy], -1)
strips = [1, 5, 8]; sens = torch.cat([torch.where(ii//NX == c)[0] for c in strips]).to(dev); S = sens.shape[0]; sxy = coords[sens]
mu, sd = F_tr.mean(0), F_tr.std(0)+1e-6; nrm = lambda f:(f-mu)/sd; phys = lambda f: f*SPAN+V1MIN
mprof = F_tr.mean(0)
def se2d(A, B, ell=.3): d2 = ((A.unsqueeze(1)-B.unsqueeze(0))**2).sum(-1); return torch.exp(-d2/(2*ell**2))
kvar = ((F_tr-mprof)**2).mean(); Kss = kvar*se2d(sxy, sxy)+(0.02**2+1e-5)*torch.eye(S, device=dev); Kxs = kvar*se2d(coords, sxy)
Kss_inv = torch.linalg.inv(Kss); W = Kxs @ Kss_inv; obsv = lambda F: F[:, sens]
krige = lambda F: mprof.unsqueeze(0)+(W @ (obsv(F)-mprof[sens]).T).T
# kriging posterior marginal std (homoscedastic across fields), in normalized units
pvar = (kvar - (Kxs * (W)).sum(1)).clamp_min(1e-8); sk = torch.sqrt(pvar)        # (D,)
f0_tr, f0_te = krige(F_tr), krige(F_te); ysn = lambda F:(obsv(F)-mu[sens])/sd[sens]; sxy_b = sxy.unsqueeze(0)
class DeepSets(nn.Module):
    def __init__(s, cd=96, hh=384):
        super().__init__(); s.phi = nn.Sequential(nn.Linear(3, hh), nn.SiLU(), nn.Linear(hh, hh), nn.SiLU(), nn.Linear(hh, cd))
        s.rho = nn.Sequential(nn.Linear(cd, hh), nn.SiLU(), nn.Linear(hh, cd))
    def forward(s, xy, y): return s.rho(s.phi(torch.cat([xy, y.unsqueeze(-1)], -1)).mean(1))
class Vel(nn.Module):
    def __init__(s, dim, cd, hh=384):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim+1+cd, hh), nn.SiLU(), nn.Linear(hh, hh), nn.SiLU(), nn.Linear(hh, hh), nn.SiLU(), nn.Linear(hh, dim))
    def forward(s, ft, t, c): return s.net(torch.cat([ft, t, c], -1))
@torch.no_grad()
def sample(vel, c, n, steps=80):
    f = torch.randn(n, D, device=dev); ts = torch.linspace(0, 1, steps+1, device=dev)
    for i in range(steps):
        t0 = ts[i]; dt = ts[i+1]-ts[i]; col = lambda tt: tt.repeat(n).unsqueeze(-1)
        k1 = vel(f, col(t0), c); k2 = vel(f+.5*dt*k1, col(t0+.5*dt), c); k3 = vel(f+.5*dt*k2, col(t0+.5*dt), c); k4 = vel(f+dt*k3, col(t0+dt), c)
        f = f+dt/6*(k1+2*k2+2*k3+k4)
    return f
R = F_tr-f0_tr; rmu, rsd = R.mean(0), R.std(0)+1e-6; Rn = (R-rmu)/rsd; f0n = nrm(f0_tr)
enc = DeepSets().to(dev); vel = Vel(D, 96+D).to(dev); opt = torch.optim.Adam(list(enc.parameters())+list(vel.parameters()), 1e-3)
N = F_tr.shape[0]
for e in range(1000):
    pm = torch.randperm(N, device=dev)
    for i in range(0, N, 256):
        b = pm[i:i+256]; f1 = Rn[b]; B = f1.shape[0]; c = torch.cat([enc(sxy_b.expand(B, -1, -1), ysn(F_tr[b])), f0n[b]], -1)
        z = torch.randn_like(f1); t = torch.rand(B, 1, device=dev); ft = (1-t)*z+t*f1
        l = ((vel(ft, t, c)-(f1-z))**2).mean()+1e-4*((ft+(1-t)*vel(ft, t, c))**2).mean(); opt.zero_grad(); l.backward(); opt.step()
    if (e+1) % 300 == 0: print(f"[wind] {e+1}", flush=True)
# ---- baseline 1: direct conditional flow (sensor-conditioned, no kriging reference) ----
encd = DeepSets().to(dev); veld = Vel(D, 96).to(dev); optd = torch.optim.Adam(list(encd.parameters())+list(veld.parameters()), 1e-3)
for e in range(1000):
    pm = torch.randperm(N, device=dev)
    for i in range(0, N, 256):
        b = pm[i:i+256]; f1 = nrm(F_tr[b]); B = f1.shape[0]; c = encd(sxy_b.expand(B, -1, -1), ysn(F_tr[b]))
        z = torch.randn_like(f1); t = torch.rand(B, 1, device=dev); ft = (1-t)*z+t*f1
        l = ((veld(ft, t, c)-(f1-z))**2).mean(); optd.zero_grad(); l.backward(); optd.step()
print("[wind] direct flow trained", flush=True)
# ---- baseline 2: DPS (unconditional flow + observation-guided sampling) ----
class UVel(nn.Module):
    def __init__(s, dim, hh=384):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim+1, hh), nn.SiLU(), nn.Linear(hh, hh), nn.SiLU(), nn.Linear(hh, hh), nn.SiLU(), nn.Linear(hh, dim))
    def forward(s, ft, t): return s.net(torch.cat([ft, t], -1))
uvel = UVel(D).to(dev); optu = torch.optim.Adam(uvel.parameters(), 1e-3)
for e in range(1000):
    pm = torch.randperm(N, device=dev)
    for i in range(0, N, 256):
        b = pm[i:i+256]; f1 = nrm(F_tr[b]); z = torch.randn_like(f1); t = torch.rand(f1.shape[0], 1, device=dev)
        ft = (1-t)*z+t*f1; l = ((uvel(ft, t)-(f1-z))**2).mean(); optu.zero_grad(); l.backward(); optu.step()
print("[wind] DPS flow trained", flush=True)
def dps(yn, n, steps=80, zeta=0.2):
    f = torch.randn(n, D, device=dev); ts = torch.linspace(0, 1, steps+1, device=dev)
    for i in range(steps):
        t0 = ts[i]; dt = ts[i+1]-ts[i]
        with torch.no_grad(): f = f+dt*uvel(f, t0.repeat(n).unsqueeze(-1))
        f = f.detach().requires_grad_(True); v2 = uvel(f, (t0+dt).repeat(n).unsqueeze(-1)); f1h = f+(1-(t0+dt))*v2
        g = torch.autograd.grad(((f1h[:, sens]-yn)**2).sum(), f)[0]; f = (f-zeta*g).detach()
    return f
# ---- ensembles over full test set ----
K = 64; Nte = F_te.shape[0]
ens = torch.zeros(Nte, K, D, device=dev)
with torch.no_grad():
    for j in range(Nte):
        c = torch.cat([enc(sxy_b, ysn(F_te[j:j+1])), nrm(f0_te[j:j+1])], -1).expand(K, -1)
        ens[j] = f0_te[j:j+1] + (sample(vel, c, K)*rsd+rmu)
mr = ens.mean(1); sr_ours = ens.std(1)                                    # (Nte,D)
# baseline ensembles (physical F-units): direct flow + DPS
dn = lambda f: f*sd+mu
ens_di = torch.zeros(Nte, K, D, device=dev); ens_dp = torch.zeros(Nte, K, D, device=dev)
with torch.no_grad():
    for j in range(Nte): ens_di[j] = dn(sample(veld, encd(sxy_b, ysn(F_te[j:j+1])).expand(K, -1), K))
for j in range(Nte): ens_dp[j] = dn(dps(ysn(F_te[j:j+1])[0], K))
Y = F_te                                                                   # truth (Nte,D)
def norm_cdf(x): return 0.5*(1+torch.erf(x/math.sqrt(2)))
def norm_pdf(x): return torch.exp(-0.5*x*x)/math.sqrt(2*math.pi)
# ---- metrics (physical m/s units via *SPAN) ----
rmse_k = (((f0_te-Y)*SPAN)**2).mean().sqrt().item()
rmse_o = (((mr-Y)*SPAN)**2).mean().sqrt().item()
# best-of-K RMSE (sample closest to truth)
bok = (((ens-Y.unsqueeze(1))*SPAN)**2).mean(-1).min(1).values.sqrt().mean().item()
# CRPS kriging (Gaussian closed form), sigma = sk broadcast
skb = sk.unsqueeze(0).expand(Nte, D)
w = (Y-f0_te)/skb; crps_k = (skb*(w*(2*norm_cdf(w)-1)+2*norm_pdf(w)-1/math.sqrt(math.pi))*SPAN).mean().item()
# CRPS KR-Flow (empirical ensemble): E|X-y| - 0.5 E|X-X'|
ax_ = (ens-Y.unsqueeze(1)).abs().mean(1)
pair = (ens.unsqueeze(2)-ens.unsqueeze(1)).abs().mean((1, 2))
crps_o = (((ax_-0.5*pair))*SPAN).mean().item()
# coverage + MPIW at nominal levels
levels = np.array([.5, .6, .7, .8, .9, .95])
cov_k, cov_o, cov_di, cov_dp = [], [], [], []
def _covq(E, p): lo = torch.quantile(E, (1-p)/2, dim=1); hi = torch.quantile(E, 1-(1-p)/2, dim=1); return ((Y >= lo) & (Y <= hi)).float().mean().item()
for p in levels:
    zq = float(torch.distributions.Normal(0., 1.).icdf(torch.tensor(0.5+p/2)))
    cov_k.append(((Y-f0_te).abs() <= zq*skb).float().mean().item())
    cov_o.append(_covq(ens, p)); cov_di.append(_covq(ens_di, p)); cov_dp.append(_covq(ens_dp, p))
cov_k = np.array(cov_k); cov_o = np.array(cov_o); cov_di = np.array(cov_di); cov_dp = np.array(cov_dp)
cal_di = float(np.abs(cov_di-levels).mean()); cal_dp = float(np.abs(cov_dp-levels).mean())
# 95% MPIW
zq95 = 1.959964; mpiw_k = (2*zq95*skb*SPAN).mean().item()
lo95 = torch.quantile(ens, .025, dim=1); hi95 = torch.quantile(ens, .975, dim=1); mpiw_o = ((hi95-lo95)*SPAN).mean().item()
cov95_k, cov95_o = cov_k[-1], cov_o[-1]
# calibration error (mean |emp-nom|)
cal_k = float(np.abs(cov_k-levels).mean()); cal_o = float(np.abs(cov_o-levels).mean())
# Energy Score (multivariate generalization of CRPS; captures spatial dependence), physical m/s
ens_k = f0_te.unsqueeze(1) + skb.unsqueeze(1)*torch.randn(Nte, K, D, device=dev)
es_k = (((ens_k-Y.unsqueeze(1)).norm(dim=2).mean() - 0.5*torch.cdist(ens_k, ens_k).mean((1, 2)).mean())*SPAN).item()
es_o = (((ens-Y.unsqueeze(1)).norm(dim=2).mean() - 0.5*torch.cdist(ens, ens).mean((1, 2)).mean())*SPAN).item()
# per-location 95% interval-width reduction map (kriging width - KR-Flow width), m/s
w_k_pix = (2*zq95*sk*SPAN); w_o_pix = ((hi95-lo95)*SPAN).mean(0)
gain = (w_k_pix-w_o_pix).reshape(NX, NX).cpu().numpy()
pct = (100.0*(w_k_pix-w_o_pix)/w_k_pix.clamp_min(1e-6)).reshape(NX, NX).cpu().numpy()
print("==METRICS (physical m/s)==", flush=True)
print(f"RMSE  kriging={rmse_k:.3f}  KR-Flow(mean)={rmse_o:.3f}  KR-Flow(best-of-{K})={bok:.3f}", flush=True)
print(f"CRPS  kriging={crps_k:.3f}  KR-Flow={crps_o:.3f}  (lower better)", flush=True)
print(f"ES    kriging={es_k:.3f}  KR-Flow={es_o:.3f}  (lower better)", flush=True)
print(f"MPIW@95 kriging={mpiw_k:.3f}  KR-Flow={mpiw_o:.3f}  cov95 krig={cov95_k:.3f} ours={cov95_o:.3f}", flush=True)
print(f"calib-err kriging={cal_k:.3f}  KR-Flow={cal_o:.3f}", flush=True)
print("cov_k", np.round(cov_k, 3), flush=True); print("cov_o", np.round(cov_o, 3), flush=True)
# ---- figure: 1x3 advantage panels ----
sxp = (sens % NX).cpu().numpy(); syp = (sens // NX).cpu().numpy()
from matplotlib.patches import Patch
sxp = (sens % NX).cpu().numpy(); syp = (sens // NX).cpu().numpy()
# ===== Main figure (wind_kr.png): (a) calibration, (b) spatial interval-width reduction =====
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
DIRC = "#8c564b"; DPSC = "#e377c2"
ax[0].plot([0, 1], [0, 1], color="0.6", ls=":", lw=1.5, label="ideal")
ax[0].plot(levels, cov_k, "-o", color=KRIG, lw=1.8, ms=5, label="kriging (Gauss.)")
ax[0].plot(levels, cov_di, "-^", color=DIRC, lw=1.8, ms=5, label="direct flow")
ax[0].plot(levels, cov_dp, "-v", color=DPSC, lw=1.8, ms=5, label="DPS")
ax[0].plot(levels, cov_o, "-s", color=OURS, lw=2, ms=6, label="KR-Flow (ours)")
ax[0].set_xlabel("Nominal coverage"); ax[0].set_ylabel("Empirical coverage")
ax[0].set_title("Calibration"); panel(ax[0], "(a)")
ax[0].legend(frameon=True, framealpha=.92, edgecolor="0.85", loc="lower right"); ax[0].grid(True, alpha=.25, lw=.5)
im = ax[1].imshow(pct, origin="lower", cmap="viridis")
ax[1].scatter(sxp, syp, s=18, facecolor="none", edgecolor="white", linewidth=1.1, zorder=5)
ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].set_title("Interval-width reduction"); panel(ax[1], "(b)")
cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04); cb.set_label("Interval-width reduction (%)")
fig.tight_layout(); fig.subplots_adjust(wspace=0.28)
fig.savefig(f"{out}/wind_kr.pdf", bbox_inches="tight"); fig.savefig(f"{out}/wind_kr.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ===== Appendix figure (wind_recon.png): true field vs KR-Flow posterior mean =====
jj = 0
tru = phys(F_te[jj]).reshape(NX, NX).cpu().numpy(); krf = phys(mr[jj]).reshape(NX, NX).cpu().numpy()
vmn = float(min(tru.min(), krf.min())); vmx = float(max(tru.max(), krf.max()))
from mpl_toolkits.axes_grid1 import make_axes_locatable
figr, axr = plt.subplots(1, 2, figsize=(7.2, 3.3))
for a_, fld, ti, lb in [(axr[0], tru, "True field", "(a)"), (axr[1], krf, "KR-Flow posterior mean", "(b)")]:
    imr = a_.imshow(fld, origin="lower", cmap="viridis", vmin=vmn, vmax=vmx, aspect="equal")
    a_.scatter(sxp, syp, s=16, facecolor="none", edgecolor="red", linewidth=1.0, zorder=5)
    a_.set_xticks([]); a_.set_yticks([]); a_.set_title(ti); panel(a_, lb)
# colorbar matched to the heatmap height; append an equal spacer to (a) so both panels stay the same size
cax0 = make_axes_locatable(axr[0]).append_axes("right", size="5%", pad=0.06); cax0.axis("off")
cax = make_axes_locatable(axr[1]).append_axes("right", size="5%", pad=0.06)
cbr = figr.colorbar(imr, cax=cax); cbr.set_label("Velocity (m/s)")
figr.savefig(f"{out}/wind_recon.pdf", bbox_inches="tight"); figr.savefig(f"{out}/wind_recon.png", dpi=300, bbox_inches="tight"); plt.close(figr)
import os as _os; _os.makedirs(f"{out}/weights", exist_ok=True)
torch.save({"enc": enc.state_dict(), "vel": vel.state_dict(), "encd": encd.state_dict(), "veld": veld.state_dict(), "uvel": uvel.state_dict(), "rmu": rmu.detach().cpu(), "rsd": rsd.detach().cpu(), "mu": mu.detach().cpu(), "sd": sd.detach().cpu(), "W": W.detach().cpu(), "mprof": mprof.detach().cpu()}, f"{out}/weights/wind_krflow.pt"); print("[weights] wind_krflow.pt saved", flush=True)
print("wind adv fig done", flush=True)
