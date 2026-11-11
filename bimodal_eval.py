"""Fig 4 redo: KR-Flow's advantage on the semi-analytic BIMODAL benchmark.
(a) double-peak marginal density at x*, (b) energy distance to the exact posterior,
(c) fraction of ambiguous instances whose BOTH modes are captured."""
import numpy as np, torch, torch.nn as nn, os
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 8.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 400})
TRUE = "#222222"; OURS = "#1f77b4"; KRIG = "#ff7f0e"; SENS = "#d62728"; DET = "#2ca02c"
def panel(ax, lab): ax.text(-0.06, 1.06, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
dev = "cuda"; D = 100; out = "./results_pretty"; os.makedirs(out, exist_ok=True)
x = torch.linspace(0, 1, D, device=dev); xnp = x.cpu().numpy()
class DeepSets(nn.Module):
    def __init__(s, cd=64, h=256):
        super().__init__(); s.phi = nn.Sequential(nn.Linear(2, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cd))
        s.rho = nn.Sequential(nn.Linear(cd, h), nn.SiLU(), nn.Linear(h, cd))
    def forward(s, xs, ys): return s.rho(s.phi(torch.stack([xs, ys], -1)).mean(1))
class Vel(nn.Module):
    def __init__(s, dim, cd, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim+1+cd, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, ft, t, c): return s.net(torch.cat([ft, t, c], -1))
class DetNet(nn.Module):
    def __init__(s, dim, cd, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim+cd, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, fK, c): return s.net(torch.cat([fK, c], -1))
@torch.no_grad()
def sample(vel, c, n, steps=80):
    f = torch.randn(n, D, device=dev); ts = torch.linspace(0, 1, steps+1, device=dev)
    for i in range(steps):
        t0 = ts[i]; dt = ts[i+1]-ts[i]; col = lambda tt: tt.repeat(n).unsqueeze(-1)
        k1 = vel(f, col(t0), c); k2 = vel(f+.5*dt*k1, col(t0+.5*dt), c); k3 = vel(f+.5*dt*k2, col(t0+.5*dt), c); k4 = vel(f+dt*k3, col(t0+dt), c)
        f = f+dt/6*(k1+2*k2+2*k3+k4)
    return f
def se(u, v, ell, var): d = u.reshape(-1, 1)-v.reshape(1, -1); return var*torch.exp(-(d**2)/(2*ell**2))
torch.manual_seed(0); np.random.seed(0)
klen, kvar, bamp, bw, bn = .08, .25, 2.0, .08, .05
mu0 = bamp*torch.exp(-((x-.35)/bw)**2); mu1 = bamp*torch.exp(-((x-.65)/bw)**2); mubar = .5*(mu0+mu1)
Kf = se(x, x, klen, kvar)+1e-4*torch.eye(D, device=dev); Lf = torch.linalg.cholesky(Kf)
sx = torch.tensor([.05, .2, .5, .8, .95], device=dev); sidx = torch.stack([(x-v).abs().argmin() for v in sx]); xs = x[sidx]; S = sidx.shape[0]
def genBM(n):
    z = (torch.rand(n, device=dev) < .5).long(); m = torch.where(z.unsqueeze(1) == 0, mu0.unsqueeze(0), mu1.unsqueeze(0))
    return m+(Lf @ torch.randn(D, n, device=dev)).T
F_tr = genBM(6000); F_te = genBM(400); obs = lambda F: F[:, sidx]+bn*torch.randn(F.shape[0], S, device=dev)
ys_tr, ys_te = obs(F_tr), obs(F_te)
Kss = se(xs, xs, klen, kvar)+bn**2*torch.eye(S, device=dev); Kss_inv = torch.linalg.inv(Kss); Kxs = se(x, xs, klen, kvar)
pcov = Kf-Kxs @ Kss_inv @ Kxs.T; Lp = torch.linalg.cholesky(pcov+1e-5*torch.eye(D, device=dev))
def tpw(y): ll = [-.5*(((y-m[sidx]) @ Kss_inv)*(y-m[sidx])).sum(1) for m in (mu0, mu1)]; return torch.softmax(torch.stack(ll, 1), 1)
def tpm(y, md): m = mu0 if md == 0 else mu1; return m.unsqueeze(0)+(Kxs @ Kss_inv @ (y-m[sidx]).T).T
@torch.no_grad()
def samp_true(y, n):
    w = tpw(y)[0]; zc = (torch.rand(n, device=dev) < w[1]).long(); m0 = tpm(y, 0)[0]; m1 = tpm(y, 1)[0]
    mm = torch.where(zc.unsqueeze(1) == 0, m0.unsqueeze(0), m1.unsqueeze(0)); return mm+(Lp @ torch.randn(D, n, device=dev)).T
f0_tr = mubar.unsqueeze(0)+(Kxs @ Kss_inv @ (ys_tr-mubar[sidx]).T).T
f0_te = mubar.unsqueeze(0)+(Kxs @ Kss_inv @ (ys_te-mubar[sidx]).T).T
mu, sd = F_tr.mean(0), F_tr.std(0)+1e-6; nrm = lambda f:(f-mu)/sd; ysn = lambda y:(y-mu[sidx])/sd[sidx]; xb = xs.unsqueeze(0)
R = F_tr-f0_tr; rmu, rsd = R.mean(0), R.std(0)+1e-6; Rn = (R-rmu)/rsd; f0n = nrm(f0_tr)
enc = DeepSets().to(dev); vel = Vel(D, 64+D).to(dev); opt = torch.optim.Adam(list(enc.parameters())+list(vel.parameters()), 1e-3)
for e in range(1200):
    pm = torch.randperm(6000, device=dev)
    for i in range(0, 6000, 256):
        b = pm[i:i+256]; f1 = Rn[b]; B = f1.shape[0]; c = torch.cat([enc(xb.expand(B, -1), ysn(ys_tr[b])), f0n[b]], -1)
        z = torch.randn_like(f1); t = torch.rand(B, 1, device=dev); ft = (1-t)*z+t*f1
        l = ((vel(ft, t, c)-(f1-z))**2).mean()+1e-4*((ft+(1-t)*vel(ft, t, c))**2).mean(); opt.zero_grad(); l.backward(); opt.step()
    if (e+1) % 400 == 0: print(f"[gen] {e+1}", flush=True)
enc3 = DeepSets().to(dev); det = DetNet(D, 64+D).to(dev); o3 = torch.optim.Adam(list(enc3.parameters())+list(det.parameters()), 1e-3)
for e in range(1200):
    pm = torch.randperm(6000, device=dev)
    for i in range(0, 6000, 256):
        b = pm[i:i+256]; c = torch.cat([enc3(xb.expand(len(b), -1), ysn(ys_tr[b])), f0n[b]], -1)
        l = ((det(f0n[b], c)-(F_tr[b]-f0_tr[b]))**2).mean(); o3.zero_grad(); l.backward(); o3.step()
    if (e+1) % 400 == 0: print(f"[det] {e+1}", flush=True)
denorm = lambda f: f*sd+mu
# ---- PCA/KLE + ridge baseline ----
mu_pca = F_tr.mean(0); _U, _S, _Vh = torch.linalg.svd(F_tr-mu_pca, full_matrices=False); Vr = _Vh[:20].T
Ctr = (F_tr-mu_pca) @ Vr; Xa = torch.cat([ys_tr, torch.ones(F_tr.shape[0], 1, device=dev)], 1)
Bpca = torch.linalg.solve(Xa.T @ Xa + 1e-3*torch.eye(S+1, device=dev), Xa.T @ Ctr)
def pca_pred(ys): Xe = torch.cat([ys, torch.ones(ys.shape[0], 1, device=dev)], 1); return mu_pca.unsqueeze(0)+(Xe @ Bpca) @ Vr.T
# ---- DPS-guidance (unconditional flow over the bimodal prior + likelihood guidance) ----
class UVel(nn.Module):
    def __init__(s, dim, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim+1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, ft, t): return s.net(torch.cat([ft, t], -1))
uvel = UVel(D).to(dev); ou = torch.optim.Adam(uvel.parameters(), 1e-3); Fn = nrm(F_tr)
for e in range(1200):
    pm = torch.randperm(6000, device=dev)
    for i in range(0, 6000, 256):
        b = pm[i:i+256]; f1 = Fn[b]; z = torch.randn_like(f1); t = torch.rand(f1.shape[0], 1, device=dev)
        ft = (1-t)*z+t*f1; l = ((uvel(ft, t)-(f1-z))**2).mean(); ou.zero_grad(); l.backward(); ou.step()
    if (e+1) % 400 == 0: print(f"[dps] {e+1}", flush=True)
def dps_sample(yn, n, steps=60, zeta=0.2):
    f = torch.randn(n, D, device=dev); ts = torch.linspace(0, 1, steps+1, device=dev)
    for i in range(steps):
        t0 = ts[i]; dt = ts[i+1]-ts[i]
        with torch.no_grad(): f = f + dt*uvel(f, t0.repeat(n).unsqueeze(-1))
        f = f.detach().requires_grad_(True); v2 = uvel(f, (t0+dt).repeat(n).unsqueeze(-1)); f1h = f+(1-(t0+dt))*v2
        g = torch.autograd.grad(((f1h[:, sidx]-yn)**2).sum(), f)[0]; f = (f-zeta*g).detach()
    return f
# ---- Direct conditional flow (sensor-conditioned, no kriging reference) ----
encd = DeepSets().to(dev); veld = Vel(D, 64).to(dev); optd = torch.optim.Adam(list(encd.parameters())+list(veld.parameters()), 1e-3)
for e in range(1200):
    pm = torch.randperm(6000, device=dev)
    for i in range(0, 6000, 256):
        b = pm[i:i+256]; f1 = Fn[b]; B = f1.shape[0]; c = encd(xb.expand(B, -1), ysn(ys_tr[b]))
        z = torch.randn_like(f1); t = torch.rand(B, 1, device=dev); ft = (1-t)*z+t*f1
        l = ((veld(ft, t, c)-(f1-z))**2).mean(); optd.zero_grad(); l.backward(); optd.step()
    if (e+1) % 400 == 0: print(f"[dir] {e+1}", flush=True)
@torch.no_grad()
def direct_samples(j, K): return denorm(sample(veld, encd(xb, ysn(ys_te[j:j+1])).expand(K, -1), K))
# ---- sampler wrappers ----
@torch.no_grad()
def ours_samples(j, K):
    c = torch.cat([enc(xb, ysn(ys_te[j:j+1])), nrm(f0_te[j:j+1])], -1).expand(K, -1)
    return f0_te[j:j+1]+(sample(vel, c, K)*rsd+rmu)
@torch.no_grad()
def krig_samples(j, K): return f0_te[j:j+1]+(Lp @ torch.randn(D, K, device=dev)).T
@torch.no_grad()
def det_field(j):
    c = torch.cat([enc3(xb, ysn(ys_te[j:j+1])), nrm(f0_te[j:j+1])], -1)
    return f0_te[j]+det(nrm(f0_te[j:j+1]), c).squeeze(0)
@torch.no_grad()
def kres_samples(j, K):  # kriging posterior samples around the det.-corrected mean (KR+res)
    return det_field(j).unsqueeze(0) + (Lp @ torch.randn(D, K, device=dev)).T
def energy_dist(A, B):  # (n,D),(m,D)
    xy = torch.cdist(A, B).mean(); xx = torch.cdist(A, A).mean(); yy = torch.cdist(B, B).mean(); return (2*xy-xx-yy).item()
# ---- metrics over ambiguous instances ----
amb = torch.where(tpw(ys_te).min(1).values > 0.2)[0]
dvec = (mu1-mu0); dvec = dvec/dvec.norm(); bnd = (mubar*dvec).sum().item()
K = 200; Nm = min(80, len(amb))
METHODS = ["Direct", "DPS", "KR-Flow"]
ed = {k: [] for k in METHODS}; bok = {k: [] for k in METHODS}; cov = {k: [] for k in METHODS}; esc = {k: [] for k in METHODS}; rms = {k: [] for k in METHODS}
Mm = {k: [] for k in METHODS}; SDm = {k: [] for k in METHODS}; Ys = []
sdk_t = torch.sqrt(torch.clamp(torch.diag(pcov), min=1e-8))                       # kriging marginal std (Kriging+res UQ)
pca_sd_t = (F_tr-pca_pred(ys_tr)).std(0)+1e-6                                     # PCA residual std (its UQ)
for jj in amb[:Nm].tolist():
    St = samp_true(ys_te[jj:jj+1], K); So = ours_samples(jj, K)
    Sdps = denorm(dps_sample(ysn(ys_te[jj:jj+1])[0], K)); Sdir = direct_samples(jj, K)
    Ys.append(F_te[jj])
    m0 = tpm(ys_te[jj:jj+1], 0)[0]; m1 = tpm(ys_te[jj:jj+1], 1)[0]; gap = (m0-m1).norm()
    for nm, Sm in [("Direct", Sdir), ("DPS", Sdps), ("KR-Flow", So)]:
        ed[nm].append(energy_dist(Sm, St))
        esc[nm].append(((Sm-F_te[jj].unsqueeze(0)).norm(dim=1).mean() - 0.5*torch.cdist(Sm, Sm).mean()).item())
        rms[nm].append(((Sm.mean(0)-F_te[jj])**2).mean().sqrt().item())
        bok[nm].append(((Sm-F_te[jj].unsqueeze(0))**2).mean(1).min().sqrt().item())
        d0 = (Sm-m0.unsqueeze(0)).norm(dim=1); d1 = (Sm-m1.unsqueeze(0)).norm(dim=1)
        near0 = ((d0 < 0.4*gap) & (d0 <= d1)).float().mean().item(); near1 = ((d1 < 0.4*gap) & (d1 < d0)).float().mean().item()
        cov[nm].append(1.0 if (near0 >= 0.15 and near1 >= 0.15) else 0.0)
        Mm[nm].append(Sm.mean(0)); SDm[nm].append(pca_sd_t if nm == "KLE" else (sdk_t if nm == "Kriging+res" else Sm.std(0)+1e-6))
ED = {k: float(np.mean(v)) for k, v in ed.items()}; BOK = {k: float(np.mean(v)) for k, v in bok.items()}; COV = {k: 100*float(np.mean(v)) for k, v in cov.items()}
ESC = {k: float(np.mean(v)) for k, v in esc.items()}; RMS = {k: float(np.mean(v)) for k, v in rms.items()}
# MPIW@0.95: scale each method's std to 0.95 empirical coverage, report mean interval width
# --- matched-coverage MPIW@0.95: calibrate gamma on a HELD-OUT validation set, apply to test ---
F_va = genBM(400); ys_va = obs(F_va); amb_va = torch.where(tpw(ys_va).min(1).values > 0.2)[0][:Nm]
f0_of = lambda y1: mubar.unsqueeze(0)+(Kxs @ Kss_inv @ (y1-mubar[sidx]).T).T
def msd_val(nm, y1):
    if nm == "DPS":
        E = denorm(dps_sample(ysn(y1)[0], K)); return E.mean(0).detach(), E.std(0).detach()+1e-6
    with torch.no_grad():
        if nm == "KLE": return pca_pred(y1)[0], pca_sd_t
        if nm == "Kriging+res":
            fk = f0_of(y1); c = torch.cat([enc3(xb, ysn(y1)), nrm(fk)], -1); return fk[0]+det(nrm(fk), c).squeeze(0), sdk_t
        if nm == "Direct": E = denorm(sample(veld, encd(xb, ysn(y1)).expand(K, -1), K))
        else: fk = f0_of(y1); c = torch.cat([enc(xb, ysn(y1)), nrm(fk)], -1).expand(K, -1); E = fk+(sample(vel, c, K)*rsd+rmu)
        return E.mean(0), E.std(0)+1e-6
Mval = {k: [] for k in METHODS}; SDval = {k: [] for k in METHODS}; Yval = []
for jj in amb_va.tolist():
    Yval.append(F_va[jj])
    for nm in METHODS:
        mv_, sv_ = msd_val(nm, ys_va[jj:jj+1]); Mval[nm].append(mv_); SDval[nm].append(sv_)
Yvt = torch.stack(Yval)
def gamma_val(Ms, SDs):
    M = torch.stack(Ms); SD = torch.stack(SDs); g = 1.0
    for _ in range(80):
        c = ((Yvt-M).abs() <= 1.959964*g*SD).float().mean().item()
        if abs(c-0.95) < 0.003: break
        g *= 1.05 if c < 0.95 else 0.96
    return g
MPIW = {k: (2*1.959964*gamma_val(Mval[k], SDval[k])*torch.stack(SDm[k])).mean().item() for k in METHODS}
print("==BIMODAL METRICS (mean over", Nm, "ambiguous instances)==", flush=True)
print("RMSE (mean pred)   :", {k: round(v, 3) for k, v in RMS.items()}, flush=True)
print("energy-score (ES)  :", {k: round(v, 3) for k, v in ESC.items()}, flush=True)
print("energy-dist-to-post:", {k: round(v, 3) for k, v in ED.items()}, flush=True)
print("MPIW@0.95          :", {k: round(v, 3) for k, v in MPIW.items()}, flush=True)
print("best-of-K RMSE     :", {k: round(v, 3) for k, v in BOK.items()}, flush=True)
print("both-mode coverage%:", {k: round(v, 1) for k, v in COV.items()}, flush=True)
# ---- bar chart (bimodal metrics): energy distance / best-of-K RMSE / mode coverage ----
order = ["Direct", "DPS", "KR-Flow"]
BARC = {"Direct": "#4a7ab5", "DPS": "#4a7ab5", "KR-Flow": "#b2182b"}
cols = [BARC[m] for m in order]; xpos = np.arange(len(order))
figb, axb = plt.subplots(1, 3, figsize=(13.0, 3.9))
def _barpanel(a, vals, title, logy=False, pct=False):
    bars = a.bar(xpos, [vals[m] for m in order], color=cols, edgecolor="0.3", linewidth=0.6, width=0.72)
    a.set_xticks(xpos); a.set_xticklabels(order, rotation=18, ha="right"); a.set_title(title)
    if logy: a.set_yscale("log")
    top = max(vals[m] for m in order)
    for m, b in zip(order, bars):
        v = vals[m]; lab = f"{v:.0f}" if pct else f"{v:.2f}"
        a.annotate(lab, (b.get_x()+b.get_width()/2, b.get_height()), xytext=(0, 2), textcoords="offset points",
                   ha="center", va="bottom", fontsize=8.5, fontweight=("bold" if m == "KR-Flow" else "normal"))
    a.grid(True, axis="y", alpha=.22, lw=.5); a.set_axisbelow(True)
    if pct: a.set_ylim(0, 112)
_barpanel(axb[0], ED, "Energy distance $\\downarrow$", logy=False)
_barpanel(axb[1], BOK, "Best-of-$K$ RMSE $\\downarrow$")
_barpanel(axb[2], COV, "Mode coverage (%) $\\uparrow$", pct=True)
for k, lb in enumerate(["(a)", "(b)", "(c)"]): panel(axb[k], lb)
figb.tight_layout(); figb.subplots_adjust(wspace=0.28)
figb.savefig(f"{out}/bimodal_metrics_bar.pdf", bbox_inches="tight"); figb.savefig(f"{out}/bimodal_metrics_bar.png", dpi=400, bbox_inches="tight"); plt.close(figb)
print("bimodal bar fig done", flush=True)
# ---- figure ----
j0 = int(amb[(tpw(ys_te[amb])[:, 1]-0.5).abs().argmin()])
xstar_i = int(np.argmax(np.abs((mu1-mu0).cpu().numpy()))); xstar = xnp[xstar_i]
with torch.no_grad():
    St = samp_true(ys_te[j0:j0+1], 800).cpu().numpy(); So = ours_samples(j0, 800).cpu().numpy()
    Sk = krig_samples(j0, 800).cpu().numpy(); dv = float(det_field(j0)[xstar_i].item())
vt, vo, vk = St[:, xstar_i], So[:, xstar_i], Sk[:, xstar_i]
gm = np.linspace(min(vt.min(), vo.min(), vk.min())-0.4, max(vt.max(), vo.max(), vk.max())+0.4, 400)
mu0n, mu1n = mu0.cpu().numpy(), mu1.cpu().numpy(); fd_v = det_field(j0).cpu().numpy(); ft_v = F_te[j0].cpu().numpy()
fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.4))
# (a) KR-Flow 95% band vs kriging (Gaussian) 95% band on one ambiguous instance
om = So.mean(0); olo = np.quantile(So, .025, 0); ohi = np.quantile(So, .975, 0)
sdk = torch.sqrt(torch.clamp(torch.diag(pcov), min=1e-8)).cpu().numpy(); f0j = f0_te[j0].cpu().numpy()
klo, khi = f0j-1.959964*sdk, f0j+1.959964*sdk
ax[0].fill_between(xnp, klo, khi, color=KRIG, alpha=.12, lw=0, label="kriging 95% band")
ax[0].fill_between(xnp, olo, ohi, color=OURS, alpha=.22, lw=0, label="KR-Flow 95% band")
ax[0].plot(xnp, f0j, color=KRIG, ls="--", lw=1.8, label="kriging mean")
ax[0].plot(xnp, om, color=OURS, lw=2.2, label="KR-Flow mean")
ax[0].plot(xnp, ft_v, color=TRUE, lw=2.0, label="true field")
ax[0].scatter(xs.cpu(), ys_te[j0].cpu(), s=40, facecolor=SENS, edgecolor="white", linewidth=.8, zorder=6, label="sensors")
ax[0].set_title("Posterior mean and 95% band"); ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$u(x)$"); panel(ax[0], "(a)")
# (b) double-peak marginal density at x*
ax[1].plot(gm, gaussian_kde(vt)(gm), color=TRUE, lw=2.4, label="true posterior")
ko = gaussian_kde(vo)(gm); ax[1].fill_between(gm, ko, color=OURS, alpha=.20, lw=0); ax[1].plot(gm, ko, color=OURS, lw=2.2, label="KR-Flow (ours)")
ax[1].plot(gm, gaussian_kde(vk)(gm), color=KRIG, lw=2.0, ls="--", label="kriging (unimodal)")
ax[1].set_title(f"Marginal posterior at $x^*={xstar:.2f}$"); ax[1].set_xlabel("Field value $u(x^*)$"); ax[1].set_ylabel("Density"); panel(ax[1], "(b)")
# (c) joint distribution at the two bump locations -> two modes appear as two clusters
i0 = int(np.argmin(np.abs(xnp-0.35))); i1 = int(np.argmin(np.abs(xnp-0.65))); detf = det_field(j0).cpu().numpy()
_rs = np.random.RandomState(0)
def _pick(A, n=420): idx = _rs.permutation(A.shape[0])[:n]; return A[idx, i0], A[idx, i1]
ax[2].scatter(*_pick(St), s=9, color="0.68", alpha=.30, edgecolor="none", rasterized=True, zorder=2, label="true posterior")
ax[2].scatter(*_pick(Sk), s=9, color=KRIG, alpha=.34, edgecolor="none", rasterized=True, zorder=3, label="kriging")
ax[2].scatter(*_pick(So), s=10, color=OURS, alpha=.34, edgecolor="none", rasterized=True, zorder=4, label="KR-Flow")
ax[2].set_xlabel("$u(0.35)$"); ax[2].set_ylabel("$u(0.65)$"); ax[2].set_title("Posterior samples at the two bumps"); panel(ax[2], "(c)")
from matplotlib.lines import Line2D
_lh = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.6", markeredgecolor="none", markersize=6, label="true posterior"),
       Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=KRIG, markeredgecolor="none", markersize=6, label="kriging"),
       Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=OURS, markeredgecolor="none", markersize=6, label="KR-Flow")]
ax[2].legend(handles=_lh, frameon=True, framealpha=.92, edgecolor="0.85", loc="upper right", fontsize=8.0, borderpad=0.5, handletextpad=0.4)
_y0b, _y1b = ax[0].get_ylim(); ax[0].set_ylim(_y0b, _y1b + 0.20*(_y1b-_y0b))
ax[0].legend(loc="upper center", ncol=3, frameon=True, framealpha=.92, edgecolor="0.85", fontsize=8, columnspacing=1.0, handlelength=1.6, handletextpad=0.5, borderpad=0.5)
ax[1].legend(loc="best", frameon=True, framealpha=.90, edgecolor="0.85", fontsize=8.2)
fig.tight_layout(); fig.subplots_adjust(wspace=0.30)
import os as _os; _os.makedirs(f"{out}/weights", exist_ok=True)
torch.save({"enc": enc.state_dict(), "vel": vel.state_dict(), "rmu": rmu.detach().cpu(), "rsd": rsd.detach().cpu(), "mu": mu.detach().cpu(), "sd": sd.detach().cpu()}, f"{out}/weights/bimodal_krflow.pt"); print("[weights] bimodal_krflow.pt saved", flush=True)
fig.savefig(f"{out}/bimodal_semianalytic.pdf", bbox_inches="tight"); fig.savefig(f"{out}/bimodal_semianalytic.png", dpi=400, bbox_inches="tight"); plt.close(fig); print("bimodal adv fig done", flush=True)
