"""Polished Fig 2 (Gaussian), Fig 3 (channel), Fig 4 (bimodal + double-peak KDE)."""
import numpy as np, torch, torch.nn as nn
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 8.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 300})
TRUE = "#222222"; OURS = "#1f77b4"; KRIG = "#ff7f0e"; SENS = "#d62728"; DET = "#2ca02c"
def panel(ax, lab): ax.text(-0.06, 1.06, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
dev = "cuda"; D = 100; out = "./results_pretty"
import os; os.makedirs(out, exist_ok=True); x = torch.linspace(0, 1, D, device=dev)
def sens(ax, sx, sy, lab="sensors"):
    ax.scatter(sx, sy, s=46, facecolor=SENS, edgecolor="white", linewidth=0.9, zorder=6, label=lab)
def shared_legend(fig, axes, ncol=None):
    h, l = [], []
    for a in axes:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=ncol or len(l), frameon=False, bbox_to_anchor=(0.5, -0.01))

class DeepSets(nn.Module):
    def __init__(s, cd=64, h=256):
        super().__init__(); s.phi = nn.Sequential(nn.Linear(2, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cd))
        s.rho = nn.Sequential(nn.Linear(cd, h), nn.SiLU(), nn.Linear(h, cd))
    def forward(s, xs, ys): return s.rho(s.phi(torch.stack([xs, ys], -1)).mean(1))
class Vel(nn.Module):
    def __init__(s, dim, cd, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim + 1 + cd, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                                  nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, ft, t, c): return s.net(torch.cat([ft, t, c], -1))
class DetNet(nn.Module):
    def __init__(s, dim, cd, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(dim + cd, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                                  nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dim))
    def forward(s, fK, c): return s.net(torch.cat([fK, c], -1))
@torch.no_grad()
def sample(vel, c, n, steps=80):
    f = torch.randn(n, D, device=dev); ts = torch.linspace(0, 1, steps + 1, device=dev)
    for i in range(steps):
        t0 = ts[i]; dt = ts[i + 1] - ts[i]; col = lambda tt: tt.repeat(n).unsqueeze(-1)
        k1 = vel(f, col(t0), c); k2 = vel(f + .5 * dt * k1, col(t0 + .5 * dt), c)
        k3 = vel(f + .5 * dt * k2, col(t0 + .5 * dt), c); k4 = vel(f + dt * k3, col(t0 + dt), c)
        f = f + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return f
def se(u, v, ell, var): d = u.reshape(-1, 1) - v.reshape(1, -1); return var * torch.exp(-(d ** 2) / (2 * ell ** 2))
def train_resflow(F_tr, ys_tr, f0_tr, sidx, ep=1000, lam=1e-4):
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6; nrm = lambda f: (f - mu) / sd
    xs = x[sidx]; ysn = lambda y: (y - mu[sidx]) / sd[sidx]; xb = xs.unsqueeze(0)
    R = F_tr - f0_tr; rmu, rsd = R.mean(0), R.std(0) + 1e-6; Rn = (R - rmu) / rsd; f0n = nrm(f0_tr)
    enc = DeepSets().to(dev); vel = Vel(D, 64 + D).to(dev); opt = torch.optim.Adam(list(enc.parameters()) + list(vel.parameters()), 1e-3)
    N = F_tr.shape[0]
    for e in range(ep):
        pm = torch.randperm(N, device=dev)
        for i in range(0, N, 256):
            b = pm[i:i+256]; f1 = Rn[b]; B = f1.shape[0]
            c = torch.cat([enc(xb.expand(B, -1), ysn(ys_tr[b])), f0n[b]], -1)
            z = torch.randn_like(f1); t = torch.rand(B, 1, device=dev); ft = (1-t)*z + t*f1
            l = ((vel(ft, t, c) - (f1-z))**2).mean(); l = l + lam*((ft+(1-t)*vel(ft, t, c))**2).mean()
            opt.zero_grad(); l.backward(); opt.step()
    return enc, vel, rmu, rsd, mu, sd, xb, ysn

# ================= Fig 2: Gaussian =================
torch.manual_seed(0); np.random.seed(0)
def genGP(n, ell=.1, var=.5):
    Kc = se(x, x, ell, var) + 1e-4*torch.eye(D, device=dev); L = torch.linalg.cholesky(Kc)
    return x.unsqueeze(0) + (L @ torch.randn(D, n, device=dev)).T
ns = 3; sidx = torch.linspace(0, D-1, ns, dtype=torch.long, device=dev); xs = x[sidx]
F_tr = genGP(4000); F_cand = genGP(60); ell, var = .1, .5
Kss = se(xs, xs, ell, var) + 1e-6*torch.eye(ns, device=dev); Kxs = se(x, xs, ell, var); Kff = se(x, x, ell, var) + 1e-4*torch.eye(D, device=dev)
km = lambda y: x.unsqueeze(0) + (Kxs @ torch.linalg.solve(Kss, (y - x[sidx]).T)).T
ys_tr = F_tr[:, sidx]; f0_tr = km(ys_tr)
enc, vel, rmu, rsd, mu, sd, xb, ysn = train_resflow(F_tr, ys_tr, f0_tr, sidx)
import os as _os; _os.makedirs(f"{out}/weights", exist_ok=True)
torch.save({"enc": enc.state_dict(), "vel": vel.state_dict(), "rmu": rmu.detach().cpu(), "rsd": rsd.detach().cpu(), "mu": mu.detach().cpu(), "sd": sd.detach().cpu()}, f"{out}/weights/gaussian_krflow.pt"); print("[weights] gaussian_krflow.pt saved", flush=True)
xnp = x.cpu().numpy()
# analytic kriging (GP) posterior std: diag(Kff - Kxs Kss^-1 Kxs^T)
kstd_t = torch.sqrt(torch.clamp(torch.diag(Kff - Kxs @ torch.linalg.solve(Kss, Kxs.T)), min=1e-8))
# choose 3 typical test fields whose true field stays within the 95% band
kmk_c = km(F_cand[:, sidx]); zmax = ((F_cand - kmk_c).abs() / kstd_t).max(1).values
F_te = F_cand[torch.argsort(zmax)[:3]]
ys_te = F_te[:, sidx]; f0_te = km(ys_te); kstd = kstd_t.cpu().numpy()
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
for k in range(3):
    with torch.no_grad():
        c = torch.cat([enc(xb, ysn(ys_te[k:k+1])), (f0_te[k:k+1]-mu)/sd], -1).expand(64, -1)
        fs = f0_te[k:k+1] + (sample(vel, c, 64)*rsd + rmu)
    m = fs.mean(0).cpu().numpy(); s = fs.std(0).cpu().numpy(); kmk = km(ys_te)[k].cpu().numpy()
    ax[k].fill_between(xnp, kmk-1.96*kstd, kmk+1.96*kstd, color=KRIG, alpha=.16, lw=0, label="kriging 95% CI (analytic)")
    ax[k].fill_between(xnp, m-1.96*s, m+1.96*s, color=OURS, alpha=.20, lw=0, label="KR-Flow 95% CI")
    ax[k].plot(xnp, m, color=OURS, lw=2.2, label="KR-Flow mean")
    ax[k].plot(xnp, kmk, "--", color=KRIG, lw=1.8, label="kriging posterior mean")
    ax[k].plot(xnp, F_te[k].cpu(), color=TRUE, lw=1.8, label="true field")
    sens(ax[k], xs.cpu(), ys_te[k].cpu()); ax[k].set_title(f"Test field {k+1}"); ax[k].set_xlabel("$x$"); panel(ax[k], f"({chr(97+k)})")
ax[0].set_ylabel("$u(x)$"); shared_legend(fig, [ax[0]])
fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(f"{out}/gp_posterior.pdf", bbox_inches="tight"); fig.savefig(f"{out}/gp_posterior.png", dpi=300, bbox_inches="tight"); plt.close(fig); print("fig2 done", flush=True)

# ================= Fig 3: channel =================
torch.manual_seed(0); np.random.seed(0)
def genCH(n, w=.02, amp=1.5):
    xi = .25 + .5*torch.rand(n, 1, device=dev); sg = torch.where(torch.rand(n, 1, device=dev) < .5, -1., 1.)
    base = .2*torch.randn(n, 1, device=dev); step = torch.sigmoid((x.unsqueeze(0)-xi)/w); wig = torch.zeros(n, D, device=dev)
    for kk in range(1, 4): wig += (.05/kk)*torch.randn(n, 1, device=dev)*torch.sin(kk*np.pi*x.unsqueeze(0)*2)
    return base + sg*amp*step + wig
ns = 10; sidx = torch.linspace(5, D-6, ns, dtype=torch.long, device=dev); xs = x[sidx]
F_tr = genCH(6000); F_te = genCH(3); ell, var, sn = .1, 1., .02
Kss = se(xs, xs, ell, var) + (sn**2+1e-6)*torch.eye(ns, device=dev); W = se(x, xs, ell, var) @ torch.linalg.inv(Kss)
ys_tr = F_tr[:, sidx] + sn*torch.randn(F_tr.shape[0], ns, device=dev); f0_tr = (W @ ys_tr.T).T
enc, vel, rmu, rsd, mu, sd, xb, ysn = train_resflow(F_tr, ys_tr, f0_tr, sidx)
ys_te = F_te[:, sidx] + sn*torch.randn(3, ns, device=dev); f0_te = (W @ ys_te.T).T
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
for k in range(3):
    with torch.no_grad():
        c = torch.cat([enc(xb, ysn(ys_te[k:k+1])), (f0_te[k:k+1]-mu)/sd], -1).expand(40, -1)
        fs = (f0_te[k:k+1] + (sample(vel, c, 40)*rsd + rmu)).cpu().numpy()
    ftc = F_te[k].cpu().numpy(); f0c = f0_te[k].cpu().numpy()
    for s_ in range(40): ax[k].plot(xnp, fs[s_], color=OURS, lw=.5, alpha=.30)
    ax[k].plot([], [], color=OURS, lw=2, label="KR-Flow samples")
    ax[k].plot(xnp, f0c, "--", color=KRIG, lw=1.9, label="kriging (oversmoothed)")
    ax[k].plot(xnp, ftc, color=TRUE, lw=2.0, label="true field")
    sens(ax[k], xs.cpu(), ys_te[k].cpu()); ax[k].set_title(f"test field {k+1}"); ax[k].set_xlabel("$x$")
ax[0].set_ylabel("$u(x)$"); shared_legend(fig, [ax[0]])
fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(f"{out}/legacy_pretty1d_fig3.png", dpi=150, bbox_inches="tight"); plt.close(fig); print("fig3 done", flush=True)

# ================= Fig 4: bimodal + double-peak KDE =================
torch.manual_seed(0); np.random.seed(0)
klen, kvar, bamp, bw, bn = .08, .25, 2.0, .08, .05
mu0 = bamp*torch.exp(-((x-.35)/bw)**2); mu1 = bamp*torch.exp(-((x-.65)/bw)**2); mubar = .5*(mu0+mu1)
Kf = se(x, x, klen, kvar) + 1e-4*torch.eye(D, device=dev); Lf = torch.linalg.cholesky(Kf)
sx = torch.tensor([.05, .2, .5, .8, .95], device=dev); sidx = torch.stack([(x-v).abs().argmin() for v in sx]); xs = x[sidx]; S = sidx.shape[0]
def genBM(n):
    z = (torch.rand(n, device=dev) < .5).long(); m = torch.where(z.unsqueeze(1) == 0, mu0.unsqueeze(0), mu1.unsqueeze(0))
    return m + (Lf @ torch.randn(D, n, device=dev)).T
F_tr = genBM(6000); F_te = genBM(400); obs = lambda F: F[:, sidx] + bn*torch.randn(F.shape[0], S, device=dev)
ys_tr, ys_te = obs(F_tr), obs(F_te)
Kss = se(xs, xs, klen, kvar) + bn**2*torch.eye(S, device=dev); Kss_inv = torch.linalg.inv(Kss); Kxs = se(x, xs, klen, kvar)
pcov = Kf - Kxs @ Kss_inv @ Kxs.T; Lp = torch.linalg.cholesky(pcov + 1e-5*torch.eye(D, device=dev))
def tpw(y): ll = [-.5*(((y-m[sidx]) @ Kss_inv)*(y-m[sidx])).sum(1) for m in (mu0, mu1)]; return torch.softmax(torch.stack(ll, 1), 1)
def tpm(y, md): m = mu0 if md == 0 else mu1; return m.unsqueeze(0) + (Kxs @ Kss_inv @ (y-m[sidx]).T).T
@torch.no_grad()
def samp_true(y, n):
    w = tpw(y)[0]; zc = (torch.rand(n, device=dev) < w[1]).long(); m0 = tpm(y, 0)[0]; m1 = tpm(y, 1)[0]
    mm = torch.where(zc.unsqueeze(1) == 0, m0.unsqueeze(0), m1.unsqueeze(0)); return mm + (Lp @ torch.randn(D, n, device=dev)).T
f0_tr = mubar.unsqueeze(0) + (Kxs @ Kss_inv @ (ys_tr-mubar[sidx]).T).T
f0_te = mubar.unsqueeze(0) + (Kxs @ Kss_inv @ (ys_te-mubar[sidx]).T).T
# KR-Flow residual + det residual
mu, sd = F_tr.mean(0), F_tr.std(0)+1e-6; nrm = lambda f:(f-mu)/sd; ysn = lambda y:(y-mu[sidx])/sd[sidx]; xb = xs.unsqueeze(0)
R = F_tr - f0_tr; rmu, rsd = R.mean(0), R.std(0)+1e-6; Rn = (R-rmu)/rsd; f0n = nrm(f0_tr)
enc = DeepSets().to(dev); vel = Vel(D, 64+D).to(dev); opt = torch.optim.Adam(list(enc.parameters())+list(vel.parameters()), 1e-3)
for e in range(1200):
    pm = torch.randperm(6000, device=dev)
    for i in range(0, 6000, 256):
        b = pm[i:i+256]; f1 = Rn[b]; B = f1.shape[0]; c = torch.cat([enc(xb.expand(B, -1), ysn(ys_tr[b])), f0n[b]], -1)
        z = torch.randn_like(f1); t = torch.rand(B, 1, device=dev); ft = (1-t)*z+t*f1
        l = ((vel(ft, t, c)-(f1-z))**2).mean()+1e-4*((ft+(1-t)*vel(ft, t, c))**2).mean(); opt.zero_grad(); l.backward(); opt.step()
enc3 = DeepSets().to(dev); det = DetNet(D, 64+D).to(dev); o3 = torch.optim.Adam(list(enc3.parameters())+list(det.parameters()), 1e-3)
for e in range(1200):
    pm = torch.randperm(6000, device=dev)
    for i in range(0, 6000, 256):
        b = pm[i:i+256]; c = torch.cat([enc3(xb.expand(len(b), -1), ysn(ys_tr[b])), f0n[b]], -1)
        l = ((det(f0n[b], c)-(F_tr[b]-f0_tr[b]))**2).mean(); o3.zero_grad(); l.backward(); o3.step()
# ---- Fig 4: bimodal posterior, line plots (a) double-peak density at x*, (b,c) two ambiguous instances ----
amb = torch.where(tpw(ys_te).min(1).values > 0.2)[0]
j0 = int(amb[(tpw(ys_te[amb])[:, 1] - 0.5).abs().argmin()])   # most-ambiguous case
mu0n, mu1n = mu0.cpu().numpy(), mu1.cpu().numpy()
xstar_i = int(np.argmax(np.abs(mu1n-mu0n))); xstar = xnp[xstar_i]
with torch.no_grad():
    S_true = samp_true(ys_te[j0:j0+1], 500).cpu().numpy()
    c = torch.cat([enc(xb, ysn(ys_te[j0:j0+1])), nrm(f0_te[j0:j0+1])], -1).expand(500, -1)
    S_ours = (f0_te[j0:j0+1] + (sample(vel, c, 500)*rsd+rmu)).cpu().numpy()
    cd = torch.cat([enc3(xb, ysn(ys_te[j0:j0+1])), nrm(f0_te[j0:j0+1])], -1)
    m_det0 = (f0_te[j0] + det(nrm(f0_te[j0:j0+1]), cd).squeeze(0)).cpu().numpy()
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))
# (a) double-peak marginal posterior at x*
vt, vo, dv = S_true[:, xstar_i], S_ours[:, xstar_i], float(m_det0[xstar_i])
gm = np.linspace(min(vt.min(), vo.min())-0.4, max(vt.max(), vo.max())+0.4, 400)
ax[0].plot(gm, gaussian_kde(vt)(gm), color=TRUE, lw=2.3, label="true posterior")
ko = gaussian_kde(vo)(gm); ax[0].fill_between(gm, ko, color=OURS, alpha=.20, lw=0); ax[0].plot(gm, ko, color=OURS, lw=2.1, label="KR-Flow (ours)")
ax[0].axvline(dv, color=DET, ls="-.", lw=2.3, label="det. residual")
ax[0].set_title(f"(a) posterior at $x^*={xstar:.2f}$ (double peak)"); ax[0].set_xlabel("field value $u(x^*)$"); ax[0].set_ylabel("density")
ax[0].legend(frameon=True, framealpha=.92, edgecolor="0.85", loc="upper center", fontsize=9)
# (b,c) two ambiguous instances: KR-Flow samples split between the two modes
wte = tpw(ys_te[amb])[:, 1]
ex = [int(amb[(wte > .5).nonzero()[0][0]]), int(amb[(wte < .5).nonzero()[0][0]])]
for ci, j in enumerate(ex):
    a = ax[ci+1]
    with torch.no_grad():
        c = torch.cat([enc(xb, ysn(ys_te[j:j+1])), nrm(f0_te[j:j+1])], -1).expand(40, -1)
        fg = (f0_te[j:j+1] + (sample(vel, c, 40)*rsd+rmu)).cpu().numpy()
        cd = torch.cat([enc3(xb, ysn(ys_te[j:j+1])), nrm(f0_te[j:j+1])], -1)
        fd = (f0_te[j] + det(nrm(f0_te[j:j+1]), cd).squeeze(0)).cpu().numpy()
    for s_ in range(40): a.plot(xnp, fg[s_], color=OURS, lw=.5, alpha=.28)
    a.plot([], [], color=OURS, lw=2, label="KR-Flow samples")
    a.plot(xnp, mu0n, color="0.55", ls="--", lw=1.0); a.plot(xnp, mu1n, color="0.55", ls="--", lw=1.0, label=r"mode means $\mu_0,\mu_1$")
    a.plot(xnp, fd, color=DET, ls="-.", lw=2.0, label="det. residual")
    a.plot(xnp, F_te[j].cpu(), color=TRUE, lw=2.0, label="true field")
    sens(a, xs.cpu(), ys_te[j].cpu()); a.set_title(f"({'bc'[ci]}) ambiguous case {ci+1}"); a.set_xlabel("$x$")
ax[1].set_ylabel("$u(x)$")
shared_legend(fig, [ax[1]])
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(f"{out}/legacy_pretty1d_fig4.png", dpi=150, bbox_inches="tight"); plt.close(fig); print("fig4 done", flush=True)
print("DONE")
