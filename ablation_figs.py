"""Data-efficiency and sensor-sparsity ablation figures (replace tables 5 & 6).
Reads the empirical-covariance multi-seed JSONs and plots mean +/- std bands."""
import json, glob, numpy as np
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, FuncFormatter
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 9,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 300})
def panel(ax, lab): ax.text(-0.10, 1.06, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
def logy(a):
    a.set_yscale("log"); a.yaxis.set_major_formatter(ScalarFormatter())
    a.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5))); a.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.tick_params(axis="y", which="minor", labelsize=7.5)
    a.grid(True, which="both", axis="y", alpha=0.16, lw=0.4); a.grid(True, which="major", alpha=0.28, lw=0.5)
KRIG = "#ff7f0e"; DIRC = "#8c564b"; OURS = "#1f77b4"
base = "."; out = base + "/results_pretty"
import os; os.makedirs(out, exist_ok=True)
mets = [("RMSE", "RMSE $\\downarrow$"), ("ES", "Energy score $\\downarrow$"), ("MPIW", "$\\mathrm{MPIW}_{0.95}$ $\\downarrow$")]

def parse(s):  # "0.126±0.010" -> (mean, std)
    p = s.replace("±", "|").split("|"); return float(p[0]), (float(p[1]) if len(p) > 1 else 0.0)

def load(path):
    d = json.load(open(glob.glob(path)[0])); return d

def series(d, keys, method, metric):  # keys sorted -> arrays of mean, std
    m, s = [], []
    for k in keys:
        mm, ss = parse(d[k][method][metric]); m.append(mm); s.append(ss)
    return np.array(m), np.array(s)

def band(ax, x, m, s, color, ls, marker, lw, ms, label):
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.16, lw=0)
    ax.plot(x, m, ls=ls, marker=marker, color=color, lw=lw, ms=ms, label=label)

# ===================== Data efficiency =====================
d = load(base + "/results_emp_dataeff_mpiw/*.json"); Nk = sorted(d, key=int); N = np.array([int(k) for k in Nk])
fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.7))
for k, (mk, mt) in enumerate(mets):
    m, s = series(d, Nk, "direct", mk); band(ax[k], N, m, s, DIRC, "--", "^", 1.8, 6, "Direct conditional flow")
    m, s = series(d, Nk, "genres", mk); band(ax[k], N, m, s, OURS, "-", "s", 2.2, 7, "KR-Flow (ours)")
    ax[k].set_xscale("log"); ax[k].set_xticks(N); ax[k].set_xticklabels([str(n) for n in N])
    ax[k].set_xlabel("Training fields $N_{\\mathrm{train}}$"); ax[k].set_title(mt); panel(ax[k], f"({chr(97+k)})"); logy(ax[k])
ax[0].legend(loc="upper right", frameon=True, framealpha=.9, edgecolor="0.85")
fig.tight_layout(); fig.subplots_adjust(wspace=0.28)
fig.savefig(f"{out}/dataeff.pdf", bbox_inches="tight"); fig.savefig(f"{out}/dataeff.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("dataeff fig done", flush=True)

# ===================== Sensor sparsity =====================
d = load(base + "/results_emp_sparsity_mpiw/*.json"); Sk = sorted(d, key=int); S = np.array([int(k) for k in Sk])
fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.7))
for k, (mk, mt) in enumerate(mets):
    m, s = series(d, Sk, "kriging", mk); band(ax[k], S, m, s, KRIG, "-", "o", 1.8, 6, "Kriging")
    m, s = series(d, Sk, "direct", mk); band(ax[k], S, m, s, DIRC, "--", "^", 1.8, 6, "Direct conditional flow")
    m, s = series(d, Sk, "genres", mk); band(ax[k], S, m, s, OURS, "-", "s", 2.2, 7, "KR-Flow (ours)")
    ax[k].set_xscale("log"); ax[k].set_xticks(S); ax[k].set_xticklabels([str(s_) for s_ in S])
    ax[k].set_xlabel("Number of sensors $S$"); ax[k].set_title(mt); panel(ax[k], f"({chr(97+k)})"); logy(ax[k])
ax[0].legend(loc="upper right", frameon=True, framealpha=.9, edgecolor="0.85")
fig.tight_layout(); fig.subplots_adjust(wspace=0.28)
fig.savefig(f"{out}/sparsity.pdf", bbox_inches="tight"); fig.savefig(f"{out}/sparsity.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("sparsity fig done", flush=True)
