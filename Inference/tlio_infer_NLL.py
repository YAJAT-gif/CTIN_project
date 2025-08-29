import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ---------------- User paths ----------------
tlio_seq_dir = Path("../results/137102747096458")     # TLIO sequence folder
ctin_npz     = Path("../results/ctin_results.npz")    # saved by your CTIN inference

# ---------------- Load TLIO trajectory ----------------
# TLIO test script saves: [t, pred_x, pred_y, pred_z, gt_x, gt_y, gt_z]
try:
    arr = np.loadtxt(tlio_seq_dir / "trajectory.txt", delimiter=",")
except ValueError:
    arr = np.loadtxt(tlio_seq_dir / "trajectory.txt", delimiter=",", skiprows=1)

t_tlio = arr[:, 0].astype(np.float64)
P_tlio = arr[:, 1:3].astype(np.float64)   # predicted XY
GT_tlio = arr[:, 4:6].astype(np.float64)  # ground-truth XY

# ---------------- Load CTIN results (robust to optional keys) ----------------
C = np.load(ctin_npz)
def get(key, default=None):
    return C[key] if key in C.files else default

t_ctin  = get("t").astype(np.float64)
pos_gt  = get("pos_gt")                     # [N,3] or [N,2]
pos_pr  = get("pos_pred")
# Optional (if you saved these)
vel_gt  = get("vel_gt")                     # [N, D]
vel_pr  = get("vel_pred")
logstd_v= get("logstd_vel")
sigma_v = get("sigma_v")                    # same as exp(logstd_v)
pos_std = get("pos_std")                    # if you learned σ_pos, [N,2 or 3]
logstd_p= get("logstd_pos")

if pos_gt is None or pos_pr is None:
    raise ValueError("ctin_results.npz must contain 'pos_gt' and 'pos_pred'.")

# Use only XY for RONIN plots
GT_ctin = pos_gt[:, :2].astype(np.float64)
P_ctin  = pos_pr[:, :2].astype(np.float64)

# ---------------- Utilities ----------------
def interp_traj(t_src, P_src, t_tgt):
    out = np.empty((len(t_tgt), P_src.shape[1]), dtype=np.float64)
    for d in range(P_src.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, P_src[:, d])
    return out

def metrics_xy(P, GT):
    e_vec  = P - GT
    e_norm = np.linalg.norm(e_vec, axis=1)
    mean   = float(e_norm.mean())
    p95    = float(np.percentile(e_norm, 95))
    rmse   = float(np.sqrt(np.mean((P - GT) ** 2)))
    emax   = float(e_norm.max())
    return dict(mean=mean, p95=p95, rmse=rmse, max=emax, series=e_norm, e_vec=e_vec)

# ---------------- Interpolate to common (relative) timebase ----------------
t_tlio_rel = t_tlio - t_tlio[0]
t_ctin_rel = t_ctin - t_ctin[0]

P_tlio_i  = interp_traj(t_tlio_rel, P_tlio,  t_ctin_rel)
GT_tlio_i = interp_traj(t_tlio_rel, GT_tlio, t_ctin_rel)

# ---------------- Metrics ----------------
m_ctin = metrics_xy(P_ctin,  GT_ctin)
m_tlio = metrics_xy(P_tlio_i, GT_tlio_i)

print("CTIN  -> mean {:.3f}  p95 {:.3f}  RMSE {:.3f}  max {:.3f}"
      .format(m_ctin['mean'], m_ctin['p95'], m_ctin['rmse'], m_ctin['max']))
print("TLIO  -> mean {:.3f}  p95 {:.3f}  RMSE {:.3f}  max {:.3f}"
      .format(m_tlio['mean'], m_tlio['p95'], m_tlio['rmse'], m_tlio['max']))

# ---------------- Visual alignment (translation only) ----------------
O    = GT_ctin[0]
GT0  = GT_ctin  - O
C0   = P_ctin   - O
T0   = P_tlio_i - O

# ---------------- Colors ----------------
COL_GT   = "#000000"
COL_CTIN = "#E69F00"
COL_TLIO = "#009E73"
COL_CONE = "#D55E00"
COL_SCAT = "#0072B2"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "savefig.dpi": 300
})

def arrow_at_end(ax, P, color):
    if len(P) < 2: return
    a = FancyArrowPatch((P[-2,0], P[-2,1]), (P[-1,0], P[-1,1]),
                        arrowstyle="-|>", mutation_scale=10, lw=0, color=color, zorder=6)
    ax.add_patch(a)

# ---------------- 2×2 Comparison Panel ----------------
fig = plt.figure(figsize=(7.2, 5.4))
fig.suptitle("RoNIN (TLIO_golden) — CTIN vs TLIO", y=0.98, fontsize=14, fontweight="bold")

# (1) CDF of ATE
ax1 = fig.add_subplot(2,2,1)
s_ctin, s_tlio = np.sort(m_ctin['series']), np.sort(m_tlio['series'])
y_ctin = np.linspace(0, 1, len(s_ctin))
y_tlio = np.linspace(0, 1, len(s_tlio))
ax1.plot(s_ctin, y_ctin, color=COL_CTIN, lw=2, label="CTIN")
ax1.plot(s_tlio, y_tlio, color=COL_TLIO, lw=2, ls="--", label="TLIO")
med = float(np.percentile(m_ctin['series'], 50))
ax1.axvline(med, color=COL_CTIN, ls=":", lw=1)
ax1.axvline(m_ctin['p95'], color=COL_CTIN, ls="--", lw=1, alpha=0.9)
ax1.set_xlabel("ATE [m]"); ax1.set_ylabel("Cumulative probability")
ax1.set_title("CDF of ATE"); ax1.grid(True, alpha=0.3)
ax1.legend(frameon=False, loc="lower right")

# (2) Error vs Time
ax2 = fig.add_subplot(2,2,2)
ax2.plot(t_ctin_rel, m_ctin['series'], color=COL_CTIN, lw=2, label="CTIN")
ax2.plot(t_ctin_rel, m_tlio['series'], color=COL_TLIO, lw=2, ls="--", label="TLIO")
ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Error [m]")
ax2.set_title("Error vs Time"); ax2.grid(True, alpha=0.3)
ax2.legend(frameon=False, loc="upper right")

# (3) Grouped metrics
ax3 = fig.add_subplot(2,2,3)
metrics = ["ATE mean", "ATE p95", "RMSE", "Max error"]
vals_ctin = [m_ctin['mean'], m_ctin['p95'], m_ctin['rmse'], m_ctin['max']]
vals_tlio = [m_tlio['mean'], m_tlio['p95'], m_tlio['rmse'], m_tlio['max']]
x = np.arange(len(metrics)); w = 0.38
b1 = ax3.bar(x - w/2, vals_ctin, width=w, color=COL_CTIN, alpha=0.95, label="CTIN")
b2 = ax3.bar(x + w/2, vals_tlio, width=w, color=COL_TLIO, alpha=0.95, label="TLIO")
y_max = max(max(vals_ctin), max(vals_tlio))
for bars in (b1, b2):
    for b in bars:
        v = b.get_height()
        ax3.text(b.get_x()+b.get_width()/2, v + 0.02*y_max, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
ax3.set_xticks(x); ax3.set_xticklabels(metrics)
ax3.set_ylabel("Error [m]")
ax3.set_title("CTIN vs TLIO Metrics")
ax3.grid(True, axis="y", alpha=0.3)
ax3.legend(frameon=False, loc="upper left")

# (4) XY overlay
ax4 = fig.add_subplot(2,2,4)
ax4.plot(GT0[:,0], GT0[:,1], color=COL_GT,   lw=2.2, label="GT")
ax4.plot(C0[:,0],  C0[:,1],  color=COL_CTIN, lw=2.2, ls="--", label="CTIN")
ax4.plot(T0[:,0],  T0[:,1],  color=COL_TLIO, lw=2.2, ls=":",  label="TLIO")
ax4.scatter(GT0[0,0], GT0[0,1], c=COL_GT,   s=28, marker="o", zorder=5)
ax4.scatter(GT0[-1,0],GT0[-1,1],c=COL_GT,   s=38, marker="X", zorder=5)
ax4.scatter(C0[0,0],  C0[0,1],  c=COL_CTIN, s=24, marker="o", zorder=5)
ax4.scatter(C0[-1,0], C0[-1,1], c=COL_CTIN, s=34, marker="X", zorder=5)
ax4.scatter(T0[0,0],  T0[0,1],  c=COL_TLIO, s=24, marker="o", zorder=5)
ax4.scatter(T0[-1,0], T0[-1,1], c=COL_TLIO, s=34, marker="X", zorder=5)
arrow_at_end(ax4, C0, COL_CTIN); arrow_at_end(ax4, T0, COL_TLIO)
ax4.set_aspect("equal", adjustable="box"); ax4.margins(0.03)
ax4.grid(True, alpha=0.25, linewidth=0.8)
ax4.set_xlabel("X [m]"); ax4.set_ylabel("Y [m]")
ax4.set_title("Trajectory Overlay — GT vs CTIN vs TLIO")
ax4.legend(frameon=False, loc="lower left")

fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("ronin_ctin_vs_tlio_panel.svg")
fig.savefig("ronin_ctin_vs_tlio_panel.png", dpi=300)
plt.show()

# =========================
# Optional: Velocity 3σ cone (if vel + σ_v available)
# =========================
if (vel_gt is not None) and (vel_pr is not None) and ((sigma_v is not None) or (logstd_v is not None)):
    print("\n[Info] Found velocity predictions & uncertainties — plotting velocity cone.")
    if sigma_v is None and logstd_v is not None:
        sigma_v = np.exp(logstd_v).astype(np.float64)

    # Align to CTIN timebase if needed
    if vel_gt.shape[0] != len(t_ctin_rel):
        # best-effort: assume vel_gt is already at t_ctin cadence; else skip
        print("[Warn] vel arrays length mismatch; skipping alignment.")
    D = vel_pr.shape[1]
    labs = ["Vx", "Vy", "Vz"][:D]

    vel_err = (vel_pr - vel_gt).astype(np.float64)
    def cone_violation_rate(e, s, k=3.0):
        return 100.0 * np.mean(s < (np.abs(e)/k))


    fig2, axs = plt.subplots(D, 1, figsize=(3.5, 3.5 * D), sharex=True)  # D rows, 1 col
    if D == 1: axs = [axs]
    fig2.suptitle("Velocity Uncertainty vs Error — 3σ Cone", y=0.98, fontsize=12, fontweight="bold")

    for i, ax in enumerate(axs):
        e = vel_err[:, i]
        s = sigma_v[:, i]
        xmax = np.percentile(np.abs(e), 99.5) if e.size else 1.0
        xs = np.linspace(-xmax, xmax, 400)
        ax.plot(xs, np.abs(xs) / 3.0, ls="--", lw=1.2, color=COL_CONE)
        ax.fill_between(xs, 0.0, np.abs(xs) / 3.0, alpha=0.10, color=COL_CONE)
        ax.scatter(e, s, s=6, alpha=0.35, color=COL_SCAT)
        ax.set_title(f"{labs[i]} • outside 3σ: {cone_violation_rate(e, s):.1f}%")
        ax.set_xlabel("Error (m/s)")
        ax.set_ylabel("Predicted σ_v (m/s)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-xmax, xmax);
        ax.set_ylim(bottom=0)

    fig2.tight_layout()
    fig2.savefig("ronin_velocity_cone_vertical.png", dpi=300, bbox_inches="tight")
    plt.show()

else:
    print("\n[Info] No velocity uncertainties found in NPZ — skipping velocity cone.")

# =========================
# Optional: Position 3σ cone (if σ_pos available)
# =========================
if pos_std is not None or logstd_p is not None:
    print("[Info] Found position uncertainties — plotting position cone.")
    if pos_std is None and logstd_p is not None:
        pos_std = np.exp(logstd_p).astype(np.float64)

    e_vec = (P_ctin - GT_ctin).astype(np.float64)  # XY error for RoNIN
    labs = ["X", "Y"]
    def cone_violation_rate(e, s, k=3.0):
        return 100.0 * np.mean(s < (np.abs(e)/k))

    fig3, axs = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    fig3.suptitle("Position Uncertainty vs Error — 3σ Cone (XY)", y=1.02, fontsize=12, fontweight="bold")

    for i, ax in enumerate(axs):
        e = e_vec[:, i]
        s = pos_std[:, i] if pos_std.shape[1] >= 2 else pos_std[:, 0]
        xmax = np.percentile(np.abs(e), 99.5) if e.size else 1.0
        xs = np.linspace(-xmax, xmax, 400)
        ax.plot(xs, np.abs(xs)/3.0, ls="--", lw=1.2, color=COL_CONE)
        ax.fill_between(xs, 0.0, np.abs(xs)/3.0, alpha=0.10, color=COL_CONE)
        ax.scatter(e, s, s=6, alpha=0.35, color=COL_SCAT)
        ax.set_title(f"{labs[i]} • outside 3σ: {cone_violation_rate(e,s):.1f}%")
        ax.set_xlabel("Error (m)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-xmax, xmax); ax.set_ylim(bottom=0)
    axs[0].set_ylabel("Predicted σ_pos (m)")
    fig3.tight_layout()
    fig3.savefig("ronin_position_cone.png", dpi=300)
    plt.show()
else:
    print("[Info] No position uncertainties found in NPZ — skipping position cone.")
