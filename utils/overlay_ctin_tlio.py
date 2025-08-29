import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# paths
tlio_seq_dir = Path("../results/145820422949970")  # pick your TLIO sequence folder
ctin_npz = Path("../results/ctin_results.npz")

# ---- load TLIO trajectory.txt (format: t, x_gt, y_gt, z_gt, x_pred, y_pred, z_pred) ----
arr = np.loadtxt(tlio_seq_dir / "trajectory.txt", delimiter=",")
t = arr[:, 0]
P = arr[:, 1:3]  # predicted XY
GT = arr[:, 4:6]  # XY

# ---- load CTIN npz ----
ctin = np.load(ctin_npz)
t_ctin  = ctin["t"]
GT_ctin = ctin["pos_gt"]
P_ctin  = ctin["pos_pred"]

# ---- interpolate TLIO to CTIN time base (so curves line up visually) ----
def interp_traj(t_src, P_src, t_tgt):
    out = np.empty((len(t_tgt), P_src.shape[1]))
    for d in range(P_src.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, P_src[:, d])
    return out

P_tlio_i  = interp_traj(t, P,  t_ctin)
GT_tlio_i = interp_traj(t, GT, t_ctin)

# ---- translation-only alignment for visualization (don’t use for metrics) ----
O    = GT_ctin[0]
GT0  = GT_ctin - O
C0   = P_ctin  - O
T0   = P_tlio_i - O

# ---- XY overlay ----
# plt.figure(figsize=(5.2,5.2))
# plt.plot(GT0[:,0], GT0[:,1], label="GT", linewidth=2)
# plt.plot(C0[:,0],  C0[:,1]  , label="CTIN", linewidth=2)
# plt.plot(T0[:,0],  T0[:,1],  label="TLIO", linewidth=2)
# plt.axis("equal"); plt.xlabel("X [m]"); plt.ylabel("Y [m]")
# plt.title("Trajectory Overlay — GT vs CTIN vs TLIO")
# plt.grid(True); plt.legend()
# plt.tight_layout(); plt.savefig("traj_overlay_ctin_vs_tlio.png", dpi=300)
# plt.show()
# print("Saved overlay -> traj_overlay_ctin_vs_tlio.png")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# GT0, C0, T0 are your aligned XY arrays (N×2) for GT, CTIN, TLIO
# Colors (Okabe–Ito)
COL_GT   = "#000000"  # black
COL_CTIN = "#E69F00"  # orange
COL_TLIO = "#009E73"  # bluish-green



plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 12,
})

fig, ax = plt.subplots(figsize=(5.6, 5.6))
ax.plot(GT0[:,0],   GT0[:,1],   color=COL_GT,   lw=2.6, label="GT")
ax.plot(C0[:,0],    C0[:,1],    color=COL_CTIN, lw=2.6, ls="--", label="CTIN")
ax.plot(T0[:,0],    T0[:,1],    color=COL_TLIO, lw=2.6, ls="--", label="TLIO")



# Start / End markers
ax.scatter(GT0[0,0], GT0[0,1], c=COL_GT,   s=36, marker="o", zorder=5)
ax.scatter(GT0[-1,0],GT0[-1,1],c=COL_GT,   s=50, marker="X", zorder=5)
ax.scatter(C0[0,0],  C0[0,1],  c=COL_CTIN, s=30, marker="o", zorder=5)
ax.scatter(C0[-1,0], C0[-1,1], c=COL_CTIN, s=45, marker="X", zorder=5)
ax.scatter(T0[0,0],  T0[0,1],  c=COL_TLIO, s=30, marker="o", zorder=5)
ax.scatter(T0[-1,0], T0[-1,1], c=COL_TLIO, s=45, marker="X", zorder=5)

# Small arrows to show direction
def arrow_at_end(P, color):
    a = FancyArrowPatch(
        posA=(P[-2,0], P[-2,1]), posB=(P[-1,0], P[-1,1]),
        arrowstyle="-|>", mutation_scale=12, lw=0, color=color, zorder=6)
    ax.add_patch(a)
arrow_at_end(GT0, COL_GT); arrow_at_end(C0, COL_CTIN); arrow_at_end(T0, COL_TLIO)


ax.set_aspect("equal", adjustable="box")
ax.margins(0.03)
ax.grid(True, alpha=0.25, linewidth=0.8)
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_title("GT vs CTIN vs TLIO (Unseen)")

# Legend outside top-center
ax.legend(
    loc="lower left",
    fontsize=8,
    framealpha=0.6         # transparency
)


fig.tight_layout()
fig.savefig("traj_overlay_ctin_vs_tlio.svg")   # best for poster printing
fig.savefig("traj_overlay_ctin_vs_tlio.png", dpi=200)
plt.show()
