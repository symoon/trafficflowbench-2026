"""Figures of the final report (docs/report/fig_*.svg). Numbers are copied from README.md (Results)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

QUI = Path(__file__).parent
SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "svg.fonttype": "none",
})


def pulisci(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def classifica():
    """Public leaderboard score of every submission from v3 on (v1 = 0.72215 is off the scale)."""
    pub = {3: .83843, 4: .80818, 5: .78720, 6: .85232, 7: .85128, 8: .85098, 9: .85050, 10: .85449, 11: .85566,
           12: .85606, 13: .85664, 14: .85689, 15: .85607, 16: .85118, 17: .85624, 18: .85998, 19: .86106}
    rif = [3, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19]  # versions that became the new reference
    scartate = [v for v in pub if v not in rif]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(rif, [pub[v] for v in rif], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter(rif, [pub[v] for v in rif], s=42, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3, label="adopted as new reference")
    ax.scatter(scartate, [pub[v] for v in scartate], s=42, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3, label="discarded")
    for v, testo, dy in ((4, "v4 Task 4 KL", 6), (5, "v5 Task 4 mean prior", 6), (7, "v7", -12), (9, "v9", -12),
                         (16, "v16 onset stack", -12), (19, "v19", 6), (10, "v10 Task 2 holes", 8)):
        ax.annotate(testo, (v, pub[v]), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(3, 20))
    ax.set_xticklabels([f"v{v}" for v in range(3, 20)], fontsize=7.5)
    ax.set_ylim(0.78, 0.868)
    ax.set_ylabel("public leaderboard score")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    pulisci(ax)
    fig.tight_layout()
    fig.savefig(QUI / "fig_classifica.svg")


def atteso_osservato():
    """Change expected from local validation (written before submitting) vs change observed on the public leaderboard."""
    t1 = {"v7": (.0012, -.0010), "v10": (.0040, .0035), "v11": (.0012, .0012), "v12": (.0005, .0004),
          "v13": (.0004, .0006), "v14": (.0003, .0003), "v19": (.0008, .0011)}
    t2 = {"v15": (.0020, -.0008), "v16": (.0020, -.0049), "v17": (.0010, .0002), "v18": (.0027, .0037)}
    SPOSTA = {"v13": (-18, 4), "v12": (6, -2), "v14": (-6, -11), "v19": (-4, 6), "v11": (6, -6)}
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot([-.001, .0045], [-.001, .0045], color=INK2, lw=1, zorder=1)
    ax.annotate("observed = expected", (.0040, .0042), fontsize=7.5, color=INK2, ha="right")
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    for d, col, nome in ((t1, BLUE, "Task 1 change"), (t2, ORANGE, "Task 2 change")):
        x, y = zip(*d.values())
        ax.scatter(x, y, s=42, color=col, edgecolor=SURFACE, linewidth=2, zorder=3, label=nome)
        for k, (a, b) in d.items():
            dx, dy = SPOSTA.get(k, (6, -3))
            ax.annotate(k, (a, b), textcoords="offset points", xytext=(dx, dy), fontsize=7.5, color=INK2)
    ax.set_xlim(0, .0045)
    ax.set_ylim(-.0055, .0045)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x * 1000:g}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x * 1000:g}"))
    ax.set_xlabel("expected change of the total score, ×10⁻³ (local validation)")
    ax.set_ylabel("observed change, ×10⁻³ (public LB)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    pulisci(ax)
    fig.tight_layout()
    fig.savefig(QUI / "fig_atteso_osservato.svg")


def buchi():
    """Task 1 flow error inside the reproduced Task 2 holes (3-fold block CV on train)."""
    m = [("LightGBM trained without holes (v6–v9)", 237), ("time/space interpolation (v1)", 95),
         ("LightGBM trained on reproduced holes (v10)", 79), ("+ three extra copies of the holes (v12)", 72.9),
         ("+ average with the E2 network (v14)", 71.6)]
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    y = range(len(m))[::-1]
    ax.barh(list(y), [b for _, b in m], height=0.5, color=BLUE)
    for yi, (_, b) in zip(y, m):
        ax.annotate(f"{b:g}", (b, yi), textcoords="offset points", xytext=(4, -3), fontsize=8, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels([n for n, _ in m], fontsize=8)
    ax.set_xlabel("flow RMSE per lane inside the holes (veh/h)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 265)
    pulisci(ax)
    fig.tight_layout()
    fig.savefig(QUI / "fig_buchi.svg")


if __name__ == "__main__":
    classifica()
    atteso_osservato()
    buchi()
