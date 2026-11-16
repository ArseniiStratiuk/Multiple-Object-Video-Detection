"""Shared matplotlib style: fixed categorical order, thin marks, recessive axes."""

import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated light-mode palette); one slot per class id.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 2,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )
