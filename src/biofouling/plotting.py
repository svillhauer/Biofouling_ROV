"""Shared matplotlib styling for presentation figures."""
import matplotlib.pyplot as plt

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eda100"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": SECONDARY_INK,
        "xtick.color": SECONDARY_INK,
        "ytick.color": SECONDARY_INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
