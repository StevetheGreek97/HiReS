"""Generate the HiReS pipeline overview diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

TEAL = "#009688"
TEAL_LIGHT = "#E0F2F1"
ORANGE = "#F57C00"
ORANGE_LIGHT = "#FFF3E0"
BLUE = "#1565C0"
BLUE_LIGHT = "#E3F2FD"
PURPLE = "#6A1B9A"
PURPLE_LIGHT = "#F3E5F5"
GRAY = "#546E7A"
BG = "#FAFAFA"

stages = [
    {"label": "1 · Chunking",      "sub": "slide window\n1024 × 1024 px"},
    {"label": "2 · Inference &\n    Filtering", "sub": "YOLO segmentation\nedge filter"},
    {"label": "3 · Merging\n   (NMS)",         "sub": "coord transform\nduplicate removal"},
    {"label": "4 · Output",         "sub": "CSV + overlays\nmorphometrics"},
]

BOX_W = 1.7
BOX_H = 1.1
GAP   = 0.55
ARROW_W = 0.35

xs = [i * (BOX_W + GAP) for i in range(4)]
BOX_Y = 1.8

commands = [
    {"label": "hires chunk", "x0": xs[0], "x1": xs[0] + BOX_W, "color": TEAL,   "bg": TEAL_LIGHT},
    {"label": "hires run",   "x0": xs[0], "x1": xs[3] + BOX_W, "color": BLUE,   "bg": BLUE_LIGHT},
    {"label": "hires plot",  "x0": xs[3], "x1": xs[3] + BOX_W, "color": PURPLE, "bg": PURPLE_LIGHT},
]

CMD_Y_TOP   = BOX_Y - 0.28
CMD_H       = 0.28
CMD_GAP     = 0.36

fig_w = xs[-1] + BOX_W + 1.4
fig_h = BOX_Y + BOX_H + 0.5

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(-1.1, xs[-1] + BOX_W + 0.15)
ax.set_ylim(0, BOX_Y + BOX_H + 0.35)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# "Image ──►" input arrow
ax.annotate(
    "", xy=(xs[0] - 0.02, BOX_Y + BOX_H / 2),
    xytext=(-0.95, BOX_Y + BOX_H / 2),
    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6),
)
ax.text(-0.97, BOX_Y + BOX_H / 2, "Image", ha="right", va="center",
        fontsize=9, color=GRAY, fontweight="bold")

# Stage boxes and inter-stage arrows
for i, (x, stage) in enumerate(zip(xs, stages)):
    box = FancyBboxPatch(
        (x, BOX_Y), BOX_W, BOX_H,
        boxstyle="round,pad=0.06",
        linewidth=1.4, edgecolor=TEAL, facecolor=TEAL_LIGHT,
    )
    ax.add_patch(box)
    ax.text(x + BOX_W / 2, BOX_Y + BOX_H * 0.63, stage["label"],
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#00695C")
    ax.text(x + BOX_W / 2, BOX_Y + BOX_H * 0.25, stage["sub"],
            ha="center", va="center", fontsize=7.0, color=GRAY, linespacing=1.4)

    if i < len(stages) - 1:
        ax.annotate(
            "", xy=(xs[i + 1] - 0.02, BOX_Y + BOX_H / 2),
            xytext=(x + BOX_W + 0.02, BOX_Y + BOX_H / 2),
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.5),
        )

# Command spans below the boxes
colors = {
    "hires chunk": (TEAL,   TEAL_LIGHT,   "#004D40"),
    "hires run":   (BLUE,   BLUE_LIGHT,   "#0D47A1"),
    "hires plot":  (PURPLE, PURPLE_LIGHT, "#4A148C"),
}

for row, cmd in enumerate(commands):
    y_top = CMD_Y_TOP - row * CMD_GAP
    y_bot = y_top - CMD_H
    y_mid = (y_top + y_bot) / 2

    c_edge, c_face, c_text = colors[cmd["label"]]

    span = FancyBboxPatch(
        (cmd["x0"], y_bot), cmd["x1"] - cmd["x0"], CMD_H,
        boxstyle="round,pad=0.04",
        linewidth=1.2, edgecolor=c_edge, facecolor=c_face, alpha=0.85,
    )
    ax.add_patch(span)

    ax.text((cmd["x0"] + cmd["x1"]) / 2, y_mid, cmd["label"],
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=c_text,
            fontfamily="monospace")

    # vertical connector line from box bottom to span top
    cx = (cmd["x0"] + cmd["x1"]) / 2
    ax.plot([cx, cx], [BOX_Y - 0.01, y_top + 0.01],
            color=c_edge, lw=0.8, linestyle="dotted", alpha=0.6)

plt.tight_layout(pad=0.1)
out = "/home/steve/Documents/Packages/hireseg/docs/assets/pipeline_overview.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.close()
