"""
Builds the README charts (images/02 to 05) from the company-level data.
images/01 is the summary infographic and is not generated here.

Usage:
    python src/charts.py --data data/glassdoor_sp500_medians.csv
Fonts: IBM Plex Sans and IBM Plex Mono (falls back to the default font if missing).
"""

import argparse, glob, os
import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.linear_model import LinearRegression

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/glassdoor_sp500_medians.csv")
ap.add_argument("--out", default="images")
args = ap.parse_args()
for f in glob.glob(os.path.expanduser("~/.fonts/IBMPlex*.ttf")) + glob.glob("fonts/IBMPlex*.ttf"):
    fm.fontManager.addfont(f)
BG = "#121614"
PANEL = "#1b1f1e"
TXT = "#e8ebe9"
MUT = "#9aa59f"
GRID = "#2c3230"
TEAL = "#4fc1a6"
BLUE = "#86a3e6"
ORNG = "#e3a062"
WHITE = "#e8ebe9"
mpl.rcParams.update(
    {
        "font.family": "IBM Plex Sans",
        "text.color": TXT,
        "axes.labelcolor": MUT,
        "xtick.color": MUT,
        "ytick.color": MUT,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "axes.edgecolor": GRID,
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
df = pd.read_csv(args.data)
df["Climate"] = df[["WLB", "Culture", "SeniorMgmt", "DI"]].mean(axis=1)


def head(fig, kicker, title, sub):
    fig.text(0.06, 0.95, kicker, family="IBM Plex Mono", color=TEAL, fontsize=12, va="top")
    fig.text(0.06, 0.905, title, fontsize=22, weight="bold", va="top")
    fig.text(0.06, 0.835, sub, fontsize=13.5, color=MUT, va="top", linespacing=1.45)


def foot(fig, t):
    fig.text(0.06, 0.035, t, fontsize=10.5, color=MUT, family="IBM Plex Sans")


def save(fig, n):
    fig.savefig(os.path.join(args.out, n), dpi=200, facecolor=BG)
    plt.close(fig)


os.makedirs(args.out, exist_ok=True)

# A: average ratings
names = {
    "DI": "Diversity & Inclusion",
    "Overall": "Overall rating",
    "Comp": "Comp & Benefits",
    "WLB": "Work/Life Balance",
    "Culture": "Culture & Values",
    "Career": "Career Opportunities",
    "SeniorMgmt": "Senior Management",
}
col = {
    "DI": TEAL,
    "WLB": TEAL,
    "Culture": TEAL,
    "SeniorMgmt": TEAL,
    "Career": BLUE,
    "Comp": ORNG,
    "Overall": WHITE,
}
m = df[list(names)].mean().sort_values()
fig = plt.figure(figsize=(10, 6.2))
head(
    fig,
    "WHAT EMPLOYEES SAY",
    "Employees rate inclusion highest and leadership lowest",
    "Average Glassdoor score (out of 5) across 495 S&P 500 companies. Green bars are the four ratings\ncombined into the Organizational Climate index.",
)
ax = fig.add_axes([0.3, 0.16, 0.62, 0.57])
y = np.arange(len(m))
ax.barh(y, m.values, color=[col[k] for k in m.index], height=0.62)
ax.set_yticks(y, [names[k] for k in m.index], color=TXT, fontsize=13.5)
ax.set_xlim(1, 4.2)
for i, v in enumerate(m.values):
    ax.text(v + 0.04, i, f"{v:.2f}", va="center", fontsize=13, color=TXT, weight="semibold")
ax.xaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(length=0)
ax.set_xlabel("Average rating (stars)")
foot(
    fig,
    "Axis starts at 1 star, the lowest possible Glassdoor rating. Data: Glassdoor, median of monthly values May to Oct 2026.",
)
save(fig, "02_average_ratings.png")

# B: distribution
o = df.Overall
q1, q3 = o.quantile([0.25, 0.75])
fig = plt.figure(figsize=(10, 6.2))
head(
    fig,
    "THE SPREAD",
    "Most companies land between 3.5 and 3.9 stars",
    f"Distribution of Overall Glassdoor ratings for 495 S&P 500 companies. The middle half of companies\nfall in the shaded band ({q1:.2f} to {q3:.2f}). The average is {o.mean():.2f}.",
)
ax = fig.add_axes([0.08, 0.17, 0.86, 0.55])
ax.axvspan(q1, q3, color=TEAL, alpha=0.12, lw=0)
ax.hist(o, bins=np.arange(2.5, 4.8, 0.1), color=BLUE, edgecolor=BG, linewidth=1.5)
ax.set_ylim(0, 78)
ax.axvline(o.mean(), color=WHITE, lw=1.5, ls="--", ymax=0.86)
ax.text(
    o.mean() + 0.015,
    65,
    f"Average {o.mean():.2f}",
    fontsize=12.5,
    color=TXT,
    va="center",
    ha="left",
)
ax.text((q1 + q3) / 2, 73, "Middle 50%", ha="center", fontsize=12, color=TEAL, va="center")
ax.annotate(
    f"Lowest: {df.Overall.min():.2f}",
    (df.Overall.min(), 1.5),
    xytext=(df.Overall.min(), 14),
    ha="center",
    fontsize=11.5,
    color=MUT,
    arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8),
)
ax.annotate(
    f"Highest: {df.Overall.max():.2f}",
    (df.Overall.max(), 3.5),
    xytext=(df.Overall.max() - 0.05, 14),
    ha="center",
    fontsize=11.5,
    color=MUT,
    arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8),
)
ax.set_xlabel("Overall Glassdoor rating (stars)")
ax.set_xlim(2.5, 4.8)
ax.set_ylabel("Number of companies")
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
foot(
    fig,
    "Each bar covers a 0.1-star range. Data: Glassdoor, median of monthly values May to Oct 2026.",
)
save(fig, "03_overall_distribution.png")

# C: why combine
fig = plt.figure(figsize=(12, 6.6))
head(
    fig,
    "WHY COMBINE FOUR RATINGS INTO ONE",
    "Four culture ratings rise and fall together, so the model treats them as one",
    "Left: how closely each pair of culture ratings moves together (1.0 = in lockstep). Right: overlap between inputs\n(VIF). Above 5 means an input mostly repeats the others, which makes the model unstable.",
)
ax = fig.add_axes([0.07, 0.15, 0.36, 0.56])
ks = ["Culture", "SeniorMgmt", "DI", "WLB"]
lab = ["Culture &\nValues", "Senior\nMgmt", "Diversity &\nInclusion", "Work/Life\nBalance"]
C = df[ks].corr().values
cm = mpl.colors.LinearSegmentedColormap.from_list("t", [PANEL, TEAL])
ax.imshow(C, cmap=cm, vmin=0.4, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(
            j,
            i,
            f"{C[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=13.5,
            color=BG if C[i, j] > 0.8 else TXT,
            weight="semibold",
        )
ax.set_xticks(range(4), lab, fontsize=11, color=TXT)
ax.set_yticks(range(4), lab, fontsize=11, color=TXT)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)


def vifs(cols):
    out = []
    for c in cols:
        oth = [x for x in cols if x != c]
        r2 = LinearRegression().fit(df[oth], df[c]).score(df[oth], df[c])
        out.append(1 / (1 - r2))
    return out


b6 = ["Culture", "SeniorMgmt", "DI", "WLB", "Career", "Comp"]
v6 = vifs(b6)
v3 = vifs(["Climate", "Career", "Comp"])
ax2 = fig.add_axes([0.6, 0.17, 0.36, 0.54])
labs = [
    "Culture & Values",
    "Senior Mgmt",
    "Diversity & Incl.",
    "Work/Life",
    "Career Opps",
    "Comp & Benefits",
    "",
    "Org. Climate",
    "Career Opps",
    "Comp & Benefits",
]
vals = v6 + [np.nan] + v3
cols = [TEAL] * 4 + [BLUE, ORNG, BG, TEAL, BLUE, ORNG]
yy = np.arange(len(vals))[::-1]
ax2.barh(yy, np.nan_to_num(vals), color=cols, height=0.65)
ax2.set_yticks(yy, labs, fontsize=11.5, color=TXT)
ax2.tick_params(length=0)
for y_, v in zip(yy, vals):
    if not np.isnan(v):
        ax2.text(v + 0.25, y_, f"{v:.1f}", va="center", fontsize=12, color=TXT, weight="semibold")
ax2.axvline(5, color=MUT, ls="--", lw=1)
ax2.text(5.2, yy[9], "Rule of thumb: 5", fontsize=10.5, color=MUT, va="center")
ax2.text(
    0,
    yy[0] + 0.9,
    "BEFORE: 6 separate ratings",
    fontsize=11,
    family="IBM Plex Mono",
    color=MUT,
    bbox=dict(facecolor=BG, edgecolor="none", pad=1),
    zorder=5,
)
ax2.text(
    0,
    yy[7] + 0.75,
    "AFTER: 3 inputs",
    fontsize=11,
    family="IBM Plex Mono",
    color=MUT,
    bbox=dict(facecolor=BG, edgecolor="none", pad=1),
    zorder=5,
)
ax2.set_xlim(0, 17.5)
ax2.set_ylim(-0.6, yy[0] + 1.4)
ax2.xaxis.grid(True, color=GRID)
ax2.set_axisbelow(True)
ax2.spines["left"].set_visible(False)
ax2.set_xlabel("Variance inflation factor (VIF)")
foot(
    fig,
    f"Max VIF drops from {max(v6):.1f} to {max(v3):.2f} after the four culture ratings are averaged into one Organizational Climate index.",
)
save(fig, "04_why_combine.png")

# D: what a 0.1 star bump is associated with
X = df[["Climate", "Career", "Comp"]]
mr = LinearRegression().fit(X, df.Overall)
eff = mr.coef_ * 0.1
fig = plt.figure(figsize=(10, 5.6))
head(
    fig,
    "IN PLAIN TERMS",
    "Climate moves the needle most",
    "If a company scored 0.1 stars higher on one input, with the other two unchanged, here is how much\nhigher its Overall rating tends to be. This is an association across companies, not a guarantee.",
)
ax = fig.add_axes([0.3, 0.18, 0.6, 0.48])
lb = ["Organizational Climate", "Career Opportunities", "Comp & Benefits"]
yy = [2, 1, 0]
ax.barh(yy, eff, color=[TEAL, BLUE, ORNG], height=0.6)
for y_, v in zip(yy, eff):
    ax.text(
        v + 0.001, y_, f"+{v:.3f} stars", va="center", fontsize=14, color=TXT, weight="semibold"
    )
ax.set_yticks(yy, lb, fontsize=14, color=TXT)
ax.tick_params(length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 0.075)
ax.xaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.set_xlabel("Change in Overall rating (stars)")
foot(
    fig,
    f"Unstandardized OLS coefficients x 0.1 (Climate {mr.coef_[0]:.2f}, Career {mr.coef_[1]:.2f}, Comp {mr.coef_[2]:.2f}). Climate is the average of its four ratings.",
)
save(fig, "05_plain_terms.png")
print("Charts written to", args.out)
