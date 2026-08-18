# ST21_final_comparison.py
# Pulls ALL results together: one visual + one printed table.
# Uses the measured values from your experiments (edit them here if you re-run).

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")

# If graphs don't pop up, uncomment these two lines:
# import matplotlib
# matplotlib.use("TkAgg")


# ======================================================================
# ALL YOUR MEASURED RESULTS (from your actual runs)
# Edit any number here if you re-run an experiment.
# ======================================================================

results = {
    # method name:        [energy, sla, completed]
    "Random":            [62.28, 18.6, 165.6],   # from ST7/ST12
    "Round Robin":       [63.05,  3.2, 181.0],   # from ST12
    "Q-learning (Shaw)": [56.50, 56.0, 125.4],   # from ST20
    "SARSA (Shaw)":      [57.96, 59.0, 124.4],   # from ST20
    "DQN (no predict)":  [54.56, 46.6, 133.4],   # from ST19 (without prediction)
    "DQN + Prediction":  [47.43, 10.8, 169.8],   # from ST19 (YOUR best result)
}

# order from "dumb" to "sophisticated" for a clean progression
order = ["Random", "Round Robin", "Q-learning (Shaw)", "SARSA (Shaw)",
         "DQN (no predict)", "DQN + Prediction"]

# colours: greys/blues for baselines, green for your best
colors = {
    "Random":            "#B0AEA6",
    "Round Robin":       "#9AA7B4",
    "Q-learning (Shaw)": "#D8956A",
    "SARSA (Shaw)":      "#C97B45",
    "DQN (no predict)":  "#85B7EB",
    "DQN + Prediction":  "#1D9E75",   # your winner = green
}


# ======================================================================
# PRINT A CLEAN COMPARISON TABLE
# ======================================================================

print("=" * 68)
print(f"{'Method':<22} {'Energy(Wh)':>11} {'SLA':>8} {'Completed':>11}")
print("-" * 68)
for name in order:
    e, s, c = results[name]
    marker = "  <-- best" if name == "DQN + Prediction" else ""
    print(f"{name:<22} {e:>11.2f} {s:>8.1f} {c:>11.1f}{marker}")
print("=" * 68)

# quick honest summary vs Shaw's methods
best = results["DQN + Prediction"]
qlearn = results["Q-learning (Shaw)"]
print(f"""
Your DQN+Prediction vs Q-learning (Shaw's method):
  Energy:    {best[0]:.1f} vs {qlearn[0]:.1f} Wh  ->  {100*(qlearn[0]-best[0])/qlearn[0]:.0f}% less energy
  SLA:       {best[1]:.1f} vs {qlearn[1]:.1f}     ->  {100*(qlearn[1]-best[1])/qlearn[1]:.0f}% fewer violations
  Completed: {best[2]:.1f} vs {qlearn[2]:.1f}     ->  {100*(best[2]-qlearn[2])/qlearn[2]:.0f}% more jobs done
""")


# ======================================================================
# VISUAL: three bar charts side by side (energy, SLA, completed)
# ======================================================================

names = order
energy = [results[n][0] for n in names]
sla    = [results[n][1] for n in names]
completed = [results[n][2] for n in names]
bar_colors = [colors[n] for n in names]

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

# short labels for the x-axis (so they fit)
short = ["Random", "Round\nRobin", "Q-learn\n(Shaw)", "SARSA\n(Shaw)",
         "DQN\n(no pred)", "DQN+\nPredict"]

def draw(ax, values, title, ylabel, lower_better):
    bars = ax.bar(short, values, color=bar_colors, edgecolor="black", linewidth=0.5)
    # highlight the best with a thick border
    best_idx = int(np.argmin(values)) if lower_better else int(np.argmax(values))
    bars[best_idx].set_linewidth(2.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=8)
    # numbers on top of bars
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

draw(axes[0], energy, "Energy Consumption (lower = better)", "Energy (Wh)", True)
draw(axes[1], sla, "SLA Violations (lower = better)", "Violations", True)
draw(axes[2], completed, "Jobs Completed (higher = better)", "Completed", False)

fig.suptitle("Full Comparison: baselines vs Shaw's methods vs your DQN+Prediction",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# ======================================================================
# BONUS: a single "trade-off" scatter (energy vs SLA) — very thesis-friendly
# ======================================================================

fig2, ax = plt.subplots(figsize=(9, 6))
for name in order:
    e, s, c = results[name]
    ax.scatter(e, s, s=200, color=colors[name], edgecolor="black",
               zorder=3, label=name)
    ax.annotate(name, (e, s), textcoords="offset points", xytext=(8, 5), fontsize=9)

ax.set_xlabel("Energy (Wh)  — lower is better →", fontsize=11)
ax.set_ylabel("SLA Violations — lower is better →", fontsize=11)
ax.set_title("Energy vs SLA Trade-off\n(bottom-left = best: low energy AND low violations)",
             fontsize=12)
ax.grid(True, alpha=0.3)
# arrow pointing to the ideal corner
ax.annotate("BEST\ncorner", xy=(min(energy), min(sla)),
            xytext=(min(energy)+3, min(sla)+8),
            fontsize=10, color="#1D9E75", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1D9E75"))
plt.tight_layout()
plt.show()

print("Two figures shown: (1) three-metric bar comparison, (2) energy-vs-SLA trade-off.")
print("The trade-off scatter is especially good for your thesis — it shows your")
print("method sits closest to the ideal bottom-left corner.")
