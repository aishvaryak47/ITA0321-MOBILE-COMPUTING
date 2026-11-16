"""
Ex. No. 19 - Handoff Simulation with Hysteresis
ITA0321 Mobile Computing Laboratory

Aim:
    To simulate a mobile moving in a straight line from base station BS1 to
    base station BS2, and to study how the hysteresis margin in the handoff
    decision affects the number of handoffs (ping-pong effect), the point at
    which the handoff happens and the signal quality.

Model:
    Received signal (dBm)  RSS(d) = Pt - PL(d) + X
    Log-distance path loss PL(d) = PL(d0) + 10 n log10(d / d0)
    Shadowing X            zero-mean Gaussian (sigma dB), correlated along
                           the route (Gudmundson model):
                           X[k] = rho X[k-1] + sqrt(1 - rho^2) * N(0, sigma),
                           rho = exp(-step / d_corr)

Handoff rule (relative signal strength with hysteresis H):
    Hand off from the serving BS to the other BS only when
        RSS_other > RSS_serving + H
    H = 0 dB gives many unnecessary back-and-forth handoffs ("ping-pong");
    a large H avoids them but delays the handoff, so the mobile stays on a
    weak signal for longer.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Input parameters -------------------------
Pt_dbm = 43.0            # BS transmit power (20 W)
f_hz = 900e6             # carrier frequency
n_exp = 3.5              # path-loss exponent
d0 = 100.0               # reference distance (m)
D = 4000.0               # distance between BS1 and BS2 (m)
step = 10.0              # measurement spacing along the route (m)
sigma_db = 6.0           # shadowing standard deviation (dB)
d_corr = 50.0            # shadowing decorrelation distance (m)
rss_min_dbm = -90.0      # weakest usable signal; below this the call is at risk
hysteresis_demo = [0, 6] # values compared in the single-run plot (dB)
hysteresis_sweep = np.arange(0, 13, 1)   # values for the Monte Carlo study (dB)
monte_carlo_runs = 500
seed = 7
# --------------------------------------------------------------------

PL0 = 20 * np.log10(4 * np.pi * d0 * f_hz / 3e8)   # free-space loss at d0
x = np.arange(d0, D - d0 + step, step)              # mobile positions from BS1 (m)
rng = np.random.default_rng(seed)


def path_loss_db(d):
    return PL0 + 10 * n_exp * np.log10(np.maximum(d, 1.0) / d0)


def correlated_shadowing(num_samples):
    rho = np.exp(-step / d_corr)
    s = np.empty(num_samples)
    s[0] = rng.normal(0, sigma_db)
    for k in range(1, num_samples):
        s[k] = rho * s[k - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sigma_db)
    return s


def rss_profiles():
    """Received power from BS1 and BS2 along the route (one random realisation)."""
    rss1 = Pt_dbm - path_loss_db(x) + correlated_shadowing(len(x))
    rss2 = Pt_dbm - path_loss_db(D - x) + correlated_shadowing(len(x))
    return np.vstack([rss1, rss2])


def simulate_handoff(rss, hyst_db):
    """Apply the hysteresis rule. Returns serving BS index per position and handoff indices."""
    serving = 0                                  # start on BS1
    serving_trace = np.empty(rss.shape[1], dtype=int)
    handoffs = []
    for k in range(rss.shape[1]):
        other = 1 - serving
        if rss[other, k] > rss[serving, k] + hyst_db:
            serving = other
            handoffs.append(k)
        serving_trace[k] = serving
    return serving_trace, handoffs


# -------------------- Part 1: one mobile journey --------------------
rss = rss_profiles()
print("HANDOFF SIMULATION WITH HYSTERESIS")
print("=" * 66)
print(f"BS separation {D / 1000:.1f} km, path-loss exponent {n_exp},"
      f" shadowing sigma {sigma_db} dB")
print()
print("Part 1 - single journey from BS1 to BS2")
traces = {}
for H in hysteresis_demo:
    trace, ho = simulate_handoff(rss, H)
    traces[H] = trace
    positions = ", ".join(f"{x[k]:.0f}" for k in ho)
    print(f"  H = {H:>2} dB : {len(ho):>2} handoff(s) at x = {positions} m")

# ------------------- Part 2: Monte Carlo study ----------------------
# Every hysteresis value is tested on the same set of random journeys.
num_h = len(hysteresis_sweep)
counts = np.zeros((num_h, monte_carlo_runs))
first_pos = np.full((num_h, monte_carlo_runs), np.nan)
weak = np.zeros((num_h, monte_carlo_runs), dtype=bool)
for run in range(monte_carlo_runs):
    r = rss_profiles()
    for h_idx, H in enumerate(hysteresis_sweep):
        trace, ho = simulate_handoff(r, H)
        counts[h_idx, run] = len(ho)
        if ho:
            first_pos[h_idx, run] = x[ho[0]]
        serving_rss = r[trace, np.arange(len(x))]
        weak[h_idx, run] = np.any(serving_rss < rss_min_dbm)

avg_handoffs = counts.mean(axis=1)
avg_first_pos = np.nanmean(first_pos, axis=1)
pct_weak = 100 * weak.mean(axis=1)

print()
print(f"Part 2 - average over {monte_carlo_runs} journeys (ideal = 1 handoff near"
      f" {D / 2:.0f} m)")
print(f"  {'H (dB)':>6} | {'Avg handoffs':>12} | {'Ping-pong':>9} |"
      f" {'1st handoff (m)':>15} | {'Runs < ' + str(int(rss_min_dbm)) + ' dBm':>15}")
print("  " + "-" * 70)
for H, n_ho, pos, weak in zip(hysteresis_sweep, avg_handoffs, avg_first_pos, pct_weak):
    print(f"  {H:>6} | {n_ho:>12.2f} | {n_ho - 1:>9.2f} | {pos:>15.0f} | {weak:>14.1f}%")

print()
print("Observation: increasing the hysteresis removes ping-pong handoffs, but the")
print("handoff happens later along the route, so the mobile stays longer on a")
print("weakening signal and more journeys dip below the usable level. A margin of")
print("a few dB balances the two.")

# ------------------------------ Plots ------------------------------
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a63"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "lines.linewidth": 2, "axes.axisbelow": True,
})

fig = plt.figure(figsize=(12, 10))
grid = fig.add_gridspec(3, 2, height_ratios=[2.2, 0.9, 1.8])

ax_rss = fig.add_subplot(grid[0, :])
ax_rss.plot(x, rss[0], color=BLUE, label="BS1")
ax_rss.plot(x, rss[1], color=ORANGE, label="BS2")
ax_rss.axvline(D / 2, color=GRAY, linestyle="--", linewidth=1.2)
ax_rss.text(D / 2 + 30, rss.max() - 2, "Cell boundary", color=GRAY, fontsize=9)
ax_rss.set_title("Received signal strength along the route (one journey)")
ax_rss.set_xlabel("Distance from BS1 (m)")
ax_rss.set_ylabel("RSS (dBm)")
ax_rss.legend()

ax_srv = fig.add_subplot(grid[1, :], sharex=ax_rss)
for row, H in enumerate(hysteresis_demo):
    trace = traces[H]
    for bs, color in [(0, BLUE), (1, ORANGE)]:
        segments, start = [], None
        for k, s in enumerate(trace):
            if s == bs and start is None:
                start = k
            if (s != bs or k == len(trace) - 1) and start is not None:
                end = k if s != bs else k + 1
                segments.append((x[start] - step / 2, (end - start) * step))
                start = None
        ax_srv.broken_barh(segments, (row - 0.35, 0.7), facecolors=color)
ax_srv.set_yticks(range(len(hysteresis_demo)))
ax_srv.set_yticklabels([f"H = {H} dB" for H in hysteresis_demo])
ax_srv.set_title("Serving base station (blue = BS1, orange = BS2)")
ax_srv.set_xlabel("Distance from BS1 (m)")
ax_srv.grid(False)

ax_ho = fig.add_subplot(grid[2, 0])
ax_ho.plot(hysteresis_sweep, avg_handoffs, marker="o", markersize=6, color=BLUE)
ax_ho.axhline(1, color=GRAY, linestyle="--", linewidth=1.2)
ax_ho.set_title("Average handoffs per journey")
ax_ho.set_xlabel("Hysteresis H (dB)")
ax_ho.set_ylabel("Handoffs")

ax_pos = fig.add_subplot(grid[2, 1])
ax_pos.plot(hysteresis_sweep, avg_first_pos, marker="o", markersize=6, color=AQUA)
ax_pos.axhline(D / 2, color=GRAY, linestyle="--", linewidth=1.2)
ax_pos.set_title("Average position of first handoff")
ax_pos.set_xlabel("Hysteresis H (dB)")
ax_pos.set_ylabel("Distance from BS1 (m)")

plt.tight_layout()
plt.savefig("ex19_output.png", dpi=150)
plt.show()
