"""
Ex. No. 17 - Frequency (Co-channel) Reuse: Cluster Size, Capacity and S/I
ITA0321 Mobile Computing Laboratory

Aim:
    To find the valid cluster sizes of a hexagonal cellular system, and to
    calculate the co-channel reuse ratio, signal-to-interference ratio (S/I)
    and system capacity for each, so that the smallest cluster size meeting a
    required S/I can be chosen.

Formulas:
    N   = i^2 + i*j + j^2       valid cluster sizes (i, j = 0, 1, 2, ...)
    Q   = D / R = sqrt(3N)      co-channel reuse ratio
    S/I = (sqrt(3N))^n / i0     i0 = 6 first-tier co-channel cells,
                                n  = path-loss exponent
    k   = S / N                 channels per cell (S = total duplex channels)
    C   = (number of cells) * k system capacity (simultaneous calls)

Check (Rappaport, Example 3.2): for a required S/I of 15 dB the program picks
N = 7 when n = 4 and N = 12 when n = 3.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Input parameters -------------------------
total_bandwidth_hz = 33e6      # total spectrum allocated
channel_bw_hz = 25e3           # simplex channel bandwidth
total_cells = 84               # cells in the coverage area
required_si_db = 15.0          # minimum acceptable S/I (dB)
i0 = 6                         # first-tier co-channel interferers
path_loss_exponents = [3, 4]
max_ij = 5                     # search range for i and j
max_cluster = 28               # largest cluster size to list
# --------------------------------------------------------------------

S = int(total_bandwidth_hz / (2 * channel_bw_hz))   # duplex channels (2 simplex each)


def cluster_sizes(limit):
    """Return a sorted list of (N, i, j) for all valid hexagonal cluster sizes."""
    found = {}
    for i in range(limit + 1):
        for j in range(i + 1):          # j <= i avoids repeating (i, j) / (j, i)
            if i == 0 and j == 0:
                continue
            n_cluster = i * i + i * j + j * j
            found.setdefault(n_cluster, (i, j))
    return [(n, ij[0], ij[1]) for n, ij in sorted(found.items())]


def si_ratio_db(n_cluster, n_exp, interferers=6):
    q = math.sqrt(3 * n_cluster)
    return 10 * math.log10(q ** n_exp / interferers)


clusters = [c for c in cluster_sizes(max_ij) if c[0] <= max_cluster]

print("FREQUENCY REUSE - CLUSTER SIZE, CAPACITY AND S/I")
print("=" * 78)
print(f"Total bandwidth     : {total_bandwidth_hz / 1e6:.0f} MHz")
print(f"Duplex channel BW   : 2 x {channel_bw_hz / 1e3:.0f} kHz")
print(f"Total duplex chans  : S = {S}")
print(f"Cells in the area   : {total_cells}")
print(f"Required S/I        : {required_si_db} dB")
print()

header = (f"{'N':>3} {'(i,j)':>6} {'Q=D/R':>7} {'S/I n=3':>9} {'S/I n=4':>9}"
          f" {'k=S/N':>7} {'Capacity':>9}")
print(header)
print("-" * len(header))
for n_cluster, i, j in clusters:
    q = math.sqrt(3 * n_cluster)
    k = S // n_cluster
    capacity = total_cells * k
    print(f"{n_cluster:>3} {f'({i},{j})':>6} {q:>7.2f} {si_ratio_db(n_cluster, 3):>8.2f}dB"
          f" {si_ratio_db(n_cluster, 4):>7.2f}dB {k:>7} {capacity:>9}")

print()
for n_exp in path_loss_exponents:
    best = next(n for n, _, _ in clusters if si_ratio_db(n, n_exp) >= required_si_db)
    print(f"Path-loss exponent n = {n_exp}: smallest N meeting {required_si_db} dB is N = {best}"
          f"  (S/I = {si_ratio_db(best, n_exp):.2f} dB, {S // best} channels/cell,"
          f" capacity = {total_cells * (S // best)} calls)")

print()
print("Observation: a larger cluster gives better S/I but fewer channels per cell,")
print("so capacity falls. The designer picks the smallest N that meets the S/I target.")

# ------------------------------ Plots ------------------------------
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "lines.linewidth": 2, "axes.axisbelow": True,
})

N_values = np.array([c[0] for c in clusters])
positions = np.arange(len(N_values))       # evenly spaced, labelled with N
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

for n_exp, color in zip([2, 3, 4], COLORS):
    si = [si_ratio_db(n, n_exp) for n in N_values]
    ax1.plot(positions, si, marker="o", markersize=6, color=color, label=f"n = {n_exp}")
ax1.axhline(required_si_db, color="#6b6a63", linestyle="--", linewidth=1.2)
ax1.text(positions[-1], required_si_db + 0.6, f"Required {required_si_db:.0f} dB",
         ha="right", color="#6b6a63", fontsize=9)
ax1.set_title("S/I vs cluster size")
ax1.set_xlabel("Cluster size N")
ax1.set_ylabel("S/I (dB)")
ax1.set_xticks(positions)
ax1.set_xticklabels(N_values)
ax1.legend(title="Path-loss exponent")

capacity = [total_cells * (S // n) for n in N_values]
ax2.bar([str(n) for n in N_values], capacity, color=COLORS[0], width=0.7)
ax2.set_title(f"System capacity vs cluster size ({total_cells} cells, S = {S})")
ax2.set_xlabel("Cluster size N")
ax2.set_ylabel("Capacity (simultaneous calls)")
ax2.grid(axis="x", visible=False)

plt.tight_layout()
plt.savefig("ex17_output.png", dpi=150)
plt.show()
