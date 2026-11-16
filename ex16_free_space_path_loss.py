import numpy as np
import matplotlib.pyplot as plt

C = 3e8  # speed of light (m/s)


def watt_to_dbm(p_watt):
    return 10 * np.log10(p_watt * 1000)


def dbm_to_watt(p_dbm):
    return 10 ** (p_dbm / 10) / 1000


def fspl_db(d_m, f_hz):
    """Free-space path loss in dB (distance in metres, frequency in Hz)."""
    return 20 * np.log10(d_m) + 20 * np.log10(f_hz) + 20 * np.log10(4 * np.pi / C)


def received_power_dbm(pt_dbm, gt_dbi, gr_dbi, d_m, f_hz):
    """Friis equation in dB form."""
    return pt_dbm + gt_dbi + gr_dbi - fspl_db(d_m, f_hz)


def max_range_m(pt_dbm, gt_dbi, gr_dbi, sensitivity_dbm, f_hz):
    """Distance at which the received power falls to the receiver sensitivity."""
    allowed_loss = pt_dbm + gt_dbi + gr_dbi - sensitivity_dbm
    return 10 ** ((allowed_loss - 20 * np.log10(f_hz) - 20 * np.log10(4 * np.pi / C)) / 20)


# ------------------------- Input parameters -------------------------
Pt_watt = 50.0                 # transmitter power (W)
Gt_dbi = 0.0                   # transmit antenna gain (dBi)
Gr_dbi = 0.0                   # receive antenna gain (dBi)
f_main = 900e6                 # carrier frequency for the table (Hz)
sensitivity_dbm = -100.0       # receiver sensitivity (dBm)
frequencies = [900e6, 1800e6, 2400e6, 3500e6]          # for the comparison plot
table_distances = [100, 500, 1000, 2000, 5000, 10000]  # metres
# --------------------------------------------------------------------

Pt_dbm = watt_to_dbm(Pt_watt)
print("FREE-SPACE PATH LOSS AND RECEIVED POWER")
print("=" * 62)
print(f"Transmit power  : {Pt_watt} W = {Pt_dbm:.2f} dBm")
print(f"Antenna gains   : Gt = {Gt_dbi} dBi, Gr = {Gr_dbi} dBi")
print(f"Carrier freq.   : {f_main / 1e6:.0f} MHz  (wavelength = {C / f_main:.3f} m)")
print()

print(f"{'Distance (m)':>12} | {'FSPL (dB)':>10} | {'Pr (dBm)':>10} | {'Pr (W)':>12}")
print("-" * 62)
for d in table_distances:
    loss = fspl_db(d, f_main)
    pr = received_power_dbm(Pt_dbm, Gt_dbi, Gr_dbi, d, f_main)
    print(f"{d:>12} | {loss:>10.2f} | {pr:>10.2f} | {dbm_to_watt(pr):>12.3e}")

print()
print(f"Maximum range for a sensitivity of {sensitivity_dbm} dBm:")
for f in frequencies:
    d_max = max_range_m(Pt_dbm, Gt_dbi, Gr_dbi, sensitivity_dbm, f)
    print(f"   {f / 1e6:>6.0f} MHz -> {d_max / 1000:8.1f} km")

print()
print("Observations:")
print(" * Doubling the distance adds 6 dB of path loss (20 log10 2).")
print(" * Doubling the frequency also adds 6 dB, so higher bands cover less area.")

# ------------------------------ Plots ------------------------------
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "lines.linewidth": 2, "axes.axisbelow": True,
})

d = np.logspace(1, 4, 300)  # 10 m to 10 km
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

for f, color in zip(frequencies, COLORS):
    label = f"{f / 1e6:.0f} MHz"
    ax1.semilogx(d, fspl_db(d, f), color=color, label=label)
    ax2.semilogx(d, received_power_dbm(Pt_dbm, Gt_dbi, Gr_dbi, d, f), color=color, label=label)

ax1.set_title("Free-space path loss vs distance")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Path loss (dB)")
ax1.legend(title="Carrier")

ax2.set_title(f"Received power vs distance (Pt = {Pt_watt:g} W)")
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Received power (dBm)")
ax2.legend(title="Carrier")

plt.tight_layout()
plt.savefig("ex16_output.png", dpi=150)
plt.show()
