"""
Ex. No. 18 - Call Blocking Probability Using the Erlang B Formula
ITA0321 Mobile Computing Laboratory

Aim:
    To calculate the call blocking probability (grade of service) of a trunked
    cellular system using the Erlang B formula, to find the number of channels
    needed for a target blocking probability, and to find how many users a
    given number of channels can support.

Formulas:
    Traffic per user   Au = lambda * H        (calls/hour x hours per call)
    Total traffic      A  = U * Au            (U = number of users)

    Erlang B           B(A, C) = (A^C / C!) / sum_{k=0..C} (A^k / k!)

    The factorials overflow for large C, so the program uses the equivalent
    recursion:  B(A, 0) = 1,   B(A, c) = A*B(A, c-1) / (c + A*B(A, c-1))

Check (Rappaport, Example 3.4): at 0.5 % blocking with 0.1 Erlang per user,
5, 10, 20 and 100 channels support about 11, 39, 110 and 809 users.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def erlang_b(A, C):
    """Blocking probability for offered traffic A (Erlangs) and C channels."""
    B = 1.0
    for c in range(1, C + 1):
        B = A * B / (c + A * B)
    return B


def erlang_b_direct(A, C):
    """Textbook formula, used only to cross-check the recursion for small C."""
    numerator = A ** C / math.factorial(C)
    denominator = sum(A ** k / math.factorial(k) for k in range(C + 1))
    return numerator / denominator


def channels_required(A, target_gos):
    """Smallest number of channels with blocking <= target_gos."""
    C, B = 0, 1.0
    while B > target_gos:
        C += 1
        B = A * B / (C + A * B)
    return C, B


def max_traffic(C, target_gos, tol=1e-6):
    """Largest traffic (Erlangs) that C channels carry at target_gos (bisection)."""
    low, high = 0.0, 1.0
    while erlang_b(high, C) < target_gos:   # widen the bracket
        high *= 2
    while high - low > tol:
        mid = (low + high) / 2
        if erlang_b(mid, C) > target_gos:
            high = mid
        else:
            low = mid
    return low


# ------------------------- Input parameters -------------------------
calls_per_hour = 2           # lambda, calls per user per hour
holding_time_min = 3         # H, average call duration (minutes)
users = 300                  # subscribers in the cell
channels = 15                # channels available in the cell
target_gos = 0.02            # 2 % blocking
channel_list = [5, 10, 20, 100]
gos_for_table = 0.005        # 0.5 % blocking for the capacity table
# --------------------------------------------------------------------

Au = calls_per_hour * holding_time_min / 60      # Erlangs per user
A = users * Au

print("ERLANG B - CALL BLOCKING PROBABILITY")
print("=" * 60)
print(f"Traffic per user : {calls_per_hour} calls/h x {holding_time_min} min = {Au:.3f} E")
print(f"Total traffic    : {users} users x {Au:.3f} E = {A:.2f} E")
print()

B = erlang_b(A, channels)
print(f"1) Blocking with {channels} channels    : B = {B:.4f}  ({B * 100:.2f} %)")
print(f"   Cross-check with direct formula : B = {erlang_b_direct(A, channels):.4f}")

C_needed, B_achieved = channels_required(A, target_gos)
print(f"2) Channels needed for {target_gos * 100:.0f} % GoS    : C = {C_needed}"
      f"  (B = {B_achieved * 100:.2f} %)")

A_max = max_traffic(channels, target_gos)
print(f"3) Max traffic on {channels} channels at {target_gos * 100:.0f} %: A = {A_max:.2f} E"
      f"  -> {int(A_max / Au)} users")

print()
print(f"4) Users supported at {gos_for_table * 100:.1f} % blocking"
      f" (each user = {Au:.2f} E)")
print(f"   {'Channels':>8} | {'Max traffic (E)':>15} | {'Users':>6} | {'Trunking eff.':>13}")
print("   " + "-" * 52)
for C in channel_list:
    a = max_traffic(C, gos_for_table)
    print(f"   {C:>8} | {a:>15.2f} | {int(a / Au):>6} | {a / C * 100:>12.1f}%")

print()
print("Observation: larger trunk groups carry more traffic per channel")
print("(trunking efficiency), so 100 channels serve far more than 20 x 5 channels.")

# ------------------------------ Plot ------------------------------
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "lines.linewidth": 2, "axes.axisbelow": True,
})

traffic = np.logspace(-1, 2, 400)   # 0.1 to 100 Erlangs
fig, ax = plt.subplots(figsize=(9, 5.5))
for C, color in zip([1, 2, 5, 10, 20, 50], COLORS):
    blocking = [erlang_b(a, C) for a in traffic]
    ax.loglog(traffic, blocking, color=color, label=f"C = {C}")

ax.axhline(target_gos, color="#6b6a63", linestyle="--", linewidth=1.2)
ax.text(0.11, target_gos * 1.15, f"GoS = {target_gos * 100:.0f} %", color="#6b6a63", fontsize=9)
ax.set_ylim(1e-3, 1)
ax.set_title("Erlang B: blocking probability vs offered traffic")
ax.set_xlabel("Offered traffic A (Erlangs)")
ax.set_ylabel("Blocking probability B")
ax.legend(title="Channels")
ax.grid(True, which="both", color="#e1e0d9", linewidth=0.6)

plt.tight_layout()
plt.savefig("ex18_output.png", dpi=150)
plt.show()
