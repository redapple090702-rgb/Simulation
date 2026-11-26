"""
Radical-pair magnetic-field toy simulation
------------------------------------------
A pedagogical *effective kinetic model*, not a full quantum-spin simulation.

States:
  S(t): singlet radical-pair population
  T(t): triplet radical-pair population
  P(t): accumulated triplet-channel product
  R(t): accumulated singlet recombination product

ODEs:
 dS/dt = -(kS + kST(B))*S + kTS(B)*T
 dT/dt =  kST(B)*S - (kT + kTS(B))*T
 dP/dt = kT*T
 dR/dt = kS*S

The field-dependent mixing rate below is a deliberately simple illustrative
function. Its parameters are NOT fitted biological constants and the model
does not predict a real organism's response to Earth's magnetic field.
"""
import numpy as np
import matplotlib.pyplot as plt

# Illustrative parameters (time unit: microsecond, rate: 1/microsecond)
kS = 0.35
kT = 0.12
k0 = 0.04
kB = 0.20
B_half = 50.0  # microtesla; illustrative scale only

def mixing_rate(B):
    """Effective singlet-triplet mixing rate for a chosen field magnitude."""
    return k0 + kB * (B**2 / (B**2 + B_half**2))

def simulate(B, t_end=30.0, dt=0.02):
    """Euler integration; returns time and S,T,P,R arrays."""
    t = np.arange(0, t_end + dt, dt)
    S = np.zeros_like(t); T = np.zeros_like(t)
    P = np.zeros_like(t); R = np.zeros_like(t)
    S[0] = 1.0  # start with a singlet radical pair
    km = mixing_rate(B)
    for i in range(len(t)-1):
        dS = -(kS + km)*S[i] + km*T[i]
        dT = km*S[i] - (kT + km)*T[i]
        dP = kT*T[i]
        dR = kS*S[i]
        S[i+1] = max(0, S[i] + dt*dS)
        T[i+1] = max(0, T[i] + dt*dT)
        P[i+1] = P[i] + dt*dP
        R[i+1] = R[i] + dt*dR
    return t, S, T, P, R

def main():
    fields = [0, 25, 50, 100]  # microtesla; comparison settings
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for B in fields:
        t,S,T,P,R = simulate(B)
        axes[0].plot(t, S, label=f"S, {B} µT")
        axes[0].plot(t, T, linestyle="--", label=f"T, {B} µT")
        axes[1].plot(t, P, label=f"{B} µT")
    axes[0].set_title("Singlet / triplet populations")
    axes[0].set_xlabel("Time (µs)"); axes[0].set_ylabel("Relative population")
    axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=.25)
    axes[1].set_title("Accumulated triplet-channel product")
    axes[1].set_xlabel("Time (µs)"); axes[1].set_ylabel("Relative product")
    axes[1].legend(title="Field"); axes[1].grid(alpha=.25)
    fig.suptitle("Illustrative radical-pair kinetic model (not experimental data)")
    fig.tight_layout()
    plt.show()

    print("Field (µT) | final triplet product | final singlet recombination")
    for B in fields:
        t,S,T,P,R = simulate(B)
        print(f"{B:>9} | {P[-1]:>21.4f} | {R[-1]:>27.4f}")

if __name__ == "__main__":
    main()
