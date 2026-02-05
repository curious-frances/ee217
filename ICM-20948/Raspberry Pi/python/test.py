#!/usr/bin/env python3
"""
Offline 1D IMU simulator + Kalman Filter test (EE217 style)

- Simulates known ground-truth motion (6ft out, stop, 6ft back)
- Adds accel bias, noise, gravity leakage
- Runs KF with state [p, v, b] + ZUPT
- Saves plots + CSVs (NO plt.show)

This is the reference testbed before using real RPi data.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =============================
# Constants
# =============================
FT_TO_M = 0.3048
G = 9.81

OUT_PREFIX = "offline_test"

# =============================
# Utilities
# =============================
def trapz_integrate(x, dt):
    y = np.zeros_like(x)
    for k in range(1, len(x)):
        y[k] = y[k-1] + 0.5 * (x[k-1] + x[k]) * dt
    return y

# =============================
# Ground-truth motion
# =============================
def make_6ft_out_and_back(dt=0.01):
    t = np.arange(0, 14.0, dt)

    Ta, Tc, Ts = 0.6, 2.0, 1.0
    dist = 6 * FT_TO_M

    a = np.zeros_like(t)

    def pulse(tloc, T, A, s):
        return s * A * 0.5 * (1 - np.cos(2*np.pi*tloc/T))

    t0 = 0.0
    for sgn in (+1, -1):
        idx = (t >= t0) & (t < t0+Ta)
        a[idx] = pulse(t[idx]-t0, Ta, 1.0, sgn)
        t0 += Ta + Tc
        idx = (t >= t0) & (t < t0+Ta)
        a[idx] = pulse(t[idx]-t0, Ta, 1.0, -sgn)
        t0 += Ta + Ts

    v = trapz_integrate(a, dt)
    p = trapz_integrate(v, dt)

    scale = dist / np.max(p)
    a *= scale
    v = trapz_integrate(a, dt)
    p = trapz_integrate(v, dt)

    return t, a, v, p

# =============================
# IMU simulation
# =============================
@dataclass
class ImuParams:
    accel_bias: float = 0.12
    accel_noise_std: float = 0.03
    bias_rw_std: float = 0.01
    pitch_deg_std: float = 1.0
    pitch_deg_bias: float = 0.5

def simulate_imu(t, a_true, dt, prm: ImuParams, seed=1):
    rng = np.random.default_rng(seed)

    b = np.zeros_like(t)
    b[0] = prm.accel_bias
    for k in range(1, len(t)):
        b[k] = b[k-1] + prm.bias_rw_std*np.sqrt(dt)*rng.standard_normal()

    pitch = prm.pitch_deg_bias + prm.pitch_deg_std*rng.standard_normal(len(t))
    g_leak = G * np.sin(np.deg2rad(pitch))
    noise = prm.accel_noise_std*rng.standard_normal(len(t))

    a_meas = a_true + b + noise + g_leak
    return a_meas, b, pitch, g_leak

# =============================
# Kalman Filter [p v b] + ZUPT
# =============================
def run_kf(t, a_meas, pitch, dt,
           sigma_a, sigma_bias_rw,
           sigma_v_zupt=0.05,
           zupt_acc=0.08, zupt_vel=0.05):

    n = len(t)
    x = np.zeros((3,1))
    P = np.diag([1,1,0.5])
    I = np.eye(3)

    H = np.array([[0,1,0]])
    R = np.array([[sigma_v_zupt**2]])

    xs = np.zeros((n,3))
    zupt = np.zeros(n)

    for k in range(n):
        g_x = G*np.sin(np.deg2rad(pitch[k]))
        a_lin = a_meas[k] - x[2,0] - g_x

        F = np.array([
            [1, dt, -0.5*dt*dt],
            [0, 1, -dt],
            [0, 0, 1]
        ])
        B = np.array([[0.5*dt*dt], [dt], [0]])

        Q = np.array([
            [0.25*dt**4*sigma_a**2, 0.5*dt**3*sigma_a**2, 0],
            [0.5*dt**3*sigma_a**2,  dt**2*sigma_a**2,  0],
            [0, 0, (sigma_bias_rw**2)*dt]
        ])

        x = F@x + B*(a_meas[k]-g_x)
        P = F@P@F.T + Q

        if abs(a_lin) < zupt_acc and abs(x[1,0]) < zupt_vel:
            z = np.array([[0.0]])
            S = H@P@H.T + R
            K = P@H.T@np.linalg.inv(S)
            x = x + K@(z - H@x)
            P = (I-K@H)@P
            zupt[k] = 1

        xs[k] = x.ravel()

    return xs, zupt

# =============================
# Main
# =============================
def main():
    dt = 0.01
    t, a_true, v_true, p_true = make_6ft_out_and_back(dt)

    prm = ImuParams()
    a_meas, b_true, pitch, g_leak = simulate_imu(t, a_true, dt, prm)

    xs, zupt = run_kf(
        t, a_meas, pitch, dt,
        sigma_a=prm.accel_noise_std,
        sigma_bias_rw=prm.bias_rw_std
    )

    p_est, v_est, b_est = xs.T

    print(f"Truth final position: {p_true[-1]:.3f} m")
    print(f"KF final position:    {p_est[-1]:.3f} m")
    print(f"Final error:          {p_est[-1]-p_true[-1]:.3f} m")

    # =============================
    # Plots (SAVE ONLY)
    # =============================
    def savefig(name):
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{OUT_PREFIX}_{name}.png", dpi=200)
        plt.close()

    plt.figure()
    plt.plot(t, a_true, label="a_true")
    plt.plot(t, a_meas, label="a_meas", alpha=0.6)
    plt.plot(t, g_leak, label="g_leak", alpha=0.7)
    plt.legend(); plt.xlabel("t"); plt.ylabel("m/s^2")
    savefig("accel")

    plt.figure()
    plt.plot(t, p_true, label="p_true")
    plt.plot(t, p_est, label="p_est")
    plt.legend(); plt.xlabel("t"); plt.ylabel("m")
    savefig("position")

    plt.figure()
    plt.plot(t, p_est - p_true)
    plt.xlabel("t"); plt.ylabel("m")
    plt.title("Position Error")
    savefig("position_error")

    plt.figure()
    plt.plot(t, b_true, label="bias true")
    plt.plot(t, b_est, label="bias est")
    plt.legend(); plt.xlabel("t"); plt.ylabel("m/s^2")
    savefig("bias")

    plt.figure()
    plt.plot(t, zupt)
    plt.xlabel("t"); plt.ylabel("flag")
    plt.title("ZUPT active")
    savefig("zupt")

if __name__ == "__main__":
    main()
