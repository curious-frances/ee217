#!/usr/bin/env python3
# Reads YOUR csv: time, ax_raw, ay_raw, az_raw, ax, ay, az, gx, gy, gz

import csv
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


def read_imu_csv(csv_path, use_bias_corrected=True):
    """
    Returns:
      t: Nx1 time (seconds)
      a: Nx3 accel (m/s^2)
      g: Nx3 gyro  (deg/s)
    """
    t_list = []
    a_list = []
    g_list = []

    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)

        if use_bias_corrected:
            axk, ayk, azk = "ax", "ay", "az"
        else:
            axk, ayk, azk = "ax_raw", "ay_raw", "az_raw"

        required = {"time", axk, ayk, azk, "gx", "gy", "gz"}
        if not required.issubset(set(r.fieldnames or [])):
            raise ValueError(f"Missing columns. Need {sorted(required)}. Got {r.fieldnames}")

        for row in r:
            t_list.append(float(row["time"]))
            a_list.append([float(row[axk]), float(row[ayk]), float(row[azk])])
            g_list.append([float(row["gx"]), float(row["gy"]), float(row["gz"])])

    t = np.asarray(t_list, dtype=float)
    a = np.asarray(a_list, dtype=float)
    g = np.asarray(g_list, dtype=float)
    return t, a, g


def estimate_dt(time_s, fallback_dt=0.01):
    if time_s is None or len(time_s) < 2:
        return fallback_dt
    d = np.diff(time_s)
    d = d[np.isfinite(d)]
    d = d[d > 0]
    if len(d) == 0:
        return fallback_dt
    return float(np.median(d))


def run_kf_position_from_accel(raw_accel_x, dt,
                               meas_var=1e2,
                               clamp_accel=0.1,
                               vel_damp_threshold=0.05,
                               vel_damp_factor=0.9,
                               q_scale=1.0):
    """
    Kalman state: [position, velocity, acceleration]
    Measurement:  acceleration (x-axis)
    """
    kf = KalmanFilter(dim_x=3, dim_z=1)

    kf.F = np.array([
        [1.0, dt, 0.5 * dt * dt],
        [0.0, 1.0, dt],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    kf.H = np.array([[0.0, 0.0, 1.0]], dtype=float)

    kf.x = np.zeros(3, dtype=float)
    kf.P = np.eye(3, dtype=float) * 1e-2

    kf.Q = np.eye(3, dtype=float) * float(q_scale)
    kf.R = np.array([[float(meas_var)]], dtype=float)

    p_estimates = []
    v_estimates = []
    a_estimates = []
    kalman_gains = []

    for z in raw_accel_x:
        z = float(z)
        z = 0.0 if (abs(z) < clamp_accel) else z

        kf.predict()
        kf.update(z)

        # prevent end drift (same idea as Ege)
        if abs(kf.x[1]) < vel_damp_threshold:
            kf.x[1] *= vel_damp_factor

        p_estimates.append(kf.x[0])
        v_estimates.append(kf.x[1])
        a_estimates.append(kf.x[2])
        kalman_gains.append(kf.K.copy())

    return (np.asarray(p_estimates),
            np.asarray(v_estimates),
            np.asarray(a_estimates),
            np.asarray(kalman_gains))


def main():
    # ====== EDIT THESE ======
    CSV_PATH = "/Users/francesraphael/school/ee217/project/ee217/ICM-20948/Raspberry Pi/python/6ft_medium_round_straight_c.csv"
    USE_BIAS_CORRECTED_ACCEL = True


    MEAS_VAR = 1e2          # measurement noise variance (tune)
    Q_SCALE = 1.0           # process noise scale (tune)
    CLAMP_ACCEL = 0.1       # m/s^2, clamp small accel to 0
    VEL_DAMP_THRESH = 0.05  # m/s
    VEL_DAMP_FACTOR = 0.9
    # ========================

    t_csv, raw_accel, raw_gyro = read_imu_csv(CSV_PATH, use_bias_corrected=USE_BIAS_CORRECTED_ACCEL)

    dt = estimate_dt(t_csv, fallback_dt=0.01)

    num_samples = raw_accel.shape[0]
    taxis = np.arange(num_samples, dtype=float) * dt

    x_accel = raw_accel[:, 0]

    p_estimates, v_estimates, a_estimates, K = run_kf_position_from_accel(
        raw_accel_x=x_accel,
        dt=dt,
        meas_var=MEAS_VAR,
        clamp_accel=CLAMP_ACCEL,
        vel_damp_threshold=VEL_DAMP_THRESH,
        vel_damp_factor=VEL_DAMP_FACTOR,
        q_scale=Q_SCALE,
    )

    # ---------------- PLOTS ----------------

    # Position / Velocity / Accel estimates
    fig1, axs1 = plt.subplots(3, 1, figsize=(8, 10))
    fig1.suptitle("Kalman Filter Estimates (CSV)")

    axs1[0].plot(taxis, p_estimates, label="KF position estimate")
    axs1[0].set_ylabel("Position (m)")
    axs1[0].grid()
    axs1[0].legend()

    axs1[1].plot(taxis, v_estimates, label="KF velocity estimate")
    axs1[1].set_ylabel("Velocity (m/s)")
    axs1[1].grid()
    axs1[1].legend()

    axs1[2].plot(taxis, a_estimates, label="KF accel estimate")
    axs1[2].set_ylabel("Acceleration (m/s^2)")
    axs1[2].set_xlabel("Time (s)")
    axs1[2].grid()
    axs1[2].legend()

    # Raw accel (all axes)
    fig2 = plt.figure()
    plt.title("Raw Acceleration (CSV)")
    plt.plot(taxis, raw_accel[:, 0], label="raw accel x")
    plt.plot(taxis, raw_accel[:, 1], label="raw accel y")
    plt.plot(taxis, raw_accel[:, 2], label="raw accel z")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.grid(True)
    plt.legend()

    # Kalman gains
    fig3, axs3 = plt.subplots(3, 1, figsize=(8, 10))
    fig3.suptitle("Kalman Gains (CSV)")

    axs3[0].plot(taxis, K[:, 0, :], label="Kalman Gain Position")
    axs3[0].set_ylabel("Position Gain")
    axs3[0].grid()

    axs3[1].plot(taxis, K[:, 1, :], label="Kalman Gain Velocity")
    axs3[1].set_ylabel("Velocity Gain")
    axs3[1].grid()

    axs3[2].plot(taxis, K[:, 2, :], label="Kalman Gain Accel")
    axs3[2].set_ylabel("Accel Gain")
    axs3[2].set_xlabel("Time (s)")
    axs3[2].grid()

    # ------------------------------------------------------------
    # (TUNE ME!)
    # ------------------------------------------------------------
    fig4 = plt.figure()

    # Define the parameters (CHANGE THESE FOR YOUR DATASET)
    x_start = 2.1      # seconds: start of forward motion
    x_end = 8.1        # seconds: end of forward motion
    slope = 1.8 / 6.0  # m/s: assumed constant velocity during forward segment

    x_start_b = 10.6   # seconds: start of backward motion (if round-trip)
    x_end_b = 16.6     # seconds: end of backward motion

    y = np.zeros(len(taxis), dtype=float)

    # Helper indices (clamped to valid range)
    def idx(t):
        return int(np.clip(t / dt, 0, len(taxis) - 1))

    i0 = idx(x_start)
    i1 = idx(x_end)
    i2 = idx(x_start_b)
    i3 = idx(x_end_b)

    # Forward segment: ramp up linearly from 0 with slope
    if i1 > i0:
        y[i0:i1] = slope * (taxis[i0:i1] - x_start)

    # Hold segment: keep constant position between forward and backward
    if i2 > i1 and i1 > 0:
        y[i1:i2] = y[i1 - 1]

    # Backward segment: ramp down with -slope, starting from end-of-forward position
    if i3 > i2 and i1 > 0:
        y[i2:i3] = -slope * (taxis[i2:i3] - x_start_b) + y[i1 - 1]

    plt.plot(taxis, y, label="ideal constant velocity path")
    plt.plot(taxis, p_estimates, label="kalman position estimate")
    plt.plot(taxis, np.abs(y - p_estimates), label="error", c="red")

    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("Position estimate error as a function of time")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(True)
    plt.legend()

    # Side-by-side: KF accel vs raw x accel (like Ege)
    fig5, axs5 = plt.subplots(1, 2, figsize=(12, 5))

    axs5[0].plot(taxis, a_estimates, label="KF accel estimate")
    axs5[0].set_title("Kalman Filter Acceleration Estimate")
    axs5[0].set_ylabel("Acceleration (m/s^2)")
    axs5[0].set_xlabel("Time (s)")
    axs5[0].grid()
    axs5[0].legend()

    axs5[1].plot(taxis, x_accel, label="raw x accel")
    axs5[1].set_title("Raw X Acceleration")
    axs5[1].set_xlabel("Time (s)")
    axs5[1].grid()
    axs5[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
