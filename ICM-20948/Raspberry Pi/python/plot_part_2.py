
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


AXIS_TO_IDX = {"x": 0, "y": 1, "z": 2}


def load_collect_csv(csv_path: str | Path, *, axis: str = "x", use_raw: bool = False):
    csv_path = Path(csv_path)
    if axis not in AXIS_TO_IDX:
        raise ValueError(f"axis must be one of {list(AXIS_TO_IDX)}, got: {axis}")
    i = AXIS_TO_IDX[axis]

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")

    t = np.asarray(data["time"], dtype=float)

    if use_raw:
        ax = np.asarray(data["ax_raw"], dtype=float)
        ay = np.asarray(data["ay_raw"], dtype=float)
        az = np.asarray(data["az_raw"], dtype=float)
    else:
        ax = np.asarray(data["ax"], dtype=float)
        ay = np.asarray(data["ay"], dtype=float)
        az = np.asarray(data["az"], dtype=float)

    a = np.vstack([ax, ay, az]).T  # (N,3)
    return t, a[:, i], a


def infer_dt(t: np.ndarray, default: float = 0.01) -> float:
    """Infer a representative dt from timestamps (median of diffs), with fallbacks."""
    if t.size < 2:
        return default
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return default
    dt = float(np.median(diffs))
    # Guard against wild values if time column is malformed
    if not (1e-4 <= dt <= 1.0):
        return default
    return dt


def run_kalman_1d(
    accel_1d: np.ndarray,
    *,
    dt: float,
    accel_clamp: float = 0.1,
    vel_damp_thresh: float = 0.05,
    vel_damp_factor: float = 0.9,
    Q_scale: float = 0.1,
    R_var: float = 1e2,
    P0: float = 1e-2,
):
    """
    Replicates the filter setup/loop from part2_kalman_position_1D.py as closely as possible.
    Returns dict with estimates and gains.
    """
    accel_1d = np.asarray(accel_1d, dtype=float).reshape(-1)
    n = accel_1d.size

    kf = KalmanFilter(dim_x=3, dim_z=1)

    # Constant-acceleration model
    kf.F = np.array([
        [1.0, dt, 0.5 * dt * dt],
        [0.0, 1.0, dt],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    # Measure acceleration
    kf.H = np.array([[0.0, 0.0, 1.0]], dtype=float)

    kf.x = np.zeros(3, dtype=float)
    kf.P = np.eye(3, dtype=float) * P0
    kf.Q = np.eye(3, dtype=float) * Q_scale
    kf.R = np.array([[R_var]], dtype=float)

    p_est, v_est, a_est, K_list = [], [], [], []

    for z in accel_1d:
        # Minimum clamping of acceleration
        z_use = 0.0 if (abs(z) < accel_clamp) else float(z)

        kf.predict()
        kf.update(z_use)

        # Prevent end drift by damping of velocity
        if abs(kf.x[1]) < vel_damp_thresh:
            kf.x[1] = kf.x[1] * vel_damp_factor

        p_est.append(float(kf.x[0]))
        v_est.append(float(kf.x[1]))
        a_est.append(float(kf.x[2]))
        K_list.append(kf.K.copy())

    return {
        "p": np.array(p_est),
        "v": np.array(v_est),
        "a": np.array(a_est),
        "K": np.array(K_list),  # shape (N,3,1)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV output from collect_data.py")
    ap.add_argument("--axis", default="x", choices=["x", "y", "z"])
    ap.add_argument("--use-raw", action="store_true", help="Use ax_raw/ay_raw/az_raw instead of ax/ay/az")
    ap.add_argument("--dt", type=float, default=None, help="Override dt (seconds). Default: infer from time column.")
    ap.add_argument("--no-plots", action="store_true", help="Run filter without showing plots.")
    ap.add_argument("--accel-clamp", type=float, default=0.1)
    ap.add_argument("--vel-damp-thresh", type=float, default=0.05)
    ap.add_argument("--vel-damp-factor", type=float, default=0.9)
    ap.add_argument("--Q", type=float, default=0.1, help="Process noise scale (multiplies I).")
    ap.add_argument("--R", type=float, default=1e2, help="Measurement variance for accel.")
    args = ap.parse_args()

    t, a_axis, a_xyz = load_collect_csv(args.csv, axis=args.axis, use_raw=args.use_raw)
    dt = args.dt if args.dt is not None else infer_dt(t, default=0.01)

    res = run_kalman_1d(
        a_axis,
        dt=dt,
        accel_clamp=args.accel_clamp,
        vel_damp_thresh=args.vel_damp_thresh,
        vel_damp_factor=args.vel_damp_factor,
        Q_scale=args.Q,
        R_var=args.R,
    )

    if args.no_plots:
        # Print a tiny summary so it doesn't look like it did nothing
        print(f"Loaded {len(t)} samples. dt={dt:.6f}s. Final position={res['p'][-1]:.3f} m")
        return

    taxis = t  # already in seconds since start

    # === Figure 1: KF estimates ===
    fig1, axs1 = plt.subplots(3, 1, figsize=(8, 10))
    fig1.suptitle('Kalman Filter Estimates (collect_data.py CSV)')

    axs1[0].plot(taxis, res["p"], label='KF position estimate')
    axs1[0].set_ylabel('Position (m)')
    axs1[0].grid(True)
    axs1[0].legend()

    axs1[1].plot(taxis, res["v"], label='KF velocity estimate')
    axs1[1].set_ylabel('Velocity (m/s)')
    axs1[1].grid(True)
    axs1[1].legend()

    axs1[2].plot(taxis, res["a"], label='KF accel estimate')
    axs1[2].set_ylabel('Acceleration (m/s^2)')
    axs1[2].set_xlabel('Time (s)')
    axs1[2].grid(True)
    axs1[2].legend()

    # === Figure 2: raw accel (all axes) to match original script's quick look ===
    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(taxis, a_xyz, label=['accel x', 'accel y', 'accel z'])
    plt.title("Accelerometer")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.grid(True)
    plt.legend()

    # === Figure 3: Kalman gains ===
    fig3, axs3 = plt.subplots(3, 1, figsize=(8, 10))
    fig3.suptitle('Kalman Gains')

    K = res["K"]  # (N,3,1)
    axs3[0].plot(taxis, K[:, 0, 0], label='Kalman Gain Position')
    axs3[0].set_ylabel('Position Gain')
    axs3[0].grid(True)
    axs3[0].legend()

    axs3[1].plot(taxis, K[:, 1, 0], label='Kalman Gain Velocity')
    axs3[1].set_ylabel('Velocity Gain')
    axs3[1].grid(True)
    axs3[1].legend()

    axs3[2].plot(taxis, K[:, 2, 0], label='Kalman Gain Accel')
    axs3[2].set_ylabel('Accel Gain')
    axs3[2].set_xlabel('Time (s)')
    axs3[2].grid(True)
    axs3[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
