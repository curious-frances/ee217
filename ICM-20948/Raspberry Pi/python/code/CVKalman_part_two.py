#!/Users/s/bin/python3

import sys
import math as m
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

csv_path = "good_data/PERFECT_imu_2026_02_06_07_59_03.csv"
df = pd.read_csv(csv_path)

taxis = df["time"].values
ax_meas = df["ax"].values   # bias-corrected acc
ay_meas = df["ay"].values
az_meas = df["az"].values 

dt = np.mean(np.diff(taxis))
print(f"Estimated dt: {dt:.6f} s")

kf = KalmanFilter(dim_x=3, dim_z=1)

kf.F = np.array([
    [1., dt, 0.5 * dt * dt],
    [0., 1.,        dt],
    [0., 0.,        1.]
])

# Measurement is acceleration only
kf.H = np.array([[0., 0., 1.]])

# Initial state: assume starting at rest
kf.x = np.array([0., 0., 0.])

kf.P = np.diag([1., 1., 1.]) * 0.01

sigma_a = 0.1 # guess

kf.Q = Q_discrete_white_noise(
    dim=3,
    dt=dt,
    var=sigma_a**2
)

# Measurement noise variance (from calibration)
accel_noise_var = 0.00004
kf.R = np.array([[accel_noise_var]])

xs = []
vs = []
accs = []

for ax, ay, az in zip(ax_meas, ay_meas, az_meas):

    ax = 0.0 if (abs(ax) < 0.2) else ax

    kf.predict()

    kf.update(np.array([ax]))

    if abs(kf.x[1]) < 0.3:
            kf.x[1] *= 0.9

    xs.append(kf.x[0])
    vs.append(kf.x[1])
    accs.append(kf.x[2])

xs = np.array(xs)
vs = np.array(vs)
accs = np.array(accs)

plt.figure()
plt.plot(taxis, xs, label="KF position estimate")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Position Estimate from Acceleration (Kalman Filter)")
plt.grid()
plt.legend()

plt.figure()
plt.plot(taxis, vs, label="KF velocity estimate")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Estimate")
plt.grid()
plt.legend()

plt.figure()
plt.plot(taxis, accs, label="KF acceleration estimate")
plt.plot(taxis, ax_meas, alpha=0.5, label="Measured ax")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration")
plt.grid()
plt.legend()

# Calculate distance traveled
distances = np.abs(np.diff(xs))
cumulative_distance = np.concatenate(([0], np.cumsum(distances)))

# Calculate error from origin
error_from_origin = np.abs(xs)
final_error = error_from_origin[-1]

print(f"\nFinal position: {xs[-1]:.4f} m")
print(f"Final error from origin: {final_error:.4f} m")
print(f"Total distance traveled: {cumulative_distance[-1]:.4f} m")

# Error vs Distance plot
plt.figure()
plt.plot(cumulative_distance, error_from_origin, 'r-', linewidth=2, label="Position error")
plt.xlabel("Distance Traveled (m)")
plt.ylabel("Error from Origin (m)")
plt.title("Sensor Error vs Distance Traveled")
plt.grid(True, alpha=0.3)
plt.axhline(y=final_error, color='k', linestyle='--', alpha=0.5, label=f'Final Error: {final_error:.4f} m')
plt.legend()
plt.tight_layout()

plt.show()
