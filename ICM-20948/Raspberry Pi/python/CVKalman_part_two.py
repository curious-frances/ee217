#!/Users/s/bin/python3

import sys
import math as m
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# ------------------------------------------------------------
# Load IMU data (bias already removed)
# ------------------------------------------------------------
csv_path = "6ft_medium_round_straight_c.csv"
df = pd.read_csv(csv_path)

taxis = df["time"].values
ax_meas = df["ax"].values   # bias-corrected acceleration (m/s^2)
ay_meas = df["ay"].values
az_meas = df["az"].values 

# Compute dt from timestamps (assumes nearly uniform sampling)
dt = np.mean(np.diff(taxis))
print(f"Estimated dt: {dt:.6f} s")

# ------------------------------------------------------------
# Kalman Filter setup
# ------------------------------------------------------------
kf = KalmanFilter(dim_x=3, dim_z=1)

kf.F = np.array([
    [1., dt, 0.5 * dt * dt],
    [0., 1.,        dt],
    [0., 0.,        1.]
])

# We measure acceleration only
kf.H = np.array([[0., 0., 1.]])

# Initial state: assume starting at rest
kf.x = np.array([0., 0., 0.])

kf.P = np.diag([1., 1., 1.]) * 0.01

# ------------------------------------------------------------
# Process & measurement noise
# ------------------------------------------------------------
# How much we trust the motion model (tune this!)
sigma_a = 0.1

kf.Q = Q_discrete_white_noise(
    dim=3,
    dt=dt,
    var=sigma_a**2
)
# kf.Q = np.eye(3, dtype=float)
# Measurement noise variance (from IMU characterization)
accel_noise_var = 0.02 # 0.02
kf.R = np.array([[accel_noise_var]])

# ------------------------------------------------------------
# Run filter
# ------------------------------------------------------------
xs = []
vs = []
accs = []

for ax, ay, az in zip(ax_meas, ay_meas, az_meas):

    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    ax = 0.0 if (abs(ax) < 0.1) else ax

    kf.predict()

    kf.update(np.array([ax]))

    if abs(kf.x[1]) < 0.05:
            kf.x[1] *= 0.9

    xs.append(kf.x[0])
    vs.append(kf.x[1])
    accs.append(kf.x[2])

xs = np.array(xs)
vs = np.array(vs)
accs = np.array(accs)

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
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

plt.show()
