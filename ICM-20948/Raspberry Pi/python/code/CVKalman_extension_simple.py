#!/Users/s/bin/python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

csv_path = "good_extension_data/imu_2026_02_08_00_05_27_0.0423_0.01damp_0.5zupt.csv"
txt_path = "good_extension_data/imu_2026_02_08_00_05_27_metadata_0.0423.txt"
df = pd.read_csv(csv_path)

t = df["time"].values
ax_m = df["ax"].values
ay_m = df["ay"].values
gz   = df["gz"].values  # deg/s

dt = np.mean(np.diff(t))
with open(txt_path, 'r') as f:
    lines = f.readlines()
gyro_bias_line = [l for l in lines if l.startswith("Gyro Bias")][0]
gyro_bias = np.array(eval(gyro_bias_line.split(":")[1].strip()))
gz_corrected = gz - gyro_bias[2]
gz_corrected = np.deg2rad(gz_corrected)

kf = KalmanFilter(dim_x=7, dim_z=3)
kf.x = np.zeros(7)

kf.F = np.array([
    [1, dt, 0.5*dt**2, 0, 0, 0, 0],
    [0, 1, dt, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, dt, 0.5*dt**2, 0],
    [0, 0, 0, 0, 1, dt, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
])

# Measurement: [ax_world, ay_world, yaw_rate]
kf.H = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
])

kf.P = np.eye(7) * 1e-2

sigma_a = 0.1        # acceleration process noise
sigma_psi = 0.01     # yaw random walk

# Process noise for x and y motion (pos, vel, accel)
Q1 = Q_discrete_white_noise(dim=3, dt=dt, var=sigma_a**2)

# Yaw process noise
Q_yaw = np.array([[sigma_psi**2 * dt]])

kf.Q = np.block([
    [Q1,               np.zeros((3,3)), np.zeros((3,1))],
    [np.zeros((3,3)),  Q1,               np.zeros((3,1))],
    [np.zeros((1,3)),  np.zeros((1,3)),  Q_yaw]
])

kf.R = np.diag([0.00004, 0.0001, 0.006])

# acceleration and velocity damping
acc_threshold_x = 0.5
acc_threshold_y = 0.4
vel_damping_threshold_x = 0.05
vel_damping_threshold_y = 0.01

xs, ys = [], []
vxs, vys = [], []
axs_world, ays_world = [], []
yaws = []
for ax_body, ay_body, wz in zip(ax_m, ay_m, gz_corrected):
    
    # Predict step
    kf.predict()
    
    # Rotate sensor accelerations to world acc using current yaw
    yaw = kf.x[6]
    c = np.cos(yaw)
    s = np.sin(yaw)
    ax_w = c*ax_body - s*ay_body
    ay_w = s*ax_body + c*ay_body
    
    # Zero small accelerations
    if abs(ax_w) < acc_threshold_x:
        ax_w = 0.0
    if abs(ay_w) < acc_threshold_y:
        ay_w = 0.0
    
    kf.update(np.array([ax_w, ay_w, wz]))
    
    # Velocity damping
    if abs(kf.x[1]) < vel_damping_threshold_x:
        kf.x[1] *= 0.9
    if abs(kf.x[4]) < vel_damping_threshold_y:
        kf.x[4] *= 0.9
    
    xs.append(kf.x[0])
    vxs.append(kf.x[1])
    axs_world.append(kf.x[2])
    ys.append(kf.x[3])
    vys.append(kf.x[4])
    ays_world.append(kf.x[5])
    yaws.append(kf.x[6])

xs = np.array(xs)
ys = np.array(ys)
vxs = np.array(vxs)
vys = np.array(vys)
axs_world = np.array(axs_world)
ays_world = np.array(ays_world)
yaws = np.array(yaws)

distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
cumulative_distance = np.concatenate(([0], np.cumsum(distances)))

error_x = np.abs(xs)
error_y = np.abs(ys)
error_from_origin = np.sqrt(xs**2 + ys**2)

final_error_x = error_x[-1]
final_error_y = error_y[-1]
final_error = error_from_origin[-1]

plt.figure(figsize=(10,8))
plt.plot(xs, ys, 'b-', linewidth=1.5, label='Trajectory')
plt.scatter(xs[0], ys[0], color='green', label='Start')
plt.scatter(xs[-1], ys[-1], color='red', label='End')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('2D Position Trajectory (X/Y Accel + Yaw)')
plt.grid(True)
plt.axis('equal')
plt.legend()

plt.figure(figsize=(10,6))
plt.plot(cumulative_distance, error_x, linewidth=2, label='X position error')
plt.plot(cumulative_distance, error_y, linewidth=2, label='Y position error')
plt.plot(cumulative_distance, error_from_origin, 'k--', alpha=0.7, label='Total error')

plt.axhline(y=final_error_x, linestyle=':', label=f'Final X error: {final_error_x:.3f} m')
plt.axhline(y=final_error_y, linestyle=':', label=f'Final Y error: {final_error_y:.3f} m')

plt.xlabel('Distance Traveled (m)')
plt.ylabel('Position Error (m)')
plt.title('Position Error vs Distance Traveled (Per Axis)')
plt.grid(True)
plt.legend()

plt.figure(figsize=(10,6))
plt.plot(t, xs, label='X position')
plt.plot(t, ys, label='Y position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position vs Time')
plt.grid(True)
plt.legend()

plt.figure(figsize=(10,6))
plt.plot(t, vxs, label='X velocity')
plt.plot(t, vys, label='Y velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs Time')
plt.grid(True)
plt.legend()

plt.figure(figsize=(10,6))
plt.plot(t, axs_world, label='X accel (world)', linewidth=1.5)
plt.plot(t, ays_world, label='Y accel (world)', linewidth=1.5)
plt.plot(t, ax_m, alpha=0.3, label='X accel (raw)')
plt.plot(t, ay_m, alpha=0.3, label='Y accel (raw)')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Acceleration vs Time')
plt.grid(True)
plt.legend()
plt.show()