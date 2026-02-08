#!/Users/s/bin/python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter

# ------------------------------------------------------------
# Load IMU data
# ------------------------------------------------------------
csv_path = "imu_2026_02_07_23_26_02_-0.0854_0.01th.csv"
df = pd.read_csv(csv_path)

t = df["time"].values
ax_m = df["ax"].values
ay_m = df["ay"].values
gz   = df["gz"].values  # deg/s

dt = np.mean(np.diff(t))
gz_rad = np.deg2rad(gz)  # convert to rad/s

# ------------------------------------------------------------
# Gyro bias estimation
# ------------------------------------------------------------
calib_samples = min(100, len(gz_rad))
gyro_bias = np.mean(gz_rad[:calib_samples])
gz_corrected = gz_rad - gyro_bias

# ------------------------------------------------------------
# Kalman Filter with yaw
# State: [px, vx, ax, py, vy, ay, yaw]
# ------------------------------------------------------------
kf = KalmanFilter(dim_x=7, dim_z=3)
kf.x = np.zeros(7)

kf.F = np.array([
    [1, dt, 0.5*dt**2, 0, 0, 0, 0],
    [0, 1, dt, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, dt, 0.5*dt**2, 0],
    [0, 0, 0, 0, 1, dt, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]  # yaw evolves independently
])

# Measurement: [ax_world, ay_world, yaw_rate]
kf.H = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
])

kf.P = np.eye(7) * 1e-2
kf.Q = np.eye(7) * 1e-4
kf.R = np.diag([0.02, 0.02, 1e-3])

# ZUPT and velocity damping
zupt_threshold = 0.1
vel_damping_threshold_x = 0.01
vel_damping_threshold_y = 0.05

# ------------------------------------------------------------
# Run filter
# ------------------------------------------------------------
xs, ys = [], []
vxs, vys = [], []
axs_world, ays_world = [], []
yaws = []
# prev_ax = 0
# NOISE_SCALE = 0.2
# NOISE_THRESHOLD = 0.1
for ax_body, ay_body, wz in zip(ax_m, ay_m, gz_corrected):
    
    # Predict step
    kf.predict()
    
    # if abs(ax_body - prev_ax) > NOISE_THRESHOLD:
    #     ax_body = ax_body * NOISE_SCALE
    # prev_ax = ax_body
    # if abs(ay_body - prev_ay) < NOISE_THRESHOLD:
    #     ay_body = ay_body * NOISE_SCALE
    # prev_ay = ay_body
    
    # Rotate body-frame accelerations to world-frame using current yaw
    yaw = kf.x[6]
    c = np.cos(yaw)
    s = np.sin(yaw)
    ax_w = c*ax_body - s*ay_body
    ay_w = s*ax_body + c*ay_body
    
    # ZUPT
    if abs(ax_w) < zupt_threshold:
        ax_w = 0.0
    if abs(ay_w) < zupt_threshold:
        ay_w = 0.0
    
    # Update step
    kf.update(np.array([ax_w, ay_w, wz]))
    
    # Velocity damping
    if abs(kf.x[1]) < vel_damping_threshold_x:
        kf.x[1] *= 0.9
    if abs(kf.x[4]) < vel_damping_threshold_y:
        kf.x[4] *= 0.9
    
    # Store results
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

# ------------------------------------------------------------
# Error calculations
# ------------------------------------------------------------
distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
cumulative_distance = np.concatenate(([0], np.cumsum(distances)))
error_from_origin = np.sqrt(xs**2 + ys**2)
final_error = error_from_origin[-1]

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

# 1. 2D Trajectory
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
# plt.show()

# 2. Error vs Distance
plt.figure(figsize=(10,6))
plt.plot(cumulative_distance, error_from_origin, 'r-', linewidth=2)
plt.axhline(y=final_error, color='k', linestyle='--', label=f'Final error: {final_error:.4f} m')
plt.xlabel('Distance Traveled (m)')
plt.ylabel('Error from Origin (m)')
plt.title('Error vs Distance Traveled')
plt.grid(True)
plt.legend()
# plt.show()

# 3. Position vs Time
plt.figure(figsize=(10,6))
plt.plot(t, xs, label='X position')
plt.plot(t, ys, label='Y position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position vs Time')
plt.grid(True)
plt.legend()
# plt.show()

# 4. Velocity vs Time
plt.figure(figsize=(10,6))
plt.plot(t, vxs, label='X velocity')
plt.plot(t, vys, label='Y velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs Time')
plt.grid(True)
plt.legend()
# plt.show()

# 5. Acceleration vs Time
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
# plt.show()

# 6. Yaw vs Time
plt.figure(figsize=(10,6))
plt.plot(t, np.rad2deg(yaws), label='Yaw (deg)', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Yaw Angle (deg)')
plt.title('Yaw Angle vs Time')
plt.grid(True)
plt.legend()
plt.show()

# #!/Users/s/bin/python3

# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
# import sys
# import os

# # ------------------------------------------------------------
# # Load IMU data
# # ------------------------------------------------------------
# # Allow command line argument for data file, default to good data
# if len(sys.argv) > 1:
#     csv_path = sys.argv[1]
# else:
#     csv_path = "good_data/PERFECT_imu_2026_02_06_07_59_03.csv"

# df = pd.read_csv(csv_path)

# t = df["time"].values
# ax_m = df["ax"].values
# ay_m = df["ay"].values
# gz   = df["gz"].values  # deg/s

# dt = np.mean(np.diff(t))
# print(f"dt = {dt:.6f} s")

# # Convert gyro to rad/s
# gz_rad = np.deg2rad(gz)

# # ------------------------------------------------------------
# # Gyro bias estimation (from initial stationary period)
# # ------------------------------------------------------------
# # Use first 1 second (or first 100 samples) to estimate gyro bias
# calib_samples = min(100, len(gz_rad))
# gyro_bias = np.mean(gz_rad[:calib_samples])
# print(f"Estimated gyro bias: {np.rad2deg(gyro_bias):.4f} deg/s ({gyro_bias:.6f} rad/s)")

# # Remove gyro bias
# gz_corrected = gz_rad - gyro_bias

# # ------------------------------------------------------------
# # Kalman Filter
# # State: [px, py, vx, vy, ax, ay]
# # ------------------------------------------------------------
# kf = KalmanFilter(dim_x=6, dim_z=2)
# kf.x = np.zeros(6)

# kf.F = np.array([
#     [1, 0, dt, 0, 0.5*dt*dt, 0],
#     [0, 1, 0, dt, 0, 0.5*dt*dt],
#     [0, 0, 1, 0, dt, 0],
#     [0, 0, 0, 1, 0, dt],
#     [0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 1],
# ])

# kf.H = np.array([
#     [0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 1],
# ])

# # Initial covariance - small uncertainty in position/velocity, larger in acceleration
# kf.P = np.diag([0.01, 0.01, 0.01, 0.01, 0.1, 0.1])

# # Process noise (decoupled X/Y, identical to two 1D filters)
# # Tune sigma_a based on expected acceleration variations
# sigma_a = 0.1  # Same as Part 2
# Q1 = Q_discrete_white_noise(dim=3, dt=dt, var=sigma_a**2)
# kf.Q = np.block([
#     [Q1, np.zeros((3,3))],
#     [np.zeros((3,3)), Q1]
# ])

# # Measurement noise - both axes should have similar noise levels
# # Part 2 used 0.02, so use similar values for both X and Y
# accel_noise_var = 0.02
# kf.R = np.diag([accel_noise_var, accel_noise_var])

# # ------------------------------------------------------------
# # Configuration parameters
# # ------------------------------------------------------------
# # Yaw sign: depends on sensor orientation and coordinate system
# # Try +1 or -1 to see which gives better results
# YAW_SIGN = -1  # Negative is typical for most IMU orientations

# # ZUPT threshold (same as Part 2)
# zupt_threshold = 0.1

# # ------------------------------------------------------------
# # Run filter
# # ------------------------------------------------------------
# yaw = 0.0  # Initial yaw angle (assume starting aligned with world frame)

# xs, ys = [], []
# vxs, vys = [], []
# axs, ays = [], []
# yaws = []  # Track yaw for debugging

# for ax, ay, wz in zip(ax_m, ay_m, gz_corrected):
    
#     # Integrate yaw angle
#     # The sign depends on sensor orientation - adjust YAW_SIGN if needed
#     yaw += YAW_SIGN * wz * dt
    
#     # Rotate accelerations from body frame to world frame
#     # Standard 2D rotation matrix
#     c = np.cos(yaw)
#     s = np.sin(yaw)
    
#     # Rotate body-frame accelerations to world frame
#     ax_w = c*ax - s*ay
#     ay_w = s*ax + c*ay
    
#     # ZUPT: Zero velocity update - if acceleration magnitude is small, assume stationary
#     # Apply threshold on world-frame acceleration magnitude
#     accel_mag = np.sqrt(ax_w**2 + ay_w**2)
#     if accel_mag < zupt_threshold:
#         ax_w = 0.0
#         ay_w = 0.0
    
#     # Predict step
#     kf.predict()
    
#     # Update step with world-frame accelerations
#     kf.update(np.array([ax_w, ay_w]))
    
#     # Velocity damping near zero (same as Part 2)
#     if abs(kf.x[2]) < 0.05:
#         kf.x[2] *= 0.9
#     if abs(kf.x[3]) < 0.05:
#         kf.x[3] *= 0.9
    
#     xs.append(kf.x[0])
#     ys.append(kf.x[1])
#     vxs.append(kf.x[2])
#     vys.append(kf.x[3])
#     axs.append(kf.x[4])
#     ays.append(kf.x[5])
#     yaws.append(yaw)

# xs = np.array(xs)
# ys = np.array(ys)
# vxs = np.array(vxs)
# vys = np.array(vys)
# axs = np.array(axs)
# ays = np.array(ays)
# yaws = np.array(yaws)

# # ------------------------------------------------------------
# # Error analysis
# # ------------------------------------------------------------
# # Calculate distance traveled
# distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
# cumulative_distance = np.concatenate(([0], np.cumsum(distances)))

# # Calculate error from origin (assuming we start and end at origin)
# error_from_origin = np.sqrt(xs**2 + ys**2)
# final_error = error_from_origin[-1]

# print(f"\nFinal position: ({xs[-1]:.4f}, {ys[-1]:.4f}) m")
# print(f"Final error from origin: {final_error:.4f} m")
# print(f"Total distance traveled: {cumulative_distance[-1]:.4f} m")

# # ------------------------------------------------------------
# # Plots
# # ------------------------------------------------------------
# # Plot 1: 2D Trajectory
# plt.figure(figsize=(10, 8))
# plt.plot(xs, ys, 'b-', linewidth=1.5, label="Trajectory")
# plt.scatter(xs[0], ys[0], color='green', s=100, marker='o', label="Start", zorder=5)
# plt.scatter(xs[-1], ys[-1], color='red', s=100, marker='x', label="End", zorder=5)
# plt.axis("equal")
# plt.xlabel("X Position (m)")
# plt.ylabel("Y Position (m)")
# plt.title("2D Position Trajectory (X/Y Accel + Yaw Gyro Fusion)")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()

# # Plot 2: Error vs Distance Traveled (key metric for assignment)
# plt.figure(figsize=(10, 6))
# plt.plot(cumulative_distance, error_from_origin, 'r-', linewidth=2)
# plt.xlabel("Distance Traveled (m)")
# plt.ylabel("Error from Origin (m)")
# plt.title("Sensor Error vs Distance Traveled")
# plt.grid(True, alpha=0.3)
# plt.axhline(y=final_error, color='k', linestyle='--', alpha=0.5, label=f'Final Error: {final_error:.4f} m')
# plt.legend()
# plt.tight_layout()

# # Plot 3: Position vs Time
# plt.figure(figsize=(10, 6))
# plt.plot(t, xs, label="X position", linewidth=1.5)
# plt.plot(t, ys, label="Y position", linewidth=1.5)
# plt.xlabel("Time (s)")
# plt.ylabel("Position (m)")
# plt.title("Position vs Time")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()

# # Plot 4: Velocity vs Time
# plt.figure(figsize=(10, 6))
# plt.plot(t, vxs, label="X velocity", linewidth=1.5)
# plt.plot(t, vys, label="Y velocity", linewidth=1.5)
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (m/s)")
# plt.title("Velocity vs Time")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()

# # Plot 5: Acceleration vs Time
# plt.figure(figsize=(10, 6))
# plt.plot(t, axs, label="X accel (KF)", linewidth=1.5)
# plt.plot(t, ays, label="Y accel (KF)", linewidth=1.5)
# plt.plot(t, ax_m, alpha=0.3, label="X accel (raw)", linewidth=0.5)
# plt.plot(t, ay_m, alpha=0.3, label="Y accel (raw)", linewidth=0.5)
# plt.xlabel("Time (s)")
# plt.ylabel("Acceleration (m/s²)")
# plt.title("Acceleration vs Time")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()

# # Plot 6: Yaw angle vs Time (for debugging)
# plt.figure(figsize=(10, 6))
# plt.plot(t, np.rad2deg(yaws), label="Yaw angle", linewidth=1.5)
# plt.xlabel("Time (s)")
# plt.ylabel("Yaw Angle (deg)")
# plt.title("Yaw Angle vs Time")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()

# plt.show()


# #!/Users/s/bin/python3

# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from filterpy.kalman import ExtendedKalmanFilter

# # ------------------------------------------------------------
# # Load IMU data
# # ------------------------------------------------------------
# # df = pd.read_csv("good_data/PERFECT_imu_2026_02_06_07_59_03.csv")
# # df = pd.read_csv("extension_data/imu_2026_02_06_08_21_33.csv")
# # df = pd.read_csv("extension_data/imu_2026_02_06_08_23_13.csv")
# # df = pd.read_csv("extension_data/imu_2026_02_06_08_24_32.csv")
# # df = pd.read_csv("extension_data/imu_2026_02_06_08_25_46.csv")
# # df = pd.read_csv("extension_data/imu_2026_02_06_08_26_41.csv")
# # df = pd.read_csv("extension_data/imu_2026_02_06_08_27_58.csv")
# df = pd.read_csv("extension_data/imu_2026_02_06_08_29_17.csv")

# t = df["time"].values
# ax = df["ax"].values
# ay = df["ay"].values
# gyro_z = np.deg2rad(df["gz"].values)    # rad/s (convert if needed!)

# dt = np.mean(np.diff(t))
# print(f"dt = {dt:.6f}")

# # ------------------------------------------------------------
# # EKF setup
# # State: [px, py, vx, vy, yaw, bax, bay]
# # ------------------------------------------------------------
# ekf = ExtendedKalmanFilter(dim_x=7, dim_z=3)

# ekf.x = np.zeros(7)

# ekf.P = np.diag([
#     0.01, 0.01,   # position
#     0.01, 0.01,   # velocity
#     0.01,         # yaw
#     0.1, 0.1      # accel bias
# ])

# # Measurement noise
# ekf.R = np.diag([
#     0.02,  # ax
#     0.02,  # ay
#     0.001  # gyro
# ])

# # Process noise
# sigma_a = 0.2
# sigma_g = 0.01
# sigma_b = 0.001

# ekf.Q = np.diag([
#     0, 0,
#     sigma_a**2, sigma_a**2,
#     sigma_g**2,
#     sigma_b**2, sigma_b**2
# ]) * dt

# # ------------------------------------------------------------
# # State transition function
# # ------------------------------------------------------------
# def fx(x, dt, u):
#     px, py, vx, vy, yaw, bax, bay = x
#     ax_b, ay_b, wz = u

#     # Remove bias
#     ax_b -= bax
#     ay_b -= bay

#     # Rotate to world frame
#     c = np.cos(yaw)
#     s = np.sin(yaw)

#     ax_w = c*ax_b - s*ay_b
#     ay_w = s*ax_b + c*ay_b

#     px += vx*dt + 0.5*ax_w*dt**2
#     py += vy*dt + 0.5*ay_w*dt**2
#     vx += ax_w*dt
#     vy += ay_w*dt
#     yaw += wz*dt

#     return np.array([px, py, vx, vy, yaw, bax, bay])

# # ------------------------------------------------------------
# # Jacobian of fx
# # ------------------------------------------------------------
# def F_jacobian(x, dt, u):
#     _, _, _, _, yaw, _, _ = x
#     ax_b, ay_b, _ = u

#     c = np.cos(yaw)
#     s = np.sin(yaw)

#     F = np.eye(7)
#     F[0,2] = dt
#     F[1,3] = dt
#     F[2,4] = (-s*ax_b - c*ay_b) * dt
#     F[3,4] = ( c*ax_b - s*ay_b) * dt

#     return F

# # ------------------------------------------------------------
# # Measurement function
# # z = [ax, ay, gyro]
# # ------------------------------------------------------------
# def hx(x):
#     _, _, _, _, _, bax, bay = x
#     return np.array([bax, bay, 0.0])

# def H_jacobian(x):
#     H = np.zeros((3,7))
#     H[0,5] = 1.0
#     H[1,6] = 1.0
#     return H

# # ------------------------------------------------------------
# # Run EKF
# # ------------------------------------------------------------
# xs, ys = [], []

# for ax_i, ay_i, wz_i in zip(ax, ay, gyro_z):

#     u = np.array([ax_i, ay_i, wz_i])

#     # --- Predict step ---
#     ekf.F = F_jacobian(ekf.x, dt, u)
#     ekf.x = fx(ekf.x, dt, u)     # <-- YOU propagate state manually
#     ekf.predict()                # <-- covariance only

#     # --- Update step (bias pseudo-measurement) ---
#     ekf.update(
#         z=np.array([0.0, 0.0, 0.0]),
#         HJacobian=H_jacobian,
#         Hx=hx
#     )

#     xs.append(ekf.x[0])
#     ys.append(ekf.x[1])



# xs = np.array(xs)
# ys = np.array(ys)

# # ------------------------------------------------------------
# # Plot trajectory
# # ------------------------------------------------------------
# plt.figure()
# plt.plot(xs, ys)
# plt.scatter(xs[0], ys[0], label="Start")
# plt.scatter(xs[-1], ys[-1], label="End")
# plt.axis("equal")
# plt.xlabel("X (m)")
# plt.ylabel("Y (m)")
# plt.title("2D Position from Accel + Gyro Fusion")
# plt.grid()
# plt.legend()
# plt.show()
