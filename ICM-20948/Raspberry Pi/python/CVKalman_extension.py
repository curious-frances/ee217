#!/Users/s/bin/python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import sys
import os

# ------------------------------------------------------------
# Load IMU data
# ------------------------------------------------------------
# Allow command line argument for data file, default to good data
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = "extension_data/imu_2026_02_06_08_29_17.csv"

df = pd.read_csv(csv_path)

t = df["time"].values
ax_m = df["ax"].values
ay_m = df["ay"].values
gz   = df["gz"].values  # deg/s

dt = np.mean(np.diff(t))
print(f"dt = {dt:.6f} s")

# Convert gyro to rad/s
gz_rad = np.deg2rad(gz)

# ------------------------------------------------------------
# Gyro bias estimation (from initial stationary period)
# ------------------------------------------------------------
# Use first 1 second (or first 100 samples) to estimate gyro bias
calib_samples = min(50, len(gz_rad))
gyro_bias = np.mean(gz_rad[:calib_samples])
print(f"Estimated gyro bias: {np.rad2deg(gyro_bias):.4f} deg/s ({gyro_bias:.6f} rad/s)")

# Remove gyro bias
gz_corrected = gz_rad - gyro_bias

# ------------------------------------------------------------
# Extended Kalman Filter
# State: [px, py, vx, vy, yaw]
# Measurements: [ax_body, ay_body, gyro_z]
# ------------------------------------------------------------
ekf = ExtendedKalmanFilter(dim_x=5, dim_z=3)
ekf.x = np.zeros(5)  # [px, py, vx, vy, yaw]

# Initial covariance
ekf.P = np.diag([0.01, 0.01, 0.01, 0.01, 0.01])  # Small uncertainty in all states

# Process noise
sigma_a = 0.1  # Acceleration process noise (same as Part 2)
sigma_yaw = 0.01  # Yaw rate process noise (rad/s)

# Build process noise Q matrix
# For position/velocity: use discrete white noise for constant acceleration model
Q_pos_vel = Q_discrete_white_noise(dim=2, dt=dt, var=sigma_a**2, block_size=2)
# Q_pos_vel is 4x4 for [px, vx, py, vy] but we need to reorganize for [px, py, vx, vy]
# Actually, let's build it manually for clarity
Q_pv = np.array([
    [0.25*dt**4*sigma_a**2, 0.5*dt**3*sigma_a**2],
    [0.5*dt**3*sigma_a**2, dt**2*sigma_a**2]
])

ekf.Q = np.zeros((5, 5))
ekf.Q[0:2, 0:2] = Q_pv  # X position/velocity
ekf.Q[2:4, 2:4] = Q_pv  # Y position/velocity  
ekf.Q[4, 4] = sigma_yaw**2 * dt  # Yaw

# Measurement noise
accel_noise_var = 0.02  # Same as Part 2
gyro_noise_var = 0.001  # Gyro measurement noise (rad/s)^2
ekf.R = np.diag([accel_noise_var, accel_noise_var, gyro_noise_var])

# ------------------------------------------------------------
# EKF Functions
# ------------------------------------------------------------

def fx(x, dt, ax_world, ay_world):
    """
    State transition function
    State: [px, py, vx, vy, yaw]
    Integrates acceleration into velocity and position
    """
    px, py, vx, vy, yaw = x
    
    # Integrate acceleration -> velocity -> position
    vx_new = vx + ax_world * dt
    vy_new = vy + ay_world * dt
    px_new = px + vx * dt + 0.5 * ax_world * dt**2
    py_new = py + vy * dt + 0.5 * ay_world * dt**2
    
    # Yaw doesn't change in prediction (updated by gyro measurement)
    return np.array([px_new, py_new, vx_new, vy_new, yaw])

def F_jacobian(x, dt, ax_body, ay_body):
    """
    Jacobian of state transition function
    """
    _, _, _, _, yaw = x
    c = np.cos(yaw)
    s = np.sin(yaw)
    
    # Rotate body accelerations to world frame
    ax_world = c * ax_body - s * ay_body
    ay_world = s * ax_body + c * ay_body
    
    F = np.eye(5)
    F[0, 2] = dt  # d(px)/d(vx)
    F[0, 4] = 0.5 * dt**2 * (-s * ax_body - c * ay_body)  # d(px)/d(yaw) via rotation
    F[1, 3] = dt  # d(py)/d(vy)
    F[1, 4] = 0.5 * dt**2 * (c * ax_body - s * ay_body)  # d(py)/d(yaw) via rotation
    F[2, 4] = dt * (-s * ax_body - c * ay_body)  # d(vx)/d(yaw)
    F[3, 4] = dt * (c * ax_body - s * ay_body)  # d(vy)/d(yaw)
    return F

def hx(x, ax_body, ay_body):
    """
    Measurement function
    Returns: [ax_world, ay_world, yaw_rate_expected]
    We measure body-frame accelerations and gyro, but need to predict world-frame accelerations
    """
    _, _, _, _, yaw = x
    
    # Rotate body-frame accelerations to world frame using current yaw estimate
    c = np.cos(yaw)
    s = np.sin(yaw)
    
    ax_world = c * ax_body - s * ay_body
    ay_world = s * ax_body + c * ay_body
    
    # Expected yaw rate is 0 (we measure it, but don't predict it from state)
    return np.array([ax_world, ay_world, 0.0])

def H_jacobian(x, ax_body, ay_body):
    """
    Jacobian of measurement function
    """
    _, _, _, _, yaw = x
    c = np.cos(yaw)
    s = np.sin(yaw)
    
    H = np.zeros((3, 5))
    # d(ax_world)/d(yaw) = -s*ax_body - c*ay_body
    H[0, 4] = -s * ax_body - c * ay_body
    # d(ay_world)/d(yaw) = c*ax_body - s*ay_body
    H[1, 4] = c * ax_body - s * ay_body
    # yaw_rate measurement doesn't depend on state
    return H

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
zupt_threshold = 0.1
velocity_threshold = 0.05

# ------------------------------------------------------------
# Run filter
# ------------------------------------------------------------
xs, ys = [], []
vxs, vys = [], []
yaws = []
axs_world, ays_world = [], []

for ax_body, ay_body, wz in zip(ax_m, ay_m, gz_corrected):
    
    # ZUPT: If acceleration is small, set to zero
    if abs(ax_body) < zupt_threshold:
        ax_body = 0.0
    if abs(ay_body) < zupt_threshold:
        ay_body = 0.0
    
    # Get current yaw estimate to rotate accelerations for prediction
    yaw_current = ekf.x[4]
    c = np.cos(yaw_current)
    s = np.sin(yaw_current)
    ax_world_pred = c * ax_body - s * ay_body
    ay_world_pred = s * ax_body + c * ay_body
    
    # Predict step (need to manually update state since EKF.predict() doesn't take inputs)
    ekf.F = F_jacobian(ekf.x, dt, ax_body, ay_body)
    ekf.x = fx(ekf.x, dt, ax_world_pred, ay_world_pred)
    # Now update covariance
    ekf.P = ekf.F @ ekf.P @ ekf.F.T + ekf.Q
    
    # Update step with measurements: [ax_body, ay_body, gyro_z]
    # The measurement function hx will rotate body-frame accels to world frame
    ekf.update(
        z=np.array([ax_body, ay_body, wz]),
        HJacobian=lambda x: H_jacobian(x, ax_body, ay_body),
        Hx=lambda x: hx(x, ax_body, ay_body)
    )
    
    # Velocity damping (same as Part 2)
    if abs(ekf.x[2]) < velocity_threshold:
        ekf.x[2] *= 0.9
    if abs(ekf.x[3]) < velocity_threshold:
        ekf.x[3] *= 0.9
    
    # Get world-frame accelerations for plotting
    ax_w, ay_w, _ = hx(ekf.x, ax_body, ay_body)
    
    xs.append(ekf.x[0])
    ys.append(ekf.x[1])
    vxs.append(ekf.x[2])
    vys.append(ekf.x[3])
    yaws.append(ekf.x[4])
    axs_world.append(ax_w)
    ays_world.append(ay_w)

xs = np.array(xs)
ys = np.array(ys)
vxs = np.array(vxs)
vys = np.array(vys)
axs_world = np.array(axs_world)
ays_world = np.array(ays_world)
yaws = np.array(yaws)

# ------------------------------------------------------------
# Error analysis
# ------------------------------------------------------------
# Calculate distance traveled
distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
cumulative_distance = np.concatenate(([0], np.cumsum(distances)))

# Calculate error from origin (assuming we start and end at origin)
error_from_origin = np.sqrt(xs**2 + ys**2)
final_error = error_from_origin[-1]

print(f"\nFinal position: ({xs[-1]:.4f}, {ys[-1]:.4f}) m")
print(f"Final error from origin: {final_error:.4f} m")
print(f"Total distance traveled: {cumulative_distance[-1]:.4f} m")
# ZUPT tracking removed - using simple thresholding instead
print(f"Final velocity: ({vxs[-1]:.6f}, {vys[-1]:.6f}) m/s")
print(f"Final yaw: {np.rad2deg(yaws[-1]):.2f} deg (drift from start: {np.rad2deg(yaws[-1] - yaws[0]):.2f} deg)")
print(f"Max velocity during motion: {np.max(np.sqrt(vxs**2 + vys**2)):.4f} m/s")
print(f"X position range: [{np.min(xs):.4f}, {np.max(xs):.4f}] m")
print(f"Y position range: [{np.min(ys):.4f}, {np.max(ys):.4f}] m")
print(f"X velocity range: [{np.min(vxs):.4f}, {np.max(vxs):.4f}] m/s")
print(f"Y velocity range: [{np.min(vys):.4f}, {np.max(vys):.4f}] m/s")

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
# Plot 1: 2D Trajectory
plt.figure(figsize=(10, 8))
plt.plot(xs, ys, 'b-', linewidth=1.5, label="Trajectory")
plt.scatter(xs[0], ys[0], color='green', s=100, marker='o', label="Start", zorder=5)
plt.scatter(xs[-1], ys[-1], color='red', s=100, marker='x', label="End", zorder=5)
plt.axis("equal")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("2D Position Trajectory (X/Y Accel + Yaw Gyro Fusion)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Plot 2: Error vs Distance Traveled (key metric for assignment)
plt.figure(figsize=(10, 6))
plt.plot(cumulative_distance, error_from_origin, 'r-', linewidth=2)
plt.xlabel("Distance Traveled (m)")
plt.ylabel("Error from Origin (m)")
plt.title("Sensor Error vs Distance Traveled")
plt.grid(True, alpha=0.3)
plt.axhline(y=final_error, color='k', linestyle='--', alpha=0.5, label=f'Final Error: {final_error:.4f} m')
plt.legend()
plt.tight_layout()

# Plot 3: Position vs Time
plt.figure(figsize=(10, 6))
plt.plot(t, xs, label="X position", linewidth=1.5)
plt.plot(t, ys, label="Y position", linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Position vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Plot 4: Velocity vs Time
plt.figure(figsize=(10, 6))
plt.plot(t, vxs, label="X velocity", linewidth=1.5)
plt.plot(t, vys, label="Y velocity", linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Plot 5: Acceleration vs Time
plt.figure(figsize=(10, 6))
plt.plot(t, axs_world, label="X accel (KF world)", linewidth=1.5)
plt.plot(t, ays_world, label="Y accel (KF world)", linewidth=1.5)
plt.plot(t, ax_m, alpha=0.3, label="X accel (raw)", linewidth=0.5)
plt.plot(t, ay_m, alpha=0.3, label="Y accel (raw)", linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Plot 6: Yaw angle vs Time (for debugging)
plt.figure(figsize=(10, 6))
plt.plot(t, np.rad2deg(yaws), label="Yaw angle", linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Yaw Angle (deg)")
plt.title("Yaw Angle vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Plot 7: ZUPT activity and velocity magnitude
plt.figure(figsize=(12, 8))
vel_mag = np.sqrt(vxs**2 + vys**2)
ax1 = plt.subplot(2, 1, 1)
plt.plot(t, vel_mag, 'b-', linewidth=1.5, label="Velocity magnitude")
plt.axhline(y=velocity_threshold, color='r', linestyle='--', alpha=0.5, label=f'ZUPT threshold ({velocity_threshold} m/s)')
plt.xlabel("Time (s)")
plt.ylabel("Velocity Magnitude (m/s)")
plt.title("Velocity Magnitude vs Time")
plt.grid(True, alpha=0.3)
plt.legend()

# ZUPT tracking plot removed - using simple thresholding approach
plt.tight_layout()

plt.show()


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
