#!/Users/s/bin/python3

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

<<<<<<< HEAD
# ------------------------------------------------------------
# Load IMU data
# ------------------------------------------------------------
csv_path = "good_data/imu_2026_02_06_07_57_32.csv"
=======
csv_path = "imu_2026_02_07_23_26_02_-0.0854.csv"
>>>>>>> 5b29d8be3359b892fb0bccba1ad49fcc5be024d1
df = pd.read_csv(csv_path)

taxis = df["time"].values
ax_meas = df["ax"].values   # bias-corrected acc
ay_meas = df["ay"].values
az_meas = df["az"].values 

dt = np.mean(np.diff(taxis))
print(f"Estimated dt: {dt:.6f} s")

# KF parameters
sigma_a = 0.1 # process noise guess
accel_noise_var = 0.02 # measurement noise variance

# ------------------------------------------------------------
# Kalman Filter function
# ------------------------------------------------------------
def run_kf(ax_meas, threshold=False, thresh_val=0.1):
    xs, vs, accs = [], [], []

    # Initialize KF
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.F = np.array([
        [1., dt, 0.5 * dt * dt],
        [0., 1.,        dt],
        [0., 0.,        1.]
    ])
    kf.H = np.array([[0., 0., 1.]])
    kf.x = np.array([0., 0., 0.])
    kf.P = np.diag([0.01, 0.01, 0.01])
    kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=sigma_a**2)
    kf.R = np.array([[accel_noise_var]])

    for ax in ax_meas:
        if threshold:
            ax = 0.0 if abs(ax) < thresh_val else ax

        kf.predict()
        kf.update(np.array([ax]))

<<<<<<< HEAD
        if threshold and abs(kf.x[1]) < 0.05:
=======
sigma_a = 0.1 # guess

kf.Q = Q_discrete_white_noise(
    dim=3,
    dt=dt,
    var=sigma_a**2
)

# Measurement noise variance (from calibration)
accel_noise_var = 0.02
kf.R = np.array([[accel_noise_var]])

xs = []
vs = []
accs = []
prev_ax = 0
NOISE_SCALE = 0.2
NOISE_THRESHOLD = 0.1
for ax, ay, az in zip(ax_meas, ay_meas, az_meas):

    # a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    ax = 0.0 if (abs(ax) < 0.1) else ax
    
    if abs(ax - prev_ax) > NOISE_THRESHOLD:
        ax = ax * NOISE_SCALE
    prev_ax = ax
    
    kf.predict()

    kf.update(np.array([ax]))

    if abs(kf.x[1]) < 0.05:
>>>>>>> 5b29d8be3359b892fb0bccba1ad49fcc5be024d1
            kf.x[1] *= 0.9

        xs.append(kf.x[0])
        vs.append(kf.x[1])
        accs.append(kf.x[2])

    return np.array(xs), np.array(vs), np.array(accs)

# ------------------------------------------------------------
# Run KF with and without threshold
# ------------------------------------------------------------
xs_thresh, vs_thresh, accs_thresh = run_kf(ax_meas, threshold=True)
xs_no_thresh, vs_no_thresh, accs_no_thresh = run_kf(ax_meas, threshold=False)

# ------------------------------------------------------------
# Position plot
# ------------------------------------------------------------
plt.figure()
plt.plot(taxis, xs_thresh, label="KF with threshold")
plt.plot(taxis, xs_no_thresh, label="KF without threshold", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Position Estimate Comparison")
plt.grid()
plt.legend()

# ------------------------------------------------------------
# Velocity plot
# ------------------------------------------------------------
plt.figure()
plt.plot(taxis, vs_thresh, label="KF velocity with threshold")
plt.plot(taxis, vs_no_thresh, label="KF velocity without threshold", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Estimate Comparison")
plt.grid()
plt.legend()

# ------------------------------------------------------------
# Acceleration plot
# ------------------------------------------------------------
plt.figure()
plt.plot(taxis, accs_thresh, label="KF acceleration with threshold")
plt.plot(taxis, accs_no_thresh, label="KF acceleration without threshold", alpha=0.7)
plt.plot(taxis, ax_meas, 'k--', alpha=0.3, label="Measured ax")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration Comparison")
plt.grid()
plt.legend()

# ------------------------------------------------------------
# Distance and error analysis for both KF runs
# ------------------------------------------------------------
def compute_distance_error(xs):
    distances = np.abs(np.diff(xs))
    cumulative_distance = np.concatenate(([0], np.cumsum(distances)))
    error_from_origin = np.abs(xs)
    final_error = error_from_origin[-1]
    total_distance = cumulative_distance[-1]
    return cumulative_distance, error_from_origin, final_error, total_distance

cumdist_thresh, err_thresh, final_err_thresh, total_dist_thresh = compute_distance_error(xs_thresh)
cumdist_no, err_no, final_err_no, total_dist_no = compute_distance_error(xs_no_thresh)

print(f"\nKF with threshold:")
print(f"Final position: {xs_thresh[-1]:.4f} m")
print(f"Final error from origin: {final_err_thresh:.4f} m")
print(f"Total distance traveled: {total_dist_thresh:.4f} m")

print(f"\nKF without threshold:")
print(f"Final position: {xs_no_thresh[-1]:.4f} m")
print(f"Final error from origin: {final_err_no:.4f} m")
print(f"Total distance traveled: {total_dist_no:.4f} m")

# Plot sensor error vs distance for both KF runs
plt.figure()
plt.plot(cumdist_thresh, err_thresh, 'r-', linewidth=2, label="KF error with threshold")
plt.plot(cumdist_no, err_no, 'b-', linewidth=2, alpha=0.7, label="KF error without threshold")
plt.xlabel("Distance Traveled (m)")
plt.ylabel("Error from Origin (m)")
plt.title("Sensor Error vs Distance Traveled")
plt.grid(True, alpha=0.3)
plt.axhline(y=final_err_thresh, color='r', linestyle='--', alpha=0.5, label=f'Final Error (thresh): {final_err_thresh:.4f} m')
plt.axhline(y=final_err_no, color='b', linestyle='--', alpha=0.5, label=f'Final Error (no thresh): {final_err_no:.4f} m')
plt.legend()
plt.tight_layout()

# ------------------------------------------------------------
# Difference between thresholded and unthresholded position
# ------------------------------------------------------------
plt.figure()
plt.plot(taxis, xs_thresh - xs_no_thresh, label="Position difference (thresh - no thresh)", color='m')
plt.xlabel("Time (s)")
plt.ylabel("Position difference (m)")
plt.title("Effect of Thresholding on Position Estimate")
plt.grid()
plt.legend()
plt.tight_layout()

plt.show()


# #!/Users/s/bin/python3

# import sys
# import math as m
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise

# csv_path = "good_data/imu_2026_02_06_07_57_32.csv"
# df = pd.read_csv(csv_path)

# taxis = df["time"].values
# ax_meas = df["ax"].values   # bias-corrected acc
# ay_meas = df["ay"].values
# az_meas = df["az"].values 

# dt = np.mean(np.diff(taxis))
# print(f"Estimated dt: {dt:.6f} s")

# kf = KalmanFilter(dim_x=3, dim_z=1)

# kf.F = np.array([
#     [1., dt, 0.5 * dt * dt],
#     [0., 1.,        dt],
#     [0., 0.,        1.]
# ])

# # We measure acceleration only
# kf.H = np.array([[0., 0., 1.]])

# # Initial state: assume starting at rest
# kf.x = np.array([0., 0., 0.])

# kf.P = np.diag([1., 1., 1.]) * 0.01

# sigma_a = 0.1 # guess

# kf.Q = Q_discrete_white_noise(
#     dim=3,
#     dt=dt,
#     var=sigma_a**2
# )

# # Measurement noise variance (from calibration)
# accel_noise_var = 0.02
# kf.R = np.array([[accel_noise_var]])

# xs = []
# vs = []
# accs = []

# for ax, ay, az in zip(ax_meas, ay_meas, az_meas):

#     # a_mag = np.sqrt(ax**2 + ay**2 + az**2)
#     ax = 0.0 if (abs(ax) < 0.1) else ax

#     kf.predict()

#     kf.update(np.array([ax]))

#     if abs(kf.x[1]) < 0.05:
#             kf.x[1] *= 0.9

#     xs.append(kf.x[0])
#     vs.append(kf.x[1])
#     accs.append(kf.x[2])

# xs = np.array(xs)
# vs = np.array(vs)
# accs = np.array(accs)

# plt.figure()
# plt.plot(taxis, xs, label="KF position estimate")
# plt.xlabel("Time (s)")
# plt.ylabel("Position (m)")
# plt.title("Position Estimate from Acceleration (Kalman Filter)")
# plt.grid()
# plt.legend()

# plt.figure()
# plt.plot(taxis, vs, label="KF velocity estimate")
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (m/s)")
# plt.title("Velocity Estimate")
# plt.grid()
# plt.legend()

# plt.figure()
# plt.plot(taxis, accs, label="KF acceleration estimate")
# plt.plot(taxis, ax_meas, alpha=0.5, label="Measured ax")
# plt.xlabel("Time (s)")
# plt.ylabel("Acceleration (m/s²)")
# plt.title("Acceleration")
# plt.grid()
# plt.legend()

# # Calculate distance traveled
# distances = np.abs(np.diff(xs))
# cumulative_distance = np.concatenate(([0], np.cumsum(distances)))

# # Calculate error from origin
# error_from_origin = np.abs(xs)
# final_error = error_from_origin[-1]

# print(f"\nFinal position: {xs[-1]:.4f} m")
# print(f"Final error from origin: {final_error:.4f} m")
# print(f"Total distance traveled: {cumulative_distance[-1]:.4f} m")

# # Error vs Distance plot
# plt.figure()
# plt.plot(cumulative_distance, error_from_origin, 'r-', linewidth=2, label="Position error")
# plt.xlabel("Distance Traveled (m)")
# plt.ylabel("Error from Origin (m)")
# plt.title("Sensor Error vs Distance Traveled")
# plt.grid(True, alpha=0.3)
# plt.axhline(y=final_error, color='k', linestyle='--', alpha=0.5, label=f'Final Error: {final_error:.4f} m')
# plt.legend()
# plt.tight_layout()

# plt.show()
