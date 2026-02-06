#!/Users/s/bin/python3
# Constant-acceleration Kalman filter using acceleration measurements
# Adapted from CV example
#
# State: [position, velocity, acceleration]
# Measurement: acceleration
#
# Units:
#   position: meters
#   velocity: m/s
#   acceleration: m/s^2
#

import sys
import math as m
import numpy as np
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# ------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------
sampling_rate = 100
dt = 1 / 100
stopTime = 12.0           # seconds
taxis = np.arange(0, stopTime, dt)

accel_bias = 0.04         # m/s^2
accel_noise_var = 0.02    # accelerometer noise variance

# ------------------------------------------------------------
# Simulate hand motion: +6 ft, stop, -6 ft
# ------------------------------------------------------------
def simulate_accel(t):
    quarter = stopTime / 4

    if quarter < t < 2*quarter:
        return 1.5
    elif 2*quarter <= t < 3*quarter:
        return -1.5
    elif 3*quarter < t < 4*quarter:
        return 1.5
    else:
        return 0.0


def read_noisy_accel(t):
    true_acc = simulate_accel(t)
    noise = np.random.normal(0.0, m.sqrt(accel_noise_var))
    return true_acc + accel_bias + noise


# ------------------------------------------------------------
# Kalman Filter setup
# ------------------------------------------------------------
kf = KalmanFilter(dim_x=3, dim_z=1)

kf.F = np.array([
    [1., dt, 0.5*dt*dt],
    [0., 1.,        dt],
    [0., 0.,        1.]
])

kf.H = np.array([[0., 0., 1.]])

# Initial state (at rest)
kf.x = np.array([0., 0., 0.])

kf.P = np.diag([1., 1., 1.])

# Process noise (acceleration random walk)
sigma_a = 0.5
# kf.Q = sigma_a**2 * np.array([
#     [0.25*dt**4, 0.5*dt**3, 0.5*dt**2],
#     [0.5*dt**3,     dt**2,       dt],
#     [0.5*dt**2,        dt,        1]
# ])
kf.Q = Q_discrete_white_noise(
    dim=3,
    dt=dt,
    var=sigma_a**2
)

kf.R = np.array([[accel_noise_var]])

# ------------------------------------------------------------
# Bias calibration (sensor held still initially)
# ------------------------------------------------------------
calib_samples = 200
calib_data = [read_noisy_accel(0.0) for _ in range(calib_samples)]
estimated_bias = np.mean(calib_data)

print("Estimated accelerometer bias:", estimated_bias)

# ------------------------------------------------------------
# Run filter
# ------------------------------------------------------------
xs = []
vs = []
accs = []
zs = []

true_x = 0.0
true_v = 0.0
true_pos = []

for t in taxis:
    z = read_noisy_accel(t) - estimated_bias

    # Ground truth integration (for error plot)
    a_true = simulate_accel(t)
    true_v += a_true * dt
    true_x += true_v * dt
    true_pos.append(true_x)

    kf.predict()
    kf.update(z)

    zs.append(z)
    xs.append(kf.x[0])
    vs.append(kf.x[1])
    accs.append(kf.x[2])

# ------------------------------------------------------------
# Error analysis
# ------------------------------------------------------------
pos_error = np.array(xs) - np.array(true_pos)
final_error = xs[-1]

print("Final position estimate (m):", xs[-1])
print("Final position error (m):", final_error)

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
plt.figure()
plt.plot(taxis, xs, label='KF position estimate')
plt.plot(taxis, true_pos, label='Ground truth')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position Estimate vs Time')
plt.grid()
plt.legend()


plt.figure()
plt.plot(taxis, pos_error, label='Position error')
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
plt.title('Position Estimation Error')
plt.grid()
plt.legend()

plt.figure()
plt.plot(taxis, accs, label='Estimated acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Acceleration: Estimated')
plt.grid()
plt.legend()

plt.figure()
plt.plot(taxis, zs, label='Measured acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Acceleration: Measured')
plt.grid()
plt.legend()

plt.show()
