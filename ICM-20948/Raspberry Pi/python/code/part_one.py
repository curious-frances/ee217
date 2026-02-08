from ICM20948 import ICM20948, Accel, Gyro, Mag
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
from filterpy.kalman import KalmanFilter


ACCEL_FULL_SCALE = 2.0 #+-2g
GYRO_FULL_SCALE = 1000 #+-1000 DPS
ADC_RESOLUTION = 32768.0 #16 bit
g = 9.81 #m/s^2
g_masured = 10.070282 #m/s^2 measured value
SCALE_FIX = g / g_masured

# Data vars
data_accel = []
data_accel_raw = []
data_gyro  = []
data_comp_pitch = []
data_raw_comp_pitch = []
time_data = []
COLLECTION_TIME = 120 # seconds 2400 3600

# Arrays for part one test 
accel_pitch_data = []
accel_roll_data = []

gyro_pitch_data = []
gyro_roll_data = []

# allen deviation vars
gyro_x_data = []
gyro_y_data = []
gyro_z_data = []

accel_x_data = []
accel_y_data = []
accel_z_data = []



SAMPLE_RATE = 10 #Hz
NOISE_SCALE = 0.2 # Attenuation 80%
NOISE_THRESHOLD = 5 # Accel Noise Threshold

ACCEL_SCALE = ACCEL_FULL_SCALE / ADC_RESOLUTION  #+-2g range
GYRO_SCALE = GYRO_FULL_SCALE / ADC_RESOLUTION  #+-1000 DPS

# 1D Kalman Filter vars 
BIAS_X = 0
VAR_AX = 0
BIAS_GY = 0
CALIBRATION_TIME = 5 # seconds 
POSITION_DATA = []


def sensor_calibration(run_time=CALIBRATION_TIME):
    print("Initializing ICM-20948...")
    imu = ICM20948()
    print("Calibrating sensors... Please keep the IMU stationary.")
    ax_samples = []
    az_samples = []
    gy_samples = []
    init_time = time.time()
    while time.time() - init_time < run_time:
        (ax,ay,az), (gx,gy,gz) = read_sensor(imu)
        ax_samples.append(ax)
        az_samples.append(az)
        gy_samples.append(gy)
        time.sleep(1.0 / SAMPLE_RATE)

    ax_samples = np.array(ax_samples)
    az_samples = np.array(az_samples)
    gy_samples = np.array(gy_samples)
    BIAS_X = float(np.mean(ax_samples))
    VAR_AX = float(np.var(ax_samples, ddof=1))  # ddof=1 for sample variance
    BIAS_GY = float(np.mean(gy_samples))
    print("Calibration complete.")
    print("Bias X:", BIAS_X)
    print("Variance AX:", VAR_AX)
    az_mean = float(np.mean(az_samples))
    print(f"calibration: mean az={az_mean:.3f} m/s^2 (should be near {g:.3f})")

    return BIAS_X, VAR_AX, BIAS_GY

def write_data_to_csv(filename, time_data, accel_x_data, position_data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time (s)', 'Filtered Accel X (m/s^2)', 'Position (m)'])
        for t, d, p in zip(time_data, accel_x_data, position_data):
            writer.writerow([t, d, p])


def collect_allen_deviation_data():
    print("Initializing ICM-20948...")
    imu = ICM20948()
    start_time = time.time()

    while time.time() - start_time < COLLECTION_TIME:
        accel, gyro = read_sensor(imu)
        time_data.append(time.time() - start_time)
        # Store data for Allan Deviation
        gyro_x_data.append(gyro[0])
        gyro_y_data.append(gyro[1])
        gyro_z_data.append(gyro[2])
        accel_x_data.append(accel[0])
        accel_y_data.append(accel[1])
        accel_z_data.append(accel[2])

        time.sleep(1.0 / SAMPLE_RATE)
    # Estimate sample rate
    dt = np.diff(np.array(time_data))
    print("Estimated fs:", 1/np.median(dt))
    print("dt std/mean:", np.std(dt)/np.mean(dt))

def measure_allan_deviation(gyro_x_data, gyro_y_data, gyro_z_data,
                            accel_x_data, accel_y_data, accel_z_data,
                            time_data, num_clusters=100):
    gyro_x_arr = np.array(gyro_x_data)
    gyro_y_arr = np.array(gyro_y_data)
    gyro_z_arr = np.array(gyro_z_data)

    accel_x_arr = np.array(accel_x_data)
    accel_y_arr = np.array(accel_y_data)
    accel_z_arr = np.array(accel_z_data)

    time_arr = np.array(time_data)

    ts = float(np.median(np.diff(time_arr)))

    # Calculate Allan Deviation
    theta_x = np.cumsum(gyro_x_arr) * ts
    theta_y = np.cumsum(gyro_y_arr) * ts
    theta_z = np.cumsum(gyro_z_arr) * ts

    tau_gx, ad_gx = AllanDeviation(theta_x, ts, num_clusters=num_clusters)
    tau_gy, ad_gy = AllanDeviation(theta_y, ts, num_clusters=num_clusters)
    tau_gz, ad_gz = AllanDeviation(theta_z, ts, num_clusters=num_clusters)

    # Integrate acceleration to get velocity seem to have worse allan deviation
    vel_x = np.cumsum(accel_x_arr) * ts
    vel_y = np.cumsum(accel_y_arr) * ts
    vel_z = np.cumsum(accel_z_arr) * ts

    tau_ax, ad_ax = AllanDeviation(vel_x, ts, num_clusters=num_clusters)
    tau_ay, ad_ay = AllanDeviation(vel_y, ts, num_clusters=num_clusters)
    tau_az, ad_az = AllanDeviation(vel_z, ts, num_clusters=num_clusters)

    return (tau_gx, ad_gx, tau_gy, ad_gy, tau_gz, ad_gz,
            tau_ax, ad_ax, tau_ay, ad_ay, tau_az, ad_az)
    

def plot_allen_deviation(tau_gx, ad_gx, tau_gy, ad_gy, tau_gz, ad_gz,
                          tau_ax, ad_ax, tau_ay, ad_ay, tau_az, ad_az):
    plt.figure(figsize=(8, 5))
    plt.title("Gyro Allan Deviation")
    plt.plot(tau_gx, ad_gx, label="gx")
    plt.plot(tau_gy, ad_gy, label="gy")
    plt.plot(tau_gz, ad_gz, label="gz")
    plt.xlabel(r"$\tau$ [s]")
    plt.ylabel("Deviation [deg]")  
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", color="0.65")
    plt.legend()
    plt.savefig("gyro_allan_deviation.png", dpi=300)

    plt.figure(figsize=(8, 5))
    plt.title("Accel Allan Deviation")
    plt.plot(tau_ax, ad_ax, label="ax")
    plt.plot(tau_ay, ad_ay, label="ay")
    plt.plot(tau_az, ad_az, label="az")
    plt.xlabel(r"$\tau$ [s]")
    plt.ylabel("Deviation [m/s]")  
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", color="0.65")
    plt.legend()
    plt.savefig("accel_allan_deviation.png", dpi=300)
    # Save to CSV
    writer = csv.writer(open("gyro_allan_deviation.csv", 'w', newline=''))
    writer.writerow(['Tau (s)', 'AD_gx (deg)', 'AD_gy (deg)', 'AD_gz (deg)'])
    for t, gx, gy, gz in zip(tau_gx, ad_gx, ad_gy, ad_gz):
        writer.writerow([t, gx, gy, gz])
    writer = csv.writer(open("accel_allan_deviation.csv", 'w', newline=''))
    writer.writerow(['Tau (s)', 'AD_ax (m/s)', 'AD_ay (m/s)', 'AD_az (m/s)'])
    for t, ax, ay, az in zip(tau_ax, ad_ax, ad_ay, ad_az):
        writer.writerow([t, ax, ay, az])


def AllanDeviation(data_arr: np.ndarray, ts: float, num_clusters: int=100):
    """
    Algorithm obtained from Mathworks:
    https://www.mathworks.com/help/fusion/ug/inertial-sensor-noise-analysis-using-allan-variance.html

    """
    N = len(data_arr)
    Mmax = 2**np.floor(np.log2(N / 2))
    M = np.logspace(np.log10(1), np.log10(Mmax), num=num_clusters)
    M = np.ceil(M)  # Round up to integer
    M = np.unique(M) # Remove duplicates

    taus = M * ts  # Compute 'cluster durations' tau

    # Compute Allan variance
    allanVar = np.zeros(len(M))
    for i, mi in enumerate(M):
        twoMi = int(2 * mi)
        mi = int(mi)
        allanVar[i] = np.sum(
            (data_arr[twoMi:N] - (2.0 * data_arr[mi:N-mi]) + data_arr[0:N-twoMi])**2
        )
    
    allanVar /= (2.0 * taus**2) * (N - (2.0 * M))
    return (taus, np.sqrt(allanVar))  # Return deviation (dev = sqrt(var))

def plot_data(time_data, data, label, title):
    # Save to CSV
    writer = csv.writer(open(f"{title.replace(' ', '_').lower()}.csv", 'w', newline=''))
    writer.writerow(['Time (s)', label])
    for t, d in zip(time_data, data):
        writer.writerow([t, d])

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, data, label=label)
    plt.xlabel('Time (s)')
    plt.ylabel(label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")

def complimetary_filter(alpha, gyro_angle, accel_angle):
    return (alpha * gyro_angle) + ((1 - alpha) * accel_angle)

def gyro_integrate(prev_angle, dt, angular_vel):
    return prev_angle + (angular_vel * dt)

def read_sensor(imu):
    imu.icm20948_Gyro_Accel_Read()

    accel_x = Accel[0] * ACCEL_SCALE * g * SCALE_FIX
    accel_y = Accel[1] * ACCEL_SCALE * g * SCALE_FIX
    accel_z = Accel[2] * ACCEL_SCALE * g * SCALE_FIX

    gyro_x = Gyro[0] * GYRO_SCALE
    gyro_y = Gyro[1] * GYRO_SCALE
    gyro_z = Gyro[2] * GYRO_SCALE
    return (accel_x, accel_y, accel_z), (gyro_x, gyro_y, gyro_z)

def accel_to_angle(accel_x, accel_y, accel_z):
    pitch = math.atan2(accel_x, accel_z) * 180 / math.pi
    roll = math.atan2(accel_y, accel_z) * 180 / math.pi
    return pitch, -roll

def main():
    print("Initializing ICM-20948...")
    imu = ICM20948()
    gyro_pitch, gyro_roll, gyro_yaw = 0,0,0
    prev_accel_pitch, prev_accel_roll = 0,0
    raw_accel_pitch, raw_accel_roll = 0,0
    prev_time = time.time()
    start_time = prev_time
    # dt = 1.0 / SAMPLE_RATE

    while time.time() - start_time < COLLECTION_TIME:
        accel, gyro = read_sensor(imu)
        accel_pitch, accel_roll = accel_to_angle(accel[0], accel[1], accel[2])
        raw_accel_pitch, raw_accel_roll = accel_pitch, accel_roll


        delta_accel_pitch = abs(accel_pitch - prev_accel_pitch)
        delta_accel_roll = abs(accel_roll - prev_accel_roll)

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        gyro_pitch = gyro_integrate(gyro_pitch, dt, -gyro[1])
        gyro_roll = gyro_integrate(gyro_roll, dt, -gyro[0])
        gyro_yaw = gyro_integrate(gyro_yaw, dt, -gyro[2])

        # Attenuate Noise
        if delta_accel_pitch > NOISE_THRESHOLD:
            accel_pitch = accel_pitch * NOISE_SCALE 
        if delta_accel_roll > NOISE_THRESHOLD:
            accel_roll = accel_roll * NOISE_SCALE 

        comp_pitch = complimetary_filter(0.55, gyro_pitch, accel_pitch)
        comp_pitch_no_lp = complimetary_filter(0.55, gyro_pitch, raw_accel_pitch)
        comp_roll = complimetary_filter(0.55, gyro_roll, accel_roll)
        
        prev_accel_pitch, prev_accel_roll = accel_pitch, accel_roll
        data_accel.append(accel_pitch)
        data_accel_raw.append(raw_accel_pitch)
        data_gyro.append(gyro_pitch)
        data_comp_pitch.append(comp_pitch)
        data_raw_comp_pitch.append(comp_pitch_no_lp)


        # data for part one 
        gyro_pitch_data.append(gyro_pitch)
        gyro_roll_data.append(gyro_roll)

        accel_pitch_data.append(raw_accel_pitch)
        accel_roll_data.append(raw_accel_roll)

        time_data.append(time.time() - start_time)
        time.sleep(1.0 / SAMPLE_RATE)
        
if __name__ == "__main__":
    BIAS_X, VAR_AX, BIAS_GY = sensor_calibration()
    write_data_to_csv("groundth_data.csv", time_data, accel_x_data, POSITION_DATA)

    plot_data(time_data, accel_x_data, label="Accel X (m/s^2)", title="Acceleration X")
    plot_data(time_data, POSITION_DATA, label="Position (m)", title="Position X")
    
