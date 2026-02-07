from ICM20948 import ICM20948, Accel, Gyro
import time
import numpy as np
from datetime import datetime
import csv

ACCEL_FULL_SCALE = 2.0      # +-2g
GYRO_FULL_SCALE  = 1000.0   # +-1000 dps
ADC_RESOLUTION   = 32768.0  # 16-bit
g = 9.81                   # m/s^2

ACCEL_SCALE = ACCEL_FULL_SCALE / ADC_RESOLUTION
GYRO_SCALE  = GYRO_FULL_SCALE  / ADC_RESOLUTION

TARGET_HZ = 100
CALIBRATION_TIME_S = 8
COLLECTION_TIME = 100


def read_sensor(imu: ICM20948):
    imu.icm20948_Gyro_Accel_Read()

    ax = Accel[0] * ACCEL_SCALE * g
    ay = Accel[1] * ACCEL_SCALE * g
    az = Accel[2] * ACCEL_SCALE * g

    gx = Gyro[0] * GYRO_SCALE  # deg/s
    gy = Gyro[1] * GYRO_SCALE
    gz = Gyro[2] * GYRO_SCALE

    return (ax, ay, az), (gx, gy, gz)



def calibrate_accel_bias(imu: ICM20948, calibration_time_s=CALIBRATION_TIME_S, fs=TARGET_HZ):
    print("Calibrating accelerometer... keep IMU stationary.")
    
    dt = 1.0 / fs
    n = max(1, int(calibration_time_s * fs))

    accel_x = []
    accel_y = []
    accel_z = []
    gyro_x = []
    gyro_y = []
    gyro_z = []

    for _ in range(n):
        (ax, ay, az), (gx, gy, gz) = read_sensor(imu)
        accel_x.append(ax)
        accel_y.append(ay)
        accel_z.append(az)
        gyro_x.append(gx)
        gyro_y.append(gy)
        gyro_z.append(gz)
        time.sleep(dt)
    acc = np.array([accel_x, accel_y, accel_z])
    gyro = np.array([gyro_x, gyro_y, gyro_z])
        
   
    gyro_bias = tuple(float(np.mean(v)) for v in (gyro_x, gyro_y, gyro_z))
    accel_bias = tuple(float(np.mean(v)) for v in (accel_x, accel_y, accel_z))
    accel_var_x = np.var(acc[0])
    accel_var_y = np.var(acc[1])
    accel_var_z = np.var(acc[2])
    accel_var = tuple(float(np.var(v)) for v in acc)    
    gyro_var_x = np.var(gyro[0])
    gyro_var_y = np.var(gyro[1])
    gyro_var_z = np.var(gyro[2])
    gyro_var = tuple(float(np.var(v)) for v in gyro)
    
    
    print(f"Calibration complete. Accel bias (m/s^2): {accel_bias}")
    return accel_bias, gyro_bias, accel_var, gyro_var


def main():
    print("\nCollecting data from ICM-20948 sensor\n")
    imu = ICM20948()

    accel_bias, gyro_bias, accel_var, gyro_var = calibrate_accel_bias(imu)

    dt_target = 1.0 / TARGET_HZ
    start_time = time.time()

    timestamp = datetime.now().strftime("imu_%Y_%m_%d_%H_%M_%S")
    filename_data = f"{timestamp}_{accel_bias[0]:.03}.csv"
    filename_metadata = f"{timestamp}_metadata_{accel_bias[0]:.03}.txt"
    
    with open(filename_metadata, "w") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Accel Bias: {accel_bias}\n")
        f.write(f"Gyro Bias: {gyro_bias}\n")
        f.write(f"accel_variance: {accel_var}\n")
        f.write(f"gyro_variance: {gyro_var}\n")

    print(f"Logging to: {filename_data}\n")
    print("Press Ctrl+C to stop.\n")

    with open(filename_data, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time",
            "ax_raw", "ay_raw", "az_raw",
            "ax", "ay", "az",
            "gx", "gy", "gz"
        ])

        
        while True:
            loop_start = time.time()

            (ax, ay, az), (gx, gy, gz) = read_sensor(imu)
            a_raw = np.array([ax, ay, az], dtype=float)
            a = a_raw - accel_bias

            t = loop_start - start_time

            w.writerow([
                    f"{t:.6f}",
                    f"{a_raw[0]:.6f}", f"{a_raw[1]:.6f}", f"{a_raw[2]:.6f}",
                    f"{a[0]:.6f}",     f"{a[1]:.6f}",     f"{a[2]:.6f}",
                    f"{gx:.6f}", f"{gy:.6f}", f"{gz:.6f}"
            ])

            # Stop after collection time
            if COLLECTION_TIME is not None and t >= COLLECTION_TIME:
                break

            # Maintain fixed Hz
            loop_dur = time.time() - loop_start
            sleep_left = dt_target - loop_dur
            if sleep_left > 0:
                time.sleep(sleep_left)

        print("Done!")


if __name__ == "__main__":
    main()
