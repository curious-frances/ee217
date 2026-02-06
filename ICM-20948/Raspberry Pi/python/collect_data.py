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
COLLECTION_TIME = 20


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

    acc = np.zeros(3, dtype=float)

    for _ in range(n):
        (ax, ay, az), _ = read_sensor(imu)
        acc += np.array([ax, ay, az], dtype=float)
        time.sleep(dt)

    bias = acc / n
    print(f"Calibration complete. Accel bias (m/s^2): {bias}")
    return bias


def main():
    print("\nCollecting data from ICM-20948 sensor\n")
    imu = ICM20948()

    bias = calibrate_accel_bias(imu)

    dt_target = 1.0 / TARGET_HZ
    start_time = time.time()

    timestamp = datetime.now().strftime("imu_%Y_%m_%d_%H_%M_%S")
    filename = f"{timestamp}.csv"

    print(f"Logging to: {filename}")
    print("Press Ctrl+C to stop.\n")

    with open(filename, "w", newline="") as f:
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
            a = a_raw - bias

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
