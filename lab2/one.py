import time
import numpy as np
import RPi.GPIO as GPIO
from filterpy.common import Q_discrete_white_noise

USE_GUI = True
if USE_GUI:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

from filterpy.kalman import KalmanFilter

from util import (
    lfsr_prbs,
    bits_to_pm1,
    circular_cross_correlation,
    setup_gpio,
    adc_setup,
    read_adc_volts,
)

DRIVE_PINS = [20, 21, 12, 16, 7]
SENSE_PINS = [7, 6, 5, 4, 3, 2, 1]

SETUP_SPI = True
SET_CHANNEL = False

SEED = 0x01
TAP_MASKS = {
    2: 0x3,
    3: 0x6,
    4: 0xC,
    5: 0x14,
    6: 0x30,
    7: 0x60,
    8: 0xB8,
    9: 0x110,
    10: 0x240,
    11: 0x500,
    12: 0xE08,
}
N_BITS = 5

USE_BASELINE = True
BASELINE_FRAMES = 50
THRESHOLD = 5.0
TOPK = 5

PROCESS_VAR = 0.2
MEAS_VAR = 0.0076

PRINT_EVERY = 100


def make_prbs_matrix(prbs0_bits, n_drive):
    L = len(prbs0_bits)
    shift = L // n_drive
    prbs_mat = np.vstack([np.roll(prbs0_bits, i * shift) for i in range(n_drive)])
    return prbs_mat, shift


def drive_outputs(prbs_mat, s_idx):
    for i, pin in enumerate(DRIVE_PINS):
        GPIO.output(pin, int(prbs_mat[i, s_idx]))


def acquire_frame(ADC, prbs_mat, L):
    raw = np.zeros((len(SENSE_PINS), L), dtype=float)
    for j, ch in enumerate(SENSE_PINS):
        for s in range(L):
            drive_outputs(prbs_mat, s)
            raw[j, s] = read_adc_volts(ADC, ch)
    return raw


def correlate_full(prbs0_pm1, raw):
    n_sense, L = raw.shape
    xcor_raw = np.zeros((n_sense, L), dtype=float)
    for j in range(n_sense):
        xcor_raw[j] = circular_cross_correlation(prbs0_pm1, raw[j])
    return xcor_raw


def sample_xcor_map(xcor_raw, shift):
    n_sense = xcor_raw.shape[0]
    xcor_map = np.zeros((n_sense, len(DRIVE_PINS)), dtype=float)
    for j in range(n_sense):
        xcor_map[j, :] = [xcor_raw[j][i * shift] for i in range(len(DRIVE_PINS))]
    return xcor_map


def centroid_topk(touch_map, k=5):
    flat = touch_map.flatten()
    idx = np.argsort(flat)[-k:]
    w = flat[idx]
    if np.sum(w) <= 1e-12:
        return None
    ys, xs = np.unravel_index(idx, touch_map.shape)
    cx = float(np.sum(xs * w) / np.sum(w))
    cy = float(np.sum(ys * w) / np.sum(w))
    return (cx, cy)


def main():
    tap_mask = TAP_MASKS[N_BITS]
    L = (1 << N_BITS) - 1

    ADC = adc_setup(SETUP_SPI, SET_CHANNEL)
    setup_gpio(DRIVE_PINS)

    prbs0_bits = np.array(lfsr_prbs(N_BITS, tap_mask, SEED), dtype=np.int8)
    prbs0_pm1 = bits_to_pm1(prbs0_bits)
    prbs_mat, shift = make_prbs_matrix(prbs0_bits, len(DRIVE_PINS))

    baseline = np.zeros((len(SENSE_PINS), len(DRIVE_PINS)), dtype=float)
    if USE_BASELINE:
        for _ in range(BASELINE_FRAMES):
            raw = acquire_frame(ADC, prbs_mat, L)
            xcor_raw = correlate_full(prbs0_pm1, raw)
            xcor_map = sample_xcor_map(xcor_raw, shift)
            baseline += xcor_map
        baseline /= float(BASELINE_FRAMES)

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.zeros((4, 1), dtype=float)
    kf.H = np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0]], dtype=float)
    kf.P = np.eye(4, dtype=float) * 0.5
    kf.R = np.eye(2, dtype=float) * float(MEAS_VAR)
    kf.F = np.eye(4, dtype=float)
    kf.Q = np.eye(4, dtype=float)

    initialized = False
    last_t = None

    if USE_GUI:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Centroid: raw vs filtered (grid coords)")
        ax.set_xlim(-0.5, len(DRIVE_PINS) - 0.5)
        ax.set_ylim(-0.5, len(SENSE_PINS) - 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        raw_pt = ax.plot([], [], marker="o", linestyle="None", label="raw")[0]
        filt_pt = ax.plot([], [], marker="x", linestyle="None", label="filtered")[0]
        vel_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
        ax.legend(loc="lower right")
        fig.canvas.draw()
        fig.show()

    t_start = time.time()
    frame = 0

    while True:
        raw = acquire_frame(ADC, prbs_mat, L)
        xcor_raw = correlate_full(prbs0_pm1, raw)
        xcor_map = sample_xcor_map(xcor_raw, shift)

        delta = xcor_map - baseline if USE_BASELINE else xcor_map.copy()
        touch_map = delta.copy()
        touch_map[touch_map < THRESHOLD] = 0.0

        c = centroid_topk(touch_map, k=TOPK)

        if c is None:
            if USE_GUI:
                raw_pt.set_data([], [])
                filt_pt.set_data([], [])
                vel_txt.set_text("no touch")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                time.sleep(0.001)
            frame += 1
            continue

        t_now = time.time()

        if not initialized:
            kf.x = np.array([[c[0]], [c[1]], [0.0], [0.0]], dtype=float)
            last_t = t_now
            initialized = True
            x_f, y_f, vx, vy = float(c[0]), float(c[1]), 0.0, 0.0
        else:
            dt = t_now - last_t
            if dt <= 0:
                dt = 1e-3

            kf.F = np.array([[1.0, 0.0, dt, 0.0],
                             [0.0, 1.0, 0.0, dt],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]], dtype=float)

            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt2 * dt2
            q = float(PROCESS_VAR)
            kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=float(PROCESS_VAR), block_size=2)

            kf.predict()
            kf.update(np.array([c[0], c[1]], dtype=float))
            last_t = t_now

            x_f = float(kf.x[0, 0])
            y_f = float(kf.x[1, 0])
            vx = float(kf.x[2, 0])
            vy = float(kf.x[3, 0])

        speed = float(np.sqrt(vx * vx + vy * vy))

        if (frame % PRINT_EVERY) == 0:
            fps_avg = (frame + 1) / (time.time() - t_start)
            print(
                f"[frame {frame:6d}] raw=({c[0]:.3f},{c[1]:.3f}) "
                f"filt=({x_f:.3f},{y_f:.3f}) "
                f"v=({vx:.3f},{vy:.3f}) |v|={speed:.3f} grid/s "
                f"fps_avg={fps_avg:.2f}"
            )

        if USE_GUI:
            raw_pt.set_data([c[0]], [c[1]])
            filt_pt.set_data([x_f], [y_f])
            vel_txt.set_text(f"v=({vx:.2f},{vy:.2f})  |v|={speed:.2f} grid/s")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.001)

        frame += 1


if __name__ == "__main__":
    main()