#!/usr/bin/env python3
"""
EE217 Lab2 — Part 2 (Last bullet): Kalman filter smoothing + velocity

What this file does:
- Runs your sensing pipeline:
  1) Scan 7 sense channels sequentially across one PRBS period (restart PRBS per sense line)
  2) Compute full circular cross-correlation per sense line (7 x L)
  3) Sample correlation at the 5 drive phase offsets -> 7x5 map
  4) Build a baseline (no-touch) map, subtract baseline to get delta map
  5) Threshold delta map -> touch map, compute centroid (grid coords)
- Applies a Kalman filter on [x, y, vx, vy] to smooth centroid and estimate velocity.

Requires util.py providing:
  lfsr_prbs, bits_to_pm1, circular_cross_correlation,
  setup_gpio, adc_setup, read_adc_volts

Run:
  python3 kalman_touch.py
"""

import time
import numpy as np
import RPi.GPIO as GPIO

# Optional live plots (needs X11)
USE_GUI = True
if USE_GUI:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt


from util import (
    lfsr_prbs,
    bits_to_pm1,
    circular_cross_correlation,
    setup_gpio,
    adc_setup,
    read_adc_volts,
)

# ---------------- Hardware config ----------------
DRIVE_PINS = [20, 21, 12, 16, 7]
SENSE_PINS = [7, 6, 5, 4, 3, 2, 1]

SETUP_SPI = True
SET_CHANNEL = False

# ---------------- PRBS config ----------------
SEED = 0x01
TAP_MASKS = {
    2:  0x3,
    3:  0x6,
    4:  0xC,
    5:  0x14,
    6:  0x30,
    7:  0x60,
    8:  0xB8,
    9:  0x110,
    10: 0x240,
    11: 0x500,
    12: 0xE08,
}
N_BITS = 6  # choose 5/6 for higher FPS, 8/9 for more coding gain (slower)

# ---------------- Baseline + touch detection ----------------
USE_BASELINE = True
BASELINE_FRAMES = 50          # collect baseline with NO TOUCH
THRESHOLD = 5.0               # threshold on (xcor_map - baseline). Tune based on your magnitudes.
TOPK = 5                      # centroid computed from top-K cells in touch map

# ---------------- Kalman tuning ----------------
# If filter is too laggy: increase PROCESS_VAR
# If still jittery: increase MEAS_VAR
PROCESS_VAR = 0.2             # process noise (acceleration uncertainty)
MEAS_VAR = 0.05               # measurement noise (centroid noise)

PRINT_EVERY = 10              # print every N frames


# ================= Kalman Filter =================
class KalmanXYVel:
    """
    State: [x, y, vx, vy]^T
    Measurement: [x, y]^T
    Constant-velocity model
    """
    def __init__(self, process_var=0.2, meas_var=0.05):
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float) * 1.0

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        self.R = np.eye(2, dtype=float) * meas_var
        self.process_var = process_var

        self.last_t = None
        self.initialized = False

    def _Q(self, dt):
        q = self.process_var
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return q * np.array([
            [dt4/4, 0,     dt3/2, 0],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0],
            [0,     dt3/2, 0,     dt2],
        ], dtype=float)

    def predict(self, t_now):
        if self.last_t is None:
            self.last_t = t_now
            return

        dt = t_now - self.last_t
        if dt <= 0:
            dt = 1e-3

        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0,  dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=float)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

        self.last_t = t_now

    def update(self, z_xy, t_now):
        z = np.array([[float(z_xy[0])], [float(z_xy[1])]], dtype=float)

        if not self.initialized:
            self.x[0, 0] = z[0, 0]
            self.x[1, 0] = z[1, 0]
            self.x[2, 0] = 0.0
            self.x[3, 0] = 0.0
            self.P = np.eye(4) * 0.5
            self.last_t = t_now
            self.initialized = True
            return self.x.copy()

        self.predict(t_now)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy()

    def reset(self):
        self.x[:] = 0
        self.P[:] = np.eye(4) * 1.0
        self.last_t = None
        self.initialized = False


# ================= Pipeline helpers =================
def make_prbs_matrix(prbs0_bits, n_drive):
    L = len(prbs0_bits)
    shift = L // n_drive
    prbs_mat = np.vstack([np.roll(prbs0_bits, i * shift) for i in range(n_drive)])
    return prbs_mat, shift


def drive_outputs(prbs_mat, s_idx):
    for i, pin in enumerate(DRIVE_PINS):
        GPIO.output(pin, int(prbs_mat[i, s_idx]))


def acquire_frame(ADC, prbs_mat, L):
    """
    Spec-compliant: scan each sense line across full PRBS length, restarting PRBS per sense line.
    raw shape: (7, L)
    """
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
    """
    Sample full correlation at phase offsets i*shift -> 7x5 map
    """
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


# ================= Main =================
def main():
    if N_BITS not in TAP_MASKS:
        raise ValueError(f"N_BITS={N_BITS} not in TAP_MASKS")

    tap_mask = TAP_MASKS[N_BITS]
    L = (1 << N_BITS) - 1

    ADC = adc_setup(SETUP_SPI, SET_CHANNEL)
    setup_gpio(DRIVE_PINS)

    prbs0_bits = np.array(lfsr_prbs(N_BITS, tap_mask, SEED), dtype=np.int8)
    prbs0_pm1 = bits_to_pm1(prbs0_bits)
    prbs_mat, shift = make_prbs_matrix(prbs0_bits, len(DRIVE_PINS))

    print(f"[kalman] n_bits={N_BITS}, L={L}, tap_mask=0x{tap_mask:X}, shift={shift}")
    print(f"[kalman] threshold={THRESHOLD}, baseline={USE_BASELINE} ({BASELINE_FRAMES} frames)")
    print(f"[kalman] KF process_var={PROCESS_VAR}, meas_var={MEAS_VAR}")

    # ---------- baseline ----------
    baseline = np.zeros((len(SENSE_PINS), len(DRIVE_PINS)), dtype=float)
    if USE_BASELINE:
        print("[kalman] Collecting baseline (NO TOUCH)...")
        for i in range(BASELINE_FRAMES):
            raw = acquire_frame(ADC, prbs_mat, L)
            xcor_raw = correlate_full(prbs0_pm1, raw)
            xcor_map = sample_xcor_map(xcor_raw, shift)
            baseline += xcor_map
            if (i % 10) == 0:
                print(f"  baseline frame {i}/{BASELINE_FRAMES}")
        baseline /= float(BASELINE_FRAMES)
        print("[kalman] Baseline done. Now slide finger slowly across sensor.")

    # ---------- Kalman ----------
    kf = KalmanXYVel(process_var=PROCESS_VAR, meas_var=MEAS_VAR)

    # ---------- optional GUI ----------
    if USE_GUI:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Centroid: raw vs filtered (grid coords)")
        ax.set_xlim(-0.5, len(DRIVE_PINS) - 0.5)
        ax.set_ylim(-0.5, len(SENSE_PINS) - 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()  # matches many heatmap conventions (optional)
        raw_pt = ax.plot([], [], marker="o", linestyle="None", label="raw")[0]
        filt_pt = ax.plot([], [], marker="x", linestyle="None", label="filtered")[0]
        vel_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
        ax.legend(loc="lower right")
        fig.canvas.draw()
        fig.show()

    # ---------- loop ----------
    frame = 0
    t_start = time.time()
    try:
        while True:
            t0 = time.time()

            raw = acquire_frame(ADC, prbs_mat, L)
            xcor_raw = correlate_full(prbs0_pm1, raw)
            xcor_map = sample_xcor_map(xcor_raw, shift)

            # delta map
            delta = xcor_map - baseline if USE_BASELINE else xcor_map.copy()

            # touch map by threshold
            touch_map = delta.copy()
            touch_map[touch_map < THRESHOLD] = 0.0

            c = centroid_topk(touch_map, k=TOPK)  # (x,y) or None

            if c is None:
                # No touch detected: optionally reset filter
                # kf.reset()
                if USE_GUI:
                    raw_pt.set_data([], [])
                    filt_pt.set_data([], [])
                    vel_txt.set_text("no touch")
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

            else:
                state = kf.update(c, time.time()).flatten()
                x_f, y_f, vx, vy = state
                speed = float(np.sqrt(vx * vx + vy * vy))

                if (frame % PRINT_EVERY) == 0:
                    fps_avg = (frame + 1) / (time.time() - t_start)
                    print(
                        f"[frame {frame:6d}] raw=({c[0]:.3f},{c[1]:.3f}) "
                        f"filt=({x_f:.3f},{y_f:.3f}) "
                        f"v=({vx:.3f},{vy:.3f}) speed={speed:.3f} grid/s "
                        f"fps_avg={fps_avg:.2f}"
                    )

                if USE_GUI:
                    raw_pt.set_data([c[0]], [c[1]])
                    filt_pt.set_data([x_f], [y_f])
                    vel_txt.set_text(f"v=({vx:.2f},{vy:.2f})  |v|={speed:.2f} grid/s")
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

            frame += 1

            # small yield so GUI stays responsive (optional)
            if USE_GUI:
                time.sleep(0.001)

    finally:
        GPIO.cleanup()
        if USE_GUI:
            plt.ioff()


if __name__ == "__main__":
    main()