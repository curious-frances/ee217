import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import RPi.GPIO as GPIO

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

SEED = 0x01

# Live debug plot
RUN_LIVE_DEBUG = False
LIVE_DEBUG_N_BITS = 5
LIVE_DEBUG_FRAMES = 100

# Data logging (NEW)
LOG_ENABLE = True
LOG_CSV_PATH = "touch_frames.csv"
LOG_N_BITS = 5                # PRBS length for logging
LOG_FRAMES = 500             # how many frames to log
LOG_MIN_CLIP = 0.0             # baseline subtract on heatmap before centroid (often 0 is fine)
LOG_BG_FRACTION = 0.70         # background = bottom 70% of weights (used for bg_std)

# Still-touch jitter test (NEW)
RUN_STILL_TEST = False
STILL_FRAMES = 1000


# ---------- Helpers ----------
def make_prbs_matrix(prbs0_bits, n_drive):
    L = len(prbs0_bits)
    shift = L // n_drive
    prbs_mat = np.vstack([np.roll(prbs0_bits, i * shift) for i in range(n_drive)])
    return prbs_mat, shift

def drive_outputs(drive_pins, prbs_mat, s_idx):
    for i, pin in enumerate(drive_pins):
        GPIO.output(pin, int(prbs_mat[i, s_idx]))

def acquire_frame(ADC, drive_pins, sense_pins, prbs_mat, L):
    raw = np.zeros((len(sense_pins), L), dtype=float)
    for j, ch in enumerate(sense_pins):
        for s in range(L):
            drive_outputs(drive_pins, prbs_mat, s)
            raw[j, s] = read_adc_volts(ADC, ch)
    return raw

def correlate_frame(prbs0_pm1, raw_sense, shift, n_drive):
    n_sense = raw_sense.shape[0]
    xcor_map = np.zeros((n_sense, n_drive), dtype=float)
    for j in range(n_sense):
        r = circular_cross_correlation(prbs0_pm1, raw_sense[j])
        xcor_map[j, :] = [r[i * shift] for i in range(n_drive)]
    return xcor_map

def preprocess_heatmap(H, min_clip=0.0):
    W = np.array(H, dtype=float)
    W = np.clip(W - float(min_clip), 0.0, None)
    return W

def centroid_and_ellipse(H, min_clip=0.0, eps=1e-12):
    """
    Returns:
      x_meas, y_meas: centroid in grid coords (x=0..4 drive, y=0..6 sense)
      major, minor: 1-sigma axis lengths (grid units)
      theta: radians, orientation of major axis
      strength_sum, strength_max
    """
    W = preprocess_heatmap(H, min_clip=min_clip)
    strength_sum = float(W.sum())
    strength_max = float(W.max()) if W.size else 0.0
    if strength_sum < eps:
        return np.nan, np.nan, np.nan, np.nan, np.nan, strength_sum, strength_max

    ys = np.arange(W.shape[0])[:, None]   # (7,1)
    xs = np.arange(W.shape[1])[None, :]   # (1,5)

    x_meas = float((W * xs).sum() / strength_sum)
    y_meas = float((W * ys).sum() / strength_sum)

    dx = xs - x_meas
    dy = ys - y_meas

    cov_xx = float((W * dx * dx).sum() / strength_sum)
    cov_yy = float((W * dy * dy).sum() / strength_sum)
    cov_xy = float((W * dx * dy).sum() / strength_sum)

    C = np.array([[cov_xx, cov_xy],
                  [cov_xy, cov_yy]], dtype=float)

    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    major = float(np.sqrt(max(evals[0], 0.0)))
    minor = float(np.sqrt(max(evals[1], 0.0)))

    vx, vy = evecs[:, 0]
    theta = float(np.arctan2(vy, vx))

    return x_meas, y_meas, major, minor, theta, strength_sum, strength_max

def noise_features(H, min_clip=0.0, bg_fraction=0.7):
    """
    Useful 'noise-ish' stats for Kalman tuning:
      - bg_std: std of background nodes
      - snr_proxy: peak / (bg_std + eps)
    """
    W = preprocess_heatmap(H, min_clip=min_clip).flatten()
    if W.size == 0:
        return np.nan, np.nan

    # Define "background" as the lowest bg_fraction of values
    k = int(max(1, np.floor(bg_fraction * W.size)))
    bg = np.partition(W, k-1)[:k]
    bg_std = float(np.std(bg))

    peak = float(np.max(W))
    snr_proxy = float(peak / (bg_std + 1e-9))
    return bg_std, snr_proxy

def csv_header(n_sense=7, n_drive=5):
    cols = [
        "t", "frame",
        "x_meas", "y_meas",
        "major_sigma", "minor_sigma", "theta",
        "strength_sum", "strength_max",
        "bg_std", "snr_proxy",
    ]
    # add heatmap values (row-major)
    for r in range(n_sense):
        for c in range(n_drive):
            cols.append(f"xcor_r{r}_c{c}")
    return cols

def append_row(writer, t, frame_idx, H, x_meas, y_meas, major, minor, theta, ssum, smax, bg_std, snr_proxy):
    row = {
        "t": t,
        "frame": frame_idx,
        "x_meas": x_meas,
        "y_meas": y_meas,
        "major_sigma": major,
        "minor_sigma": minor,
        "theta": theta,
        "strength_sum": ssum,
        "strength_max": smax,
        "bg_std": bg_std,
        "snr_proxy": snr_proxy,
    }
    for r in range(H.shape[0]):
        for c in range(H.shape[1]):
            row[f"xcor_r{r}_c{c}"] = float(H[r, c])
    writer.writerow(row)

def jitter_rms(xs, ys):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    if len(xs) == 0 or len(ys) == 0:
        return np.nan, np.nan, np.nan
    x0 = xs.mean()
    y0 = ys.mean()
    dx = xs - x0
    dy = ys - y0
    rms_x = float(np.sqrt(np.mean(dx*dx)))
    rms_y = float(np.sqrt(np.mean(dy*dy)))
    rms_r = float(np.sqrt(np.mean(dx*dx + dy*dy)))
    return rms_x, rms_y, rms_r


def run_live_debug(ADC, n_bits, frames=50):
    tap_mask = TAP_MASKS[n_bits]
    L = (1 << n_bits) - 1

    prbs0_bits = np.array(lfsr_prbs(n_bits, tap_mask, SEED), dtype=np.int8)
    prbs0_pm1 = bits_to_pm1(prbs0_bits)
    prbs_mat, shift = make_prbs_matrix(prbs0_bits, len(DRIVE_PINS))

    plt.ion()
    fig, ax = plt.subplots()
    heat = ax.imshow(np.zeros((len(SENSE_PINS), len(DRIVE_PINS))),
                     interpolation="nearest", aspect="auto")
    plt.colorbar(heat, ax=ax)
    ax.set_title(f"Correlator outputs (7x5) — n_bits={n_bits}, L={L}")
    ax.set_xlabel("Drive index (0..4)")
    ax.set_ylabel("Sense index (0..6)")
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(ax.bbox)

    t_start = time.time()
    for k in range(frames):
        t0 = time.time()

        raw = acquire_frame(ADC, DRIVE_PINS, SENSE_PINS, prbs_mat, L)
        xcor = correlate_frame(prbs0_pm1, raw, shift, len(DRIVE_PINS))

        heat.set_data(xcor)
        fig.canvas.restore_region(bg)
        ax.draw_artist(heat)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()

        t1 = time.time()
        fps_inst = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0
        fps_avg = (k + 1) / (t1 - t_start)
        if (k % 5) == 0:
            print(f"[LIVE n_bits={n_bits}, L={L}] frame={k+1:3d}/{frames} | fps_inst={fps_inst:.2f} | fps_avg={fps_avg:.2f}")

    plt.ioff()
    plt.show()


def log_frames_to_csv(ADC, n_bits, frames, csv_path):
    tap_mask = TAP_MASKS[n_bits]
    L = (1 << n_bits) - 1

    prbs0_bits = np.array(lfsr_prbs(n_bits, tap_mask, SEED), dtype=np.int8)
    prbs0_pm1 = bits_to_pm1(prbs0_bits)
    prbs_mat, shift = make_prbs_matrix(prbs0_bits, len(DRIVE_PINS))

    print(f"=== Logging frames to {csv_path} (n_bits={n_bits}, L={L}, frames={frames}) ===")
    print("Tip: keep your touch steady; for a 'still' test, tape your finger down or touch a coin while touching it with your finger.")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header(len(SENSE_PINS), len(DRIVE_PINS)))
        writer.writeheader()

        xs, ys = [], []
        t0 = time.time()

        for k in range(frames):
            raw = acquire_frame(ADC, DRIVE_PINS, SENSE_PINS, prbs_mat, L)
            xcor = correlate_frame(prbs0_pm1, raw, shift, len(DRIVE_PINS))

            x_meas, y_meas, major, minor, theta, ssum, smax = centroid_and_ellipse(
                xcor, min_clip=LOG_MIN_CLIP
            )
            bg_std, snr_proxy = noise_features(
                xcor, min_clip=LOG_MIN_CLIP, bg_fraction=LOG_BG_FRACTION
            )

            t = time.time()
            append_row(writer, t=t, frame_idx=k, H=xcor,
                       x_meas=x_meas, y_meas=y_meas,
                       major=major, minor=minor, theta=theta,
                       ssum=ssum, smax=smax,
                       bg_std=bg_std, snr_proxy=snr_proxy)

            xs.append(x_meas)
            ys.append(y_meas)

            if (k % 50) == 0 and k > 0:
                fps = (k + 1) / (time.time() - t0)
                print(f"logged {k+1}/{frames} | avg fps={fps:.2f} | last centroid=({x_meas:.2f},{y_meas:.2f}) | snr~{snr_proxy:.1f}")

    print(f"Done. Wrote {frames} frames -> {csv_path}")


def still_touch_stats_from_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    xs = df["x_meas"].to_numpy()
    ys = df["y_meas"].to_numpy()
    rms_x, rms_y, rms_r = jitter_rms(xs, ys)
    print("=== Still-touch jitter stats (grid units) ===")
    print(f"RMS jitter: x={rms_x:.4f}, y={rms_y:.4f}, radial={rms_r:.4f}")
    print("Interpretation: this RMS sets your positional resolution floor (in grid cells).")


def main():
    ADC = adc_setup(SETUP_SPI, SET_CHANNEL)
    setup_gpio(DRIVE_PINS)

    try:
        if LOG_ENABLE:
            log_frames_to_csv(ADC, LOG_N_BITS, LOG_FRAMES, LOG_CSV_PATH)

            if RUN_STILL_TEST:
                # If you only want 1000 still samples, set LOG_FRAMES=1000 and RUN_STILL_TEST=True
                still_touch_stats_from_csv(LOG_CSV_PATH)

        if RUN_LIVE_DEBUG:
            print("=== Live correlator debug (heatmap) ===")
            run_live_debug(ADC, LIVE_DEBUG_N_BITS, frames=LIVE_DEBUG_FRAMES)

    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()