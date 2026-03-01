import time
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

N_BITS_SWEEP = [2, 3, 4, 5, 6, 7]
FRAMES_PER_TEST = 2

RUN_LIVE_DEBUG = True
LIVE_DEBUG_N_BITS = 5
LIVE_DEBUG_FRAMES = 10000


def save_crosscorr_plot(xcor_raw, shift, n_bits, frame_idx, out_dir="plots"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    n_sense, L = xcor_raw.shape
    fig, axes = plt.subplots(n_sense, 1, figsize=(10, 12), sharex=True)

    drive_marks = [i * shift for i in range(len(DRIVE_PINS))]

    for j in range(n_sense):
        ax = axes[j]
        ax.plot(np.arange(L), xcor_raw[j])
        for m in drive_marks:
            ax.axvline(m, linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"S{j}")

    axes[-1].set_xlabel("Lag (samples)")
    fig.suptitle(f"Cross-correlations (n_bits={n_bits}, L={L}) frame={frame_idx}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(f"{out_dir}/xcor_n{n_bits}_frame{frame_idx:05d}.png", dpi=150)
    plt.close(fig)


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


def measure_fps_for_nbits(ADC, n_bits, frames=2):
    tap_mask = TAP_MASKS[n_bits]
    L = (1 << n_bits) - 1

    prbs0_bits = np.array(lfsr_prbs(n_bits, tap_mask, SEED), dtype=np.int8)
    prbs0_pm1 = bits_to_pm1(prbs0_bits)
    prbs_mat, shift = make_prbs_matrix(prbs0_bits, len(DRIVE_PINS))

    t0 = time.time()
    for _ in range(frames):
        raw = acquire_frame(ADC, DRIVE_PINS, SENSE_PINS, prbs_mat, L)
        _ = correlate_frame(prbs0_pm1, raw, shift, len(DRIVE_PINS))
    dt = time.time() - t0

    fps = frames / dt if dt > 0 else 0.0
    return L, tap_mask, fps


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
    ax.set_ylabel("Sense index (0..7)")
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


def main():
    ADC = adc_setup(SETUP_SPI, SET_CHANNEL)
    setup_gpio(DRIVE_PINS)

    print("n_bits   L=2^n-1   tap_mask   fps")
    results = []

    for n_bits in N_BITS_SWEEP:
        if n_bits not in TAP_MASKS:
            continue
        L, tap_mask, fps = measure_fps_for_nbits(ADC, n_bits, frames=FRAMES_PER_TEST)
        results.append((n_bits, L, tap_mask, fps))
        print(f"{n_bits:5d}  {L:8d}   0x{tap_mask:04X}   {fps:6.3f}")

    if RUN_LIVE_DEBUG:
        run_live_debug(ADC, LIVE_DEBUG_N_BITS, frames=LIVE_DEBUG_FRAMES)

    GPIO.cleanup()


if __name__ == "__main__":
    main()