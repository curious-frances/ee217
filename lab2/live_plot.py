import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
SEED = 0x01

TAP_MASKS = {
    5: 0x14,
    6: 0x30,
    7: 0x60,
    8: 0xB8,
}

N_BITS = 6
N_FRAMES = 300
SAVE_EVERY = 10
OUT_DIR = "plots_live"


def make_prbs_matrix(prbs0_bits, n_drive):
    L = len(prbs0_bits)
    shift = L // n_drive
    prbs_mat = np.vstack([np.roll(prbs0_bits, i * shift) for i in range(n_drive)])
    return prbs_mat, shift


def drive_outputs(prbs_mat, s_idx):
    for i, pin in enumerate(DRIVE_PINS):
        GPIO.output(pin, int(prbs_mat[i, s_idx]))


def acquire_frame(ADC, prbs_mat, L):
    raw = np.zeros((len(SENSE_PINS), L))
    for j, ch in enumerate(SENSE_PINS):
        for s in range(L):
            drive_outputs(prbs_mat, s)
            raw[j, s] = read_adc_volts(ADC, ch)
    return raw


def correlate_full(prbs0_pm1, raw):
    n_sense, L = raw.shape
    xcor = np.zeros((n_sense, L))
    for j in range(n_sense):
        xcor[j] = circular_cross_correlation(prbs0_pm1, raw[j])
    return xcor


def sample_map(xcor_raw, shift):
    xcor_map = np.zeros((len(SENSE_PINS), len(DRIVE_PINS)))
    for j in range(len(SENSE_PINS)):
        xcor_map[j] = [xcor_raw[j][i * shift] for i in range(len(DRIVE_PINS))]
    return xcor_map


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tap_mask = TAP_MASKS[N_BITS]
    L = (1 << N_BITS) - 1

    ADC = adc_setup(SETUP_SPI, SET_CHANNEL)
    setup_gpio(DRIVE_PINS)

    prbs0 = np.array(lfsr_prbs(N_BITS, tap_mask, SEED), dtype=np.int8)
    prbs0_pm1 = bits_to_pm1(prbs0)
    prbs_mat, shift = make_prbs_matrix(prbs0, len(DRIVE_PINS))

    print(f"Running live save: n_bits={N_BITS}, L={L}")

    for k in range(N_FRAMES):
        t0 = time.time()

        raw = acquire_frame(ADC, prbs_mat, L)
        xcor_raw = correlate_full(prbs0_pm1, raw)
        xcor_map = sample_map(xcor_raw, shift)

        if k % SAVE_EVERY == 0:
            # Heatmap
            fig, ax = plt.subplots()
            im = ax.imshow(xcor_map, aspect="auto")
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Heatmap n_bits={N_BITS} frame={k}")
            fig.savefig(f"{OUT_DIR}/heatmap_{k:04d}.png", dpi=150)
            plt.close(fig)

            # Full cross correlations
            fig, axes = plt.subplots(len(SENSE_PINS), 1, figsize=(10, 12), sharex=True)
            for j in range(len(SENSE_PINS)):
                axes[j].plot(xcor_raw[j])
                axes[j].set_ylabel(f"S{j}")
            fig.savefig(f"{OUT_DIR}/xcor_{k:04d}.png", dpi=150)
            plt.close(fig)

        fps = 1.0 / (time.time() - t0)
        if k % 5 == 0:
            print(f"Frame {k}/{N_FRAMES} | fps={fps:.2f}")

    GPIO.cleanup()
    print("Done.")


if __name__ == "__main__":
    main()