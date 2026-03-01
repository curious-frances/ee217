#!/usr/bin/env python3
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
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
    5:  0x14,
    6:  0x30,
    7:  0x60,
    8:  0xB8,
    9:  0x110,
    10: 0x240,
    11: 0x500,
    12: 0xE08,
}

N_BITS = 8
PRINT_EVERY = 10


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
    xcor = np.zeros((n_sense, L), dtype=float)
    for j in range(n_sense):
        xcor[j] = circular_cross_correlation(prbs0_pm1, raw[j])
    return xcor


def main():
    if N_BITS not in TAP_MASKS:
        raise ValueError(f"N_BITS={N_BITS} not in TAP_MASKS")

    tap_mask = TAP_MASKS[N_BITS]
    L = (1 << N_BITS) - 1

    ADC = adc_setup(SETUP_SPI, SET_CHANNEL)
    setup_gpio(DRIVE_PINS)

    prbs0_bits = np.array(lfsr_prbs(N_BITS, tap_mask, SEED), dtype=np.int8)
    prbs0_pm1 = bits_to_pm1(prbs0_bits)
    prbs_mat, _ = make_prbs_matrix(prbs0_bits, len(DRIVE_PINS))

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(L)

    lines = []
    for j in range(len(SENSE_PINS)):
        (ln,) = ax.plot(x, np.zeros(L), linewidth=1.2, label=f"Sense Line {j+1}")
        lines.append(ln)

    ax.set_title("Live Plot of Cross Correlations of Sense Lines")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.canvas.draw()
    fig.show()

    t_start = time.time()
    k = 0

    try:
        while True:
            t0 = time.time()

            raw = acquire_frame(ADC, prbs_mat, L)
            xcor = correlate_full(prbs0_pm1, raw)

            y_min = float(np.min(xcor))
            y_max = float(np.max(xcor))
            pad = 0.05 * (y_max - y_min + 1e-9)

            for j, ln in enumerate(lines):
                ln.set_ydata(xcor[j])

            ax.set_ylim(y_min - pad, y_max + pad)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            t1 = time.time()
            fps_inst = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0
            fps_avg = (k + 1) / (t1 - t_start)

            if (k % PRINT_EVERY) == 0:
                print(f"[xcor live] frame={k:6d}  fps_inst={fps_inst:5.2f}  fps_avg={fps_avg:5.2f}")

            k += 1

    finally:
        GPIO.cleanup()
        plt.ioff()


if __name__ == "__main__":
    main()