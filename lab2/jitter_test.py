#!/usr/bin/env python3
import os
import time
import csv
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

TAP_MASKS = {6: 0x30}

N_BITS = 6
N_SAMPLES = 1000
TOPK = 5
THRESHOLD = 0.0
OUT_DIR = "jitter_out"


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


def centroid_topk(touch_map, k=5):
    flat = touch_map.flatten()
    idx = np.argsort(flat)[-k:]
    w = flat[idx]
    if np.sum(w) == 0:
        return None
    ys, xs = np.unravel_index(idx, touch_map.shape)
    cx = np.sum(xs * w) / np.sum(w)
    cy = np.sum(ys * w) / np.sum(w)
    return cx, cy


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tap_mask = TAP_MASKS[N_BITS]
    L = (1 << N_BITS) - 1

    ADC = adc_setup(SETUP_SPI, SET_CHANNEL)
    setup_gpio(DRIVE_PINS)

    prbs0 = np.array(lfsr_prbs(N_BITS, tap_mask, SEED), dtype=np.int8)
    prbs0_pm1 = bits_to_pm1(prbs0)
    prbs_mat, shift = make_prbs_matrix(prbs0, len(DRIVE_PINS))

    centroids = []

    print("Place penny or hold still touch now...")

    while len(centroids) < N_SAMPLES:
        raw = acquire_frame(ADC, prbs_mat, L)
        xcor_raw = correlate_full(prbs0_pm1, raw)
        xcor_map = sample_map(xcor_raw, shift)

        xcor_map[xcor_map < THRESHOLD] = 0
        c = centroid_topk(xcor_map, TOPK)

        if c is not None:
            centroids.append(c)

        if len(centroids) % 50 == 0:
            print(f"{len(centroids)}/{N_SAMPLES}")

    centroids = np.array(centroids)
    cx = centroids[:, 0]
    cy = centroids[:, 1]

    cx0 = np.mean(cx)
    cy0 = np.mean(cy)

    rms = np.sqrt(np.mean((cx - cx0)**2 + (cy - cy0)**2))

    print(f"Mean centroid: ({cx0:.4f}, {cy0:.4f})")
    print(f"RMS jitter: {rms:.6f} grid units")

    # Save scatter
    fig, ax = plt.subplots()
    ax.scatter(cx, cy, s=8)
    ax.scatter(cx0, cy0, marker='x', s=80)
    ax.set_title(f"Centroid jitter (RMS={rms:.4f})")
    fig.savefig(f"{OUT_DIR}/jitter.png", dpi=150)
    plt.close(fig)

    GPIO.cleanup()


if __name__ == "__main__":
    main()