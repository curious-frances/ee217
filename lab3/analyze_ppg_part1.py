import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_ppg_csv(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    time_data = data[:, 0]
    ppg_data = data[:, 1]
    return time_data, ppg_data

def hamming_window(N):
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

def estimate_fs(time_data):
    dt = np.diff(time_data)
    return 1.0 / np.mean(dt)

def moving_average(x, M):
    kernel = np.ones(M) / M
    return np.convolve(x, kernel, mode="same")

def main():
    filename = sys.argv[1]

    time_data, ppg_data = load_ppg_csv(filename)

    fs = estimate_fs(time_data)

    ppg_centered = ppg_data - np.mean(ppg_data)

    # remove slow baseline drift so the very low frequencies do not dominate
    baseline_window_seconds = 0.75
    baseline_window_samples = int(baseline_window_seconds * fs)
    if baseline_window_samples < 3:
        baseline_window_samples = 3
    if baseline_window_samples % 2 == 0:
        baseline_window_samples += 1

    baseline = moving_average(ppg_centered, baseline_window_samples)
    ppg_detrended = ppg_centered - baseline

    N = len(ppg_detrended)
    window = hamming_window(N)
    ppg_windowed = ppg_detrended * window

    fft_output = np.fft.rfft(ppg_windowed)
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    magnitude = np.abs(fft_output)

    freq_resolution = fs / N

    # plausible heart-rate band for resting / light activity
    hr_mask = (freqs >= 0.6) & (freqs <= 3.0)
    hr_freqs = freqs[hr_mask]
    hr_magnitude = magnitude[hr_mask]

    # initial strongest peak
    strongest_index = np.argmax(hr_magnitude)
    strongest_freq = hr_freqs[strongest_index]

    # if the strongest peak is a harmonic, prefer a strong peak near half
    fundamental_freq = strongest_freq
    half_freq = strongest_freq / 2.0

    if 0.6 <= half_freq <= 3.0:
        half_band = np.abs(hr_freqs - half_freq) < 0.08
        if np.any(half_band):
            half_peak_mag = np.max(hr_magnitude[half_band])
            strong_peak_mag = hr_magnitude[strongest_index]

            # if there is a reasonably strong peak near half the frequency,
            # treat the larger peak as a harmonic and choose the lower one
            if half_peak_mag > 0.35 * strong_peak_mag:
                half_peak_index = np.argmax(hr_magnitude[half_band])
                fundamental_freq = hr_freqs[half_band][half_peak_index]

    heart_rate_bpm = 60 * fundamental_freq

    print(f"Number of samples: {N}")
    print(f"Estimated sample rate: {fs:.3f} Hz")
    print(f"FFT length: {N}")
    print(f"Frequency resolution: {freq_resolution:.6f} Hz")
    print(f"Strongest spectral peak: {strongest_freq:.4f} Hz ({60*strongest_freq:.2f} BPM)")
    print(f"Estimated fundamental frequency: {fundamental_freq:.4f} Hz")
    print(f"Estimated heart rate: {heart_rate_bpm:.2f} BPM")

    output_prefix = os.path.splitext(filename)[0]

    plt.figure(figsize=(10, 4))
    plt.plot(time_data, ppg_detrended, label="PPG Signal")
    plt.title("Recovered Time-Domain PPG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_time_signal.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitude, label="FFT Magnitude")
    plt.axvline(
        strongest_freq,
        linestyle="--",
        label=f"Strongest Peak = {strongest_freq:.3f} Hz ({60*strongest_freq:.2f} BPM)"
    )
    plt.axvline(
        fundamental_freq,
        linestyle=":",
        label=f"Chosen HR = {fundamental_freq:.3f} Hz ({heart_rate_bpm:.2f} BPM)"
    )
    plt.xlim(0, 5)
    plt.title("PPG Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_fft.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()