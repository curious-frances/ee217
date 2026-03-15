import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.signal import find_peaks

def load_ppg_csv(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    time_data = data[:, 0]
    ppg_data = data[:, 1]
    return time_data, ppg_data

def estimate_fs(time_data):
    dt = np.diff(time_data)
    return 1.0 / np.mean(dt)

def fft_bandpass(signal, fs, lowcut, highcut):
    N = len(signal)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    keep = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
    X_filt = np.zeros_like(X, dtype=complex)
    X_filt[keep] = X[keep]

    return np.fft.ifft(X_filt).real

def moving_average(x, M):
    if M <= 1:
        return x.copy()
    kernel = np.ones(M) / M
    return np.convolve(x, kernel, mode="same")

def estimate_hr_from_window(window_time, window_signal, fs, lowcut=1.0, highcut=4.0):
    if len(window_signal) < 10:
        return np.nan

    x = window_signal - np.mean(window_signal)
    x_clean = fft_bandpass(x, fs, lowcut, highcut)

    min_peak_distance_seconds = 0.4
    min_peak_distance_samples = int(min_peak_distance_seconds * fs)

    prominence_value = 0.15 * np.std(x_clean)
    if prominence_value <= 0:
        prominence_value = 1e-6

    peaks, _ = find_peaks(
        x_clean,
        distance=min_peak_distance_samples,
        prominence=prominence_value
    )

    if len(peaks) < 2:
        return np.nan

    peak_times = window_time[peaks]
    ibi = np.diff(peak_times)

    valid_mask = (ibi >= 0.4) & (ibi <= 1.5)
    valid_ibi = ibi[valid_mask]

    if len(valid_ibi) < 2:
        return np.nan

    mean_ibi = np.mean(valid_ibi)
    hr_bpm = 60.0 / mean_ibi
    return hr_bpm

def fit_recovery_tau(time_sec, hr_bpm):
    valid = np.isfinite(hr_bpm)
    time_sec = time_sec[valid]
    hr_bpm = hr_bpm[valid]

    if len(hr_bpm) < 10:
        return np.nan, np.nan, np.nan

    hr_final = np.mean(hr_bpm[-max(3, len(hr_bpm)//10):])
    hr_peak = np.max(hr_bpm)

    delta = hr_bpm - hr_final
    fit_mask = delta > 0

    time_fit = time_sec[fit_mask]
    delta_fit = delta[fit_mask]

    if len(delta_fit) < 5:
        return np.nan, hr_peak, hr_final

    ln_delta = np.log(delta_fit)

    A = np.vstack([time_fit, np.ones_like(time_fit)]).T
    slope, intercept = np.linalg.lstsq(A, ln_delta, rcond=None)[0]

    if slope >= 0:
        return np.nan, hr_peak, hr_final

    tau = -1.0 / slope
    return tau, hr_peak, hr_final

def hr_at_offset(time_sec, hr_bpm, peak_index, offset_sec):
    target_time = time_sec[peak_index] + offset_sec
    idx = np.argmin(np.abs(time_sec - target_time))
    return hr_bpm[idx], time_sec[idx]

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_ppg_part3_recovery.py <file.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    time_data, ppg_data = load_ppg_csv(filename)
    fs = estimate_fs(time_data)

    # Analysis settings
    lowcut = 1.0
    highcut = 4.0
    window_length_sec = 10.0
    step_sec = 1.0
    smoothing_length_samples = 5

    window_length_samples = int(window_length_sec * fs)
    step_samples = int(step_sec * fs)

    hr_times = []
    hr_values = []

    for start in range(0, len(ppg_data) - window_length_samples + 1, step_samples):
        end = start + window_length_samples

        window_time = time_data[start:end]
        window_signal = ppg_data[start:end]

        hr_bpm = estimate_hr_from_window(window_time, window_signal, fs, lowcut, highcut)

        hr_times.append(np.mean(window_time))
        hr_values.append(hr_bpm)

    hr_times = np.array(hr_times)
    hr_values = np.array(hr_values)

    hr_smooth = moving_average(hr_values, smoothing_length_samples)

    valid = np.isfinite(hr_smooth)
    hr_times_valid = hr_times[valid]
    hr_smooth_valid = hr_smooth[valid]

    if len(hr_smooth_valid) < 10:
        print("Not enough valid HR estimates.")
        sys.exit(1)

    peak_index = np.argmax(hr_smooth_valid)
    hr_peak = hr_smooth_valid[peak_index]
    peak_time = hr_times_valid[peak_index]

    tau, hr_peak_fit, hr_final = fit_recovery_tau(hr_times_valid - peak_time, hr_smooth_valid)

    hr_30, t_30 = hr_at_offset(hr_times_valid, hr_smooth_valid, peak_index, 30.0)
    hr_60, t_60 = hr_at_offset(hr_times_valid, hr_smooth_valid, peak_index, 60.0)
    hr_300, t_300 = hr_at_offset(hr_times_valid, hr_smooth_valid, peak_index, 300.0)

    drop_30 = hr_peak - hr_30
    drop_60 = hr_peak - hr_60
    drop_300 = hr_peak - hr_300

    print(f"Estimated sample rate: {fs:.3f} Hz")
    print(f"Bandpass range: {lowcut:.2f} Hz to {highcut:.2f} Hz")
    print(f"Window length: {window_length_sec:.1f} s")
    print(f"Window step: {step_sec:.1f} s")
    print(f"Peak HR: {hr_peak:.2f} BPM at t = {peak_time:.2f} s")
    print(f"Smoothed final HR: {hr_final:.2f} BPM")

    if np.isfinite(tau):
        print(f"Recovery time constant tau: {tau:.2f} s")
    else:
        print("Recovery time constant tau: could not be estimated reliably")

    print(f"HR after ~30 s: {hr_30:.2f} BPM at t = {t_30:.2f} s")
    print(f"Drop after 30 s: {drop_30:.2f} BPM")

    print(f"HR after ~60 s: {hr_60:.2f} BPM at t = {t_60:.2f} s")
    print(f"Drop after 1 min: {drop_60:.2f} BPM")

    print(f"HR after ~300 s: {hr_300:.2f} BPM at t = {t_300:.2f} s")
    print(f"Drop after 5 min: {drop_300:.2f} BPM")

    output_prefix = os.path.splitext(filename)[0]

    plt.figure(figsize=(12, 5))
    plt.plot(hr_times, hr_values, ".", alpha=0.6, label="Windowed HR estimates")
    plt.plot(hr_times, hr_smooth, linewidth=2, label="Smoothed HR")

    plt.axhline(hr_peak, linestyle="--", label=f"Peak HR = {hr_peak:.2f} BPM")
    plt.axhline(hr_final, linestyle=":", label=f"Final HR = {hr_final:.2f} BPM")
    plt.axvline(peak_time, linestyle="--", alpha=0.7, label="Peak time")

    plt.plot(t_30, hr_30, "o", label=f"30 s drop = {drop_30:.2f} BPM")
    plt.plot(t_60, hr_60, "o", label=f"1 min drop = {drop_60:.2f} BPM")
    plt.plot(t_300, hr_300, "o", label=f"5 min drop = {drop_300:.2f} BPM")

    plt.title("Heart Rate Recovery After Exercise")
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (BPM)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_part3_hr_recovery.png", dpi=300)
    plt.close()

    if np.isfinite(tau):
        valid_tau = np.isfinite(hr_smooth_valid) & (hr_smooth_valid > hr_final)
        time_tau = hr_times_valid[valid_tau] - peak_time
        hr_tau = hr_smooth_valid[valid_tau]

        plt.figure(figsize=(12, 5))
        plt.plot(time_tau, np.log(hr_tau - hr_final), ".", label="log(HR - HR_final)")
        plt.title("Log-Linear Recovery Fit")
        plt.xlabel("Time Since Peak (s)")
        plt.ylabel("log(HR - HR_final)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_part3_tau_fit.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()