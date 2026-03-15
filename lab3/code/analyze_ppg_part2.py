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

    filtered = np.fft.ifft(X_filt).real
    return filtered

def main():
    filename = sys.argv[1]
    start_time = None
    end_time = None

    if len(sys.argv) >= 3:
        start_time = float(sys.argv[2])
    if len(sys.argv) >= 4:
        end_time = float(sys.argv[3])

    time_data, ppg_data = load_ppg_csv(filename)

    if start_time is not None:
        if end_time is None:
            end_time = time_data[-1]
        mask = (time_data >= start_time) & (time_data <= end_time)
        time_data = time_data[mask]
        ppg_data = ppg_data[mask]

    if len(time_data) < 10:
        print("Not enough data after cropping.")
        sys.exit(1)

    fs = estimate_fs(time_data)

    x = ppg_data - np.mean(ppg_data)

    lowcut = 1
    highcut = 4.0
    x_clean = fft_bandpass(x, fs, lowcut, highcut)

    min_peak_distance_seconds = 0.5
    min_peak_distance_samples = int(min_peak_distance_seconds * fs)

    prominence_value = 0.20 * np.std(x_clean)
    if prominence_value <= 0:
        prominence_value = 1e-6

    peaks, properties = find_peaks(
        x_clean,
        distance=min_peak_distance_samples,
        prominence=prominence_value
    )

    if len(peaks) < 2:
        print("Not enough peaks detected.")
        sys.exit(1)

    peak_times = time_data[peaks]
    ibi = np.diff(peak_times)

    valid_mask = (ibi >= 0.5) & (ibi <= 2)
    valid_ibi = ibi[valid_mask]

    if len(valid_ibi) < 2:
        print("Not enough valid intervals after filtering.")
        sys.exit(1)

    avg_interval = np.mean(valid_ibi)
    avg_hr_bpm = 60.0 / avg_interval

    mean_ibi = np.mean(valid_ibi)
    max_hrv_sec = np.max(np.abs(valid_ibi - mean_ibi))
    rms_hrv_sec = np.sqrt(np.mean((valid_ibi - mean_ibi) ** 2))

    max_hrv_ms = 1000.0 * max_hrv_sec
    rms_hrv_ms = 1000.0 * rms_hrv_sec

    print(f"Number of samples: {len(x_clean)}")
    print(f"Estimated sample rate: {fs:.3f} Hz")
    print(f"Bandpass range: {lowcut:.2f} Hz to {highcut:.2f} Hz")
    print(f"Detected peaks: {len(peaks)}")
    print(f"Valid intervals used: {len(valid_ibi)}")
    print(f"Average Heart Rate: {avg_hr_bpm:.2f} BPM")
    print(f"Max HRV: {max_hrv_ms:.2f} ms")
    print(f"RMS HRV: {rms_hrv_ms:.2f} ms")

    output_prefix = os.path.splitext(filename)[0]
    if start_time is not None:
        output_prefix += f"_{int(start_time)}_{int(end_time)}"

    plt.figure(figsize=(12, 5))
    plt.plot(time_data, x_clean, label="Filtered PPG")
    plt.plot(time_data[peaks], x_clean[peaks], ".", label="Detected Peaks")
    plt.title("Filtered PPG Signal with Detected Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_part2_fft_peaks_full.png", dpi=300)
    plt.close()

    # Zoomed plot
    total_duration = time_data[-1] - time_data[0]
    zoom_length = min(20.0, total_duration)
    zoom_start = time_data[0] + max(0.0, (total_duration - zoom_length) / 2.0)
    zoom_end = zoom_start + zoom_length

    zoom_mask = (time_data >= zoom_start) & (time_data <= zoom_end)
    zoom_peaks_mask = (time_data[peaks] >= zoom_start) & (time_data[peaks] <= zoom_end)

    plt.figure(figsize=(12, 5))
    plt.plot(time_data[zoom_mask], x_clean[zoom_mask], label="Filtered PPG")
    plt.plot(
        time_data[peaks][zoom_peaks_mask],
        x_clean[peaks][zoom_peaks_mask],
        ".",
        label="Detected Peaks"
    )
    plt.title("Filtered PPG Signal with Detected Peaks (Zoomed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_part2_fft_peaks_zoom.png", dpi=300)
    plt.close()

    # IBI plot
    valid_peak_times = peak_times[1:][valid_mask]

    plt.figure(figsize=(12, 4))
    plt.plot(valid_peak_times, valid_ibi * 1000.0, ".-")
    plt.title("Beat-to-Beat Intervals")
    plt.xlabel("Time (s)")
    plt.ylabel("IBI (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_part2_ibi.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()