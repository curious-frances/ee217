import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import ADS1256
import RPi.GPIO as GPIO
import spidev
import matplotlib.patches as patches

def generate_prbs(polynomial, length, seed):
    num_bits = polynomial.bit_length() 
    lfsr = [int(bit) for bit in format(seed, f'0{num_bits}b')]
    prbs_sequence = []
    for _ in range(length):
        prbs_sequence.append(lfsr[-1])
        feedback = 0
        for bit_position in range(num_bits):
            if (polynomial >> bit_position) & 1:
                feedback ^= lfsr[-(bit_position + 1)]
        lfsr = lfsr[1:] + [feedback]
    return prbs_sequence

def circular_cross_correlation(x, y):
    N = max(len(x), len(y))
    x = np.pad(x, (0, N - len(x)), mode='constant')
    y = np.pad(y, (0, N - len(y)), mode='constant')
    result = np.zeros(N)
    for k in range(N):
        result[k] = np.sum(x * np.roll(y, k))
    return result

def fit_gaussian_ellipse(touch_map, num_points=5):
    """Fit an ellipse based on a weighted centroid of the top 'num_points' maximum values."""
    # Get indices of the top 'num_points' max values
    flat_map = touch_map.flatten()
    sorted_indices = np.argsort(flat_map)[-num_points:]
    
    # Convert linear indices to 2D indices (row, col)
    max_indices = np.unravel_index(sorted_indices, touch_map.shape)
    
    # Compute weighted centroid
    weights = flat_map[sorted_indices]
    centroid_x = np.sum(max_indices[1] * weights) / np.sum(weights)
    centroid_y = np.sum(max_indices[0] * weights) / np.sum(weights)
    centroid = (centroid_x, centroid_y)
    
    # Compute variances for major and minor axes
    x = np.arange(touch_map.shape[1])
    y = np.arange(touch_map.shape[0])
    X, Y = np.meshgrid(x, y)
    
    var_x = np.sum(touch_map * (X - centroid_x) ** 2) / np.sum(touch_map)
    var_y = np.sum(touch_map * (Y - centroid_y) ** 2) / np.sum(touch_map)

    major_axis_length = 2 * np.sqrt(var_x)
    minor_axis_length = 2 * np.sqrt(var_y)
    
    return centroid, major_axis_length, minor_axis_length

# Drive and sense line pins as wired
DRIVE = [21, 7, 12, 16, 20]
SENSE = [1, 2, 3, 4, 5, 6, 7]
SETUP_SPI = True

if __name__ == '__main__':
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()
    ADDR = 0x00
    ON = 0x02
    ADC.ADS1256_WriteReg(ADDR, ON)
    if SETUP_SPI:
        SPI = spidev.SpiDev(0, 0)
        SPI.mode = 0b01
        SPI.max_speed_hz = 3000000

    for drive in DRIVE:
        GPIO.setup(drive, GPIO.OUT)

    polynomial = 0x14
    length = 2**5-1
    shift = length // 5
    seed = 0x01
    prbs0 = generate_prbs(polynomial, length, seed)
    prbs = np.vstack([np.roll(prbs0, i*shift) for i in range(len(DRIVE))])

    raw_sense = np.zeros((7, length))
    xcor = np.zeros((np.shape(raw_sense)[0], len(DRIVE)))
    xcor_raw = np.zeros((np.shape(raw_sense)[0], len(prbs0)))
    sense = np.zeros((7, length))
    threshold = 22.25
    
    # Set up live plot with ellipsoid overlay
    figure, ax = plt.subplots()
    heatmap = ax.imshow(xcor, cmap='viridis', interpolation='nearest', vmin=threshold, vmax=24)
    plt.colorbar(heatmap)
    plt.title("Live Updating Heatmap")
    ax.invert_yaxis()
    ax.invert_xaxis()
    figure.canvas.draw()
    axbackground = figure.canvas.copy_from_bbox(ax.bbox)
    plt.show(block=False)

    # Standalone figure for the ellipse
    ellipse_fig, ellipse_ax = plt.subplots()
    ellipse_ax.set_aspect('equal')
    ellipse_ax.set_xlim(0, len(DRIVE))
    ellipse_ax.set_ylim(0, len(SENSE))
    ellipse_ax.set_title("Ellipse Fit Plot")
    ellipse_ax.invert_xaxis()
    plt.show(block=False)

    s = 0
    while True:
        for j in range(len(SENSE)):
            for i in range(len(DRIVE)):
                GPIO.output(DRIVE[i], int(prbs[i, s]))

            ADC.ADS1256_SetChannal(SENSE[j])
            ADC_Value = ADC.ADS1256_GetChannalValue(SENSE[j])
            raw_sense[j][s] = ADC_Value * 5.0 / 0x7fffff
            s += 1

            if s >= length:
                for idx in range(np.shape(raw_sense)[0]):
                    xcor_raw[idx] = circular_cross_correlation(prbs0, raw_sense[idx])
                    xcor[idx] = [xcor_raw[idx][i*shift] for i in range(len(DRIVE))]

                touch_map = xcor
                touch_map[touch_map < threshold] = 0
                heatmap.set_data(touch_map)

                # Fit ellipse with weighted centroid
                centroid, major_axis, minor_axis = fit_gaussian_ellipse(touch_map, num_points=5)

                # Clear previous ellipses and centroids on the ellipse-only plot
                ellipse_ax.clear()

                # Draw the new ellipse
                ellipse_only = patches.Ellipse(
                    xy=centroid, width=major_axis, height=minor_axis,
                    edgecolor='blue', facecolor='none', linestyle='--', linewidth=1.5
                )
                ellipse_ax.add_patch(ellipse_only)
                ellipse_ax.plot(centroid[0], centroid[1], 'ro')

                # Redraw the ellipse plot
                ellipse_fig.canvas.draw()

                # Update live plot
                figure.canvas.restore_region(axbackground)
                ax.draw_artist(heatmap)
                figure.canvas.blit(ax.bbox)
                figure.canvas.flush_events()

                s = 0
                # time.sleep(0.1)
