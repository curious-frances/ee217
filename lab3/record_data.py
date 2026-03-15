import numpy as np
import time
import sys
import ADS1256
import spidev
import RPi.GPIO as GPIO
import os

# Constants
SAMPLE_FREQUENCY = 500
SENSE_LINE_PIN = 7


def init_adc(SET_UP_SPI=True):
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()
    
    if SET_UP_SPI:
        SPI = spidev.SpiDev(0, 0)
        SPI.mode = 0b01
        SPI.max_speed_hz = 3000000

    ADC.ADS1256_ConfigADC(
        ADS1256.ADS1256_GAIN_E['ADS1256_GAIN_1'],
        ADS1256.ADS1256_DRATE_E['ADS1256_500SPS']
    )

    ADC.ADS1256_SetChannal(SENSE_LINE_PIN)

    return ADC


def main():
    filename = sys.argv[1]
    duration = float(sys.argv[2])

    data_file = f"{filename}.csv"

    adc = init_adc()

    if not os.path.exists(data_file):
        with open(data_file, "w") as f:
            f.write("time,ecg_signal\n")

    print(f"Recording ECG to {data_file}")
    print(f"Sampling at {SAMPLE_FREQUENCY} Hz for {duration} seconds")

    sample_period = 1.0 / SAMPLE_FREQUENCY
    start_time = time.time()

    with open(data_file, "a") as f:

        while (time.time() - start_time) < duration:

            loop_start = time.time()

            current_time = loop_start - start_time

            raw = adc.ADS1256_GetChannalValue(SENSE_LINE_PIN)
            sense = raw * 5.0 / 0x7fffff

            f.write(f"{current_time},{sense}\n")

            elapsed = time.time() - loop_start
            sleep_time = sample_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    print("Recording complete")


if __name__ == "__main__":
    main()