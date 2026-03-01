import numpy as np
import matplotlib.pyplot as plt
import spidev
import ADS1256
import RPi.GPIO as GPIO
import time

from util import (
    lfsr_prbs,
    bits_to_pm1,
    circular_cross_correlation,
    setup_gpio,
    adc_setup,
    read_adc_volts,
)
from datetime import datetime


# Configuration parameters
DRIVE_PINS = [21, 20, 16, 12, 7]  # GPIO pins used to drive the LFSR
SENSE_PINS = [1, 2, 3, 4, 5, 6, 7]  # PINS on the AD-DA board
SETUP_SPI = True
SET_CHANNEL = False


def main():
    ADC = adc_setup(SETUP_SPI, SET_CHANNEL)
    setup_gpio(DRIVE_PINS)
    tap_mask = 0x14
    seed = 0x01
    length = 2**5 - 1
    prbs_bits = lfsr_prbs(5, tap_mask, seed)

    raw_sense_data = np.zeros((len(SENSE_PINS), length))
