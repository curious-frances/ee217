import numpy as np
import time
import sys
import ADS1256
import spidev
import RPi.GPIO as GPIO

# Consts 
VREF = 5.0
ADC_RESOLUTION = 0x7fffff # 23-bit signed ADC range

def lfsr_prbs(n_bits, tap_mask, seed=None):
    if seed is None:
        seed = (1 << n_bits) - 1 # Initial state of the shift register (all bits set to 1)
    state = seed & ((1 << n_bits) - 1)
    period = (1 << n_bits) - 1
    out = []

    for _ in range(period):
        msb = (state >> (n_bits - 1)) & 1
        out.append(msb)

        
        temp = state & tap_mask
        feedback = temp & 1
        temp >>= 1
        while temp:
            feedback ^= (temp & 1)
            temp >>= 1

        state = ((state << 1) & ((1 << n_bits) - 1)) | feedback

    return out

def bits_to_pm1(bits):
     return np.array([1 if b == 0 else -1 for b in bits])
 
def setup_gpio(pins):
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
        
def adc_setup(SET_UP_SPI, SET_CHANNEL):
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()
    
    if SET_UP_SPI:
        SPI = spidev.SpiDev(0, 0)
        SPI.mode = 0b01
        SPI.max_speed_hz = 3000000
        
    if SET_CHANNEL:
        ADC.ADS1256_SetChannal(7)
        ADC.ADS1256_WriteCmd(0xFC) #sync
        ADC.ADS1256_WriteCmd(0x00) #wakeup
         
    return ADC

def circular_cross_correlation(x_pm1, y_pm1):
    x = np.array(x_pm1)
    y = np.array(y_pm1)
    N = len(x)
    r = np.zeros(N)
    for k in range(N):
        r[k] = np.sum(x * np.roll(y, k))
    return r

def read_adc_volts(ADC, ch):    
    ADC.ADS1256_SetChannal(ch)
    raw = ADC.ADS1256_GetChannalValue(ch)
    v = raw * VREF / ADC_RESOLUTION
    return v