"""
Frequency Hopping Spread Spectrum (FHSS) simulation
Python translation of fhss.m
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Generation of bit pattern
# ---------------------------------------------------------
np.random.seed()  # remove/replace with a fixed seed if you want reproducibility
num_bits = 25
samples_per_bit = 120

s = np.random.randint(0, 2, num_bits)   # 25 random bits (0/1)

signal = []
carrier = []
t = np.linspace(0, 2 * np.pi, samples_per_bit)  # 120 samples for one cosine

for k in range(num_bits):
    if s[k] == 0:
        sig = -np.ones(samples_per_bit)   # 120 minus-ones for bit 0
    else:
        sig = np.ones(samples_per_bit)    # 120 ones for bit 1
    c = np.cos(t)
    carrier.extend(c)
    signal.extend(sig)

signal = np.array(signal)
carrier = np.array(carrier)
N = len(signal)  # total number of samples (25*120 = 3000)

# ---------------------------------------------------------
# BPSK Modulation of the signal
# ---------------------------------------------------------
bpsk_sig = signal * carrier

# ---------------------------------------------------------
# Preparation of 6 new carrier frequencies
# Each is repeated so that its total length also equals
# samples_per_bit (120), just like in the original script.
# ---------------------------------------------------------
def make_carrier(num_samples, repeats):
    tt = np.linspace(0, 2 * np.pi, num_samples)
    c = np.cos(tt)
    return np.tile(c, repeats)

c1 = make_carrier(10, 12)    # 10 samples * 12 repeats = 120
c2 = make_carrier(20, 6)     # 20 * 6 = 120
c3 = make_carrier(30, 4)     # 30 * 4 = 120
c4 = make_carrier(40, 3)     # 40 * 3 = 120
c5 = make_carrier(60, 2)     # 60 * 2 = 120
c6 = make_carrier(120, 1)    # 120 * 1 = 120

carriers = {1: c1, 2: c2, 3: c3, 4: c4, 5: c5, 6: c6}

# ---------------------------------------------------------
# Random frequency hops to form a spread signal
# ---------------------------------------------------------
spread_signal = []
for n in range(num_bits):
    choice = np.random.randint(1, 7)  # random int in [1, 6]
    spread_signal.extend(carriers[choice])

spread_signal = np.array(spread_signal)

# ---------------------------------------------------------
# Spreading BPSK Signal into a wider band
# ---------------------------------------------------------
freq_hopped_sig = bpsk_sig * spread_signal

# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------
x = np.arange(1, N + 1)

fig1, axs = plt.subplots(4, 1, figsize=(10, 10))

axs[0].plot(signal,color='black')
axs[0].set_xlim([-100, 3100])
axs[0].set_ylim([-1.5, 1.5])
axs[0].set_title('Original Bit Sequence', style='italic', fontweight='bold')

axs[1].plot(bpsk_sig,color='tab:red')
axs[1].set_xlim([-100, 3100])
axs[1].set_ylim([-1.5, 1.5])
axs[1].set_title('BPSK Modulated Signal', style='italic', fontweight='bold')

axs[2].plot(x, spread_signal,color='tab:green')
axs[2].set_xlim([-100, 3100])
axs[2].set_ylim([-1.5, 1.5])
axs[2].set_title('Spread Signal with 6 frequencies', style='italic', fontweight='bold')

axs[3].plot(x, freq_hopped_sig,color='tab:blue')
axs[3].set_xlim([-100, 3100])
axs[3].set_ylim([-1.5, 1.5])
axs[3].set_title('Frequency Hopped Spread Spectrum Signal', style='italic', fontweight='bold')

fig1.tight_layout()

# Second figure: FHSS signal + its FFT
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 6))

axs2[0].plot(x, freq_hopped_sig)
axs2[0].set_xlim([-100, 3100])
axs2[0].set_ylim([-1.5, 1.5])
axs2[0].set_title('Frequency Hopped Spread Spectrum signal and its FFT',
                   style='italic', fontweight='bold')

axs2[1].plot(x, np.abs(np.fft.fft(freq_hopped_sig)))

fig2.tight_layout()

plt.show()