import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generate Original Bit Sequence
# ---------------------------------------------------------

np.random.seed(1)
num_bits = 20
samples_per_bit = 6

b = np.random.randint(0, 2, num_bits)

pattern = []
for k in range(num_bits):
    if b[k] == 0:
        sig = np.zeros(samples_per_bit)
    else:
        sig = np.ones(samples_per_bit)
    pattern.extend(sig)

pattern = np.array(pattern)

# ---------------------------------------------------------
# 2. Generate Pseudo-random Sequence
# ---------------------------------------------------------

spread_sig = np.random.randint(0, 2, len(pattern))

# ---------------------------------------------------------
# 3. XOR (Spreading)
# ---------------------------------------------------------

hopped_sig = np.logical_xor(pattern.astype(bool),
                            spread_sig.astype(bool)).astype(int)

# ---------------------------------------------------------
# 4. DSSS Modulation
# ---------------------------------------------------------

t = np.arange(0, 40)   # samples per chip
fc = 0.05

c1 = np.cos(2 * np.pi * fc * t)
c2 = np.cos(2 * np.pi * fc * t + np.pi)

dsss_sig = []
for k in range(len(hopped_sig)):
    if hopped_sig[k] == 0:
        dsss_sig.extend(c1)
    else:
        dsss_sig.extend(c2)

dsss_sig = np.array(dsss_sig)
x = np.arange(len(dsss_sig))

# ---------------------------------------------------------
# 5. Plotting (Original + Zoom)
# ---------------------------------------------------------

fig, axs = plt.subplots(5, 1, figsize=(10, 14))

# 1. Original Bits
axs[0].plot(pattern, color='blue')
axs[0].set_title('Original Bit Sequence', style='italic', fontweight='bold')
axs[0].set_ylim([-0.5, 1.5])
axs[0].grid(True)

# 2. PN Sequence
axs[1].plot(spread_sig, color='green')
axs[1].set_title('Pseudorandom Sequence', style='italic', fontweight='bold')
axs[1].set_ylim([-0.5, 1.5])
axs[1].grid(True)

# 3. Spread Bits
axs[2].plot(hopped_sig, color='orange')
axs[2].set_title('Spreaded Signal (After XOR)', style='italic', fontweight='bold')
axs[2].set_ylim([-0.5, 1.5])
axs[2].grid(True)

# 4. FULL DSSS Signal (Original)
axs[3].plot(x, dsss_sig, color='red', linewidth=0.7)
axs[3].set_title('DSSS Signal (Full View)', style='italic', fontweight='bold')
axs[3].set_ylim([-1.5, 1.5])
axs[3].grid(True)

# 5. ZOOMED DSSS Signal
zoom_samples = 5 * len(t)
axs[4].plot(x[:zoom_samples], dsss_sig[:zoom_samples], color='purple')
axs[4].set_title('DSSS Signal (Zoomed View)', style='italic', fontweight='bold')
axs[4].set_ylim([-1.5, 1.5])
axs[4].grid(True)

plt.tight_layout()
plt.show()