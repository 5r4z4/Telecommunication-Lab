import numpy as np
import matplotlib.pyplot as plt

# 1. ORIGINAL MESSAGE SIGNAL
fm = 5                       # message signal frequency (Hz)
Am = 1                       # amplitude
T = 1                        # total duration (s)

t_cont = np.linspace(0, T, 5000)             # "continuous" time axis (fine resolution)
message_signal = Am * np.sin(2 * np.pi * fm * t_cont)

# 2. SAMPLING  (Sampling Theorem: fs must be > 2*fm to avoid aliasing)
oversampling_factor = 10                     # how many times over Nyquist rate
fs = oversampling_factor * 2 * fm            # sampling frequency, fs > 2*fm  ✅ satisfies theorem
Ts = 1 / fs

n_samples = int(T * fs)
t_sampled = np.linspace(0, T, n_samples, endpoint=False)
sampled_signal = Am * np.sin(2 * np.pi * fm * t_sampled)

print(f"Message frequency (fm): {fm} Hz")
print(f"Nyquist rate (2*fm)   : {2*fm} Hz")
print(f"Sampling frequency fs : {fs} Hz  -> fs > 2*fm satisfied: {fs > 2*fm}")

# 3. QUANTIZATION
n_bits = 4                                   # bits per sample -> 2^n_bits levels
L = 2 ** n_bits                              # number of quantization levels
v_min, v_max = -Am, Am
step = (v_max - v_min) / L                   # quantization step size

# map each sample to the nearest quantization level
quant_index = np.round((sampled_signal - v_min) / step)
quant_index = np.clip(quant_index, 0, L - 1)
quantized_signal = v_min + quant_index * step + step / 2   # mid-riser levels

# 4. ENCODING + DECODING (PCM binary encode -> decode back to levels)
encoded_bits = [format(int(idx), f'0{n_bits}b') for idx in quant_index]  # binary words

decoded_index = np.array([int(b, 2) for b in encoded_bits])
decoded_signal = v_min + decoded_index * step + step / 2   # should equal quantized_signal

# 5. RECONSTRUCTION (low-pass / sinc interpolation of decoded samples)
def sinc_reconstruction(t_axis, t_samp, samples, fs):
    recon = np.zeros_like(t_axis)
    for n, s in enumerate(samples):
        recon += s * np.sinc(fs * (t_axis - t_samp[n]))
    return recon

reconstructed_signal = sinc_reconstruction(t_cont, t_sampled, decoded_signal, fs)

# PLOTS
fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

axs[0].plot(t_cont, message_signal, 'b')
axs[0].set_title('1. Original Message Signal')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

axs[1].plot(t_cont, message_signal, 'b', alpha=0.3, label='Original')
axs[1].stem(t_sampled, sampled_signal, linefmt='r-', markerfmt='ro', basefmt=' ')
axs[1].set_title(f'2. Sampled Signal (fs = {fs} Hz, satisfies fs > 2·fm = {2*fm} Hz)')
axs[1].set_ylabel('Amplitude')
axs[1].legend(loc='upper right')
axs[1].grid(True)

axs[2].step(t_sampled, quantized_signal, where='mid', color='g')
axs[2].plot(t_sampled, sampled_signal, 'ro', markersize=3, alpha=0.6, label='Sample points')
for level in np.arange(v_min, v_max, step):
    axs[2].axhline(level, color='gray', linewidth=0.3)
axs[2].set_title(f'3. Quantized Signal ({n_bits}-bit, L = {L} levels)')
axs[2].set_ylabel('Amplitude')
axs[2].legend(loc='upper right')
axs[2].grid(True)

axs[3].step(t_sampled, decoded_signal, where='mid', color='m')
axs[3].set_title('4. Decoded Signal (after PCM encode -> binary -> decode)')
axs[3].set_ylabel('Amplitude')
axs[3].grid(True)

axs[4].plot(t_cont, message_signal, 'b', alpha=0.3, label='Original')
axs[4].plot(t_cont, reconstructed_signal, 'k', label='Reconstructed')
axs[4].set_title('5. Reconstructed Signal (sinc / low-pass interpolation)')
axs[4].set_xlabel('Time (s)')
axs[4].set_ylabel('Amplitude')
axs[4].legend(loc='upper right')
axs[4].grid(True)

plt.tight_layout()
plt.show()

print("\nFirst 5 PCM binary codes:", encoded_bits[:5])