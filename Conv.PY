import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONVOLUTIONAL CODED DIGITAL COMMUNICATION SYSTEM
# Binary Data -> Encoder -> Modulation -> AWGN
# -> Demodulation -> Viterbi Decoder -> BER
# ============================================================

np.random.seed(10)

# -----------------------------
# Simulation parameters
# -----------------------------
N_BITS = 20000
SNR_dB = np.arange(0, 11, 1)

# Choose modulation:
# "BPSK", "QPSK", or "16QAM"
MODULATION = "BPSK"

# Constraint length
K = 3

# Generator polynomials in octal
# Rate 1/2 : 7, 5
# Rate 1/3 : 7, 5, 3
# Rate 1/4 : 7, 5, 3, 6
GENERATORS = {
    "1/2": [0o7, 0o5],
    "1/3": [0o7, 0o5, 0o3],
    "1/4": [0o7, 0o5, 0o3, 0o6]
}


# ============================================================
# CONVOLUTIONAL ENCODER
# ============================================================

def convolutional_encode(bits, generators, K=3):

    # Add K-1 zeros to terminate encoder
    bits = np.concatenate([bits, np.zeros(K - 1, dtype=int)])

    state = np.zeros(K - 1, dtype=int)

    encoded = []

    for bit in bits:

        # Current input + memory
        register = np.concatenate(([bit], state))

        output = []

        for g in generators:

            # Convert octal generator to binary
            g_bits = np.array(
                [(g >> i) & 1 for i in range(K - 1, -1, -1)]
            )

            # Modulo-2 addition
            value = np.sum(register * g_bits) % 2
            output.append(value)

        encoded.extend(output)

        # Shift register
        state = register[:-1]

    return np.array(encoded, dtype=int)


# ============================================================
# MODULATION
# ============================================================

def modulate(bits, modulation):

    # -------------------------
    # BPSK
    # -------------------------
    if modulation == "BPSK":

        symbols = 1 - 2 * bits
        return symbols.astype(float)

    # -------------------------
    # QPSK
    # -------------------------
    elif modulation == "QPSK":

        # Pad if necessary
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)

        pairs = bits.reshape(-1, 2)

        # Gray-coded QPSK
        I = 1 - 2 * pairs[:, 0]
        Q = 1 - 2 * pairs[:, 1]

        symbols = (I + 1j * Q) / np.sqrt(2)

        return symbols

    # -------------------------
    # 16-QAM
    # -------------------------
    elif modulation == "16QAM":

        if len(bits) % 4 != 0:
            bits = np.pad(
                bits,
                (0, 4 - len(bits) % 4)
            )

        groups = bits.reshape(-1, 4)

        # Gray-coded levels
        levels = {
            (0, 0): -3,
            (0, 1): -1,
            (1, 1): 1,
            (1, 0): 3
        }

        I = np.array([
            levels[(b[0], b[1])]
            for b in groups
        ])

        Q = np.array([
            levels[(b[2], b[3])]
            for b in groups
        ])

        # Normalize average symbol energy to 1
        symbols = (I + 1j * Q) / np.sqrt(10)

        return symbols

    else:
        raise ValueError("Unsupported modulation")


# ============================================================
# AWGN CHANNEL
# ============================================================

def awgn_channel(symbols, snr_db):

    snr_linear = 10 ** (snr_db / 10)

    signal_power = np.mean(np.abs(symbols) ** 2)

    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(symbols))
        + 1j * np.random.randn(len(symbols))
    )

    return symbols + noise


# ============================================================
# DEMODULATION
# ============================================================

def demodulate(symbols, modulation):

    # -------------------------
    # BPSK
    # -------------------------
    if modulation == "BPSK":

        bits = (np.real(symbols) < 0).astype(int)

        return bits

    # -------------------------
    # QPSK
    # -------------------------
    elif modulation == "QPSK":

        I = (np.real(symbols) < 0).astype(int)
        Q = (np.imag(symbols) < 0).astype(int)

        bits = np.column_stack((I, Q)).reshape(-1)

        return bits

    # -------------------------
    # 16-QAM
    # -------------------------
    elif modulation == "16QAM":

        # Denormalize
        symbols = symbols * np.sqrt(10)

        I = np.real(symbols)
        Q = np.imag(symbols)

        def level_to_bits(x):

            if x < -2:
                return [0, 0]

            elif x < 0:
                return [0, 1]

            elif x < 2:
                return [1, 1]

            else:
                return [1, 0]

        bits = []

        for i, q in zip(I, Q):

            bits.extend(level_to_bits(i))
            bits.extend(level_to_bits(q))

        return np.array(bits, dtype=int)

    else:
        raise ValueError("Unsupported modulation")


# ============================================================
# VITERBI DECODER
# ============================================================

def viterbi_decode(received_bits, generators, K=3):

    n_outputs = len(generators)

    n_steps = len(received_bits) // n_outputs

    received = received_bits.reshape(
        n_steps,
        n_outputs
    )

    n_states = 2 ** (K - 1)

    # Path metrics
    INF = 10**9

    metrics = np.full(n_states, INF)
    metrics[0] = 0

    # Store previous states
    previous_state = np.zeros(
        (n_steps, n_states),
        dtype=int
    )

    previous_input = np.zeros(
        (n_steps, n_states),
        dtype=int
    )

    # --------------------------------
    # Trellis construction
    # --------------------------------

    for step in range(n_steps):

        new_metrics = np.full(n_states, INF)

        for state in range(n_states):

            if metrics[state] >= INF:
                continue

            # Previous memory bits
            state_bits = np.array([
                (state >> i) & 1
                for i in range(K - 2, -1, -1)
            ])

            for input_bit in [0, 1]:

                register = np.concatenate(
                    ([input_bit], state_bits)
                )

                output = []

                for g in generators:

                    g_bits = np.array([
                        (g >> i) & 1
                        for i in range(K - 1, -1, -1)
                    ])

                    value = np.sum(
                        register * g_bits
                    ) % 2

                    output.append(value)

                output = np.array(output)

                # New state
                new_state_bits = register[:-1]

                next_state = 0

                for b in new_state_bits:
                    next_state = (
                        next_state << 1
                    ) | int(b)

                # Hamming distance
                branch_metric = np.sum(
                    output != received[step]
                )

                total_metric = (
                    metrics[state]
                    + branch_metric
                )

                if total_metric < new_metrics[next_state]:

                    new_metrics[next_state] = total_metric

                    previous_state[
                        step,
                        next_state
                    ] = state

                    previous_input[
                        step,
                        next_state
                    ] = input_bit

        metrics = new_metrics

    # --------------------------------
    # Traceback
    # --------------------------------

    # Encoder was terminated with zeros,
    # therefore final state should be zero.

    state = 0

    decoded = []

    for step in range(n_steps - 1, -1, -1):

        bit = previous_input[
            step,
            state
        ]

        decoded.append(bit)

        state = previous_state[
            step,
            state
        ]

    decoded.reverse()

    return np.array(decoded, dtype=int)


# ============================================================
# MAIN SIMULATION
# ============================================================

results = {}

# Generate original binary data
original_bits = np.random.randint(
    0,
    2,
    N_BITS
)

for rate, generators in GENERATORS.items():

    print("\nSimulating convolutional code:",
          rate)

    ber_values = []

    for snr in SNR_dB:

        # -----------------------------
        # 1. Convolutional Encoding
        # -----------------------------

        encoded_bits = convolutional_encode(
            original_bits,
            generators,
            K
        )

        # -----------------------------
        # 2. Modulation
        # -----------------------------

        tx_symbols = modulate(
            encoded_bits,
            MODULATION
        )

        # -----------------------------
        # 3. AWGN Channel
        # -----------------------------

        rx_symbols = awgn_channel(
            tx_symbols,
            snr
        )

        # -----------------------------
        # 4. Demodulation
        # -----------------------------

        received_bits = demodulate(
            rx_symbols,
            MODULATION
        )

        # Make length same as encoded bits
        received_bits = received_bits[
            :len(encoded_bits)
        ]

        # -----------------------------
        # 5. Viterbi Decoding
        # -----------------------------

        decoded_bits = viterbi_decode(
            received_bits,
            generators,
            K
        )

        # Remove termination bits
        decoded_bits = decoded_bits[
            :N_BITS
        ]

        # -----------------------------
        # 6. BER Calculation
        # -----------------------------

        errors = np.sum(
            original_bits != decoded_bits
        )

        ber = errors / N_BITS

        ber_values.append(ber)

        print(
            f"Rate={rate}, "
            f"SNR={snr:2d} dB, "
            f"BER={ber:.6f}"
        )

    results[rate] = ber_values


# ============================================================
# BER vs SNR GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

markers = {
    "1/2": "o",
    "1/3": "s",
    "1/4": "^"
}

for rate in ["1/2", "1/3", "1/4"]:

    plt.semilogy(
        SNR_dB,
        results[rate],
        marker=markers[rate],
        linewidth=2,
        markersize=6,
        label=f"Code Rate {rate}"
    )

plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Bit Error Rate (BER)", fontsize=12)

plt.title(
    f"BER vs SNR for Convolutional Codes\n"
    f"Modulation: {MODULATION}",
    fontsize=14
)

plt.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.6
)

plt.legend()
plt.ylim(1e-5, 1)
plt.tight_layout()

plt.show()