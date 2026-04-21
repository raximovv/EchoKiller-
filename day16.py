import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.signal as sig

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import sounddevice as sd
    HAS_PLAYBACK = True
except ImportError:
    HAS_PLAYBACK = False

SAMPLE_RATE = 16000
FILTER_ORDER = 2500
LEARNING_RATE = 0.02

def generate_synthetic_speech(duration=3.0, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    x = np.zeros_like(t)
    for burst_start in [0.2, 0.9, 1.6, 2.3]:
        burst_len = 0.5
        mask = (t >= burst_start) & (t < burst_start + burst_len)
        fundamental = np.random.uniform(100, 200)
        word = np.zeros_like(t)
        for harmonic in range(1, 5):
            word += np.sin(2 * np.pi * fundamental * harmonic * t) / harmonic
        envelope = np.zeros_like(t)
        envelope[mask] = np.sin(np.pi * (t[mask] - burst_start) / burst_len)
        x += word * envelope
    x = x / np.max(np.abs(x)) * 0.6
    return x.astype(np.float32)

def add_synthetic_echo(x, delay_ms=150, decay=0.5, sample_rate=SAMPLE_RATE):
    delay_samples = int(sample_rate * delay_ms / 1000)
    y = np.copy(x)
    if delay_samples < len(x):
        y[delay_samples:] += decay * x[:-delay_samples]
    return y.astype(np.float32)

def resample_audio(x, orig_sr, target_sr):
    if orig_sr == target_sr:
        return x.astype(np.float32)
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    y = sig.resample_poly(x, up, down)
    return y.astype(np.float32)

def load_or_generate():
    wav_files = [f for f in os.listdir(".") if f.lower().endswith(".wav")]
    if wav_files and HAS_SOUNDFILE:
        path = wav_files[0]
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        data = resample_audio(data, sr, SAMPLE_RATE)
        data = data / (np.max(np.abs(data)) + 1e-9) * 0.6
        return data.astype(np.float32), True, path
    x = generate_synthetic_speech()
    return x, False, None

def lms_filter(reference, mixed, filter_order, mu):
    n_samples = len(mixed)
    w = np.zeros(filter_order, dtype=np.float32)
    e = np.zeros(n_samples, dtype=np.float32)
    xbuf = np.zeros(filter_order, dtype=np.float32)

    for n in range(n_samples):
        xbuf[1:] = xbuf[:-1]
        xbuf[0] = reference[n]
        y_hat = np.dot(w, xbuf)
        err = mixed[n] - y_hat
        e[n] = err
        norm = np.dot(xbuf, xbuf) + 1e-6
        w = w + (mu / norm) * err * xbuf

    return e, w

def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-9)

def plot_results(reference, echoed, cleaned, coeffs, sample_rate):
    reference_n = normalize(reference)
    echoed_n = normalize(echoed)
    cleaned_n = normalize(cleaned)

    t = np.arange(len(reference_n)) / sample_rate
    tap_t = np.arange(len(coeffs)) / sample_rate * 1000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(t, echoed_n, label="Before")
    axes[0].plot(t, cleaned_n, label="After", alpha=0.8)
    axes[0].set_title("Before vs After Waveforms")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Normalized Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(tap_t[::5], coeffs[::5], width=0.8)
    axes[1].set_title("Learned FIR Coefficients")
    axes[1].set_xlabel("Delay (ms)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    reference, from_file, path = load_or_generate()

    if from_file:
        echoed = add_synthetic_echo(reference, delay_ms=150, decay=0.5)
        print(f"Loaded: {path}")
    else:
        echoed = add_synthetic_echo(reference, delay_ms=150, decay=0.5)
        print("No WAV found, generated synthetic speech.")

    print(f"Sample rate: {SAMPLE_RATE}")
    print(f"Filter order: {FILTER_ORDER}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Duration: {len(reference)/SAMPLE_RATE:.2f}s")

    start = time.time()
    cleaned, coeffs = lms_filter(reference, echoed, FILTER_ORDER, LEARNING_RATE)
    elapsed = time.time() - start

    echoed_energy = np.mean(echoed ** 2)
    cleaned_energy = np.mean(cleaned ** 2)
    reduction_db = 10 * np.log10((echoed_energy + 1e-9) / (cleaned_energy + 1e-9))

    print(f"Done in {elapsed:.2f}s")
    print(f"Echoed energy:  {echoed_energy:.6f}")
    print(f"Cleaned energy: {cleaned_energy:.6f}")
    print(f"Reduction: {reduction_db:.2f} dB")

    if HAS_SOUNDFILE:
        sf.write("echoed_input.wav", normalize(echoed), SAMPLE_RATE)
        sf.write("cleaned_output.wav", normalize(cleaned), SAMPLE_RATE)
        print("Saved: echoed_input.wav, cleaned_output.wav")

    if HAS_PLAYBACK:
        try:
            print("Playing echoed...")
            sd.play(normalize(echoed), SAMPLE_RATE)
            sd.wait()
            print("Playing cleaned...")
            sd.play(normalize(cleaned), SAMPLE_RATE)
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()

    plot_results(reference, echoed, cleaned, coeffs, SAMPLE_RATE)

if __name__ == "__main__":
    main()