"""QC inspection tool — sample and visualize pipeline stages.

Subcommands (via ``rf-datagen qc``):
    text       Show generated QSO messages, contest exchanges, etc.
    audio      Export TTS speech as WAV files
    modulated  Visualize clean IQ signals (spectrograms, waveforms)
    impaired   Visualize signals after channel impairments
    dataset    Inspect existing .npy training data on disk
    report     Full self-contained HTML report for one mode
"""

import base64
import csv
import glob
import os
import shutil
import sys
import tempfile
import wave

import numpy as np

from .constants import FS, WINDOW_LEN, SNR_LEVELS, SIGNAL_LABELS
from .impairments import normalize_power, extract_windows, apply_scenario, SCENARIO_NAMES
from .impairments.effects import add_awgn
from .impairments.scenarios import _SCENARIO_FUNCS


# ---------------------------------------------------------------------------
# Lazy imports for optional / heavy dependencies
# ---------------------------------------------------------------------------

def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ERROR: matplotlib required.  pip install rf-datagen[qc]",
              file=sys.stderr)
        sys.exit(1)


def _require_tts():
    try:
        from .content.tts import TTSEngine
        from .content.ham_text import gen_speech_text
        return TTSEngine, gen_speech_text
    except ImportError as e:
        print(f"ERROR: Cannot import TTS engine: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------

def iq_spectrogram(sig, fs, nfft=256, noverlap=192):
    """Compute bilateral spectrogram of complex IQ signal.

    Returns (freqs, times, power_db) with full negative+positive frequency axis.
    """
    hop = nfft - noverlap
    window = np.hanning(nfft)
    n_frames = max(1, (len(sig) - nfft) // hop + 1)

    S = np.zeros((nfft, n_frames))
    for i in range(n_frames):
        segment = sig[i * hop: i * hop + nfft] * window
        spec = np.fft.fftshift(np.fft.fft(segment))
        S[:, i] = 10 * np.log10(np.abs(spec) ** 2 + 1e-30)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    times = np.arange(n_frames) * hop / fs
    return freqs, times, S


def plot_iq_spectrogram(sig, fs, title, save_path, nfft=256):
    """Save bilateral spectrogram PNG for complex IQ signal."""
    plt = _require_matplotlib()
    freqs, times, S = iq_spectrogram(sig, fs, nfft)

    fig, ax = plt.subplots(figsize=(10, 4))
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    ax.imshow(S, aspect="auto", origin="lower", extent=extent,
              cmap="viridis", vmin=S.max() - 60, vmax=S.max())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_iq_waveform(sig, fs, title, save_path, max_samples=4096):
    """Save time-domain I/Q waveform PNG."""
    plt = _require_matplotlib()
    n = min(len(sig), max_samples)
    t = np.arange(n) / fs * 1000  # ms

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, sig[:n].real, linewidth=0.5, alpha=0.8, label="I")
    ax.plot(t, sig[:n].imag, linewidth=0.5, alpha=0.8, label="Q")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_psd(sig, fs, title, save_path, nfft=1024):
    """Save power spectral density PNG."""
    plt = _require_matplotlib()

    n_segs = max(1, len(sig) // nfft)
    psd = np.zeros(nfft)
    for i in range(n_segs):
        segment = sig[i * nfft: (i + 1) * nfft]
        if len(segment) < nfft:
            break
        windowed = segment * np.hanning(nfft)
        spec = np.fft.fftshift(np.fft.fft(windowed))
        psd += np.abs(spec) ** 2
    psd /= n_segs
    psd_db = 10 * np.log10(psd + 1e-30)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs, psd_db, linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_snr_grid(sig_clean, fs, mode, save_path, nfft=256):
    """Save grid of spectrograms at each SNR level for one clean signal."""
    plt = _require_matplotlib()

    n_snr = len(SNR_LEVELS)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    fig.suptitle(f"{mode} — SNR comparison", fontsize=14)

    for idx, snr in enumerate(SNR_LEVELS):
        ax = axes[idx // 4, idx % 4]
        impaired = add_awgn(normalize_power(sig_clean.copy()), snr)
        freqs, times, S = iq_spectrogram(impaired, fs, nfft)
        extent = [times[0], times[-1], freqs[0], freqs[-1]]
        ax.imshow(S, aspect="auto", origin="lower", extent=extent,
                  cmap="viridis", vmin=S.max() - 50, vmax=S.max())
        ax.set_title(f"SNR {snr} dB", fontsize=10)
        if idx % 4 == 0:
            ax.set_ylabel("Freq (Hz)")
        if idx >= 4:
            ax.set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_before_after(clean, impaired, fs, scenario, snr, save_path, nfft=256):
    """Save side-by-side clean vs impaired spectrogram."""
    plt = _require_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    freqs, times, S1 = iq_spectrogram(clean, fs, nfft)
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    vmax = max(S1.max(), 0)
    ax1.imshow(S1, aspect="auto", origin="lower", extent=extent,
               cmap="viridis", vmin=vmax - 50, vmax=vmax)
    ax1.set_title("Clean signal")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_xlabel("Time (s)")

    freqs, times, S2 = iq_spectrogram(impaired, fs, nfft)
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    vmax = max(S2.max(), 0)
    ax2.imshow(S2, aspect="auto", origin="lower", extent=extent,
               cmap="viridis", vmin=vmax - 50, vmax=vmax)
    ax2.set_title(f"{scenario} @ {snr} dB SNR")
    ax2.set_xlabel("Time (s)")

    fig.suptitle("Clean vs Impaired", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def sig_to_wav(sig, fs, path, stereo_iq=True):
    """Write signal to WAV file.

    If stereo_iq=True, writes I=left Q=right stereo WAV.
    Otherwise writes magnitude as mono.
    """
    if stereo_iq and np.iscomplexobj(sig):
        peak = max(np.max(np.abs(sig.real)), np.max(np.abs(sig.imag)), 1e-10)
        i_int = np.clip(sig.real / peak * 32000, -32767, 32767).astype(np.int16)
        q_int = np.clip(sig.imag / peak * 32000, -32767, 32767).astype(np.int16)
        interleaved = np.empty(len(sig) * 2, dtype=np.int16)
        interleaved[0::2] = i_int
        interleaved[1::2] = q_int
        n_channels = 2
        data = interleaved.tobytes()
    else:
        audio = sig.real if np.iscomplexobj(sig) else sig
        peak = max(np.max(np.abs(audio)), 1e-10)
        samples = np.clip(audio / peak * 32000, -32767, 32767).astype(np.int16)
        n_channels = 1
        data = samples.tobytes()

    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(data)
    return path


def _png_to_base64(path):
    """Read PNG file and return base64-encoded data URI."""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _wav_to_base64(path):
    """Read WAV file and return base64-encoded data URI."""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _audio_tag(wav_path, label=""):
    """HTML audio element with inline base64 WAV and optional label."""
    uri = _wav_to_base64(wav_path)
    lbl = f"<small>{label}</small><br>" if label else ""
    return (f'<div style="margin:4px 0">{lbl}'
            f'<audio controls preload="none" src="{uri}">'
            f'Your browser does not support audio.</audio></div>')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_active_window(mode, max_retries=20):
    """Get a WINDOW_LEN chunk with non-zero power (skip silence gaps)."""
    from .generators.synthetic import SYNTHESIZERS

    for _ in range(max_retries):
        sig = SYNTHESIZERS[mode.upper()]()
        if len(sig) > WINDOW_LEN:
            start = np.random.randint(0, len(sig) - WINDOW_LEN)
            window = sig[start:start + WINDOW_LEN]
        else:
            window = np.zeros(WINDOW_LEN, dtype=np.complex128)
            window[:len(sig)] = sig
        power = np.mean(np.abs(window) ** 2)
        if power > 1e-10:
            return normalize_power(window)
    # Fallback: use extract_windows which filters silence
    sig = SYNTHESIZERS[mode.upper()]()
    windows = extract_windows(sig)
    if len(windows) > 0:
        return normalize_power(windows[np.random.randint(len(windows))])
    return normalize_power(window)  # last resort


# ---------------------------------------------------------------------------
# Subcommand: text
# ---------------------------------------------------------------------------

def cmd_text(args):
    """Show generated text content for each generator type."""
    from .content.ham_text import gen_speech_text, get_text_for_mode, gen_packet_content

    np.random.seed(args.seed)

    if args.generator == "analog":
        print(f"=== Analog voice text generator ({args.count} samples) ===\n")
        for i in range(args.count):
            text, style = gen_speech_text()
            print(f"[{i+1:3d}] style={style:<8s} len={len(text):>4d}  "
                  f"{text[:120]}{'...' if len(text) > 120 else ''}")

    elif args.generator == "fldigi":
        mode = args.mode or "PSK31"
        print(f"=== Fldigi text generator: {mode} ({args.count} samples) ===\n")
        for i in range(args.count):
            text = get_text_for_mode(mode, target_chars=200)
            lines = text.strip().split("\n")
            preview = lines[0][:120]
            print(f"[{i+1:3d}] len={len(text):>4d}  {preview}"
                  f"{'...' if len(text) > 120 or len(lines) > 1 else ''}")

    elif args.generator == "packet":
        print(f"=== Packet content generator ({args.count} batches) ===\n")
        for i in range(args.count):
            content = gen_packet_content(n_packets=5)
            print(f"--- batch {i+1} ---")
            print(content)
            print()

    elif args.generator == "digivoice":
        print(f"=== Digital voice speech text ({args.count} samples) ===\n")
        print("(Same text pool as analog — content fed to TTS then encoded)")
        print()
        for i in range(args.count):
            text, style = gen_speech_text()
            print(f"[{i+1:3d}] style={style:<8s} len={len(text):>4d}  "
                  f"{text[:120]}{'...' if len(text) > 120 else ''}")

    elif args.generator == "all":
        args.count = min(args.count, 5)
        for gen in ["analog", "fldigi", "packet"]:
            args.generator = gen
            cmd_text(args)
            print()
    else:
        print(f"Unknown generator: {args.generator}")
        print("Available: analog, fldigi, packet, digivoice, all")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Subcommand: audio
# ---------------------------------------------------------------------------

def cmd_audio(args):
    """Generate and export TTS speech audio as WAV files."""
    TTSEngine, gen_speech_text = _require_tts()
    np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    print(f"Initializing TTS engine (voice cache: {args.voice_cache})...")
    tts = TTSEngine(args.voice_cache)
    tmpdir = tempfile.mkdtemp(prefix="qc_audio_")

    print(f"\nGenerating {args.count} speech samples:\n")

    for i in range(args.count):
        text, style = gen_speech_text()
        audio, wav_fs = tts.synthesize(text, tmpdir)

        if len(audio) < 100:
            print(f"[{i+1:3d}] SKIP (too short)")
            continue

        duration = len(audio) / wav_fs
        wav_path = os.path.join(args.output, f"speech_{i+1:03d}_{style}.wav")
        sig_to_wav(audio, wav_fs, wav_path, stereo_iq=False)

        print(f"[{i+1:3d}] style={style:<8s} dur={duration:.1f}s  "
              f"fs={wav_fs}  {wav_path}")
        print(f"      \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

        if args.play:
            os.system(f"aplay -q {wav_path} 2>/dev/null")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\nAudio samples saved to: {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: modulated
# ---------------------------------------------------------------------------

def cmd_modulated(args):
    """Generate and visualize clean modulated IQ signals."""
    from .generators.synthetic import SYNTHESIZERS

    plt = _require_matplotlib()
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    if args.all_modes:
        modes = sorted(SYNTHESIZERS.keys())
    else:
        modes = [m.upper() for m in args.mode]

    for mode in modes:
        if mode not in SYNTHESIZERS:
            print(f"WARNING: Unknown mode '{mode}', skipping")
            continue

        print(f"\n{mode}:")
        for i in range(args.count):
            window = _get_active_window(mode)
            prefix = f"{mode}_{i+1:02d}"

            spec_path = os.path.join(args.output, f"{prefix}_spectrogram.png")
            plot_iq_spectrogram(window, FS, f"{mode} — clean spectrogram",
                                spec_path)

            wave_path = os.path.join(args.output, f"{prefix}_waveform.png")
            plot_iq_waveform(window, FS, f"{mode} — I/Q waveform", wave_path)

            psd_path = os.path.join(args.output, f"{prefix}_psd.png")
            plot_psd(window, FS, f"{mode} — PSD", psd_path)

            wav_path = os.path.join(args.output, f"{prefix}_iq.wav")
            sig_to_wav(window, FS, wav_path, stereo_iq=True)

            dur_ms = len(window) / FS * 1000
            print(f"  [{i+1}] {dur_ms:.0f}ms  "
                  f"power={np.mean(np.abs(window)**2):.4f}  "
                  f"-> {spec_path}")

        if args.snr_grid:
            window = _get_active_window(mode)
            grid_path = os.path.join(args.output, f"{mode}_snr_grid.png")
            plot_snr_grid(window, FS, mode, grid_path)
            print(f"  SNR grid -> {grid_path}")

    print(f"\nOutput: {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: impaired
# ---------------------------------------------------------------------------

def cmd_impaired(args):
    """Visualize signals after channel impairments."""
    _require_matplotlib()
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    mode = args.mode.upper()
    clean = _get_active_window(mode)

    if args.all_snr:
        grid_path = os.path.join(args.output, f"{mode}_snr_grid.png")
        plot_snr_grid(clean, FS, mode, grid_path)
        print(f"SNR grid -> {grid_path}")

        for snr in SNR_LEVELS:
            impaired, scenario = apply_scenario(clean.copy(), snr)
            prefix = f"{mode}_snr{snr:+d}_{scenario}"

            ba_path = os.path.join(args.output, f"{prefix}_before_after.png")
            plot_before_after(clean, impaired, FS, scenario, snr, ba_path)

            spec_path = os.path.join(args.output, f"{prefix}_spectrogram.png")
            plot_iq_spectrogram(impaired, FS,
                                f"{mode} — {scenario} @ {snr} dB",
                                spec_path)
            print(f"  SNR {snr:+3d} dB  scenario={scenario:<20s}  -> {ba_path}")

    elif args.scenario:
        if args.scenario not in SCENARIO_NAMES:
            print(f"ERROR: Unknown scenario '{args.scenario}'")
            print(f"Available: {', '.join(SCENARIO_NAMES)}")
            return 1

        scenario_fn = _SCENARIO_FUNCS[args.scenario]
        snr = args.snr

        for i in range(args.count):
            impaired = scenario_fn(clean.copy(), snr, FS)
            prefix = f"{mode}_{args.scenario}_snr{snr:+d}_{i+1:02d}"

            ba_path = os.path.join(args.output, f"{prefix}_before_after.png")
            plot_before_after(clean, impaired, FS, args.scenario, snr, ba_path)

            wav_path = os.path.join(args.output, f"{prefix}_iq.wav")
            sig_to_wav(impaired, FS, wav_path, stereo_iq=True)

            print(f"  [{i+1}] {args.scenario} @ {snr} dB  -> {ba_path}")

    else:
        for i in range(args.count):
            snr = args.snr
            impaired, scenario = apply_scenario(clean.copy(), snr)
            prefix = f"{mode}_{scenario}_snr{snr:+d}_{i+1:02d}"

            ba_path = os.path.join(args.output, f"{prefix}_before_after.png")
            plot_before_after(clean, impaired, FS, scenario, snr, ba_path)

            wav_path = os.path.join(args.output, f"{prefix}_iq.wav")
            sig_to_wav(impaired, FS, wav_path, stereo_iq=True)

            print(f"  [{i+1}] {scenario} @ {snr} dB  -> {ba_path}")

    print(f"\nOutput: {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: dataset
# ---------------------------------------------------------------------------

def cmd_dataset(args):
    """Inspect an existing .npy + .csv training dataset on disk."""
    plt = _require_matplotlib()
    np.random.seed(args.seed)

    data_path = args.path

    npy_files = sorted(glob.glob(os.path.join(data_path, "*_iq.npy")))
    csv_files = sorted(glob.glob(os.path.join(data_path, "*_tags.csv")))

    if not npy_files:
        print(f"ERROR: No *_iq.npy files found in {data_path}", file=sys.stderr)
        return 1

    npy_path = npy_files[0]
    csv_path = csv_files[0] if csv_files else None

    print(f"Loading: {npy_path}")
    iq_data = np.load(npy_path, mmap_mode="r")
    print(f"  Shape: {iq_data.shape}, dtype: {iq_data.dtype}")
    print(f"  Size: {os.path.getsize(npy_path) / (1024**3):.1f} GB")

    labels = []
    snrs = []
    scenarios = []
    if csv_path:
        print(f"  Tags: {csv_path}")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(row.get("mode", ""))
                snrs.append(row.get("snr", ""))
                scenarios.append(row.get("scenario", ""))

    if labels:
        from collections import Counter
        counts = Counter(labels)
        print(f"\n  Class distribution ({len(counts)} classes, "
              f"{len(labels):,} samples):")
        for cls in sorted(counts, key=lambda c: SIGNAL_LABELS.index(c)
                          if c in SIGNAL_LABELS else 999):
            bar = "#" * (counts[cls] // max(1, max(counts.values()) // 40))
            print(f"    {cls:<15s} {counts[cls]:>7,d}  {bar}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)

        if args.mode and labels:
            mode = args.mode.upper()
            valid_indices = [i for i, l in enumerate(labels) if l == mode]
            if not valid_indices:
                print(f"\nERROR: Mode '{mode}' not found in dataset")
                print(f"Available: {', '.join(sorted(set(labels)))}")
                return 1
            sample_indices = np.random.choice(valid_indices,
                                               size=min(args.count, len(valid_indices)),
                                               replace=False)
        else:
            sample_indices = np.random.choice(len(iq_data),
                                               size=min(args.count, len(iq_data)),
                                               replace=False)

        print(f"\n  Generating {len(sample_indices)} sample visualizations...")

        for idx in sorted(sample_indices):
            window = iq_data[idx]
            label = labels[idx] if idx < len(labels) else "?"
            snr = snrs[idx] if idx < len(snrs) else "?"
            scenario = scenarios[idx] if idx < len(scenarios) else "?"
            prefix = f"sample_{idx:06d}_{label}_snr{snr}_{scenario}"

            spec_path = os.path.join(args.output, f"{prefix}_spectrogram.png")
            plot_iq_spectrogram(window, FS,
                                f"#{idx} {label} SNR={snr}dB {scenario}",
                                spec_path)

            psd_path = os.path.join(args.output, f"{prefix}_psd.png")
            plot_psd(window, FS, f"#{idx} {label} — PSD", psd_path)

            wav_path = os.path.join(args.output, f"{prefix}_iq.wav")
            sig_to_wav(window, FS, wav_path, stereo_iq=True)

            print(f"    #{idx:>6d}  {label:<15s}  snr={snr:>3s}  "
                  f"scenario={scenario:<20s}  -> {spec_path}")

        if labels:
            fig, ax = plt.subplots(figsize=(12, 5))
            sorted_classes = sorted(counts.keys(),
                                     key=lambda c: SIGNAL_LABELS.index(c)
                                     if c in SIGNAL_LABELS else 999)
            ax.bar(range(len(sorted_classes)),
                   [counts[c] for c in sorted_classes])
            ax.set_xticks(range(len(sorted_classes)))
            ax.set_xticklabels(sorted_classes, rotation=45, ha="right",
                                fontsize=8)
            ax.set_ylabel("Samples")
            ax.set_title("Class Distribution")
            fig.tight_layout()
            hist_path = os.path.join(args.output, "class_distribution.png")
            fig.savefig(hist_path, dpi=120)
            plt.close(fig)
            print(f"\n  Class histogram -> {hist_path}")

        print(f"\n  Output: {args.output}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------

def cmd_probe(args):
    """GNU Radio probe — analyze, decode, or transform signals at any pipeline stage."""
    from .generators.synthetic import SYNTHESIZERS
    from .gnuradio_probe import GnuRadioProbe, ProbePoint

    np.random.seed(args.seed)

    mode = args.mode.upper()
    probe = GnuRadioProbe(fs=FS)

    # Generate or load signal
    if args.input:
        iq = np.fromfile(args.input, dtype=np.complex64).astype(np.complex128)
        print(f"Loaded {len(iq)} samples from {args.input}")
    elif mode in SYNTHESIZERS:
        iq = SYNTHESIZERS[mode](fs=FS, window_len=WINDOW_LEN)
        print(f"Generated {mode}: {len(iq)} samples ({len(iq)/FS:.3f}s)")
    else:
        print(f"ERROR: Unknown mode '{mode}' and no --input file")
        return 1

    # Determine probe point
    point_map = {
        "after-generation": ProbePoint.AFTER_GENERATION,
        "after-impairments": ProbePoint.AFTER_IMPAIRMENTS,
        "after-windowing": ProbePoint.AFTER_WINDOWING,
        "custom": ProbePoint.CUSTOM,
    }
    probe_point = point_map.get(args.point, ProbePoint.CUSTOM)

    # Apply impairments if requested
    if args.snr is not None:
        from .impairments import normalize_power
        from .impairments.effects import add_awgn
        iq = add_awgn(normalize_power(iq), args.snr)
        probe_point = ProbePoint.AFTER_IMPAIRMENTS
        print(f"Applied AWGN at SNR={args.snr} dB")

    if args.action == "analyze":
        result = probe.analyze(iq, fs=FS, mode=mode, probe_point=probe_point)
        print(f"\n{'='*60}")
        print(f"Probe Analysis: {mode} @ {probe_point.value}")
        print(f"{'='*60}")
        for k, v in sorted(result.measurements.items()):
            if isinstance(v, float):
                print(f"  {k:<25s} {v:>12.3f}")
            else:
                print(f"  {k:<25s} {v!s:>12s}")

    elif args.action == "decode":
        print(f"\nAvailable decoders: {probe.available_decoders}")
        result = probe.decode(iq, fs=FS, decoder=args.decoder, mode=mode,
                              probe_point=probe_point)
        print(f"\nDecode result: {'SUCCESS' if result.success else 'FAILED'}")
        if result.decoded_text:
            print(f"Decoded text:\n{result.decoded_text}")
        if result.decode_confidence is not None:
            print(f"Confidence: {result.decode_confidence:.2f}")
        if result.error:
            print(f"Error: {result.error}")

    elif args.action == "transform":
        params = {}
        if args.transform_params:
            for p in args.transform_params:
                k, v = p.split("=", 1)
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v

        result = probe.transform(iq, fs=FS, flowgraph=args.flowgraph,
                                 params=params)
        if result.success and result.output_iq is not None:
            print(f"Transform output: {len(result.output_iq)} samples")
            if args.output:
                result.output_iq.astype(np.complex64).tofile(args.output)
                print(f"Saved to: {args.output}")
            # Also analyze the output
            analysis = probe.analyze(result.output_iq, fs=FS, mode=mode)
            print(f"\nPost-transform analysis:")
            for k, v in sorted(analysis.measurements.items()):
                if isinstance(v, float):
                    print(f"  {k:<25s} {v:>12.3f}")
        else:
            print(f"Transform failed: {result.error}")

    # Save visualizations if output dir specified
    if args.output_dir:
        _require_matplotlib()
        os.makedirs(args.output_dir, exist_ok=True)
        prefix = f"{mode}_{probe_point.value}"

        spec_path = os.path.join(args.output_dir, f"{prefix}_spectrogram.png")
        plot_iq_spectrogram(iq, FS, f"{mode} @ {probe_point.value}", spec_path)
        print(f"\nSpectrogram: {spec_path}")

        psd_path = os.path.join(args.output_dir, f"{prefix}_psd.png")
        plot_psd(iq[:min(len(iq), 8192)], FS, f"{mode} PSD", psd_path)
        print(f"PSD: {psd_path}")

        wav_path = os.path.join(args.output_dir, f"{prefix}_iq.wav")
        sig_to_wav(iq[:min(len(iq), FS * 5)], FS, wav_path, stereo_iq=True)
        print(f"WAV: {wav_path}")

    return 0


def cmd_benchmark(args):
    """Benchmark sdr library vs NumPy modulation performance."""
    from .dsp.modulation_sdr import sdr_available, benchmark as run_benchmark

    if not sdr_available():
        print("sdr library not installed. Install with: pip install rf-datagen[accel]")
        print("\nBenchmark skipped — sdr library required for comparison.")
        return 0

    print(f"Running modulation benchmark ({args.trials} trials, "
          f"{args.symbols} symbols)...\n")

    results = run_benchmark(n_trials=args.trials, n_symbols=args.symbols)

    print(f"{'Function':<15s} {'NumPy (ms)':>12s} {'sdr (ms)':>12s} {'Speedup':>10s}  Recommendation")
    print("-" * 65)
    for name, r in results.items():
        speedup = r["speedup"]
        rec = "USE sdr" if speedup >= 2.0 else "keep NumPy"
        print(f"{name:<15s} {r['numpy_ms']:>12.3f} {r['sdr_ms']:>12.3f} "
              f"{speedup:>9.1f}x  {rec}")

    print(f"\nThreshold: adopt sdr if speedup >= 2.0x")
    return 0


def cmd_report(args):
    """Generate a self-contained HTML report for one mode."""
    from .generators.synthetic import SYNTHESIZERS

    plt = _require_matplotlib()
    np.random.seed(args.seed)

    mode = args.mode.upper()
    if mode not in SYNTHESIZERS:
        print(f"ERROR: Unknown mode '{mode}'")
        print(f"Available: {', '.join(sorted(SYNTHESIZERS.keys()))}")
        return 1

    os.makedirs(args.output, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="qc_report_")

    html_parts = [f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>QC Report: {mode}</title>
<style>
body {{ font-family: monospace; max-width: 1200px; margin: 0 auto;
       padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #00d4ff; }} h2 {{ color: #ff6b6b; border-bottom: 1px solid #333; }}
img {{ max-width: 100%; border: 1px solid #333; margin: 5px 0; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.grid > img {{ flex: 1 1 45%; min-width: 400px; }}
.grid > div {{ flex: 1 1 45%; min-width: 400px; }}
.grid > div img {{ width: 100%; }}
pre {{ background: #0d0d1a; padding: 10px; overflow-x: auto;
       border: 1px solid #333; }}
audio {{ margin: 5px 0; }}
</style></head><body>
<h1>QC Report: {mode}</h1>
<p>Generated: seed={args.seed}, FS={FS} Hz, WINDOW_LEN={WINDOW_LEN}</p>
"""]

    # Section 1: Text content + TTS audio
    html_parts.append("<h2>1. Text Content &amp; TTS Audio</h2><pre>")
    tts_texts = []
    try:
        from .content.ham_text import gen_speech_text
        for i in range(5):
            text, style = gen_speech_text()
            tts_texts.append((text, style))
            html_parts.append(f"[{style:8s}] {text}\n")
    except Exception:
        html_parts.append("(text generation not available for this mode)\n")
    html_parts.append("</pre>")

    # TTS audio samples
    try:
        from .content.tts import TTSEngine
        tts = TTSEngine(args.voice_cache)
        html_parts.append("<h3>TTS Speech Samples</h3>")
        for i, (text, style) in enumerate(tts_texts[:3]):
            audio, wav_fs = tts.synthesize(text, tmpdir)
            if len(audio) > 100:
                tts_wav = os.path.join(tmpdir, f"tts_{i}.wav")
                sig_to_wav(audio, wav_fs, tts_wav, stereo_iq=False)
                html_parts.append(
                    _audio_tag(tts_wav,
                               f"TTS [{style}] {text[:60]}{'...' if len(text)>60 else ''}"))
    except Exception as e:
        html_parts.append(f"<p><em>TTS not available: {e}</em></p>")

    # Section 2: Clean signal
    html_parts.append("<h2>2. Clean Modulated Signal</h2>")
    n_samples = min(args.count, 3)
    html_parts.append('<div class="grid">')

    first_clean = None
    for i in range(n_samples):
        window = _get_active_window(mode)

        spec_path = os.path.join(tmpdir, f"clean_{i}_spec.png")
        plot_iq_spectrogram(window, FS, f"{mode} clean #{i+1}", spec_path)
        html_parts.append(f'<img src="{_png_to_base64(spec_path)}">')

        iq_wav = os.path.join(tmpdir, f"clean_{i}_iq.wav")
        sig_to_wav(window, FS, iq_wav, stereo_iq=True)
        html_parts.append(_audio_tag(iq_wav, f"Clean IQ #{i+1} (L=I, R=Q)"))

        if i == 0:
            first_clean = window.copy()

            wave_path = os.path.join(tmpdir, f"clean_{i}_wave.png")
            plot_iq_waveform(window, FS, f"{mode} I/Q waveform", wave_path)
            html_parts.append(f'<img src="{_png_to_base64(wave_path)}">')

            psd_path = os.path.join(tmpdir, f"clean_{i}_psd.png")
            plot_psd(window, FS, f"{mode} PSD", psd_path)
            html_parts.append(f'<img src="{_png_to_base64(psd_path)}">')

    html_parts.append("</div>")

    # Section 3: SNR comparison grid
    html_parts.append("<h2>3. SNR Comparison</h2>")
    grid_path = os.path.join(tmpdir, "snr_grid.png")
    plot_snr_grid(first_clean, FS, mode, grid_path)
    html_parts.append(f'<img src="{_png_to_base64(grid_path)}" '
                       f'style="max-width:100%">')

    # Section 4: Scenario comparison
    html_parts.append("<h2>4. Impairment Scenarios (SNR=10 dB)</h2>")
    html_parts.append('<div class="grid">')

    for scenario_name in SCENARIO_NAMES:
        scenario_fn = _SCENARIO_FUNCS[scenario_name]
        impaired = scenario_fn(first_clean.copy(), 10, FS)

        ba_path = os.path.join(tmpdir, f"scenario_{scenario_name}.png")
        plot_before_after(first_clean, impaired, FS, scenario_name, 10,
                           ba_path)
        html_parts.append(f'<div style="flex:1 1 45%;min-width:400px">')
        html_parts.append(f'<img src="{_png_to_base64(ba_path)}">')

        imp_wav = os.path.join(tmpdir, f"scenario_{scenario_name}.wav")
        sig_to_wav(impaired, FS, imp_wav, stereo_iq=True)
        html_parts.append(_audio_tag(imp_wav, f"{scenario_name} @ 10 dB"))
        html_parts.append("</div>")

    html_parts.append("</div>")

    # Section 5: Signal stats
    html_parts.append("<h2>5. Signal Statistics</h2><pre>")
    sig = SYNTHESIZERS[mode]()
    html_parts.append(f"Raw signal length: {len(sig)} samples "
                       f"({len(sig)/FS:.3f}s)\n")
    html_parts.append(f"Power: {np.mean(np.abs(normalize_power(sig))**2):.4f}\n")
    html_parts.append(f"Peak I: {np.max(np.abs(sig.real)):.4f}\n")
    html_parts.append(f"Peak Q: {np.max(np.abs(sig.imag)):.4f}\n")
    html_parts.append(f"I/Q correlation: "
                       f"{np.abs(np.corrcoef(sig.real, sig.imag)[0,1]):.4f}\n")

    # Bandwidth estimate
    spec = np.abs(np.fft.fftshift(np.fft.fft(normalize_power(sig))))**2
    freqs = np.fft.fftshift(np.fft.fftfreq(len(sig), 1/FS))
    spec_db = 10 * np.log10(spec / spec.max() + 1e-30)
    bw3_mask = spec_db > -3
    bw3_freqs = freqs[bw3_mask]
    bw3 = (bw3_freqs[-1] - bw3_freqs[0]) if len(bw3_freqs) > 1 else 0
    bw10_mask = spec_db > -10
    bw10_freqs = freqs[bw10_mask]
    bw10 = (bw10_freqs[-1] - bw10_freqs[0]) if len(bw10_freqs) > 1 else 0
    html_parts.append(f"3 dB bandwidth: {bw3:.0f} Hz\n")
    html_parts.append(f"10 dB bandwidth: {bw10:.0f} Hz\n")
    html_parts.append("</pre>")

    html_parts.append("</body></html>")

    # Write HTML
    html_path = os.path.join(args.output, f"qc_report_{mode}.html")
    with open(html_path, "w") as f:
        f.write("".join(html_parts))

    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"Report saved: {html_path}")
    print(f"  Open in browser: file://{os.path.abspath(html_path)}")
    return 0
