# rf-datagen

RF signal IQ dataset generator for ML training — 90 signal classes across three sample-rate domains with 19 realistic channel impairment scenarios.

## Overview

rf-datagen synthesizes complex IQ baseband samples for training RF signal classifiers. It operates across three sample-rate domains (narrowband 12 kHz, moderate 1 MHz, wideband 20 MHz), each with its own window length, dtype, and synthesizer registry. Output is stored as NumPy arrays with per-window metadata including signal class, SNR, scenario, and domain. The generator applies configurable propagation/receiver impairment scenarios so models train against realistic channel conditions out of the box.

### Multi-Rate Domain Architecture

| Domain | Sample Rate | Window Length | Nyquist | dtype | Classes |
|--------|------------|---------------|---------|-------|---------|
| Narrowband | 12 kHz | 2,048 | 6 kHz | complex128 | 68 |
| Moderate | 1 MHz | 131,072 | 500 kHz | complex64 | 14 |
| Wideband | 20 MHz | 2,097,152 | 10 MHz | complex64 | 8 |

### Signal Classes (90)

#### Narrowband (68 classes — 12 kHz)

| Category | Modes |
|---|---|
| Digital text | CW, PSK31, PSK63, QPSK, PSK125, 8PSK, RTTY, OLIVIA, JS8, DOMINOEX, MT63, HELLSCHREIBER, MFSK16, MFSK32, CONTESTIA, THOR, FSQ, IFKP, THROB |
| WSJT-X / weak signal | FT8, FT4, WSPR, JT65, JT9, MSK144 |
| Analog voice | SSB, AM, FM |
| Image | SSTV, FAX |
| Data / packet | NAVTEX, PACKET, ARDOP, BELL103, BELL202 |
| Digital voice | FREEDV, M17, DMR, DSTAR, YSF, P25, NXDN |
| Broadcast / digital | ATV, LORA, POCSAG, FLEX, HDRADIO, DRM, EAS |
| Tones / signaling | DTMF, SELCAL, ATIS, NDB, ACARS, AIS |
| Time signals | WWVB, DCF77 |
| IoT / telemetry | SIGFOX, TPMS, SCADA_TELEMETRY, TETRA |
| EW / jamming | SPOT_JAMMER, SWEEP_JAMMER, NOISE_JAMMER, BARRAGE_JAMMER |
| Radar | PULSE_RADAR, BARKER_RADAR |
| Other | NOISE |

#### Moderate (14 classes — 1 MHz)

| Category | Modes |
|---|---|
| Short-range wireless | BLE, ZWAVE, DECT |
| Aviation / maritime | ADS_B, VDL2 |
| Cellular | GSM_BURST, IRIDIUM |
| Radar | LFM_RADAR, FMCW_RADAR, PHASE_CODED_RADAR |
| Satellite | NOAA_APT, COSPAS_SARSAT |
| Spread spectrum | LORA_WIDE |
| Broadcast | DRM_WIDE |

#### Wideband (8 classes — 20 MHz)

| Category | Modes |
|---|---|
| WiFi / cellular | WIFI_PREAMBLE, LTE_FRAME, FIVEG_NR |
| Navigation | GPS_L1, LORAN_C_WIDE |
| IoT | ZIGBEE |
| Broadcast | DAB, DVB_T |

### Impairment Scenarios (19)

Samples are distributed across configurable propagation/receiver scenarios with configurable SNR levels (default: 25, 20, 15, 10, 5, 0, -5, -10 dB) and relative weights:

| Category | Scenarios |
|---|---|
| HF propagation | hf_clean, hf_good, hf_poor, auroral |
| VHF/UHF | vhf_mobile, uhf_urban |
| Receiver | sdr_desktop, contest_crowded, near_far |
| Transmitter | overdriven, poorly_operated, vintage |
| Indoor / urban | indoor_multipath, urban_cellular |
| Mobile | automotive |
| Satellite | leo_satellite |
| Radar | radar_clutter |
| Maritime | maritime |
| ISM band | ism_congested |

Each scenario applies a chain of 26 channel effects (AWGN, Watterson fading, Rayleigh/Rician fading, Doppler, multipath, atmospheric noise, phase noise, IQ imbalance, clock drift, adjacent signals, etc.) and optionally a transmitter model (ALC compression, RF clipping, hum, key clicks, VFO drift) with four operator profiles.

See `[impairments]` in `config.toml` for weight configuration.

## Prerequisites

- Python >= 3.11
- NumPy, SciPy, Pillow, PySSTV (installed automatically)
- Optional Python packages:
  - `matplotlib` — QC visualizations (`pip install rf-datagen[qc]`)
  - `scikit-dsp-comm` — higher-fidelity Watterson channel model (`pip install rf-datagen[sdc]`)
  - `sdr` — Numba-accelerated modulation (`pip install rf-datagen[accel]`)
- Optional external tools per generator:
  - **fldigi** — `fldigi` (for real fldigi-encoded digital modes)
  - **wsjtx** — `wsjtx` tools (`jt9`, `wsprd`, etc.)
  - **analog/digivoice** — `piper` TTS engine + voice models
  - **packet** — `direwolf` (for AX.25 packet encoding)
  - **probe** — `gnuradio` + `multimon-ng` (for signal analysis/decoding)

## Installation

### pip (Python only)

```bash
pip install -e .
```

This installs the Python package but not the external CLI tools (fldigi, wsjtx, direwolf, piper, codec2, etc.). You'll need to provide those separately for generators that require them — generators with missing tools are skipped automatically.

### Nix flake (recommended)

The flake provides a complete dev shell with all Python dependencies and external CLI tools:

```bash
nix develop
```

### Nix flake + direnv

The repo includes an `.envrc` so the environment activates automatically when you enter the directory:

```bash
# one-time setup after cloning
direnv allow
```

After this, every `cd` into the project directory loads the full environment automatically.

### devenv

If you use [devenv](https://devenv.sh) for your Nix-based development environments, you can consume the flake directly. Create a `devenv.nix` in the project root:

```nix
{ inputs, ... }: {
  imports = [
    inputs.rf-datagen.devShells.default
  ];
}
```

Or point devenv at the flake in your `devenv.yaml`:

```yaml
inputs:
  rf-datagen:
    url: github:Quantum-Serendipity/rf-datagen
```

Then activate with:

```bash
devenv shell
```

Or with direnv integration (`devenv init` generates the `.envrc` for you):

```bash
direnv allow
```

## Usage

### Generate a dataset

```bash
rf-datagen generate -c config.toml

# Generate specific domains
rf-datagen generate -c config.toml --domains narrowband,moderate
rf-datagen generate -c config.toml --domains wideband

# Generate specific generators only
rf-datagen generate -c config.toml --generators synthetic,synthetic_moderate
```

### List signal classes

```bash
rf-datagen list
```

### Validate a dataset

```bash
rf-datagen validate ./output
```

### Round-trip validation

```bash
rf-datagen validate-roundtrip              # all modes
rf-datagen validate-roundtrip --clean-only --trials 3  # quick check
```

### Inspect a signal class (from existing dataset)

```bash
rf-datagen inspect ./output --class FT8
```

### QC inspection tool

The `qc` subcommand generates on-the-fly visualizations and self-contained HTML reports for inspecting pipeline stages — no pre-generated dataset needed. Requires `matplotlib` (`pip install rf-datagen[qc]`).

```bash
# Show generated text content for each generator type
rf-datagen qc text --generator analog --count 10
rf-datagen qc text --generator fldigi --mode PSK31

# Export TTS speech audio as WAV files
rf-datagen qc audio --count 5 --output /tmp/qc/audio

# Visualize clean modulated signals (spectrograms, waveforms, PSD)
# Works across all domains — automatically resolves sample rate and window length
rf-datagen qc modulated --mode FT8 CW --output /tmp/qc/modulated
rf-datagen qc modulated --mode BLE WIFI_PREAMBLE --output /tmp/qc/wideband
rf-datagen qc modulated --all-modes --snr-grid

# Visualize signals after channel impairments
rf-datagen qc impaired --mode CW --all-snr --output /tmp/qc/impaired
rf-datagen qc impaired --mode DMR --scenario hf_poor --snr 5

# Inspect an existing .npy dataset on disk (class distribution, sample plots)
rf-datagen qc dataset --path ./output --mode FT8 --output /tmp/qc/dataset

# Full self-contained HTML report for one mode (spectrograms, audio, SNR grid,
# all 19 impairment scenarios, signal statistics)
rf-datagen qc report --mode FT8 --output /tmp/qc/reports

# GNU Radio probe — analyze, decode, or transform signals
rf-datagen qc probe --mode FT8 --action analyze
rf-datagen qc probe --mode POCSAG --action decode --decoder pocsag
rf-datagen qc probe --mode CW --action transform --flowgraph bandpass

# Benchmark sdr library vs NumPy modulation performance
rf-datagen qc benchmark
```

## Configuration

See `config.toml` for the full set of options. Key sections:

- **`[dataset]`** — sample rate, window length, output directory, seed, worker count, domains
- **`[impairments]`** — SNR levels, frequency offset, stride, power threshold
- **`[impairments.scenarios]`** — relative weights for all 19 propagation/receiver scenarios
- **`[generators.*]`** — per-generator enable/disable, samples per class, and generator-specific tuning

### Generators

| Generator | Domain | Tool Deps | Description |
|-----------|--------|-----------|-------------|
| `synthetic` | narrowband | none | Pure-Python synthesis for 68 narrowband classes |
| `synthetic_moderate` | moderate | none | Pure-Python synthesis for 14 moderate-rate classes |
| `synthetic_wideband` | wideband | none | Pure-Python synthesis for 8 wideband classes |
| `fldigi` | narrowband | fldigi | Real fldigi-encoded digital modes |
| `wsjtx` | narrowband | jt9, wsprd | WSJT-X weak-signal modes |
| `analog` | narrowband | piper | TTS-driven analog voice (SSB, AM, FM) |
| `digivoice` | narrowband | piper, codec2 | TTS-driven digital voice codecs |
| `sstv` | narrowband | none | Slow-scan TV image modes |
| `packet` | narrowband | direwolf | AX.25 packet radio |
| `cw` | narrowband | ebook2cw | CW from ebook text (opt-in) |
| `msk144` | narrowband | msk144gensim | MSK144 meteor scatter (opt-in) |
| `minimodem` | narrowband | minimodem | RTTY/Bell modem modes (opt-in) |
| `sameeas` | narrowband | sameeas | Emergency Alert System (opt-in) |
| `ardop` | narrowband | ardopcf | ARDOP HF data modem (opt-in) |
| `js8call` | narrowband | js8call | JS8Call digital mode (opt-in) |
| `op25` | narrowband | OP25 | P25 voice via OP25 (opt-in) |
| `hacktv` | narrowband | hacktv | Analog TV (opt-in) |

## Output Format

| File | Description |
|---|---|
| `rf_datagen_iq.npy` | `(N, window_length)` complex array — one row per window |
| `rf_datagen_tags.csv` | Per-window metadata: idx, mode, scenario, domain, sample_rate, window_length, category, subcategory |

When multi-domain generation is used, each domain produces its own output files with domain-specific shapes and dtypes.

### Train/Val/Test Split

`save_dataset()` supports optional stratified splitting:

```python
save_dataset(iq, tags, output_dir, split_ratios=(0.8, 0.1, 0.1), seed=42)
```

This produces `{prefix}_{train,val,test}_iq.npy` and matching CSV files with class-balanced splits.

### Checkpointing

Each signal class is checkpointed independently under `output/parts/` with atomic writes and config-hash validation. If generation is interrupted, it resumes from the last complete checkpoint. Config changes automatically invalidate stale checkpoints.

## Flake Outputs

| Output | Description |
|---|---|
| `packages.*.default` | CLI application with all external tools on `PATH` (fldigi, wsjtx, piper, codec2, etc.) |
| `packages.*.pythonPackage` | Python library package for `import rf_datagen` in downstream flakes |
| `devShells.*.default` | Development shell with Python deps + CLI tools |

To use rf-datagen as a library dependency in another flake:

```nix
# In your flake inputs:
inputs.rf-datagen.url = "github:Quantum-Serendipity/rf-datagen";
inputs.rf-datagen.inputs.nixpkgs.follows = "nixpkgs";

# In a python environment:
python3.withPackages (ps: [ inputs.rf-datagen.packages.${system}.pythonPackage ])
```

## Make Targets

```
make generate            # Generate narrowband dataset
make generate-moderate   # Generate moderate domain only
make generate-wideband   # Generate wideband domain only
make generate-all        # Generate all three domains
make list                # List all signal classes and their generators
make validate            # Validate dataset integrity
make validate-roundtrip  # Round-trip encode/decode validation
make validate-quick      # Quick validation (clean-only, 3 trials)
make validate-all        # All modes including STT variants
make qc-report           # HTML QC report (default: FT8; override: make qc-report MODE=CW)
make qc-modulated        # Spectrogram/waveform/PSD for all modes
make qc-text             # Show text content from all generators
make clean               # Remove generated output
```

## License

MIT — see [LICENSE](LICENSE).
