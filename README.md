# rf-datagen

RF signal IQ dataset generator for ML training — 90 signal classes across three sample-rate domains with 19 realistic channel impairment scenarios and a 4-layer validation pipeline.

## Overview

rf-datagen synthesizes complex IQ baseband samples for training RF signal classifiers. It operates across three sample-rate domains (narrowband 12 kHz, moderate 1 MHz, wideband 20 MHz), each with its own window length, dtype, and synthesizer registry. Output is stored as NumPy arrays with per-window metadata including signal class, SNR, scenario, and domain. The generator applies configurable propagation/receiver impairment scenarios so models train against realistic channel conditions out of the box.

Beyond generation, rf-datagen provides a layered validation system:

- **Structural** — dataset integrity (shapes, dtypes, completeness, NaN/Inf, distributions)
- **Spectral fingerprint** — per-class bandwidth and PAPR against expected ranges
- **Round-trip decode** — 48 modes through real external decoders (fldigi, WSJT-X, multimon-ng, dsdcc, etc.)
- **ML classification** — TorchSig/RFML/CGDNN inference accuracy on generated data

These layers are unified in a single `rf-datagen e2e` command that generates a dataset, runs all checks, and produces a structured pass/fail report.

### Multi-Rate Domain Architecture

| Domain | Sample Rate | Window Length | Duration | Nyquist | dtype | Classes | Default Samples/Class |
|--------|------------|---------------|----------|---------|-------|---------|----------------------|
| Narrowband | 12 kHz | 2,048 | ~170 ms | 6 kHz | complex128 | 68 | 6,000 |
| Moderate | 1 MHz | 131,072 | ~131 ms | 500 kHz | complex64 | 14 | 2,000 |
| Wideband | 20 MHz | 2,097,152 | ~105 ms | 10 MHz | complex64 | 8 | 1,000 |

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

Samples are distributed across 19 configurable propagation/receiver scenarios with configurable SNR levels (default: 25, 20, 15, 10, 5, 0, -5, -10 dB) and relative weights:

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
- Optional Python extras:
  - `matplotlib` — QC visualizations (`pip install rf-datagen[qc]`)
  - `scikit-dsp-comm` — higher-fidelity Watterson channel model (`pip install rf-datagen[sdc]`)
  - `sdr` — Numba-accelerated modulation (`pip install rf-datagen[accel]`)
  - `torch`, `torchsig`, `onnxruntime` — ML classification validation (`pip install rf-datagen[ml-validate]`)
  - `gnuradio` — GNU Radio probe integration (`pip install rf-datagen[gnuradio]`)
- Optional external tools per generator:
  - **fldigi** — `fldigi` (real fldigi-encoded digital modes)
  - **wsjtx** — `jt9`, `wsprd` (WSJT-X weak-signal modes)
  - **analog/digivoice** — `piper` TTS engine + voice models, `espeak-ng`
  - **digivoice** — `codec2`, `m17-cxx-demod`, `dsdcc` (digital voice codecs)
  - **packet** — `direwolf` (AX.25 packet encoding)
  - **cw** — `ebook2cw` (CW from ebook text)
  - **msk144** — `msk144gensim` (MSK144 meteor scatter)
  - **minimodem** — `minimodem` (RTTY/Bell modem modes)
  - **sameeas** — `sameeas` Python package (Emergency Alert System)
  - **ardop** — `ardopcf` (ARDOP HF data modem)
  - **js8call** — `js8call` + `xvfb-run` (JS8Call digital mode)
  - **op25** — OP25 `dv_tx.py` + `mbelib` (P25 voice)
  - **hacktv** — `hacktv` (analog TV)
  - **probe** — `gnuradio` + `multimon-ng` (signal analysis/decoding)
  - **round-trip validation** — `whisper-cpp` (STT for analog voice validation)

## Installation

### pip (Python only)

```bash
pip install -e .

# With optional extras
pip install -e ".[qc]"                  # QC visualizations (matplotlib)
pip install -e ".[ml-validate]"         # ML classification validation (torch, torchsig)
pip install -e ".[qc,sdc,accel]"        # Multiple extras
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

### End-to-end pipeline (generate + validate)

The `e2e` command is the primary workflow — it generates a dataset across all requested domains and runs every available validation layer:

```bash
# Full pipeline: generate all 3 domains + validate everything
rf-datagen e2e -c config.toml --domains narrowband,moderate,wideband

# Quick: narrowband only, skip slow tests
rf-datagen e2e -c config.toml --domains narrowband --skip-ml --skip-roundtrip

# Validate an existing dataset (skip generation)
rf-datagen e2e --skip-generate --domains narrowband,moderate,wideband

# Strict mode: all 4 gates must pass (not just structural + spectral)
rf-datagen e2e -c config.toml --strict
```

See [E2E Pipeline](#e2e-pipeline) for details on phases and quality gates.

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
rf-datagen validate-roundtrip              # default 13 core modes
rf-datagen validate-roundtrip --clean-only --trials 3  # quick check
rf-datagen validate-roundtrip --modes FT8 WSPR CW     # specific modes
rf-datagen validate-roundtrip --snr-only 25 20 15      # specific SNR levels
```

### ML classification validation

```bash
rf-datagen validate-ml                              # TorchSig backend, all mapped modes
rf-datagen validate-ml --model all --samples 100    # All backends, 100 samples per mode
rf-datagen validate-ml --snr-sweep                  # Sweep across SNR levels
rf-datagen validate-ml --device openvino            # Use OpenVINO acceleration
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

## Validation

rf-datagen has four validation layers, unified by the `e2e` command:

### Layer 1: Structural validation

Checks dataset integrity on disk:
- File existence (IQ `.npy` + metadata `.csv` per domain)
- Array shape and dtype correctness per domain
- No NaN or Inf values (samples up to 500 rows)
- No zero-power windows
- All expected signal classes present in metadata
- Per-class sample counts within tolerance of config target
- Scenario distribution matches configured weights
- SNR distribution across configured levels

### Layer 2: Spectral fingerprint

Per-class bandwidth and PAPR validation against expected ranges. Each of the 90 signal classes has a defined spectral specification (3 dB bandwidth range and PAPR range). The checker:

1. Samples N windows per class from the generated dataset
2. Computes PSD via Welch's method
3. Measures 3 dB bandwidth and PAPR
4. Passes if median values fall within the expected range

Classes with inherently ill-defined bandwidths (e.g., analog voice, jammers) are marked exempt from BW checks but still validated for PAPR.

### Layer 3: Round-trip decode

Encode-then-decode validation using real external software. Currently covers 48 modes across 11 validator modules:

| Group | Modes | Decoder |
|-------|-------|---------|
| WSJT-X | FT8, WSPR, FT4, JT65, JT9 | jt9, wsprd |
| fldigi | PSK31, PSK63, QPSK, PSK125, 8PSK, RTTY, OLIVIA, DOMINOEX, MT63, HELLSCHREIBER, MFSK16, MFSK32, CONTESTIA, THOR, FSQ, IFKP, THROB, NAVTEX | fldigi RX |
| Digital voice | FREEDV, M17, DMR, DSTAR, YSF, NXDN, P25 | codec2, m17-cxx-demod, dsdcc |
| Analog voice | SSB, AM, FM (+ STT variants) | sox demodulation, whisper-cpp |
| Packet | PACKET_1200 | direwolf |
| Multimon | DTMF, POCSAG, EAS, ACARS | multimon-ng |
| Modem | BELL103, BELL202 | minimodem |
| Other | CW, JS8, SSTV, IMPAIRMENT | tone detection, js8call, sstv decoder |

Each mode is tested across configurable SNR levels with multiple trials. Decode rate = correct_decodes / trials.

### Layer 4: ML classification

Feeds generated IQ windows to pre-trained neural network classifiers and checks that predicted modulation classes match expected labels. Supported backends:

- **TorchSig** — XCiT-based classifier (28 mapped signal classes)
- **RFML** — RF signal classifier
- **CGDNN** — Lightweight CNN via ONNX Runtime

## E2E Pipeline

The `e2e` command orchestrates 6 phases:

```
rf-datagen e2e -c config.toml --domains narrowband,moderate,wideband
  |
  +-- Phase 1: GENERATE        Generate all classes across all domains
  +-- Phase 2: STRUCTURAL      Dataset integrity checks (required gate)
  +-- Phase 3: SPECTRAL        Per-class BW/PAPR fingerprinting (required gate)
  +-- Phase 4: ROUND-TRIP      Fresh-generation decode tests (advisory)
  +-- Phase 5: ML              Classification accuracy (advisory)
  +-- Phase 6: REPORT          JSON + HTML report with verdict
```

### Quality gates

| Gate | Phase | Required | Criteria |
|------|-------|----------|----------|
| 1 | Structural | Yes | All checks pass |
| 2 | Spectral | Yes | >=85% of non-exempt classes pass |
| 3 | Round-trip | Advisory | >=50% decode rate per mode at 25 dB, >=70% of modes pass overall |
| 4 | ML | Advisory | >=40% accuracy per class, >=60% of classes pass overall |

- **PASS**: Gates 1 + 2 pass
- **FAIL**: Gate 1 or Gate 2 fails
- **STRICT PASS**: All 4 gates pass (with `--strict`)

### Report output

The pipeline produces:
- `e2e_report.json` — structured results with per-class, per-check detail
- `e2e_report.html` — self-contained dark-themed HTML with expandable phase sections
- Console summary with pass/fail verdict

## Configuration

Configuration uses **TOML** format. See `config.toml` for a complete annotated example.

```bash
rf-datagen generate -c config.toml
rf-datagen e2e -c config.toml
```

If no config file is provided, built-in defaults are used.

### `[dataset]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sample_rate` | int | `12000` | IQ sample rate in Hz (narrowband default) |
| `window_length` | int | `2048` | Samples per training window (~170 ms at 12 kHz) |
| `output_dir` | string | `"./output"` | Output directory |
| `seed` | int | `42` | Random seed for reproducibility |
| `workers` | int | `0` | Worker count; 0 = auto (CPU count). Fallback for generators without own workers |
| `domains` | list | `["narrowband"]` | Domains to generate: `"narrowband"`, `"moderate"`, `"wideband"` |

### `[impairments]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `snr_levels` | list[int] | `[25, 20, 15, 10, 5, 0, -5, -10]` | SNR levels in dB; samples distributed evenly |
| `max_freq_offset` | int | `500` | Max carrier offset in Hz (RTL-SDR realistic) |
| `window_stride` | int | `0` | Window extraction stride; 0 = auto (`window_length // 2`) |
| `window_power_threshold` | float | `0.001` | Windows below this mean power are discarded as silence |
| `watterson_model` | string | `"builtin"` | Watterson fading implementation: `"builtin"` or `"sdc"` (scikit-dsp-comm) |

### `[impairments.scenarios]`

Relative weights for the 19 propagation/receiver scenarios (auto-normalized to sum to 1.0):

```toml
[impairments.scenarios]
hf_clean = 0.07
hf_good = 0.16
hf_poor = 0.16
vhf_mobile = 0.08
uhf_urban = 0.04
sdr_desktop = 0.08
contest_crowded = 0.08
overdriven = 0.04
poorly_operated = 0.04
vintage = 0.03
near_far = 0.02
auroral = 0.02
indoor_multipath = 0.04
leo_satellite = 0.03
automotive = 0.02
urban_cellular = 0.03
radar_clutter = 0.02
maritime = 0.02
ism_congested = 0.02
```

### `[generators.<name>]`

Each generator has its own section. When a `[generators]` section is present in the config, only the listed generators are active — unlisted generators are not run.

#### Common keys (all generators)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Whether this generator runs |
| `samples_per_class` | int | `6000` | Target windows per signal class |
| `classes` | string or list | `"all"` | `"all"` or explicit list: `["FT8", "CW"]` |
| `workers` | int | `0` | Per-generator parallelism; 0 = inherit from `[dataset]` |

#### `[generators.<name>.boost]`

Per-class resampling multiplier. Useful for modes that produce fewer usable windows per raw sample (e.g., narrow-band PSK modes):

```toml
[generators.synthetic.boost]
PSK31 = 2.0
PSK63 = 2.0
MFSK16 = 3.0
MFSK32 = 3.0
```

#### Generator-specific keys

| Key | Generators | Type | Default | Description |
|-----|-----------|------|---------|-------------|
| `utterances_per_class` | analog, digivoice, op25 | int | `100` | TTS utterances to generate per class |
| `voice_cache` | analog, digivoice, op25 | string | `"./piper-voices"` | Directory for Piper TTS voice models |
| `rsid_probability` | synthetic | float | `0.35` | Probability of prepending RSID preamble |
| `cw_wpm_range` | synthetic, cw | list[int] | `[10, 30]` | CW words-per-minute range `[min, max]` |
| `cw_tone_range` | cw | list[int] | `[400, 800]` | CW tone frequency range in Hz |
| `messages_per_mode` | wsjtx, msk144 | int | `200` | Messages to encode per WSJT-X mode |
| `images_per_mode` | sstv | int | `10` | Images to render per SSTV sub-mode |
| `packets_per_baud` | packet | int | `100` | Packets generated per baud rate |
| `codec2_mode` | digivoice | string | `"3200"` | Codec2 bitrate (1200/1300/1400/1600/2400/3200) |
| `freedv_modes` | digivoice | list[str] | `["1600", "700C", "700D", "700E"]` | FreeDV sub-modes to generate |
| `minimodem_modes` | minimodem | list[str] | `["rtty", "bell103", "bell202"]` | Minimodem protocols |
| `ardop_speeds` | ardop | list[int] | `[200, 500, 1000, 2000]` | ARDOP data rates |

### Generators

| Generator | Domain | Tool Deps | Default | Description |
|-----------|--------|-----------|---------|-------------|
| `synthetic` | narrowband | none | enabled | Pure-Python synthesis for 68 narrowband classes |
| `synthetic_moderate` | moderate | none | enabled | Pure-Python synthesis for 14 moderate-rate classes |
| `synthetic_wideband` | wideband | none | enabled | Pure-Python synthesis for 8 wideband classes |
| `fldigi` | narrowband | fldigi | enabled | Real fldigi-encoded digital modes |
| `wsjtx` | narrowband | jt9, wsprd | enabled | WSJT-X weak-signal modes |
| `analog` | narrowband | piper | enabled | TTS-driven analog voice (SSB, AM, FM) |
| `digivoice` | narrowband | piper, codec2 | enabled | TTS-driven digital voice codecs |
| `sstv` | narrowband | none | enabled | Slow-scan TV image modes |
| `packet` | narrowband | direwolf | enabled | AX.25 packet radio |
| `cw` | narrowband | ebook2cw | opt-in | CW from ebook text (overlaps synthetic) |
| `msk144` | narrowband | msk144gensim | opt-in | MSK144 meteor scatter |
| `minimodem` | narrowband | minimodem | opt-in | RTTY/Bell modem modes (overlaps synthetic) |
| `sameeas` | narrowband | sameeas | opt-in | Emergency Alert System |
| `ardop` | narrowband | ardopcf | opt-in | ARDOP HF data modem |
| `js8call` | narrowband | js8call, xvfb | opt-in | JS8Call digital mode |
| `op25` | narrowband | OP25, mbelib | opt-in | P25 voice via OP25 |
| `hacktv` | narrowband | hacktv | opt-in | Analog TV |

## Output Format

### Generated files

| File | Description |
|---|---|
| `rf_datagen_iq.npy` | `(N, window_length)` complex array — one row per window |
| `rf_datagen_tags.csv` | Per-window metadata: idx, mode, scenario, domain, sample_rate, window_length, category, subcategory |
| `generation_report.json` | Per-class generation status, timing, total windows, file size |

### Multi-domain output

When multiple domains are enabled, each domain produces its own files in a subdirectory:

```
output/
  narrowband/
    rf_datagen_narrowband_iq.npy      # (N, 2048) complex128
    rf_datagen_narrowband_tags.csv
  moderate/
    rf_datagen_moderate_iq.npy        # (N, 131072) complex64
    rf_datagen_moderate_tags.csv
  wideband/
    rf_datagen_wideband_iq.npy        # (N, 2097152) complex64
    rf_datagen_wideband_tags.csv
  generation_report.json
```

### E2E report files

When running `rf-datagen e2e`, additional files are produced:

| File | Description |
|---|---|
| `e2e_report.json` | Structured results: per-phase, per-class, per-check with pass/fail |
| `e2e_report.html` | Self-contained HTML report with expandable phase sections |

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

### Generation

```
make generate            # Generate narrowband dataset
make generate-moderate   # Generate moderate domain only
make generate-wideband   # Generate wideband domain only
make generate-all        # Generate all three domains
make list                # List all signal classes and their generators
```

### E2E pipeline

```
make e2e                 # Full pipeline: all 3 domains + all validation
make e2e-quick           # Narrowband only, skip roundtrip + ML, 5 spectral samples
make e2e-validate        # Skip generation, validate existing output
make e2e-ml              # Narrowband, skip roundtrip, 50 ML samples
```

### Dataset validation

```
make validate            # Validate dataset integrity on ./output
```

### Round-trip validation

```
make validate-roundtrip  # Default 13 core modes
make validate-quick      # Clean-only, 3 trials, core modes
make validate-all        # All core modes including STT variants
```

Individual mode groups:

```
make validate-wsjtx      # FT8, WSPR
make validate-wsjtx-ext  # FT4, JT65, JT9
make validate-packet     # PACKET_1200
make validate-freedv     # FREEDV, M17
make validate-digivoice  # FREEDV, M17, DMR, DSTAR, YSF, NXDN
make validate-analog     # SSB, AM, FM
make validate-analog-stt # SSB_STT, AM_STT, FM_STT (requires whisper-cpp)
make validate-cw         # CW
make validate-sstv       # SSTV
make validate-fldigi     # 18 fldigi modes (PSK31, RTTY, OLIVIA, etc.)
make validate-fldigi-quick  # 3 fast fldigi modes, clean-only
make validate-minimodem  # BELL103, BELL202
make validate-p25        # P25 via dsdcc
make validate-multimon   # DTMF, POCSAG, EAS
make validate-js8        # JS8
make validate-impairment # Impairment state continuity check
make validate-all-expanded  # All 48 modes
```

### QC inspection

```
make qc-report           # HTML QC report (default: FT8; override: make qc-report MODE=CW)
make qc-modulated        # Spectrogram/waveform/PSD for all modes
make qc-text             # Show text content from all generators
```

### Housekeeping

```
make clean               # Remove generated output
```

## License

MIT — see [LICENSE](LICENSE).
