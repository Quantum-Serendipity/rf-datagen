# rf-datagen

RF signal IQ dataset generator for ML training — 39 amateur radio signal classes with realistic channel impairments.

## Overview

rf-datagen synthesizes complex IQ baseband samples for training RF signal classifiers. Window length, sample rate, SNR levels, and impairment scenario weights are all configurable (see `config.toml`). Output is stored as complex64 NumPy arrays with per-window metadata. The generator applies configurable propagation/receiver impairment scenarios so models train against realistic channel conditions out of the box.

### Signal Classes (39)

| Category | Modes |
|---|---|
| Digital text | CW, PSK31, PSK63, QPSK, PSK125, 8PSK, RTTY, OLIVIA, JS8, DOMINOEX, MT63, HELLSCHREIBER, MFSK16, MFSK32, CONTESTIA, THOR, FSQ, IFKP, THROB |
| WSJT-X | FT8, FT4, WSPR, JT65, JT9 |
| Analog voice | SSB, AM, FM |
| Image | SSTV, FAX |
| Data | NAVTEX, PACKET |
| Digital voice | FREEDV, M17, DMR, DSTAR, YSF, P25, NXDN |
| Other | NOISE |

### Impairment Scenarios

Samples are distributed across configurable propagation/receiver scenarios (HF clean, HF good/poor, VHF mobile, UHF urban, SDR desktop, contest crowded, overdriven, poorly operated, vintage, near-far, auroral) with configurable SNR levels and relative weights. See `[impairments]` in `config.toml`.

## Prerequisites

- Python >= 3.11
- NumPy, SciPy, Pillow, PySSTV (installed automatically)
- Optional external tools per generator:
  - **fldigi** — `fldigi` (for real fldigi-encoded digital modes)
  - **wsjtx** — `wsjtx` tools (`jt9`, `wsprd`, etc.)
  - **analog/digivoice** — `piper` TTS engine + voice models
  - **packet** — `direwolf` (for AX.25 packet encoding)

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
rf-datagen qc modulated --mode FT8 CW --output /tmp/qc/modulated
rf-datagen qc modulated --all-modes --snr-grid

# Visualize signals after channel impairments
rf-datagen qc impaired --mode CW --all-snr --output /tmp/qc/impaired
rf-datagen qc impaired --mode DMR --scenario hf_poor --snr 5

# Inspect an existing .npy dataset on disk (class distribution, sample plots)
rf-datagen qc dataset --path ./output --mode FT8 --output /tmp/qc/dataset

# Full self-contained HTML report for one mode (spectrograms, audio, SNR grid,
# all 12 impairment scenarios, signal statistics)
rf-datagen qc report --mode FT8 --output /tmp/qc/reports
```

## Configuration

See `config.toml` for the full set of options. Key sections:

- **`[dataset]`** — sample rate, window length, output directory, seed, worker count
- **`[impairments]`** — SNR levels, frequency offset, scenario weights
- **`[generators.*]`** — per-generator enable/disable, samples per class, and generator-specific tuning

## Output Format

| File | Description |
|---|---|
| `rf_datagen_iq.npy` | `(N, window_length)` complex64 array — one row per window |
| `rf_datagen_tags.csv` | Per-window metadata: mode, SNR, scenario, etc. |

## Flake Outputs

| Output | Description |
|---|---|
| `packages.*.default` | CLI application with all external tools on `PATH` (fldigi, wsjtx, piper, etc.) |
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
make generate           # Run full generation
make validate           # Validate dataset integrity
make validate-roundtrip # Round-trip encode/decode validation
make validate-quick     # Quick validation (clean-only, 3 trials)
make validate-all       # All modes including STT
make qc-report          # HTML QC report (default: FT8; override: make qc-report MODE=CW)
make qc-modulated       # Spectrogram/waveform/PSD for all modes
make qc-text            # Show text content from all generators
make clean              # Remove generated output
```

## License

MIT — see [LICENSE](LICENSE).
