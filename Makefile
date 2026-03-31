.PHONY: generate generate-moderate generate-wideband generate-all list validate clean cleanup-orphans
.PHONY: e2e e2e-quick e2e-validate e2e-ml
.PHONY: validate-roundtrip validate-all validate-quick
.PHONY: validate-wsjtx validate-packet validate-freedv validate-digivoice
.PHONY: validate-analog validate-analog-stt validate-cw validate-sstv
.PHONY: validate-impairment
.PHONY: validate-fldigi validate-fldigi-quick validate-minimodem validate-p25
.PHONY: validate-multimon validate-wsjtx-ext validate-js8
.PHONY: validate-all-expanded
.PHONY: qc-report qc-modulated qc-text

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

generate:
	python -m rf_datagen.cli generate -c config.toml

generate-moderate:
	python -m rf_datagen.cli generate -c config.toml --domains moderate

generate-wideband:
	python -m rf_datagen.cli generate -c config.toml --domains wideband

generate-all:
	python -m rf_datagen.cli generate -c config.toml --domains narrowband moderate wideband

list:
	python -m rf_datagen.cli list

# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

e2e:
	python -m rf_datagen.cli e2e -c config.toml --domains narrowband,moderate,wideband

e2e-quick:
	python -m rf_datagen.cli e2e -c config.toml --domains narrowband \
		--skip-ml --skip-roundtrip --spectral-samples 5

e2e-validate:
	python -m rf_datagen.cli e2e --skip-generate --domains narrowband,moderate,wideband

e2e-ml:
	python -m rf_datagen.cli e2e -c config.toml --domains narrowband \
		--skip-roundtrip --ml-samples 50

# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

validate:
	python -m rf_datagen.cli validate ./output

# ---------------------------------------------------------------------------
# Round-trip validation
# ---------------------------------------------------------------------------

validate-wsjtx:
	python -m rf_datagen.cli validate-roundtrip --modes FT8 WSPR

validate-packet:
	python -m rf_datagen.cli validate-roundtrip --modes PACKET_1200

validate-freedv:
	python -m rf_datagen.cli validate-roundtrip --modes FREEDV M17

validate-digivoice:
	python -m rf_datagen.cli validate-roundtrip --modes FREEDV M17 DMR DSTAR YSF NXDN

validate-analog:
	python -m rf_datagen.cli validate-roundtrip --modes SSB AM FM

validate-analog-stt:
	python -m rf_datagen.cli validate-roundtrip --modes SSB_STT AM_STT FM_STT

validate-cw:
	python -m rf_datagen.cli validate-roundtrip --modes CW

validate-sstv:
	python -m rf_datagen.cli validate-roundtrip --modes SSTV

validate-impairment:
	python -m rf_datagen.cli validate-roundtrip --modes IMPAIRMENT

# Default: all modes except STT (requires whisper model download)
validate-roundtrip:
	python -m rf_datagen.cli validate-roundtrip

# Quick: clean-only, 3 trials, core modes
validate-quick:
	python -m rf_datagen.cli validate-roundtrip --clean-only --trials 3

# Everything including STT (original modes)
validate-all:
	python -m rf_datagen.cli validate-roundtrip --modes \
		FT8 WSPR PACKET_1200 CW \
		FREEDV M17 DMR DSTAR YSF NXDN P25 \
		SSB AM FM SSB_STT AM_STT FM_STT SSTV \
		IMPAIRMENT

# ---------------------------------------------------------------------------
# Expanded round-trip validation (Phase 1-2 modes)
# ---------------------------------------------------------------------------

# 18 fldigi modes via TX/RX pipeline
validate-fldigi:
	python -m rf_datagen.cli validate-roundtrip --modes \
		PSK31 PSK63 QPSK PSK125 8PSK \
		RTTY OLIVIA DOMINOEX MT63 HELLSCHREIBER \
		MFSK16 MFSK32 CONTESTIA THOR \
		FSQ IFKP THROB NAVTEX

# Quick fldigi: 3 fast modes, clean-only, 3 trials
validate-fldigi-quick:
	python -m rf_datagen.cli validate-roundtrip --clean-only --trials 3 \
		--modes PSK31 RTTY OLIVIA

# Minimodem modes
validate-minimodem:
	python -m rf_datagen.cli validate-roundtrip --modes BELL103 BELL202

# P25 via dsdccx
validate-p25:
	python -m rf_datagen.cli validate-roundtrip --modes P25

# Multimon-ng modes (DTMF, POCSAG, EAS)
validate-multimon:
	python -m rf_datagen.cli validate-roundtrip --modes DTMF POCSAG EAS

# Extended WSJT-X modes (expected partial decode — synthesis approximations)
validate-wsjtx-ext:
	python -m rf_datagen.cli validate-roundtrip --modes FT4 JT65 JT9

# JS8
validate-js8:
	python -m rf_datagen.cli validate-roundtrip --modes JS8

# All expanded: everything (original + Phase 1-2 new modes)
validate-all-expanded:
	python -m rf_datagen.cli validate-roundtrip --modes \
		FT8 WSPR PACKET_1200 CW \
		FREEDV M17 DMR DSTAR YSF NXDN P25 \
		SSB AM FM SSB_STT AM_STT FM_STT SSTV \
		IMPAIRMENT \
		BELL103 BELL202 \
		DTMF POCSAG EAS \
		FT4 JT65 JT9 JS8 \
		PSK31 PSK63 QPSK PSK125 8PSK \
		RTTY OLIVIA DOMINOEX MT63 HELLSCHREIBER \
		MFSK16 MFSK32 CONTESTIA THOR \
		FSQ IFKP THROB NAVTEX

# ---------------------------------------------------------------------------
# QC inspection
# ---------------------------------------------------------------------------

qc-report:
	python -m rf_datagen.cli qc report --mode $(or $(MODE),FT8) --output /tmp/qc_inspect/reports

qc-modulated:
	python -m rf_datagen.cli qc modulated --all-modes --snr-grid --output /tmp/qc_inspect/modulated

qc-text:
	python -m rf_datagen.cli qc text --generator all

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

cleanup-orphans:
	python -m rf_datagen.cli cleanup

clean:
	rm -rf output/parts output/validation
	rm -f output/rf_datagen_iq.npy output/rf_datagen_tags.csv
