.PHONY: generate list validate clean
.PHONY: validate-roundtrip validate-all validate-quick
.PHONY: validate-wsjtx validate-packet validate-freedv validate-digivoice
.PHONY: validate-analog validate-analog-stt validate-cw validate-sstv
.PHONY: validate-impairment

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

generate:
	python -m rf_datagen.cli generate -c config.toml

list:
	python -m rf_datagen.cli list

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

# Everything including STT
validate-all:
	python -m rf_datagen.cli validate-roundtrip --modes \
		FT8 WSPR PACKET_1200 CW \
		FREEDV M17 DMR DSTAR YSF NXDN \
		SSB AM FM SSB_STT AM_STT FM_STT SSTV \
		IMPAIRMENT

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean:
	rm -rf output/parts output/validation
	rm -f output/rf_datagen_iq.npy output/rf_datagen_tags.csv
