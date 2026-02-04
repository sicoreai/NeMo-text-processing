# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeMo-text-processing is NVIDIA's Python package for text normalization (TN) and inverse text normalization (ITN) using Weighted Finite-State Transducers (WFSTs). It converts text between written form (e.g., "123") and spoken form (e.g., "one hundred twenty three") for ASR and TTS applications. Supports 18+ languages.

## Common Commands

### Installation (development)
```bash
./reinstall.sh                    # Full install with extras
pip install -e .                  # Quick install without extras
```

### Running Tests
```bash
pytest                            # Run all tests (requires GPU by default)
pytest --cpu                      # Run tests on CPU only
pytest --use_local_test_data      # Skip downloading test data from GitHub
pytest --tn_cache_dir=/path       # Use cached .far grammar files
pytest --run_audio_based          # Include audio-based TN tests
pytest tests/nemo_text_processing/en/test_cardinal.py  # Run single test file
pytest -k "test_cardinal"         # Run tests matching pattern
```

### Code Style
```bash
python setup.py style --scope=nemo_text_processing         # Check style
python setup.py style --scope=nemo_text_processing --fix   # Fix style issues
```

Style rules: line length 119, black with `--skip-string-normalization`, isort with multi-line=3

### Grammar Export (for Sparrowhawk deployment)
```bash
bash tools/text_processing_deployment/export_grammars.sh --MODE=test --LANG=en
```

## Architecture

### Core Processing Pipeline
Text normalization uses a two-stage FST pipeline:
1. **Taggers** - Parse input text into semiotic class representations (e.g., `"21.º"` → `ordinal { integer: "vigésimo primero" morphosyntactic_features: "masc" }`)
2. **Verbalizers** - Convert tagged representations to spoken form

### Directory Structure
- `nemo_text_processing/text_normalization/` - Written → spoken form (TN)
- `nemo_text_processing/inverse_text_normalization/` - Spoken → written form (ITN)
- `nemo_text_processing/hybrid/` - Context-aware TN using WFST + masked language model fusion
- `nemo_text_processing/fst_alignment/` - Input/output character alignment through FST

### Language Implementation Pattern
Each language (ar, de, en, es, fr, hi, hu, hy, it, ja, ru, zh, etc.) implements:
- Taggers and verbalizers for semiotic classes (cardinal, ordinal, date, time, money, measure, etc.)
- `graph_utils.py` - Language-specific FST components
- English (`en/`) serves as the reference implementation

**Important**: Don't duplicate code from `en/graph_utils.py` or `en/utils.py`. Import from them instead.

### Sparrowhawk Compatibility
The library targets dual deployment: Python (Pynini) and C++ (Sparrowhawk). Key constraints:
- Only use Sparrowhawk-supported semiotic classes and properties
- Use `morphosyntactic_features` property for custom token features (not arbitrary strings)
- Add `preserve_order: "true"` to skip expensive property permutation when order is fixed

## Test Organization

Test files mirror the source structure under `tests/nemo_text_processing/`:
- Each language has `data_text_normalization/` and `data_inverse_text_normalization/` test data
- Tests organized by semiotic class (test_cardinal.py, test_date.py, etc.)

Test markers:
- `@pytest.mark.run_only_on('CPU')` or `@pytest.mark.run_only_on('GPU')`
- `@pytest.mark.with_downloads` - Requires `--with_downloads` flag
- `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.system`

## Contributing

- Send PRs to `main` branch
- Sign commits with `git commit -s`
- Add `__init__.py` to every new folder
- Add tests for both pytest and Sparrowhawk
- New languages must register in `tools/text_processing_deployment/pynini_export.py`
- License header required: `Copyright (c) YEAR, NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
