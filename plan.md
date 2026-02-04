# Implementation Plan: Greek Support for TN/ITN

## general

### Grammar Standards
- All Greek grammars should reside in `nemo_text_processing/text_normalization/greek/`.
- Use the **English** implementations as templates for semiotic classes (Cardinal, Date, Measure, etc.).
- Ensure all transducers are exported via a `GraphBuilder` class.
- Handle Greek-specific characters (e.g., α, β, γ) and accentuation (tonos) carefully.

### Workflow
- Always write a small test case in `tests/nemo_text_processing/greek/` before implementing a rule.
- Use `Shift+Tab` (Plan Mode) to design the regex/logic before writing `.py` files.

## Phase 1: Environment & Scaffolding
- [ ] Create directory structure: `nemo_text_processing/text_normalization/greek/`
- [ ] Implement `nemo_text_processing/text_normalization/greek/utils.py` (Common Greek regex/mappings)
- [ ] Register `el` (Greek) in `nemo_text_processing/text_normalization/normalize.py`

## Phase 2: Fundamental Semiotic Classes (TN)
- [ ] **Cardinal**: Support for 0-999,999,999 (Handling gender/case for 1, 3, 4).
- [ ] **Decimal**: Handling the comma `,` as a decimal separator.
- [ ] **Date**: Days, months (Greek names), and years.
- [ ] **Money**: Support for Euro (€ / ευρώ).

## Phase 3: Verbalizers & Final Graphs
- [ ] Implement Greek Tagger (combines all semiotic classes).
- [ ] Implement Greek Verbalizer (formats the output string).
- [ ] Integration test: `python normalize.py --text="10€" --language=el`

## Phase 4: Inverse Text Normalization (ITN)
- [ ] (Mirror Phase 2 logic for ITN: Spoken -> Written)
