# Raman wood species identification — v1

Proof of concept with Prof. Tsao (NCHU): identify Taiwanese timber species from Raman spectra
(532 nm benchtop instrument) with ML, evaluated at the **tree** level.

```bash
uv sync                                          # Python 3.12 env from uv.lock
uv run pytest                                    # parsing + leak-free splitters
uv run python experiments/00_confound_audit.py   # confound audit -> results/<date>_00_confound_audit/
```

- `CLAUDE.md` — data facts, evaluation contract, research direction (read this first).
- `docs/literature-2026-10.md` — literature map.
- `docs/measurement-protocol-v1.md` — what to ask and how to measure the next batch.
- The v0 pipeline and Streamlit demo (Sep–Oct 2025) live at git tag `v0-legacy`.
