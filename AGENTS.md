# Project rules

- Phase 1 is closed. Phase 2 data work is authorized; use BaoStock as the single selected source.
- Do not copy or reuse source code, models, features, or architecture from another project. The supplied CSV files may be read but must not be modified or copied into this repository.
- Local-file research inputs may read only `train.csv`; never load or expose `test.csv` to research code. BaoStock records may enter only through the explicit source boundary and P2 acquisition path.
- Preserve security codes as six-character strings and enforce point-in-time availability. Use the selected source and user-provided provenance without repeated multi-provider or full-file reliability audits.
- Keep errors visible. Validate external files at the boundary and avoid speculative modules or APIs.
