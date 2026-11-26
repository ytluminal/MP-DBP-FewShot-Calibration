# Publication release checklist

- [x] Record the official repository owner, final manuscript title, and five manuscript authors.
- [ ] Decide an explicit code license, confirm any institutional intellectual-property policies, and document dataset reuse rights.
- [ ] Replace historical SI `.docx` computational inputs with versioned machine-readable parameter files (or review them before public release).
- [ ] Confirm final `data/MP-DBPs_SI.xlsx` is consistent with the exact training inputs; retain historical data for reproduction and document differences.
- [ ] Verify external Refs 28–30 were screened by the same inclusion/exclusion criteria as Refs 1–27.
- [ ] Reconcile Figure 2a small differences in the original archived replication audit with the main manuscript and SI.
- [x] Add supplied Figure 5a–e and Figure 6a–d plotting packages and their existing source data.
- [ ] Provide remaining upstream generation scripts (where unavailable), including Figure 1, standalone Figure S16 plotting, and oracle-correction analyses; confirm exact reproduction scope.
- [ ] Perform a controlled environment rerun of critical reproduction entrypoints; record Python/OS/BLAS versions and output hashes.
- [ ] Decide where optional trained-model binaries belong (GitHub release assets, DOI archive, or separate storage).
- [ ] Add permanent public GitHub URL and archived DOI to manuscript Data Availability Statement and SI.
- [ ] Avoid committing third-party publication PDFs or private documents without redistribution permission.
- [ ] Tag a stable version (e.g. v1.0.0) only after validation.
