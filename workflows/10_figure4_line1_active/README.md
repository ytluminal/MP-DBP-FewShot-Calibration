# Figure 4 — Line 1 Stage-matched LightGBM Active Few-Shot

Re-run using Python 3.10+ (numpy, scikit-learn, artifact_tool):

1. Extract package contents.
2. The bundled replay script detects bundled input files automatically, and outputs to outputs_replay without modifying the original outputs.
3. Use exactly frozen Line1 Stage 1/2/3 external predictions (no model training).
4. Active selection uses per-Stage 22 / 55 / 50 features, TRAIN-only preprocessed, centroid then Euclidean maximin independently in Ref28, Ref29, Ref30.
5. Uses the original 500 Random permutations in the input folder, and per-reference offsets in ln(1 + µg/L).
6. Reports Active/Random/Zero prediction performance on the same held-out observations per replicate; Active-own Remainder is separate.
7. Consult outputs/Experiment_Manifest.json for SHA256, feature dimensions and detailed notes.

Main results: outputs/Matched_Overall_Summary.csv and outputs/Matched_Publication_Summary.csv.
Selected row IDs: outputs/Active_Budget_Selections.csv and outputs/Active_Complete_Sequences.csv.
Complete paired results: outputs/Matched_Overall_Replicates.csv, outputs/Matched_Publication_Replicates.csv.

CAVEATS: values are retrospective few-shot adaptations with partial label information. Different selected rows are DBP-specific labels, not guaranteed independent physical water samples. Full Ref 30 source overlap audit remains necessary.
