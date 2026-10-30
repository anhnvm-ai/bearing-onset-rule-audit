# Amendments to the pre-specified analysis plan

This file records deviations from `ANALYSIS_PLAN_2026-10-24.md`. That plan is locked; its SHA-256 digest is in `ANALYSIS_PLAN_2026-10-24.sha256`.

**Rules:**
- Entries are only ever appended; existing entries are never edited or removed.
- Each entry is written before the result it affects is reported.

**Entry format:**

```
## A<number> — <YYYY-MM-DD>
- Section(s) affected:
- Change:
- Reason:
- Results seen before the change (if any):
```

---

## A1 — 2026-10-24

**Section(s) affected:** 8 (WP-A, J61).

**Change.** Implementation details that Section 8 leaves open, fixed before J61 is run:

1. **Bisection.** It is applied to the exact Monte Carlo FAR function.
   - For every simulated series and rule, the critical multiplier m* is computed once. m* is the largest multiplier at which the rule still alarms at some t ≥ t0. It equals the maximum over t ≥ t0 of the minimum of the standardised alarm statistic over t, t+1, t+2.
   - The rule alarms at multiplier m if and only if m < m*. Hence FAR(m) is the share of series with m* > m, averaged with equal weight over the trajectory lengths.
   - Bisection on m is then exact for the simulated sample. Common random numbers hold automatically, because the same series serve every rule and every m.
   - The same critical multipliers give the FAR of each of the 67 designs at its own parameters:
     - for the autocorrelation-corrected EWMA, by comparing L × inflation with the EWMA critical value of each series;
     - for the adaptive chart, by the expanding-window statistic of J26.
2. **Phase-I window of P-retro for ARL0.** On series of 10,000 records, the P-retro window is the median over the dataset's trajectories of max(5, ⌊0.2n⌋), and t0 = 0. For P-prospective it is 20, with t0 = 20. A run length that reaches 10,000 is censored there. ARL0 is reported as "> 10,000" when more than half of the runs are censored.
3. **Gaps compared.**
   - The uncalibrated reference gap is the gap of J60: the median β1 − 1 over 3sig_rms, ma3sig and ma5sig minus the median over EWMA and CUSUM, under the same protocol and on the same bearing set.
   - The calibrated gap is the median over the calibrated threshold and moving-average detectors minus the median over the calibrated EWMA and CUSUM.
4. **"Intervals overlap" in the first reading** means that the 95% interval of the calibrated detectors' median β1 − 1 overlaps that of the calibrated charts' median, over the same bearing resamples.
5. **"Different d" in the second and third readings** means that the 95% bearing-bootstrap interval of the difference in median shift d (calibrated detectors minus calibrated charts) excludes 0.

**Reason.** Section 8 does not specify these points. They are fixed before any J61 output exists, so none of them can be chosen after seeing a result.

**Results seen before the change.** None of J61. J60 had produced only a quick test run (B = 50) for checking the code; its full run was still in progress.

## A2 — 2026-10-24

**Section(s) affected:** 9 (WP-C, J62).

**Change.** Implementation details that Section 9 leaves open, fixed before J62 is run:

1. **Seeds of GRU and TCN.** The cell value of a metric is its median over the three seeds. The paired bootstrap uses the per-bearing errors of the median seed by the pooled metric, as in J47 and J48.
2. **Labeler main effect** (second reading of Section 9).
   - For each pair of labelers: the difference in the mean over the nine models of the pooled C-common RMSE. For C-rank: the difference in the mean over models of the bearing-averaged concordance on S_j.
   - Intervals: 95% bearing-bootstrap intervals, stratified by condition, B = 20,000, seed 62.
   - The effect is present if at least one pair has an interval that excludes 0.
3. **Secondary reversal analysis on C-rank.** It uses the bearing-averaged concordance index on S_j (higher is better). Concordance counts the pairs s < t in S_j with ŷ_s > ŷ_t, with ties in ŷ counted one half.
4. **Execution.**
   - Models run in parallel worker processes with one thread each.
   - torch is imported before scikit-learn in every worker (otherwise WinError 1114 on this machine).
   - Tree ensembles use n_jobs = 1, which gives results identical to n_jobs = −1 for a fixed random_state.
5. **Inputs.**
   - Elapsed record count t: the 0-based record index within the bearing, appended as a column.
   - History features:
     - the trailing mean of RMS over records max(0, t − 15) to t;
     - the OLS slope of RMS on the record index over the same records (0 when only one record);
     - the cumulative sum of RMS² over records 0 to t.
   - Scaling: inside the scikit-learn pipelines as in J35, fitted on the training rows. For GRU and TCN, J47's standardisation on the training windows.
6. **Two-stage model.**
   - T̂ = i* + u_hit from J39's exponential fit.
   - T̂ = t once the fitted value has reached θ.
   - T̂ = ∞ when there is no hitting time (fewer than MIN_FIT records since the onset, or a non-positive slope).
   - C-rank uses T̂ − t. C-common uses (T̂ − t)/T̂ clipped to [0, 1], with ∞ giving 1.
   - The target-defined prediction (T̂ − t)/(T̂ − i*) must reproduce J39 exactly; this is asserted.
7. **Models of the reversal analysis:** lin:rms, ridge, knn, gbr, et, rf, two-stage, gru and tcn, all with input variant (a).
8. **Calibrated labelers** (sensitivity) use the J61 multipliers at 5% FAR under the main null model of the same protocol. The calibrated threshold replaces 3sig_rms, the calibrated moving-average rule replaces ma5sig, and the calibrated EWMA replaces ewma; pct05 is unchanged.
9. **Bootstrap.**
   - Bearings are resampled with replacement within each condition.
   - Pooled RMSE is recomputed from per-bearing sums of squared errors and record counts, keeping the record weighting of J35.
   - B = 20,000, seed 62.
10. **Replication of J47.** The target-defined RMSE of GRU and TCN under P-retro, with canonical labelers and input variant (a), is compared with `results/tables/j47_sequence_models.csv` (main arm), and the differences are reported (Section 16).

**Reason.** Section 9 does not specify these points. They are fixed before any J62 output exists.

**Results seen before the change.** None of J62. Quick test runs of J60 (B = 50) and J61 (R = 300, B = 200) had been inspected only to check the code. The full runs of J60 and J61 were still in progress.

## A3 — 2026-10-24

**Section(s) affected:** 9 (WP-C, J62), the two-stage model.

**Change.** J39 divides the RMS series by its mean over the P-retro baseline window, max(5, ⌊0.2n⌋) records. In J62 the scaling window follows the protocol that defines the onsets:
- under P-retro it is the same as J39;
- under P-prospective it is the first 20 records, the Phase-I window of that protocol.

The failure level θ is unchanged: the median, over the other usable bearings, of the last record's scaled RMS.

**Reason.** Under P-prospective the model must not use a window whose length depends on the total life n.

**Results seen before the change.** As for A2. No output of J62 exists.

## A4 — 2026-10-24

**Section(s) affected:** 10 (WP-F, J64).

**Change.** Implementation details that Section 10 leaves open, fixed before J64 is run:

1. **Bearing sets.**
   - Primary: both protocols on the bearings eligible under P-prospective (n ≥ 80: XJTU-SY 13, FEMTO 11). This follows the common-set rule of Section 3.
   - Secondary, for continuity with the submitted manuscript: P-retro on all bearings (XJTU-SY 15, FEMTO 13).
2. **Outcome of a design.** β1 − 1 over the bearings of the set on which the design fires, with condition fixed effects. It is undefined when the design fires on fewer than four bearings (as in J26, J36, J41, J43).
3. **Functional forms.** Each model has an intercept.
   - M0: the four-level family label of J26 as dummies.
   - M1: log(nominal shift) and its square, the form of the submitted Eq. (6) and of J43.
   - M2: logit(design FAR) and its square, for parity with M1.
   - M3: M2 plus the family dummies.
   - M4 (M1 + M2) is reported as optional.
4. **Design FAR.** From `j61_design_far.csv` (main null model), for the protocol of the analysis.
   - It is a property of a design averaged over the dataset's trajectory lengths, so the same value is used for the main and the all-bearing sets.
   - It is held fixed in the bearing bootstrap.
   - The logit uses FAR clipped to [1/(2R), 1 − 1/(2R)], with R the total number of simulated series of the dataset and protocol.
5. **Splits.**
   - The bearings are split into halves within each condition. When a condition has an odd number of bearings, the extra bearing goes to either half.
   - With the main sets, fewer than 2,000 distinct unordered stratified splits exist for both datasets (XJTU-SY as well as FEMTO), so all distinct unordered splits are used once.
   - Each split is scored in both directions, and the two out-of-sample R² values are averaged (J43).
   - A split is used only when at least ten designs have a defined outcome in both halves and at least two families are present (J43).
6. **Readings.**
   - "M2 beats M0 out of sample" means a higher out-of-sample R² in the same split.
   - The third reading applies when the median out-of-sample R² is below 0 for every model.
   - The flags of the three readings are reported separately, since they can hold together.

**Reason.** Section 10 does not specify these points. They are fixed before any J64 output exists.

**Results seen before the change.** None of J64. The full results of J60 and J61 had been seen, including the design onsets and design FARs that J64 uses as inputs, but not the WP-F regressions.

## A5 — 2026-10-24

**Section(s) affected:** 11 (WP-E, J63) and 14 (Gate 1, rule G1-b).

**Change.**
1. **Gate 1, rule G1-b, applied.** H_w was not supported (J60: condition 2 gave 0.52 against the required 0.67), so the warm-up scenario of Section 11 is not simulated.
2. **Rules.**
   - At canonical settings: 3sig_rms, ma3sig, ma5sig, ewma and cusum, as in J60.
   - At matched 5% FAR: the calibrated threshold, moving-average, EWMA and CUSUM rules of J61 (XJTU-SY, main null model, for the protocol of the cell).
3. **Lengths and random numbers.**
   - Series are simulated once per cell at every XJTU-SY trajectory length, 400 per length.
   - P-retro uses all 15 lengths; P-prospective uses the 13 lengths with n ≥ 80. Both protocols see the same series (common random numbers).
   - Each cell is seeded from a CRC of its parameters (J44 convention), with the base seed 63.
4. **Outcomes per cell, protocol and rule.**
   - Share of series alarming before t_{δ'}.
   - Share never alarming.
   - Delay: i* − t_{δ'} in records and (i* − t_{δ'})/n in life fraction, median over the series that alarm at or after t_{δ'}.
   - Median u = (i* + 1)/n.
   - β1 − 1: the OLS slope of log u on log n over all firing series of the cell, pooled over lengths, without fixed effects.
   - t_{δ'} is undefined when the noiseless signal never reaches δ' (for example δ = 2 and δ' = 3); the outcomes that need it are then reported as missing.

**Reason.** Section 11 leaves these points open. They are fixed before any J63 output exists.

**Results seen before the change.** None of J63. The full results of J60 and J61, and a quick run of J64 (B = 100), had been seen.

## A6 — 2026-10-25

Written at 14:45 +07, before `scripts/J69_labeler_model_contrast.py` existed.

**Section(s) affected:** 9 (WP-C, model comparison under the common reference) and 12 (WP-G). One analysis is added; no existing analysis changes.

**Change.** A direct comparison of the size of the labeler effect with the size of the model effect on the common reference, computed by `scripts/J69_labeler_model_contrast.py`.
1. **Data.**
   - The per-bearing results of J62 (`results/tables/j62_per_bearing.csv`): input variant (a), the nine models of Section 9, and for GRU and TCN the median seed. These are exactly the arrays of the J62 model comparison.
   - The four configurations and the two datasets of J62.
   - Scoring set S with the four labelers (primary, kept at Gate 1), and the variant S3 with the three signal-based labelers.
2. **Cell value.** For labeler L and model X, as in J62:
   - C-rank: the mean over bearings of the concordance on the scored records;
   - C-common: the RMSE pooled over the scored records of all bearings.
3. **Effect sizes.**
   - Main effect of a labeler: its cell values averaged over the models. Main effect of a model: its cell values averaged over the labelers.
   - Size of a factor: the mean absolute difference between the main effects of two of its levels, over all pairs of levels (Gini mean difference). Unlike a range or a population standard deviation, its expectation does not depend on the number of levels (four or three labelers against nine models).
   - Statistic: R = size(labeler) / size(model). R < 1 means that the model matters more.
4. **Model pools.**
   - (i) The nine models (primary).
   - (ii) The eight models without the two-stage model (robustness; the two-stage model is the worst in every cell).
   - (iii) Leave-one-model-out point estimates; the largest R is reported.
5. **Inference.**
   - 95% percentile intervals from B = 20,000 bearing resamples stratified by operating condition (`strat_index` of J60), seed 69.
   - One resample set per configuration and dataset, shared by S and S3, both metrics and both pools.
   - Also reported: the share of resamples with R < 1.
6. **Self-checks before any output is written.**
   - The labeler main-effect differences reproduce `diff_mean_over_models` of `j62_main_effect.csv` within 1e-9.
   - The bearing counts match `j62_common_reference.csv`.
7. **Decision rule for the abstract (Gate 1, phrase 3A).**
   - The phrase "the model changes how well remaining life is ranked more than the labeler does" stays in the abstract only if, for C-rank, the upper 95% bound of R is below 1 in all 16 cells (4 configurations × 2 datasets × S and S3), for both pool (i) and pool (ii).
   - Otherwise the phrase is removed from the abstract, and the per-cell results are reported in the Results.
   - For C-common RMSE, no claim is made in the abstract whatever the result. The results are reported in the Results or in the supplementary material.

**Reason.**
- Gate 1 (2026-10-25) approved abstract phrase 3A, which compares the model and labeler effects, and kept S as the primary scoring set.
- This plan has no pre-specified analysis for that comparison. J65 compares the standard deviations of the mixed-model fixed effects, as a sensitivity analysis.
- A descriptive check (`GATE1_REPORT.md`, Section 11) showed that on C-common RMSE the comparison depends on the two-stage model.

**Results seen before the change.**
- All results of J57–J65, including the J65 standard deviations of the labeler and model effects.
- The descriptive check of `GATE1_REPORT.md`, Section 11: population standard deviations of the main effects with nine and with eight models, and leave-one-model-out point estimates, all without intervals.
- This analysis is therefore labeled post hoc in the R2.8 classification table.

## A7 — 2026-10-26

Written after all planned results and amendment A6 were reported. It changes where the files are released; no analysis is affected.

**Section(s) affected:** the "Integrity" paragraph at the head of the plan (deposit of the three files).

**Change.**
- The plan, its `.sha256` file and this file are released with the code and derived data in the GitHub repository https://github.com/anhnvm-ai/bearing-onset-rule-audit, release v1.1.0, instead of a Zenodo record of version 1.1.0.
- The version number and the content of the release are as the plan describes.

**Reason.** The authors chose to release the code and data on GitHub rather than on Zenodo before the release was built.

**Results seen before the change.** All planned results and the result of A6. None of them depends on where the files are released.

## A8 — 2026-10-26

Written after an internal review of the manuscript, before the resubmission. It changes no analysis and no result table.

**Section(s) affected:** 9 (secondary outputs of WP-C) and the release named in A7.

**Change.**
1. **Harrell's concordance over all records is not reported.** Section 9 plans the Spearman correlation and Harrell's concordance over S_j (main) and over all records (secondary). J62 computed the Spearman correlation over S_j and over all records, and the concordance over S_j and S3_j only. The secondary measure over all records is therefore reported as the Spearman correlation alone; the concordance over all records was never computed and is not added now.
2. **The release cited by the resubmission is v1.1.1.** It supersedes v1.1.0 (A7), which stays available. v1.1.1 adds the exporters and the supplementary tables for planned outputs that the draft checked by the internal review did not report: the reversals in C-rank concordance (secondary), Kendall's τ between model rankings, the share of resamples in which the best model differs, the Spearman correlations of item 1, the simulation delays and misses for each end-of-life size, and the median in-control run length. The result tables of J57–J69 are unchanged.

**Reason.** The plan's reporting commitments (Section 15) require every planned analysis to be reported. The internal review found the outputs listed in item 2 unreported, and found that J62 did not compute the concordance over all records.

**Results seen before the change.** All planned results, including the outputs of item 2, which were in the result tables of release v1.1.0.

## A9 — 2026-10-29

Written at 22:35 +07, before `scripts/J70_far_validation.py` and `scripts/J71_fixed_scoring_set.py` existed.

**Section(s) affected:** 8 (WP-A, calibration) and 9 (WP-C, model comparison under the common reference). Three analyses are added; no existing analysis and no existing result table changes.

**Change.**
1. **Validation of the calibrated multipliers on independent series** (`scripts/J70_far_validation.py`).
   - The 192 multipliers of `results/tables/j61_calibration.csv` (2 datasets × 2 protocols × 4 noise arms × 4 rules × 3 levels) are kept unchanged.
   - Self-check before any output is written: with the seed sequences of J61 ([61, dataset, protocol, noise arm]) and R = 4,000, the bisection of J61 must reproduce every multiplier and every achieved FAR of `j61_calibration.csv` to 1e-12.
   - New degradation-free series are simulated exactly as in J61 (same null models, trajectory lengths, protocols and c = 3), from the seed sequences [70, dataset, protocol, noise arm], with R = 10,000 series per length.
   - For each calibration: the false-alarm rate FAR_val of the new series at the calibrated multiplier, its Monte Carlo standard error SE_val, the interval FAR_val ± 1.96 SE_val, and z = (FAR_val − target) / sqrt(SE_cal² + SE_val²), where SE_cal is the `mc_se` of J61.
   - Criterion: the calibration is validated if no |z| exceeds 4 among the 192 calibrations and if, at each level, the median of FAR_val − target lies within ±0.2 percentage points. Also reported: the range of FAR_val at each level and the number of calibrations within the ±0.5-point tolerance of Section 8.
   - If the criterion fails, the affected multipliers are calibrated again on more series in a further amendment, and both results are reported.
   - The in-control run lengths of J61 were computed on separate series (seed sequence [61, 99, dataset]) and are not recomputed.
2. **A scoring set that does not depend on the labelers** (`scripts/J71_fixed_scoring_set.py`).
   - S^fix_j = {t : t ≥ ⌊0.8 n_j⌋}, the last 20% of the records of bearing j. No other fixed set is analyzed.
   - Everything else is as in J62: the four configurations, the two datasets and their bearings, the four labelers and all six labeler pairs, the nine models with input variant (a), the cached out-of-fold predictions of J62 (no model is retrained), the cell values (C-common: RMSE pooled over the scored records; C-rank: mean over bearings of Harrell's concordance over the scored records), for GRU and TCN the per-bearing values of the median seed chosen by J62 (by pooled C-common RMSE over S_j), the B = 20,000 resamples of J62 (seed 62, drawn in the same order, hence identical), the bootstrap p-value, Holm's correction over the 2 × 36 tests of each labeler pair, Kendall's τ, the share of resamples in which the best model differs, and the labeler main effects.
   - Self-check before any output is written: the same code applied to S_j and S3_j must reproduce `j62_common_reference.csv` (pooled values of variants (a) and (b), to 1e-9), `j62_rankings.csv` (reversals, best models, shares and τ) and `j62_main_effect.csv`.
   - Alignment of each labeler with each scored set (S_j, S3_j, S^fix_j), per configuration and dataset, pooled over bearings: coverage, the share of scored records with t ≥ i*_{L,j}, i.e. inside the training window that labeler L selects on bearing j; specialization, the share of the records with t ≥ i*_{L,j} that are scored.
   - Reading, made in C-common RMSE (the primary measure) on the two primary configurations (P-retro and P-prospective with the conventional labelers); the configurations with calibrated labelers are reported as sensitivity analyses, and C-rank without a reading:
     - F1: at least one significant reversal on S^fix between two of the three signal-based labelers (3sig_rms, ma5sig, ewma). The onset rule, through the training records it selects, can reverse model comparisons on a scoring set that does not depend on the labelers.
     - F2: significant reversals on S^fix only in pairs that include pct05. The reversals are attributed to the alignment of a labeler's training records with the scored records (the training window of pct05 covers only the last quarter of S^fix), not to better labels.
     - F3: no significant reversal on S^fix. The evaluation protocol, including the alignment of training and scored records, rather than the onset rule alone, reversed the comparisons on S_j.
   - The abstract, the contributions, the conclusion and item 4 of ORD-4 state the reading obtained. S_j remains the primary scoring set; S_j, S3_j and S^fix_j are reported side by side.
3. **Reversal test with the elapsed record count as input** (also `scripts/J71_fixed_scoring_set.py`).
   - Input variant (b) for the eight models that have it, and variant (a) for the two-stage model, which has no other; nine models, so the Holm family is unchanged.
   - Scoring sets S_j and S3_j, with the per-bearing values of variant (b) as in `j62_per_bearing.csv`, recomputed by the same code.
   - Same test and outputs as item 2, on the same resamples.
   - Reading, in C-common RMSE on the two primary configurations: at least one significant reversal over S_j means that the reversals do not depend on per-record inputs alone; none means that the reversal results are stated as conditional on per-record inputs. In both cases the Discussion states that the reversal results are conditional on the nine pre-specified models and their inputs.
   - Variant (c) (history features, tree models only) is not tested; its error stays as reported.
4. **Release.** The release cited by the resubmission becomes v1.2.0, which adds the two scripts, their result tables and this amendment. Releases v1.1.0 and v1.1.1 stay available.

**Reason.**
- An internal review of the resubmission package (2026-10-29) noted that (i) the achieved false-alarm rates were reported on the same series as the bisection, so their agreement with the target holds by construction; (ii) S_j depends on the labelers and coincides with the training window of pct05; (iii) the reversal test used per-record inputs only, although inputs with the elapsed record count lower the error.
- Section 8 plans the achieved FAR on the calibration series only. Section 9 plans S_j and its variant S3_j only, and reports the input variants by their error, without a reversal test.

**Results seen before the change.**
- All planned results (J57–J65) and the result of A6 (J69), including every reversal over S_j and S3_j and the error of every input variant.
- No result on S^fix_j, on the alignment indices, on reversals with variant (b) or on independent null series had been computed.
- Items 2 and 3 are therefore labeled post hoc in the R2.8 classification table; item 1 is a validation of a planned procedure, added after review.

## A10 — 2026-10-29

Written at 22:45 +07, after the results of A9 had been computed. It corrects one statement of A9 and changes no analysis and no result.

**Section(s) affected:** none of the plan; the first line of amendment A9.

**Change.** The first line of A9 gives its writing time as 22:35 +07, a time estimated before the entry was saved. The file system records the entry as saved at 22:31:26 +07. `scripts/J70_far_validation.py` was created at 22:32:23 and first run at 22:32:35; `scripts/J71_fixed_scoring_set.py` was created at 22:34:43 and first run at 22:35:23. A9 was therefore written at 22:31 +07, before either script existed, as its first line states; only the clock time was wrong. The script that appended A9 refused to run if either script existed.

**Reason.** The time stated in A9 is later than the first run of J70, so the order of events would appear reversed.

**Results seen before the change.** The results of J70 and J71.

## A11 — 2026-10-29

Written at 23:39 +07, before `scripts/J72_constant_reference.py` existed. Post hoc and descriptive.

**Section(s) affected:** 9 (WP-C). One descriptive reference is added; no analysis and no result changes.

**Change.** For each configuration, dataset and scoring set (S_j, S3_j and S^fix_j of A9), the RMSE against the common reference r_t of two constant predictions, computed by `scripts/J72_constant_reference.py`:
1. a constant equal to the mean of r_t over the training records of pct05 (t ≥ ⌊0.95 n⌋) on the training bearings of each fold, with one bearing held out at a time as in J62;
2. the best constant, the mean of r_t over the scored records of all bearings, which no constant prediction can beat.
- Inputs: the bearing lengths and the numbers of scored records per bearing in `j62_per_bearing.csv` (S_j and S3_j are the last records of each bearing), and ⌊0.8 n⌋ for S^fix_j. No model is involved.
- Self-check: the numbers of records per bearing equal those of `j62_per_bearing.csv`, and the S^fix_j counts equal n − ⌊0.8 n⌋.
- Reported as a description next to the C-common RMSE of the models trained under pct05; no reading is attached.

**Reason.** A second internal review (2026-10-29) noted that S^fix_j contains the whole training window of pct05, so a constant prediction near the end of life may score about as well as the models trained under pct05, and their low error would then say little about the labels.

**Results seen before the change.**
- All results of J70 and J71.
- The reviewer's approximate values for a reference spread evenly over the last 20% of life: about 0.095 for a constant of 0.025 and about 0.058 for a constant of 0.1.
