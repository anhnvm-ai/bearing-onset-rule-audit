# Pre-specified analysis plan for the resubmission of IEEE Access manuscript Access-2026-51094

**Manuscript.** *Effect of Onset Rules on Degradation Labels and Remaining-Useful-Life Evaluation in Run-to-Failure Bearing Data.*
**Authors.** Anh Nguyen Vu Minh, Cao-Phuc Ha, Nguyen Le Thai, Huu-Tho Nguyen, Phu-Nguyen Le.

**Lock.**
- Locked on 2026-10-24, before any analysis in Sections 7–13 was run. The exact time is recorded in `ANALYSIS_PLAN_2026-10-24.sha256`.
- All authors approved the design decisions this plan implements on 2026-10-24.

**Integrity.**
- This file is read-only and is never edited. Its SHA-256 digest is stored in `ANALYSIS_PLAN_2026-10-24.sha256`.
- Any deviation is recorded, with its date, reason and affected section, in the append-only file `ANALYSIS_PLAN_AMENDMENTS.md` before the affected result is reported.
- All three files will be deposited with the code and derived data (Zenodo, version 1.1.0).

---

## 1. Purpose and scope

This plan fixes the analyses added for the resubmission before they are run. They answer the three review reports of 23 October 2026, which are cited as R1.1–R1.7, R2.1–R2.9 and R3.1–R3.5; R1.7 is the general comment of Reviewer 1. The reports are in `REVIEWS_2026-10-23.md`.

The plan covers scripts J57–J65 of the project repository and the reuse of results that existed before the lock (Section 2).

The revised manuscript addresses one question:

> How sensitive are onset-derived degradation labels and downstream remaining-useful-life (RUL) conclusions to the implementation choices of an onset protocol?

Five implementation factors are examined:
- the baseline protocol (Phase I and Phase II);
- the false-alarm calibration;
- the confirmation count;
- the rule design (family and parameters);
- the treatment of trajectories on which a rule never fires.

## 2. Results seen before the lock (disclosure)

The analyses below were run before this plan was written, most of them before the original submission. Their results were known when the choices in this plan were made.

| Scripts | Content | Where it was reported |
|---|---|---|
| J3, J11, J13 | Ten onset rules on 33 trajectories; convention grid (baseline 10–40% of life × confirmation 1, 3, 5) | Submitted manuscript |
| J20 (j22 tables) | Null-model p-values, in-control run lengths, baseline autocorrelation | Partly in the submitted manuscript |
| J23 | Condition-adjusted slopes, within-condition correlations, condition-stratified bootstrap, retrospective changepoint reference | Partly in the submitted manuscript |
| J24, J44 | Known-change simulations: step and ramp, Gaussian and Student-t noise, φ up to 0.9, a no-change arm | J24 in the submitted manuscript; J44 in an unsubmitted supplement |
| J26, J26b, J26c, J26d | 67-design sweep; decimation; fixed-length baselines; adaptive update rules | J26 in the submitted manuscript; the others unsubmitted |
| J28 | Kaplan–Meier treatment of non-firing trajectories | Submitted manuscript |
| J32, J35–J39 | RUL targets; six regressors; per-bearing results; ratio bootstrap; two-stage model | Submitted manuscript |
| J41, J42, J43 | Nominal shift; pool-size-corrected dispersion; held-out bearing halves | Unsubmitted supplement |
| J45, J46 | Spectral and envelope indicators; indicator sweep | Unsubmitted supplement |
| J47, J48 | GRU and TCN with shuffled-order and constant-window controls; nine-model dispersion | Unsubmitted supplement |

Four checks were also run on 23–24 October 2026 on the deposited derived data:
1. The Kaplan–Meier "share of life labeled degraded" of J28 equals the bearing-averaged positive rate with non-firing trajectories labeled healthy.
2. ρ(i*/n, n) was computed for the ten rules.
3. Onset fractions and in-baseline counts were tabulated by operating condition.
4. The alarm indices of the two control charts on XJTU-SY were listed.

Of these results, the fixed-baseline slopes of J26c bear directly on the protocol choice. They concern the difference between the RMS threshold detectors and the charts in the slope of log i* on log n.
- With a fixed baseline of 10–15 records, the difference vanishes on XJTU-SY for 3sig_rms and ma3sig, while ma5sig keeps part of it. On FEMTO it reverses.
- With 20–30 records it partly returns.
- The intervals are wide throughout.

Because these results were known:
1. The primary protocol (Section 5) is fixed by a rule that does not depend on them.
2. The relation between onset and test length is reported over the full protocol curve, not only at the primary protocol.

## 3. Data, units and bearing sets

- **Data.** Datasets and trajectories are those of the submitted manuscript: XJTU-SY (15), FEMTO (13), IMS (4) and NSK 6205 (1). Health indicators, windows and record indexing are unchanged. Nine rules use RMS; one uses kurtosis.
- **Unit of analysis.** The experimental unit is the bearing. Operating conditions:

  | Dataset | Bearings per condition |
  |---|---|
  | XJTU-SY | 5 / 5 / 5 |
  | FEMTO | 6 / 5 / 2 |
  | IMS | 2 / 1 / 1 (one test per condition) |
  | NSK | 1 |

- **Intervals.** Intervals are computed for XJTU-SY and FEMTO only. IMS and NSK are reported descriptively.
- **Eligibility for fixed Phase-I windows.** A protocol with a fixed Phase-I window of N0 records is applied only to trajectories with n ≥ 4·N0.

  | N0 | XJTU-SY | FEMTO | IMS | NSK |
  |---|---|---|---|---|
  | 10 | 15 | 12 | 4 | 1 |
  | 15 | 13 | 12 | 4 | 1 |
  | 20 | 13 | 11 | 4 | 1 |
  | 30 | 12 | 9 | 4 | 1 |

  At N0 = 20 the excluded trajectories are XJTU-SY Bearing1_5 (n = 52) and Bearing2_4 (n = 42), and the two FEMTO trajectories with n = 38 and n = 72.
- **Common bearing sets.** Every comparison between protocols, calibrations or labelers is computed on one set of bearings that is eligible under every arm in the comparison. The list of bearings is written to the script's log.
- **Non-firing trajectories.** Onset statistics are computed over the trajectories on which a rule fires, and the firing coverage is reported. Positive rates are reported under the decomposition of Section 13, item 1.

## 4. Onset rules and designs

The definitions are unchanged from the submitted manuscript.

**The ten rules R1–R10:**

| Group | Rules | Definition |
|---|---|---|
| Threshold detectors | R1 (3sig_rms), R2 (3sig_kurt) | k = 3 |
| Moving-average detectors | R3 (ma2sig), R4 (ma3sig), R5 (ma5sig) | k = 2, 3, 5; trailing five-record mean, fewer records at the start |
| EWMA chart | R6 | λ = 0.2, L = 3, asymptotic limit |
| CUSUM chart | R7 | One-sided upper, κ = 0.5σ, H = 5σ |
| Percentile rules | R8–R10 | Onset at ⌊n(1 − p)⌋ |

For all rules, the confirmation count is c = 3 and the alarm index is the first record of the confirming run. σ is the population standard deviation (ddof = 0).

**The 67 designs** keep the definitions of `scripts/J26_design_sweep.py` (function `build_grid`):

| Family | Count | Parameters |
|---|---|---|
| EWMA | 25 | λ ∈ {0.05, 0.1, 0.2, 0.4, 0.8} × L ∈ {2, 3, 4, 5, 6} |
| CUSUM | 20 | κ ∈ {0.25, 0.5, 1, 2, 3}σ × H ∈ {3, 5, 8, 12}σ |
| Threshold on RMS | 9 | k ∈ {1, 1.5, 2, 2.5, 3, 4, 5, 6, 8} |
| Moving-average threshold | 3 | k ∈ {2, 3, 5} |
| Autocorrelation-corrected EWMA | 5 | λ ∈ {0.05, 0.1, 0.2, 0.4, 0.8}, L = 3 |
| Adaptive-limit chart | 5 | L ∈ {2, 3, 4, 5, 6}, expanding-window update of J26 |

J59 exports this list with every parameter.

## 5. Protocols

Each protocol is defined by four elements:
- the Phase-I window W, the records used to estimate μ and σ;
- the estimator;
- the Phase-I cleaning;
- the first record t0 at which an alarm may start.

| Arm | Phase-I window W | Estimator | Phase-I cleaning | Alarm start t0 | Role |
|---|---|---|---|---|---|
| P-retro | First n_b = max(5, ⌊0.2n⌋) records | Mean, population SD | None | 0 | Submitted manuscript; control arm |
| P-II | As P-retro | Mean, population SD | None | n_b | Changes only the alarm start |
| P-robust | As P-retro | Median, 1.4826 × MAD | None | 0 | Changes only the estimator |
| P-iter | As P-retro | Mean, population SD | Iterative | n_b | Phase I as in SPC practice |
| P-fixed(N0), N0 ∈ {10, 15, 20, 30} | First N0 records | Mean, population SD | None | N0 | The protocol curve |
| **P-prospective (primary)** | First 20 records | Mean, population SD | Iterative | 20 | Primary protocol for Sections 8 and 9 |

**Iterative cleaning:**
1. Estimate μ and σ on the retained Phase-I records.
2. Flag every retained record with |h_t − μ| > 3σ, and remove the flagged records.
3. Repeat until no record is flagged, or for at most three iterations.
4. If an iteration would leave fewer than 15 records (for P-iter, fewer than half of W), keep the set from the previous iteration.

**Monitoring:**
- Monitoring statistics are computed causally over the whole trajectory; only the alarm search starts at t0.
- At record t0 − 1, the EWMA statistic is initialised at μ and the CUSUM statistic at 0. For t0 = 0 this is exactly the submitted convention.
- Moving averages include records before t0 when those records fall inside their window.

**Alarm index:** i* = min{t ≥ t0 : the alarm condition holds for t, …, t + c − 1}.

**"Inside the baseline"** is defined only for arms with t0 = 0, as i* < |W|.

**Reproduction check.** J60 must reproduce all 330 onsets of `results/tables/fig1_fpt_matrix.csv` exactly under P-retro before it writes any other output. Otherwise the run stops.

**Rationale for the primary protocol:**
- It needs no knowledge of n (R1.3).
- It separates Phase I from Phase II (R2.7).
- It cleans Phase I (R2.6, R3.2).
- N0 = 20 is the usual minimum Phase-I sample for trial limits. With n ≥ 80, the Phase-I window covers at most 25% of any trajectory it is applied to.
- N0 = 10, which keeps more bearings, enters only as a sensitivity arm.

## 6. Quantities, estimators and inference

- **Onset fraction.** u = (i* + 1)/n. The added 1 handles alarms at record 0, as in Eq. (5) of the submitted manuscript.
- **Length scaling.**
  - Main statistic: β1 − 1, the slope of log u on log n, fitted by ordinary least squares with fixed effects for operating condition on XJTU-SY and FEMTO. This is identical to β1 − 1 in the submitted Eq. (5).
  - Rank version: Spearman ρ(u, n).
  - ρ(i*, n) and β1 are reported only for continuity, in the supplementary material.
- **Shift at the alarm.** d, as in the submitted Eq. (4), computed with each arm's μ and σ. It is descriptive.
- **Positive rate.** Reported both record-pooled (submitted Eq. (2)) and bearing-averaged.
- **Agreement and target distance.** Cohen's κ, and D_j as in the submitted Eq. (7). Both are unchanged.
- **Intervals.**
  - Percentile 95% intervals from B = 20,000 bootstrap resamples of whole bearings, stratified by operating condition: bearings are resampled with replacement within each condition, keeping each condition's size.
  - For any statistic computed over the 67 designs, every resample recomputes all 67 design summaries from the per-trajectory alarms before the fit.
  - The random seed equals the script number (J58 → 58, …, J65 → 65) unless stated otherwise.
- **Mixed-effects models** are a sensitivity analysis only (Section 12).

## 7. WP-B — protocol curve, baseline contamination and warm-up (J60)

**Outcomes.** For each arm of Section 5, on a common bearing set:
- μ̂ and σ̂ per bearing, and their ratios to P-retro;
- i*, u and firing coverage;
- positive rate under both weightings;
- D_j between rule pairs;
- d;
- β1 − 1 with its interval;
- in-baseline counts, for arms with t0 = 0.

**Protocol curve.** β1 − 1 for each rule:
- Main: under P-retro, P-II, P-robust, P-iter, P-fixed(10), P-fixed(15), P-fixed(20) and P-prospective, on the bearings eligible at N0 = 20 (XJTU-SY 13, FEMTO 11).
- P-fixed(30) on its own eligible set (XJTU-SY 12, FEMTO 9).
- Every arm on its own eligible set, in the supplementary material.

**Contamination diagnostic** (descriptive only).
- Applies to bearings whose P-retro onset falls inside the baseline and that have at least 20 records before the alarm.
- For these bearings, μ and σ are re-estimated on the records before the alarm, and the onset, positive rate and RUL target are recomputed.
- Bearings with fewer records before the alarm are listed and not used.
- For R2.6, whose quantity symbol is unreadable in the decision letter, the report covers σ̂ and μ̂, the length-scaling statistic (β1 − 1 and ρ), the positive rate and the RUL target.

**Warm-up hypothesis H_w** (pre-specified). Under P-retro, the early chart alarms occur because the first records lie above the P-retro baseline mean.
- Statistic, per bearing: e10 = (mean of records 0–9 − μ̂_retro)/σ̂_retro.
- H_w is supported if both conditions hold:
  1. For EWMA and for CUSUM, Spearman ρ(e10, i*) is negative with a 95% bootstrap interval that excludes 0 on at least one of XJTU-SY and FEMTO, and is negative on the other.
  2. At least two thirds of the chart alarms with i* < 0.05n occur on bearings with e10 > 1.
- The first 100 records of every bearing with a chart alarm before record 10 are plotted for the supplementary material.

**Pre-specified reading:**

| Result | Reading |
|---|---|
| The detector–chart difference in β1 − 1 or in in-baseline counts changes materially across arms (intervals at different arms do not overlap, or the sign changes) | The evidence favours a protocol-dependent relation between onset and test length. This becomes the main sensitivity finding of Section IV |
| The difference is stable across arms (same sign, overlapping intervals) | Reported as robust to the protocol |
| H_w is supported | The in-baseline chart alarms are described as an early-transient effect under a retrospective baseline. This replaces the submitted statement that the charts "contradict their own premise" |
| H_w is not supported | The in-baseline alarms are described as either a non-stationary baseline or false alarms, without choosing between the two |

## 8. WP-A — matched false-alarm calibration (J61)

**Protocols.** P-prospective (primary) and P-retro (control).

**Rules calibrated:**

| Rule | Calibrated parameter |
|---|---|
| Threshold on RMS | k |
| Moving-average threshold on RMS (merges R3–R5 into one calibrated rule) | k |
| EWMA, λ = 0.2 | L |
| CUSUM, κ = 0.5σ | H |

3sig_kurt is not calibrated because it uses a different indicator.

**Null model.**
- Main: Gaussian AR(1) with unit variance. φ is the dataset median of the detrended lag-1 autocorrelation of the baseline (`results/tables/j22_baseline_autocorrelation.csv`): 0.367 for XJTU-SY and 0.264 for FEMTO.
- Sensitivity: Student-t(3) innovations rescaled to unit variance; φ at the 25th and 75th percentiles of the per-trajectory values.
- Series are simulated at the actual lengths of the dataset's trajectories that are eligible under the protocol.
- The full protocol is applied to every simulated series, including Phase-I estimation and cleaning on that series, and c = 3.

**Calibration target.**
- FAR = the probability of at least one alarm at t ≥ t0 on a no-change series, averaged with equal weight over the dataset's trajectories.
- Primary level: 5%. Sensitivity levels: 1% and 10%.
- ARL0 is reported as a secondary measure, on series of up to 10,000 records. Larger values are reported as "> 10,000".

**Procedure.**
- One multiplier per rule, dataset and protocol. There is no per-length calibration.
- Bisection with common random numbers, using 4,000 series per trajectory length at each evaluation.
- The achieved FAR must lie within ±0.5 percentage points of the target. Its Monte Carlo standard error is reported.
- If a rule cannot reach the target (for example because of the floor set by c = 3), the nearest attainable level is used and reported.

**Outcomes on the real trajectories.** At the calibrated multipliers (5% in the main text; 1% and 10% in the supplementary material), on a common bearing set:
- u and β1 − 1;
- d;
- positive rate;
- κ between the calibrated chart labels and the calibrated detector labels, in place of the submitted 68.9% disagreement;
- D_j.

**Design FAR.** The FAR of each of the 67 designs at its own canonical parameters, under P-prospective and under P-retro, is computed for use in Section 10. On the logit scale it is clipped to [1/(2R), 1 − 1/(2R)], where R is the number of simulated series.

**Pre-specified reading:**

| Result at matched 5% FAR | Reading |
|---|---|
| The chart–detector difference shrinks clearly (the intervals overlap, or the difference is less than half of the uncalibrated one), or disappears | Evidence against a family-level explanation increases, and evidence for sensitivity to calibration and protocol increases. Statements comparing charts and detectors are restricted to conventional settings. This is not described as proof that calibration decides |
| The difference persists, and the calibrated families alarm at different d | Consistent with the response-size account: at equal false-alarm rates, charts respond to smaller shifts |
| The difference persists, and the families alarm at similar d | The family has an effect beyond calibration, and the response-size account is narrowed |

## 9. WP-C — RUL evaluation on a common retrospective reference (J62)

**Labelers.**
- 3sig_rms, ma5sig, ewma and pct05 at their canonical settings, under P-retro and under P-prospective.
- Calibrated labelers at 5% FAR (Section 8) are a sensitivity analysis.
- For each protocol, only bearings on which all labelers fire are used, and their list is logged.

**Models.**

| Model | Settings |
|---|---|
| Six regressors of J35 | Linear regression on RMS; ridge, α = 1; k-nearest neighbours, k = 15; gradient boosting (default settings); extra trees and random forest with 200 trees each. Standardisation as in J35 |
| Two-stage exponential model of J39 | Variant 'exp' |
| GRU and TCN | Settings of J47: window of 16 records, 30 epochs, batch size 256, Adam with learning rate 1e-3, seeds 0, 1 and 2, median over seeds |

**Input variants.**
- (a) Per-record statistics, as in J35.
- (b) (a) plus the elapsed record count t, computed within each dataset. The sampling interval is constant within a dataset, so t is proportional to elapsed time. The total length n and the ratio t/n are never inputs.
- (c) (a) plus history features for the tree models: the trailing mean and the OLS slope of RMS over the last W = 16 records, and the cumulative sum of RMS² from record 0. W = 64 is added only if Gate 1 keeps it.

All models are evaluated by leave-one-bearing-out cross-validation within each dataset.

**Common record set.** For bearing j, S_j = {t ≥ max over labelers of i*_{L,j}}. A variant uses only the three signal-based labelers, because with pct05 included S_j covers only about the last 5% of life.

**C-rank (primary).**
- Models are trained on each labeler's own target (submitted Eq. (3)).
- Within each bearing, the Spearman correlation and Harrell's concordance index are computed between ŷ_t and the number of remaining records, n − 1 − t. Ties in ŷ count one half.
- They are computed over S_j (main) and over all records (secondary), and reported as the median over bearings with an interval.
- For the two-stage model, the predicted remaining records are T̂ − t, where T̂ is its extrapolated end-of-life record.

**C-common (primary).**
- For labeler L, the training records of each training bearing are those with t ≥ i*_{L,j}.
- Every model is trained on the same target, the common retrospective RUL reference r_t = (n − 1 − t)/(n − 1). Because it uses the total life n, it is valid only for offline evaluation.
- The two-stage model outputs r̂_t = (T̂ − t)/T̂, clipped to [0, 1].
- Predictions are scored over S_j of the test bearing, as RMSE pooled over records and as RMSE per bearing.

**Model comparison under the common reference.**
- Scope: the nine models with input variant (a); the other variants are secondary.
- For each pair of labelers (A, B) and each pair of models (X, Y), the paired difference in C-common RMSE is bootstrapped over bearings.
- A significant reversal occurs when X is better than Y under A, Y is better than X under B, and both differences are significant at familywise α = 0.05.
- Holm correction is applied over the 2 × 36 tests of each labeler pair. The bootstrap p-value is p = 2·min(P(Δ ≤ 0), P(Δ ≥ 0)).
- The same analysis on C-rank is reported as secondary.
- Also reported:
  - the proportion of resamples in which the best model differs between A and B;
  - Kendall's τ between the two rankings.
- Target-defined rankings, as in the submitted manuscript, are reported as descriptive results only.

**Pre-specified reading:**

| Result | Reading |
|---|---|
| At least one significant reversal under the common reference | Headline: the choice of onset protocol can reverse model comparisons even when every model is scored on the same target and the same records |
| No reversal, but the labeler main effect on C-rank or C-common has an interval that excludes 0 | The size of the effect is reported; rankings are not a headline |
| Neither | The RUL section is shortened to one paragraph and one table, stating that the effect is small on a common reference. Target-defined rankings remain as descriptive results |

## 10. WP-F — design parameters and onset scaling (J64)

**Outcome.** β1 − 1 for each of the 67 designs, under P-retro (from `results/tables/j26_design_sweep_per_traj.csv`) and under P-prospective (designs re-run through J60).

**Models.** Every predictor is known before any bearing is seen.

| Model | Predictors |
|---|---|
| M0 | Family label of J26: detector, control chart, AR-corrected chart, adaptive-limit chart |
| M1 | Nominal shift (definitions of J41): k for thresholds; k/√5 for moving averages; L·√(λ/(2 − λ)) for EWMA, also for the AR-corrected EWMA; 2κ for CUSUM; L for adaptive limits |
| M2 | Logit of the design FAR (Section 8) |
| M3 | M2 + family |

M4 (M1 + M2) is optional. Raw design parameters enter only through the nominal shift and the FAR, because raw parameters are not comparable across families.

**Reported:**
- in-sample R² for each model;
- the distribution of R² over stratified bearing resamples, with the design FAR held at its full-sample value;
- ΔR²(M3 − M2) with its interval.

**Out of sample.**
- 2,000 random splits of the bearings into halves, stratified by condition, as in J43. Where fewer than 2,000 distinct stratified splits exist (FEMTO), all distinct splits are used once. Each model is fitted on one half and predicts the design outcomes of the other half; out-of-sample R² is recorded.
- The split-half reliability of the outcome, the correlation of the design outcomes between halves, is reported as the ceiling against which out-of-sample R² is read.

**Descriptive mediators.** Measured d and onset fraction are presented as the chain θ → FAR → d → u, without causal language.

**Pre-specified reading:**

| Result | Reading |
|---|---|
| The interval of ΔR²(M3 − M2) includes 0, and M2 beats M0 out of sample in at least 75% of splits | "A design's a-priori false-alarm calibration predicts its onset scaling better than its family label" |
| The family adds beyond FAR (the interval of ΔR² excludes 0) | Report that the family carries information beyond calibration |
| Out-of-sample R² is below 0 for every model | The in-sample ordering is reported only as descriptive, and the claim is removed from the abstract |

## 11. WP-E — simulation with gradual degradation (J63)

**Signal.** h_t = μ + σ·g(t/n) + w_t + ε_t, simulated at the XJTU-SY lengths eligible under each protocol. The change starts at u0 ∈ {0.3, 0.5, 0.7}. Let s = max(0, (t/n − u0)/(1 − u0)). The shapes are:

| Shape | Definition |
|---|---|
| Step (control) | g = δ·1[t/n ≥ u0] |
| Linear ramp | g = δ·s |
| Two-phase exponential | g = δ·(e^{5s} − 1)/(e^5 − 1) |
| Warm-up (only if Gate 1 keeps it) | Two-phase exponential plus w_t = 2σ·e^{−t/5} |

**Noise.** Gaussian AR(1) at φ = 0.367, or a heteroscedastic version with the innovation standard deviation multiplied by 1 + g/δ. The end-of-life size is δ ∈ {2, 6}.

**Protocols and calibrations.**
- P-prospective and P-retro, each at canonical settings and at matched 5% FAR.
- 400 series per cell, with seeds derived from a CRC of the cell parameters (the J44 convention).

**Reference times.** t_{δ'} is the first record at which the noiseless g reaches δ' ∈ {1, 3}. Delays are measured from t_{δ'}.

**Outcomes:**
- the share of alarms before t_{δ'};
- the share of misses;
- the delay;
- u;
- β1 − 1 across lengths.

## 12. WP-G — sensitivity models (J65)

The primary inference is that of Section 6. Linear mixed models are reported as sensitivity analyses only. They use statsmodels MixedLM with REML, a random intercept for bearing and operating condition as a fixed effect:
1. C-common RMSE per bearing, and C-rank, ~ labeler + model + (1 | bearing).
2. log u ~ log n × family + condition + (1 | bearing), fitted only if Gate 1 keeps it.

Operating condition enters as a fixed effect because it has only three levels.

## 13. Quick analyses (J57–J59)

1. **J57, non-firing decomposition.**
   - For every dataset × rule with at least one non-firing trajectory, positive rates are computed under 2 weightings (record-pooled, bearing-averaged) × 2 treatments (exclusion, labeled healthy).
   - The script asserts that the J28 Kaplan–Meier value equals the bearing-averaged, labeled-healthy rate to within 1e-9.
   - It writes the algebraic argument to its log: censoring at u = 1 follows every event, so 1 − RMST equals the bearing mean of (1 − U_j), with non-firing trajectories contributing 0.
2. **J58, onset fraction and conditions.**
   - u, β1 − 1 and ρ(u, n) for the ten rules under P-retro.
   - Per-condition tables for XJTU-SY and FEMTO, giving u, d, in-baseline counts, positive rate under both weightings, and coverage.
3. **J59, configuration and settings tables.**
   - The 67 designs and the ten rules, with every parameter and the nominal shift.
   - The model settings of Section 9.

## 14. Gates

**Gate 1** (after J57–J62):
- (G1-a) M0–M3 always run; M4 is optional.
- (G1-b) WP-E keeps the step, ramp and exponential shapes. The warm-up case is kept only if H_w is supported.
- (G1-c) If WP-C gives the third reading of Section 9, W = 64 and any further history variants are dropped.
- (G1-d) Mixed models are fitted only for the RUL analysis, unless the onset results depend on the condition structure.
- (G1-e) If the work is more than one week behind schedule, only the non-required parts of WP-E, WP-F and WP-G are cut.
- (G1-f) A provisional three-point abstract is drafted.

**Gate 2** (after WP-E, WP-F and WP-G) fixes:
- the claims;
- the three contributions;
- the split between the main text and the supplementary material;
- the figure and table list, with at most six figures and five tables.

Gate decisions choose among the analyses in this plan. They do not add analyses; anything added is an amendment.

## 15. Reporting commitments

- Every analysis in this plan is reported, in the main text or in the supplementary material, whatever its result.
- Analyses dropped at a gate are listed together with the reason.
- Deviations from this plan are recorded in `ANALYSIS_PLAN_AMENDMENTS.md` before the affected result is reported.
- The response to R2.8 cites this file and its lock time.

## 16. Software

- Python 3.12.4.
- numpy 2.2.6, pandas 2.3.2, scipy 1.14.0, scikit-learn 1.5.0, statsmodels 0.14.6.
- torch 2.14.0+cpu for GRU and TCN. The deposited J47 results were produced with torch 2.12.0+cpu, so J62 re-runs the J47 configuration under 2.14.0 and reports any difference.
