# Code

72 scripts and the 15-module library they import. File names are the ones the article, its
supplementary material and the logs refer to, so they are kept as they are rather than renumbered.

```
code/
  make_workspace.py   lays the scripts out so that their relative paths resolve
  scripts/            the analyses: J3-J49 (first submission) and J57-J69 (resubmission)
  src/                the library the scripts import (features, metrics, thresholds, splits)
```

## Running

```
python code/make_workspace.py
python scripts/<script>.py
```

`make_workspace.py` copies `code/scripts` and `code/src` to the repository root, next to `results/` and `data/`,
because every script resolves the project root as the parent of its own directory. It writes
`colab_bundle/<dataset>/records.csv` from `data/derived/record_index_*.csv`, and it creates the manuscript folders
into which the exporters write (the manuscript itself is not released).

## The pipeline

| Step | What it does | Script | Needs |
|---|---|---|---|
| 01 | Per-record vibration statistics (RMS, kurtosis, crest, p2p, ...) for every trajectory | `J7_circularity.py, J7c_femto_channel_test.py` | raw datasets |
| 02 | Onsets of the ten rules on the 33 trajectories, and the induced label sets | `J3_fpt_rules.py` | raw datasets |
| 03 | Duration signature: fire index against trajectory length, and its mechanism | `J11_arl_and_covariates.py, J20_arl_mechanism.py, J29_scaling_law.py` | this repository |
| 04 | Size of change at each alarm, and the 67-design response-size sweep | `J21_fig_shift.py, J26_design_sweep.py, J36_design_cv.py` | this repository |
| 05 | Label-set identity: pairwise agreement, prevalence screen, censoring | `J13_convention_sensitivity.py, J15_adoption_prevalence.py, J28_censoring.py` | this repository |
| 06 | RUL targets of Eq. (2) and their disagreement | `J32_rul_target_disagreement.py` | this repository |
| 07 | Downstream experiment: four labelers, six regressors, leave-one-bearing-out | `J35_downstream_rul.py, J37_downstream_per_bearing.py, J38_downstream_ratio_bootstrap.py, J39_two_stage_rul.py` | this repository |
| 08 | Release file and the ORD-4 reporting screen | `J17_release_and_screen.py` | this repository |
| 09 | What the response-size relation identifies: the shift each design responds to by construction, against the shift it is measured to alarm at | `J41_estimand_separation.py` | this repository |
| 10 | Dispersion of the downstream error corrected for pool size, and the response-size curve on held-out bearings | `J42_dispersion_statistics.py, J43_heldout_bearings.py` | this repository |
| 11 | The known-change simulation under heavier tails, more memory, and with no change at all | `J44_synthetic_robustness.py` | this repository |
| 12 | Spectral and envelope indicators of the same records | `J45_indicator_extraction.py` | raw datasets |
| 13 | The 67-design sweep repeated on every indicator | `J46_indicator_sweep.py` | this repository |
| 14 | Two sequence models in the downstream pool, with a shuffled-order and a constant-window control (needs torch) | `J47_sequence_models.py` | this repository |
| 15 | The pool-size-corrected dispersion of step 10, on the pool step 14 extends | `J48_pool_extended_dispersion.py` | this repository |
| 16 | The three treatments of a trajectory a rule never fires on, and the leave-one-bearing-out schematic | `J49_supplementary_figures.py` | this repository |
| 17 | Figures and the exported LaTeX numbers (the exporters update macro files of the manuscript, which is not part of this repository) | `J14_journal_figures.py, J25_reviewer_figures.py, J19_export_numbers.py, J39b/J41b/J43b/J44b/J46b/J47b/J48b/J49b_export_numbers.py` | this repository |
| 18 | Treatment x weighting of non-firing trajectories; Kaplan-Meier identity | `J57_censoring_decomposition.py` | this repository |
| 19 | Onset fraction and onset scaling per rule and operating condition | `J58_onset_fraction.py` | this repository |
| 20 | Rule configurations and model settings, read from the code | `J59_configurations.py` | this repository |
| 21 | Baseline protocols P-retro to P-prospective; onsets, gaps, warm-up test, contamination | `J60_baseline_protocols.py` | this repository |
| 22 | False-alarm calibration on an AR(1) null; matched onsets, ARL0, matched gaps | `J61_far_calibration.py` | this repository |
| 23 | RUL evaluation on the common retrospective reference: C-rank, C-common, reversals, main effects (GRU and TCN need torch) | `J62_common_reference_rul.py` | this repository |
| 24 | Simulated gradual degradation under each protocol | `J63_gradual_simulation.py` | this repository |
| 25 | Design-parameter models M0-M4, in and out of sample | `J64_design_parameters.py` | this repository |
| 26 | Linear mixed models with a random intercept per bearing (sensitivity) | `J65_mixed_effects.py` | this repository |
| 27 | Amendment A6 (post hoc): ratio of labeler to model effects | `J69_labeler_model_contrast.py` | this repository |
| 28 | Every number and table body of the resubmitted paper (writes into paper/access_resub/) | `J66_resub_export_numbers.py` | this repository |
| 29 | The six figures of the resubmitted paper (writes into paper/access_resub/figures/) | `J67_resub_figures.py` | this repository |
| 30 | This release | `J68_make_release_v110.py` | this repository |

Steps 01, 02 and 12 read the raw acquisitions; every other step runs from the tables in `results/tables/`.

## Differences from the code that produced the results

In the scripts that read the raw acquisitions, the local dataset folder was replaced by the environment variable
`BEARING_DATASETS` (default `./datasets`) when this release was built: J18_data_audit.py (1); J31_xjtu_20g.py (1); J45_indicator_extraction.py (1); J7c_femto_channel_test.py (1). In 14 scripts, comments
that referred to an earlier manuscript or quoted a review comment were shortened; their code is unchanged, which the
release build checks by comparing syntax trees. Nothing else differs.

Not released, because each only produced manuscript files for earlier submissions or revisions and none writes a
result table:
- J33: a graphical abstract that an earlier submission required.
- J34: a highlights file that an earlier submission required.
- J50: Overleaf projects of earlier manuscript trees.
- J51: the manuscript tree of an earlier submission, from a local template.
- J52: the figures of an earlier submission, restyled.
- J53: the submission package of an earlier submission.
- J54: Fig. 1 of an earlier revision, and two figures restyled from J14 and J25.
- J55: a redrawing of Fig. 1 of an earlier revision.
- J56: Fig. 1 of an earlier revision as a draw.io file.

## Conventions the scripts share

Bearings, not records or designs, are the independent units: bootstrap intervals resample whole bearings within each
operating condition (20 000 resamples), and cross-validation holds out whole bearings. A label set "fails the
prevalence screen" when its positive rate falls outside [0.01, 0.50]. Scripts that write a `_quick` output were run
on a reduced grid as a code check; such outputs are not evidence and are not released.
