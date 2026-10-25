# bearing-onset-rule-audit

Code and derived data for the article

> Anh Nguyen Vu Minh, Cao-Phuc Ha, Nguyen Le Thai, Huu-Tho Nguyen and Phu-Nguyen Le, "Effect of Onset Rules and Their Implementation on Degradation Labels and Remaining-Useful-Life Evaluation in Run-to-Failure Bearing Data," manuscript under review, 2026.

Run-to-failure bearing datasets record no degradation onset, so an onset rule creates one. This repository holds the
onsets that ten published rules and 67 rule designs place on 33 trajectories from four public datasets,
the labels and remaining-useful-life (RUL) targets derived from them, and the code that produced every number, table
and figure of the article and its supplementary material. Release 1.1.1 accompanies the resubmitted manuscript.

## What is here

```
bearing-onset-rule-audit/
  README.md         this file
  LICENSE           MIT, for everything under code/
  LICENSE-DATA      CC BY 4.0, for everything under data/, plan/ and results/
  CITATION.cff      how to cite this repository
  MANIFEST.csv      every file with its size and SHA-256
  code/             72 analysis scripts, a 15-module library and the pipeline map (code/README.md)
  data/             the derived data (11 files); no raw dataset is redistributed (data/README.md)
  plan/             the pre-specified analysis plan, its SHA-256 at locking, and its amendments
  results/tables/   160 result tables
  results/logs/     the run logs of the analyses added for the resubmission (J57-J69), with their self-checks
  results/figures/  every figure of the article and its supplementary material
  environment/      package versions the results were produced with
```

## Where each table and figure comes from

Each script writes the result tables named after it (`j60_*` by `J60_*.py`, and so on). The writer of each table is
shown in brackets. `J66_resub_export_numbers.py` writes every number of the article as a LaTeX macro whose comment
names the table it comes from, and writes the rows of every data table; `J67_resub_figures.py` draws the six figures
of the article.

### Article

| Item | What | Script | Result tables |
|---|---|---|---|
| Fig. 1 | Framework of the study (schematic) | `J67_resub_figures.py` | none (schematic) |
| Table 2 | Datasets and trajectories | `J66_resub_export_numbers.py` (numbers) | the tables named in the comments of the exported macro file |
| Table 3 | Rules, baseline protocols and onset scaling | `J66_resub_export_numbers.py` | `j61_design_far` (J61), `j61_calibration` (J61), `j60_protocol_curve` (J60) |
| Fig. 2 | Detector-minus-chart gap per protocol | `J67_resub_figures.py` | `j60_gap` (J60), `j61_matched_gap` (J61) |
| Fig. 3 | Onset fraction against trajectory length | `J67_resub_figures.py` | `j60_onsets` (J60) |
| Fig. 4 | Simulated gradual degradation | `J67_resub_figures.py` | `j63_gradual` (J63) |
| Fig. 5 | Label disagreement and target distance | `J67_resub_figures.py` | `j60_labels` (J60), `j61_matched_gap` (J61) |
| Fig. 6 | RUL evaluation on the common reference | `J67_resub_figures.py` | `j60_onsets` (J60), `j62_common_reference` (J62) |
| Table 4 | ORD-4 reporting items | `J66_resub_export_numbers.py` (numbers) | the tables named in the comments of the exported macro file |
| Text | Every other number of the paper | `J66_resub_export_numbers.py` | the tables named in the comments of the exported macro file |

### Supplementary material

| Item | Section | Script | Result tables |
|---|---|---|---|
| Table S1 | S1 Data, Rules and Models | `J66_resub_export_numbers.py` (rows) | `j59_rule_configurations` (J59) |
| Table S2 | S1 Data, Rules and Models | `J66_resub_export_numbers.py` (rows) | `j59_model_settings` (J59) |
| Table S3 | S2 Baseline Protocols in Detail | `J66_resub_export_numbers.py` (rows) | `j60_protocol_curve` (J60) |
| Table S4 | S2 Baseline Protocols in Detail | `J66_resub_export_numbers.py` (rows) | `j60_gap` (J60) |
| Table S5 | S2 Baseline Protocols in Detail | `J66_resub_export_numbers.py` (rows) | `j60_contamination` (J60) |
| Table S6 | S2 Baseline Protocols in Detail | `J66_resub_export_numbers.py` (rows) | `j60_warmup_test` (J60) |
| Fig. S1 | S2 Baseline Protocols in Detail | `J60_baseline_protocols.py` | see the script |
| Fig. S2 | S2 Baseline Protocols in Detail | `J14_journal_figures.py` | see the script |
| Fig. S3 | S2 Baseline Protocols in Detail | `J14_journal_figures.py` | see the script |
| Table S7 | S3 False-Alarm Calibration in Detail | `J66_resub_export_numbers.py` (rows) | `j61_calibration` (J61) |
| Table S8 | S3 False-Alarm Calibration in Detail | `J66_resub_export_numbers.py` (rows) | `j61_arl0` (J61) |
| Table S9 | S3 False-Alarm Calibration in Detail | `J66_resub_export_numbers.py` (rows) | `j61_matched_gap` (J61) |
| Table S10 | S4 Onset Scaling per Operating Condition | `J66_resub_export_numbers.py` (rows) | `j58_onset_fraction` (J58) |
| Table S11 | S4 Onset Scaling per Operating Condition | `J66_resub_export_numbers.py` (rows) | `j58_by_condition` (J58) |
| Table S12 | S4 Onset Scaling per Operating Condition | `J66_resub_export_numbers.py` (rows) | `j26b_slopes` (J26b) |
| Table S13 | S5 Simulated Degradation in Detail | `J66_resub_export_numbers.py` (rows) | `j63_gradual` (J63) |
| Table S14 | S5 Simulated Degradation in Detail | `J66_resub_export_numbers.py` (rows) | `j63_gradual` (J63) |
| Table S15 | S5 Simulated Degradation in Detail | `J66_resub_export_numbers.py` (rows) | `j63_gradual` (J63) |
| Table S16 | S5 Simulated Degradation in Detail | `J66_resub_export_numbers.py` (rows) | `j63_gradual` (J63) |
| Table S17 | S5 Simulated Degradation in Detail | `J66_resub_export_numbers.py` (rows) | `j24_sim_single_change` (J24) |
| Table S18 | S6 Design Parameters | `J66_resub_export_numbers.py` (rows) | `j64_models` (J64) |
| Table S19 | S6 Design Parameters | `J66_resub_export_numbers.py` (rows) | `j64_design_outcomes` (J64) |
| Fig. S4 | S6 Design Parameters | `J21_fig_shift.py` | see the script |
| Fig. S5 | S6 Design Parameters | `J25_reviewer_figures.py` | see the script |
| Table S20 | S7 Labels in Detail | `J66_resub_export_numbers.py` (rows) | `j57_censoring_2x2` (J57) |
| Fig. S6 | S7 Labels in Detail | `J25_reviewer_figures.py` | see the script |
| Fig. S7 | S7 Labels in Detail | `J25_reviewer_figures.py` | see the script |
| Table S21 | S7 Labels in Detail | earlier version (`supp_tables.py` of the first submission) | `j20_scored_check` (J17) |
| Table S22 | S8 RUL Evaluation in Full | `J66_resub_export_numbers.py` (rows) | `j62_common_reference` (J62) |
| Table S23 | S8 RUL Evaluation in Full | `J66_resub_export_numbers.py` (rows) | `j62_common_reference` (J62) |
| Table S24 | S8 RUL Evaluation in Full | `J66_resub_export_numbers.py` (rows) | `j62_common_reference` (J62) |
| Table S25 | S8 RUL Evaluation in Full | `J66_resub_export_numbers.py` (rows) | `j62_common_reference` (J62) |
| Table S26 | S8 RUL Evaluation in Full | `J66_resub_export_numbers.py` (rows) | `j62_rankings` (J62) |
| Table S27 | S8 RUL Evaluation in Full | `J66_resub_export_numbers.py` (rows) | `j62_rankings` (J62) |
| Table S28 | S8 RUL Evaluation in Full | `J66_resub_export_numbers.py` (rows) | `j62_common_reference` (J62) |
| Table S29 | S8 RUL Evaluation in Full | `J66_resub_export_numbers.py` (rows) | `j62_main_effect` (J62) |
| Table S30 | S8 RUL Evaluation in Full | `J66_resub_export_numbers.py` (rows) | `j62_common_reference` (J62) |
| Table S31 | S8 RUL Evaluation in Full | `J66_resub_export_numbers.py` (rows) | `j69_labeler_model_contrast` (J69) |
| Table S32 | S9 Mixed-Effects Models | `J66_resub_export_numbers.py` (rows) | `j65_mixed_onset` (J65) |
| Table S33 | S9 Mixed-Effects Models | `J66_resub_export_numbers.py` (rows) | `j65_mixed_rul` (J65) |
| Table S34 | S10 Pre-Specified Analyses and Amendments | text | the plan and its amendments |
| Table S35 | S10 Pre-Specified Analyses and Amendments | text | the plan and its amendments |

Figs. S2 and S5 are the figures of `J14_journal_figures.py` and `J25_reviewer_figures.py`, restyled for an earlier
revision of the manuscript; the restyling script is not released. Table S21 was written by the table script of the
first submission from `j20_scored_check`.

## The pre-specified analysis plan

The protocol, calibration, simulation, design-parameter and RUL analyses added for the resubmission were specified in
`plan/ANALYSIS_PLAN_2026-10-24.md` before they were run. The plan was locked on 24 October 2026, 11:11:16 (UTC+7); its SHA-256 at locking is

```
898407948fca88d93de586c4411d4173b655067a5f4f7c3b629876c91fcd9576
```

Check it with `sha256sum plan/ANALYSIS_PLAN_2026-10-24.md` (Linux, macOS) or `Get-FileHash plan/ANALYSIS_PLAN_2026-10-24.md` (Windows PowerShell).
Deviations were recorded in the append-only file `plan/ANALYSIS_PLAN_AMENDMENTS.md` (amendments A1-A7), each before
any result it affects was reported. A6 adds one analysis after the planned ones and is labeled post hoc; A7 records
that the files are released here rather than on Zenodo, as the plan had stated.

## Reproducing

The scripts resolve the project root as the parent of their own directory. One command lays them out that way:

```
python code/make_workspace.py
python scripts/J57_censoring_decomposition.py
python scripts/J58_onset_fraction.py
python scripts/J59_configurations.py
python scripts/J60_baseline_protocols.py
python scripts/J61_far_calibration.py
python scripts/J62_common_reference_rul.py
python scripts/J63_gradual_simulation.py
python scripts/J64_design_parameters.py
python scripts/J65_mixed_effects.py
python scripts/J69_labeler_model_contrast.py
python scripts/J66_resub_export_numbers.py
python scripts/J67_resub_figures.py
```

These steps read only the derived data and result tables released here; J62 needs `torch` for its GRU and TCN models
and runs for several hours on a laptop. The full pipeline, including the earlier steps, is in `code/README.md`. The
steps that read the raw acquisitions need the datasets rebuilt first (`data/README.md`); set the environment variable
`BEARING_DATASETS` to the folder that holds them.

Every script that wrote a result table of the resubmission checks itself before writing (for example, J62 reproduces
the submitted onsets and the earlier RUL models, and J69 reproduces the labeler main effects of J62 within 1e-9); the
checks and their outcomes are in `results/logs/`.

## Licences

Code under `code/` is MIT (`LICENSE`). The derived data, the plan, the result tables, logs and figures are CC BY 4.0
(`LICENSE-DATA`). The four run-to-failure datasets are third-party, are **not** redistributed here, and keep the terms
of their own providers; `data/README.md` names each source.

## How to cite

Cite the article and this repository:

> Anh Nguyen Vu Minh, Cao-Phuc Ha, Nguyen Le Thai, Huu-Tho Nguyen and Phu-Nguyen Le, "bearing-onset-rule-audit: code and derived data, release 1.1.1," https://github.com/anhnvm-ai/bearing-onset-rule-audit, 2026.

## Versions

| Version | Date | Content |
|---|---|---|
| 1.0.0 | 2026-10-02 | Code and derived data of the first submission |
| 1.1.0 | 2026-10-26 | Adds the analyses of the resubmission (J57-J69), their logs, the pre-specified plan and the figures of the resubmitted paper |
| 1.1.1 | 2026-10-26 | Cited by the resubmission (amendment A8): adds the supplementary tables of the planned secondary outputs (C-rank reversals, rank agreement, Spearman correlations, simulation delays, median run lengths), Fig. 6 in two rows, and the map of the final supplementary numbering; no result table changed |
