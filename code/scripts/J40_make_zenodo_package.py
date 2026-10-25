"""J40 -- build the Zenodo deposit: code and derived data, no raw dataset redistributed.

The deposit is the artifact the manuscript's Data and Code Availability statement points to. Two
licences apply inside it: MIT for the code, CC BY 4.0 for the derived data and figures. The four
run-to-failure datasets are third-party and are NOT redistributed; data/README.md names each source
and the directory layout the code expects if a reader reconstructs them.

Layout written under <out>/FPT-Rule-Audit:

    README.md  LICENSE-CODE  LICENSE-DATA  MANIFEST.csv
    code/README.md, code/make_workspace.py, code/scripts/*.py, code/src/*.py
    data/README.md, data/derived/{onset_labels,rule_designs,bearing_metadata,rul_targets}.csv
                    data/derived/record_index_{xjtu,ims,nsk}.csv
    results/tables/*.csv, results/figures/fig*.pdf|png
    supplementary/supplementary.pdf, supplementary/supplementary_data.csv
    environment/requirements.txt, environment/python_version.txt

Usage
    python scripts/J40_make_zenodo_package.py                  build + zip
    python scripts/J40_make_zenodo_package.py --no-zip
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
RELEASE = ROOT / "results" / "release"
BUNDLE = ROOT / "colab_bundle"
IEEE = ROOT / "paper" / "ieee"
VERSION = "1.0.0"

AUTHORS = "Anh Nguyen Vu Minh and Phu-Nguyen Le"
TITLE = "FPT Rule Audit: Code and Derived Data for Run-to-Failure Bearing Analysis"

# The ten rules of the paper against the sweep designs that share their definition (Table II).
CANONICAL = {"thr_3sd": "3sig_rms", "ma5_2sd": "ma2sig", "ma5_3sd": "ma3sig", "ma5_5sd": "ma5sig",
             "ewma_l0.2_L3": "ewma", "cusum_k0.5_h5": "cusum"}

# Per-dataset acquisition facts, as Table I of the manuscript states them.
DATASETS = {
    "xjtu": dict(name="XJTU-SY", bearings_used="15 of 15", record="1.28 s, 2 axes, 25.6 kHz, every minute",
                 stop_criterion="amplitude above ten times its normal-stage maximum, or above 20 g",
                 source="Wang et al. (2020); Lei et al. (2019)"),
    "femto": dict(name="FEMTO", bearings_used="13 of 17", record="0.1 s, 2 axes, 25.6 kHz, every minute",
                  stop_criterion="vibration amplitude exceeds 20 g",
                  source="Nectoux et al. (2012); NASA PCoE / IEEE PHM 2012 challenge"),
    "ims": dict(name="IMS", bearings_used="4 failed of 12", record="1.024 s, 1 axis, 20 kHz, about every 30 minutes",
                stop_criterion="debris on a magnetic plug exceeds a set level",
                source="Lee et al. (2007); Qiu et al. (2006)"),
    "nsk": dict(name="NSK 6205", bearings_used="1", record="about 78 s, 2 axes, 25.6 kHz, every hour",
                stop_criterion="bearing temperature exceeds 85 C and vibration exceeds 9 m/s^2",
                source="Jung et al. (2024)"),
}
CONDITIONS = {  # operating condition of each trajectory, from the dataset descriptions
    "xjtu": {"1": "35 Hz, 12 kN", "2": "37.5 Hz, 11 kN", "3": "40 Hz, 10 kN"},
    "femto": {"1": "1800 rpm, 4.0 kN", "2": "1650 rpm, 4.2 kN", "3": "1500 rpm, 5.0 kN"},
}
ONE_CONDITION = {"ims": "2000 rpm, 6000 lb", "nsk": "1770-1780 rpm"}  # datasets run at a single condition

SUPP_ITEMS = [  # supplementary item -> section, source table(s), script
    ("Fig. S1", "S3", "The duration signature: fire index against trajectory length",
     "j3_fpt_times.csv", "J14_journal_figures.py"),
    ("Fig. S2", "S4", "Why one fire index scales with test duration and the other does not",
     "j22_arl_mechanism_summary.csv", "J25_reviewer_figures.py"),
    ("Table S1", "S5", "What each family recovers when the change time is known",
     "j24_sim_single_change.csv", "J24_known_change_simulation.py"),
    ("Fig. S3", "S6", "Three measurements recomputed over twelve labeling conventions",
     "j15_rule_sensitivity.csv", "J14_journal_figures.py"),
    ("Table S2", "S8", "Firing coverage and induced positive rate of the ten rules",
     "j19_grid_coverage.csv", "J19_export_numbers.py"),
    ("Table S3", "S8", "Pooled positive rate induced by each rule on subsets of bearings",
     "j17_prevalence_by_subset.csv", "J17_release_and_screen.py"),
    ("Table S4", "S8", "Sensitivity of the prevalence screen to its bounds",
     "j23_usability_grid.csv; j23_usability_flippoints.csv", "J23_reviewer_analyses.py"),
    ("Fig. S4", "S9", "Macro-F1 of every method under every rule that passes the prevalence screen",
     "j4_ranking_reversal.csv", "J25_reviewer_figures.py"),
    ("Table S5", "S10", "The response-size curve under every fit",
     "j36_design_cv.csv; j36_design_bootstrap.csv", "J36_design_cv.py"),
    ("Table S6", "S11", "Labeler spread and model spread of the prediction RMSE, per bearing",
     "j37_per_bearing.csv", "J37_downstream_per_bearing.py"),
    ("Table S7", "S11", "Differences in prediction RMSE between two labelers on the same bearing",
     "j37_pair_summary.csv", "J37_downstream_per_bearing.py"),
    ("Table S8", "S12", "The consistency check on the non-firing defect",
     "j20_scored_check.csv", "J17_release_and_screen.py"),
    ("Table S9", "S13", "What the response-size relation identifies: the a-priori shift of each design",
     "j41_estimand_r2.csv; j41_estimand_bootstrap.csv", "J41_estimand_separation.py"),
    ("Table S10", "S14", "False alarms when nothing happens, by noise and autocorrelation",
     "j44_robustness.csv", "J44_synthetic_robustness.py"),
    ("Table S11", "S15", "The 67-design sweep repeated on every indicator, including spectral and envelope ones",
     "j46_indicator_r2.csv", "J45_indicator_extraction.py, J46_indicator_sweep.py"),
    ("Table S12", "S16", "A GRU and a TCN under each labeller, with the shuffled-order and constant-window controls",
     "j47_sequence_models.csv; j47_seed_spread.csv", "J47_sequence_models.py"),
    ("Fig. S4", "S8", "The three treatments of a trajectory a rule never fires on",
     "j49_censoring_abc.csv", "J49_supplementary_figures.py"),
    ("Fig. S6", "S11", "Leave one bearing out, and what the labeller changes",
     "fig1_fpt_matrix.csv", "J49_supplementary_figures.py"),
]

PIPELINE = [  # step, what it does, script, whether the deposit alone is enough
    ("01", "Per-record vibration statistics (RMS, kurtosis, crest, p2p, ...) for every trajectory",
     "J7_circularity.py, J7c_femto_channel_test.py", "raw"),
    ("02", "Onsets of the ten rules on the 33 trajectories, and the induced label sets",
     "J3_fpt_rules.py", "raw"),
    ("03", "Duration signature: fire index against trajectory length, and its mechanism",
     "J11_arl_and_covariates.py, J20_arl_mechanism.py, J29_scaling_law.py", "derived"),
    ("04", "Size of change at each alarm, and the 67-design response-size sweep",
     "J21_fig_shift.py, J26_design_sweep.py, J36_design_cv.py", "derived"),
    ("05", "Label-set identity: pairwise agreement, prevalence screen, censoring",
     "J13_convention_sensitivity.py, J15_adoption_prevalence.py, J28_censoring.py", "derived"),
    ("06", "RUL targets of Eq. (2) and their disagreement",
     "J32_rul_target_disagreement.py", "derived"),
    ("07", "Downstream experiment: four labelers, six regressors, leave-one-bearing-out",
     "J35_downstream_rul.py, J37_downstream_per_bearing.py, J38_downstream_ratio_bootstrap.py, J39_two_stage_rul.py",
     "derived"),
    ("08", "Release file and the ORD-4 reporting screen",
     "J17_release_and_screen.py", "derived"),
    ("09", "What the response-size relation identifies: the shift each design responds to by construction, "
           "against the shift it is measured to alarm at",
     "J41_estimand_separation.py", "derived"),
    ("10", "Dispersion of the downstream error corrected for pool size, and the response-size curve on "
           "held-out bearings",
     "J42_dispersion_statistics.py, J43_heldout_bearings.py", "derived"),
    ("11", "The known-change simulation under heavier tails, more memory, and with no change at all",
     "J44_synthetic_robustness.py", "derived"),
    ("12", "Spectral and envelope indicators of the same records", "J45_indicator_extraction.py", "raw"),
    ("13", "The 67-design sweep repeated on every indicator", "J46_indicator_sweep.py", "derived"),
    ("14", "Two sequence models in the downstream pool, with a shuffled-order and a constant-window "
           "control (needs torch)",
     "J47_sequence_models.py", "derived"),
    ("15", "The pool-size-corrected dispersion of step 10, on the pool step 14 extends",
     "J48_pool_extended_dispersion.py", "derived"),
    ("16", "The three treatments of a trajectory a rule never fires on, and the leave-one-bearing-out "
           "schematic",
     "J49_supplementary_figures.py", "derived"),
    ("17", "Figures and the exported LaTeX numbers (the exporters update macro files of the manuscript, "
           "which is not part of this deposit)",
     "J14_journal_figures.py, J25_reviewer_figures.py, J19_export_numbers.py, "
     "J39b/J41b/J43b/J44b/J46b/J47b/J48b/J49b_export_numbers.py", "derived"),
]

MIT = """MIT License

Copyright (c) 2026 {authors}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""".format(authors=AUTHORS)

DATA_LICENCE_HEADER = """Creative Commons Attribution 4.0 International (CC BY 4.0)

Copyright (c) 2026 {authors}

This licence covers the files this deposit derives from the four run-to-failure
datasets: everything under data/derived/, results/tables/, results/figures/ and
supplementary/. It does NOT cover the datasets themselves, which are not part of
this deposit and stay under the terms of their own providers (see data/README.md).

Attribution: cite the article and this deposit (see README.md, "How to cite").

The full legal code follows.

=======================================================================

""".format(authors=AUTHORS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ derived data
def onset_labels(out: Path) -> pd.DataFrame:
    """results/release/onset_labels.csv under the column names of the deposit."""
    d = pd.read_csv(RELEASE / "onset_labels.csv")
    o = pd.DataFrame({
        "dataset": d.dataset,
        "bearing_id": d.trajectory,
        "trajectory_id": d.dataset + "/" + d.trajectory,
        "n_records": d.n_records,
        "rule": d.rule,
        "rule_family": d.family,
        "indicator": d.indicator,
        "fired": d.fired,
        "onset_index": d.onset_index,
        "onset_fraction": d.onset_fraction,
        "censored": d.censored,
        "baseline_length": d.baseline_records,
        "onset_inside_baseline": d.onset_inside_baseline,
        "baseline_fraction": d.baseline_fraction,
        "confirmation_count": d.confirm_records,
        "response_size": d.response_size_sd,
        "positive_records": d.positive_records,
        "positive_rate": d.positive_rate,
    })
    o.to_csv(out / "onset_labels.csv", index=False)
    return o


def rule_designs(out: Path) -> pd.DataFrame:
    """The 67 designs of the response-size sweep, per dataset, with their measured shift and rho."""
    d = pd.read_csv(TABLES / "j26_design_sweep.csv")
    o = pd.DataFrame({
        "design_id": d.design,
        "design_kind": d.kind,
        "rule_family": d.family,
        "canonical": d.canonical,
        "rule": d.design.map(CANONICAL),
        "dataset": d.dataset,
        "n_trajectories": d.n_traj,
        "n_fired": d.n_fired,
        "median_response_size": d.median_shift_sd,
        "rho_index_length": d.rho_index_length,
        "rho_fraction_length": d.rho_fraction_length,
        "median_onset_fraction": d.median_onset_fraction,
        "fires_inside_baseline": d.inside_baseline,
    }).sort_values(["design_id", "dataset"])
    o.to_csv(out / "rule_designs.csv", index=False)
    return o


def bearing_metadata(out: Path, lab: pd.DataFrame) -> pd.DataFrame:
    """One row per trajectory: length, operating condition and the acquisition facts of its dataset."""
    base = lab[["dataset", "bearing_id", "trajectory_id", "n_records"]].drop_duplicates()
    cond = {}
    for ds in ("xjtu", "ims", "nsk"):
        r = pd.read_csv(BUNDLE / ds / "records.csv")
        for t, c in r.groupby("trajectory_id").condition_id.first().items():
            cond[(ds, t)] = c
    f = pd.read_csv(TABLES / "j7c_femto_features.csv")
    for t, c in f.groupby("bearing").condition.first().items():
        cond[("femto", t)] = str(c)
    rows = []
    for r in base.itertuples(index=False):
        c = cond.get((r.dataset, r.bearing_id), "")
        key = r.bearing_id[7] if r.dataset in CONDITIONS and r.bearing_id.startswith("Bearing") else str(c)
        rows.append(dict(
            dataset=r.dataset, dataset_name=DATASETS[r.dataset]["name"], bearing_id=r.bearing_id,
            trajectory_id=r.trajectory_id, n_records=r.n_records, condition_id=c,
            operating_condition=CONDITIONS.get(r.dataset, {}).get(key, ONE_CONDITION.get(r.dataset, "")),
            record=DATASETS[r.dataset]["record"], test_stopped_when=DATASETS[r.dataset]["stop_criterion"],
            dataset_source=DATASETS[r.dataset]["source"],
        ))
    o = pd.DataFrame(rows).sort_values(["dataset", "bearing_id"])
    o.to_csv(out / "bearing_metadata.csv", index=False)
    return o


def rul_targets(out: Path, lab: pd.DataFrame) -> pd.DataFrame:
    """Per-record target of Eq. (2): 1 before the onset, falling linearly to 0 at the last record.

    Written only for the (trajectory, rule) pairs on which the rule fires; a censored pair defines no
    target, which is the point Section V makes about non-firing trajectories.
    """
    rows = []
    for r in lab[lab.fired].itertuples(index=False):
        n, i = int(r.n_records), int(r.onset_index)
        denom = max(n - 1 - i, 1)  # J32's convention: a rule firing at the last record gives y = 0 there
        for t in range(n):
            y = 1.0 if t < i else (n - 1 - t) / denom
            rows.append((r.dataset, r.bearing_id, r.rule, t, round(y, 6)))
    o = pd.DataFrame(rows, columns=["dataset", "bearing_id", "rule", "record_index", "y_target"])
    o.to_csv(out / "rul_targets.csv", index=False)
    return o


def record_index(out: Path) -> list[str]:
    """Row order of the per-record feature tables: trajectory, condition, record index, life fraction."""
    written = []
    for ds in ("xjtu", "ims", "nsk"):
        src = BUNDLE / ds / "records.csv"
        d = pd.read_csv(src)
        d.to_csv(out / f"record_index_{ds}.csv", index=False)
        written.append(f"record_index_{ds}.csv ({len(d)} records)")
    return written


# ------------------------------------------------------------------ text files
def readme(pkg: Path, counts: dict) -> str:
    steps = "\n".join(
        f"| {s} | {what} | `{scr}` | {'raw datasets needed' if need == 'raw' else 'runs from this deposit'} |"
        for s, what, scr, need in PIPELINE)
    return f"""# {TITLE}

Version {VERSION}. Code and derived data for the article *What First-Prediction-Time Rules Respond
To: Size of Change, Test Duration and Label-Set Identity in Run-to-Failure Bearing Data* by
{AUTHORS}.

Run-to-failure bearing datasets record no degradation onset, so an onset rule creates one. This
deposit holds the onsets that ten published rules place on {counts['trajectories']} trajectories from
four public datasets, everything derived from them, and the code that produced every number, table
and figure in the article and its supplementary material.

## What is here

```
FPT-Rule-Audit/
  README.md            this file
  LICENSE-CODE         MIT, for everything under code/
  LICENSE-DATA         CC BY 4.0, for everything under data/, results/ and supplementary/
  MANIFEST.csv         every file with its size and SHA-256
  code/                {counts['scripts']} analysis scripts, a {counts['library']}-module library, and the pipeline map
  data/derived/        the derived data ({counts['derived']} files); no raw dataset is redistributed
  results/tables/      {counts['tables']} result tables, one per analysis
  results/figures/     the {counts['figures']} figures of the article, PDF and PNG
  supplementary/       supplementary.pdf and an index of its items
  environment/         package versions the results were produced with
```

## The four derived data files

| File | One row per | Holds |
|---|---|---|
| `data/derived/onset_labels.csv` | trajectory x rule ({counts['onsets']:,} rows) | where each rule places the onset, whether it fired, the size of change it responded to, the induced positive rate |
| `data/derived/rule_designs.csv` | design x dataset ({counts['designs']:,} rows) | the {counts['n_designs']} designs of the response-size sweep, their measured response size and duration signature |
| `data/derived/bearing_metadata.csv` | trajectory ({counts['trajectories']:,} rows) | trajectory length, operating condition, acquisition and stopping criterion of its dataset |
| `data/derived/rul_targets.csv` | record x rule ({counts['targets']:,} rows) | the remaining-useful-life target of Eq. (2), for the pairs on which the rule fires |

`data/derived/record_index_*.csv` gives the row order of the per-record feature tables in
`results/tables/`, so those tables can be joined back to trajectories and record indices.

## Reproducing

The scripts resolve the project root as the parent of their own directory. One command lays them out
that way inside this deposit:

```
cd FPT-Rule-Audit
python code/make_workspace.py          # copies code/scripts and code/src to the deposit root
python scripts/J17_release_and_screen.py     # example: the ORD-4 reporting screen
```

Every step that reads only derived data runs from this deposit as it stands. The two steps that read
the raw acquisitions need the datasets rebuilt first (`data/README.md`).

| Step | What it does | Script | Needs |
|---|---|---|---|
{steps}

## Licences

Code under `code/` is MIT (`LICENSE-CODE`). The derived data, tables, figures and supplementary
material are CC BY 4.0 (`LICENSE-DATA`). The four run-to-failure datasets are third-party, are **not**
redistributed here, and keep the terms of their own providers; `data/README.md` names each source.

## How to cite

Cite the article, and this deposit by its DOI:

> {AUTHORS}, "What first-prediction-time rules respond to: Size of change, test duration and
> label-set identity in run-to-failure bearing data," submitted for publication, 2026.

> {AUTHORS}, "{TITLE}," Zenodo, 2026, version {VERSION}, doi: [DOI of this record].

## Versions

| Version | When |
|---|---|
| 1.0.0 | at submission |
| 1.1.0 | after review, if the analysis changes |
| 2.0.0 | at acceptance, with the published article's numbers |
"""


def data_readme(counts: dict) -> str:
    rows = "\n".join(
        f"| {v['name']} | {v['bearings_used']} | {v['record']} | {v['stop_criterion']} | {v['source']} |"
        for v in DATASETS.values())
    return f"""# Data

## The raw datasets are not redistributed

The original datasets are not redistributed in this deposit. They are public, each carries its own
terms, and each should be obtained from its provider:

| Dataset | Bearings used | Record | Test stopped when | Source |
|---|---|---|---|---|
{rows}

- **XJTU-SY**: https://biaowang.tech/xjtu-sy-bearing-datasets/ (the tutorial of Lei et al., 2019,
  doi: 10.3901/JME.2019.16.001, describes the tests).
- **FEMTO (PRONOSTIA, IEEE PHM 2012 challenge)**: NASA Prognostics Data Repository,
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
  The challenge release is also mirrored at https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset
- **IMS (NSF I/UCRC Center for Intelligent Maintenance Systems)**: NASA Prognostics Data Repository, as above.
- **NSK 6205**: Mendeley Data, doi: 10.17632/5hcdd3tdvb.6 (version 6), released with Jung et al. (2024).

Each link was reachable on 23 September 2026.

## What this deposit does hold

`data/derived/` holds only quantities computed from those datasets: onset indices, per-record
statistics, targets and metadata. It holds no raw acquisition.

- `onset_labels.csv` -- one row per trajectory and rule ({counts['onsets']:,} rows). `onset_index` is the
  first record labelled degraded, empty when the rule never fired (`fired` is false, `censored` true).
  Records `onset_index` to `n_records - 1` are degraded, earlier records healthy. `baseline_length` is
  max(5, floor(0.2 n)); `confirmation_count` is the number of consecutive records required; both are the
  conventions of Section III, and the percentile rules use neither. `response_size` is the indicator
  level over the confirming records minus the baseline mean, in baseline standard deviations (ORD-4
  item 1): it is measured on RMS, so it is empty for `3sig_kurt` and for the percentile rules.
  `positive_rate` is the share of records labelled degraded, empty when censored.
- `rule_designs.csv` -- the designs of the response-size sweep, per dataset, with the response size each
  one answers to and the correlation between its fire index and trajectory length. `canonical` marks the
  designs that are also one of the paper's ten rules; `rule` names that rule.
- `bearing_metadata.csv` -- one row per trajectory: its length, its operating condition, and the
  acquisition and stopping criterion of its dataset.
- `rul_targets.csv` -- the target of Eq. (2) for every record of every (trajectory, rule) pair on which
  the rule fires: 1 before the onset, falling linearly to 0 at the last record. A censored pair defines
  no target and is absent, which is the point Section V makes about non-firing trajectories.
- `record_index_{{xjtu,ims,nsk}}.csv` -- the row order of `results/tables/j7_features_*.csv`, one row per
  record, so the per-record statistics can be joined back to trajectories. FEMTO carries its own
  identifiers inside `results/tables/j7c_femto_features.csv`.

A trajectory on which a rule never fires has no labels under that rule. Do not label it healthy: the
article shows that doing so changes prevalence-screen verdicts and winning methods.

## If you rebuild the raw datasets

The code reads preprocessed acquisitions from a `colab_bundle/` directory next to the scripts:

```
colab_bundle/<dataset>/records.csv     one row per record: trajectory_id, condition_id, record_index, life_fraction
colab_bundle/<dataset>/windows.npy     the windowed vibration samples of those records
```

`python code/make_workspace.py` writes the `records.csv` files from `data/derived/record_index_*.csv`.
The `windows.npy` arrays are the raw acquisitions in windowed form and are not part of this deposit;
they are rebuilt from the downloaded datasets with step 01 of the pipeline in `code/README.md`.
"""


def code_readme(counts: dict) -> str:
    steps = "\n".join(
        f"| {s} | {what} | `{scr}` | {'raw datasets' if need == 'raw' else 'this deposit'} |"
        for s, what, scr, need in PIPELINE)
    return f"""# Code

{counts['scripts']} scripts and the {counts['library']}-module library they import. File names are the ones the article's
logs and the supplementary material refer to, so they are kept as they are rather than renumbered.

```
code/
  make_workspace.py   lays the scripts out so their relative paths resolve
  scripts/            the analyses, J3 .. J40
  src/                the library the scripts import (features, metrics, thresholds, splits)
```

## Running

```
cd FPT-Rule-Audit
python code/make_workspace.py
python scripts/<script>.py
```

`make_workspace.py` copies `code/scripts` and `code/src` to the deposit root, next to `results/` and
`data/`, because every script resolves the project root as the parent of its own directory. It also
writes `colab_bundle/<dataset>/records.csv` from `data/derived/record_index_*.csv`.

## The pipeline

| Step | What it does | Script | Needs |
|---|---|---|---|
{steps}

Steps 01, 02 and 12 read the raw acquisitions; every other step runs from the tables in
`results/tables/`, which this deposit contains. So the downstream results of the article -- the
duration signature, the response-size sweep, the label-set comparisons, the RUL targets and the
downstream experiment -- can be recomputed from this deposit alone.

Checked in a clean extraction of this deposit, steps 03 to 08 on 23 September 2026 and steps 09
to 11, 13, 15 and 16 on 24-25 September 2026, with the same code in both sittings: every one of them ran to
completion and its tables matched the deposited ones exactly, the fitted models of step 07 to within
1e-15. Step 14 trains 552 models and takes about an hour and a half; in the extraction it was run on a
reduced grid, which shows the deposit carries what it needs, and its full grid was rerun in place, where
it reproduced `j47_sequence_models.csv` cell for cell. The one exception is the export step, whose
scripts edit LaTeX macro files of the manuscript in place; the manuscript source is not deposited, so
they have nothing to edit here.

## Conventions the scripts share

Baseline length max(5, floor(0.2 n)); confirmation count 3; a rule "fires inside its own baseline"
when its fire index is below the baseline length; a label set is "degenerate" on a dataset when the
induced positive rate falls outside [0.01, 0.50]. Bootstrap resampling is over whole bearings, seed 0,
20 000 resamples. Cross-validation holds out whole bearings.
"""


MAKE_WORKSPACE = '''"""Lay this deposit out so the analysis scripts resolve their relative paths.

Every script computes the project root as the parent of its own directory and reads results/tables,
data/ and colab_bundle/ from there. This copies code/scripts and code/src next to those directories
and writes the record-index files where the scripts expect them.

    python code/make_workspace.py            # into the deposit root
    python code/make_workspace.py --dest DIR # into DIR
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEPOSIT = HERE.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEPOSIT)
    a = ap.parse_args()
    dest = a.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    for name in ("scripts", "src"):
        shutil.copytree(HERE / name, dest / name, dirs_exist_ok=True)
        print(f"  {name}/ -> {dest / name}")

    if dest != DEPOSIT:
        for name in ("results", "data", "supplementary", "environment"):
            if (DEPOSIT / name).exists():
                shutil.copytree(DEPOSIT / name, dest / name, dirs_exist_ok=True)
                print(f"  {name}/ -> {dest / name}")

    # the two number exporters write LaTeX macro files into the manuscript tree, which is not
    # part of the deposit; the folders are created so they have somewhere to write
    for name in ("paper/ieee", "paper/journal"):
        (dest / name).mkdir(parents=True, exist_ok=True)

    derived = DEPOSIT / "data" / "derived"
    for f in sorted(derived.glob("record_index_*.csv")):
        ds = f.stem.replace("record_index_", "")
        out = dest / "colab_bundle" / ds
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / "records.csv")
        print(f"  {f.name} -> {out / 'records.csv'}")

    print(f"\\nworkspace ready in {dest}")
    print("the raw windows (colab_bundle/<dataset>/windows.npy) are not part of this deposit;")
    print("steps 01 and 02 of code/README.md rebuild them from the downloaded datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


# ------------------------------------------------------------------ build
def copy_code(pkg: Path) -> tuple[int, int]:
    code = pkg / "code"
    (code / "scripts").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((ROOT / "scripts").glob("J*.py")):
        shutil.copy2(f, code / "scripts" / f.name)
        n += 1
    src = code / "src"
    src.mkdir(parents=True, exist_ok=True)
    n_lib = 0
    for f in sorted((ROOT / "src").glob("*.py")):
        shutil.copy2(f, src / f.name)
        n_lib += 1
    (code / "make_workspace.py").write_text(MAKE_WORKSPACE, encoding="utf-8", newline="\n")
    return n, n_lib


def copy_results(pkg: Path) -> tuple[int, int]:
    t = pkg / "results" / "tables"
    t.mkdir(parents=True, exist_ok=True)
    nt = 0
    # j* and fig* are this study's tables; e2_raw_counts_h5.csv is the one table of the earlier
    # conference study that J19 reads when it exports the manuscript's macros
    for f in sorted(TABLES.glob("*.csv")):
        if f.name.startswith(("j", "fig")) or f.name == "e2_raw_counts_h5.csv":
            shutil.copy2(f, t / f.name)
            nt += 1
    g = pkg / "results" / "figures"
    g.mkdir(parents=True, exist_ok=True)
    used = set()
    for tex in sorted(IEEE.glob("*.tex")):
        s = tex.read_text(encoding="utf-8")
        for token in s.split("includegraphics")[1:]:
            name = token.split("{", 1)[1].split("}", 1)[0]
            used.add(Path(name).name)
    nf = 0
    for name in sorted(used):
        for ext in (".pdf", ".png"):
            f = FIGURES / (name + ext)
            if f.exists():
                shutil.copy2(f, g / f.name)
        nf += 1
    return nt, nf


def copy_supplementary(pkg: Path) -> None:
    s = pkg / "supplementary"
    s.mkdir(parents=True, exist_ok=True)
    shutil.copy2(IEEE / "supplementary.pdf", s / "supplementary.pdf")
    with (s / "supplementary_data.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "section", "title", "source_table", "script"])
        for row in SUPP_ITEMS:
            w.writerow(row)


def environment(pkg: Path) -> None:
    e = pkg / "environment"
    e.mkdir(parents=True, exist_ok=True)
    import matplotlib
    import numpy
    import scipy
    import sklearn
    try:                                     # the version without loading the runtime's DLLs
        import importlib.metadata
        torch_line = (f"torch=={importlib.metadata.version('torch')}"
                      f"   # step 14 only; every other step runs without it\n")
    except Exception:                        # the deposit still builds on a machine without it
        torch_line = "torch                       # step 14 only; every other step runs without it\n"
    e.joinpath("requirements.txt").write_text(
        "# versions the deposited results were produced with\n"
        f"numpy=={numpy.__version__}\npandas=={pd.__version__}\nscipy=={scipy.__version__}\n"
        f"scikit-learn=={sklearn.__version__}\nmatplotlib=={matplotlib.__version__}\n" + torch_line,
        encoding="utf-8", newline="\n")
    e.joinpath("python_version.txt").write_text(
        f"{sys.version.split()[0]}\n", encoding="utf-8", newline="\n")


def manifest(pkg: Path) -> int:
    files = sorted(p for p in pkg.rglob("*") if p.is_file() and p.name != "MANIFEST.csv")
    with (pkg / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes", "sha256"])
        for p in files:
            w.writerow([p.relative_to(pkg).as_posix(), p.stat().st_size, sha256(p)])
    return len(files)


def guard(pkg: Path) -> None:
    """Nothing third-party, nothing raw, nothing huge leaves this machine."""
    bad = []
    for p in pkg.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(pkg).as_posix()
        if "workflow_verification" in rel or "/sources/" in rel:
            bad.append(("third-party sources", rel))
        if p.suffix in (".npy", ".npz", ".pt", ".h5"):
            bad.append(("raw or cached array", rel))
        if p.suffix == ".pdf" and not rel.startswith(("supplementary/", "results/figures/")):
            bad.append(("unexpected pdf", rel))
        if p.stat().st_size > 20 * 1024 * 1024:
            bad.append(("larger than 20 MB", rel))
    if bad:
        for why, rel in bad:
            print(f"  REFUSED {why}: {rel}")
        raise SystemExit("the package would carry files it must not; nothing was zipped")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "zenodo")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    pkg = a.out / "FPT-Rule-Audit"
    if pkg.exists():
        # empty it rather than remove it: on Windows a shell sitting in the folder blocks the rmdir
        for p in sorted(pkg.iterdir()):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    derived = pkg / "data" / "derived"
    derived.mkdir(parents=True)

    print("derived data")
    lab = onset_labels(derived)
    des = rule_designs(derived)
    met = bearing_metadata(derived, lab)
    tgt = rul_targets(derived, lab)
    idx = record_index(derived)
    for name, d in (("onset_labels", lab), ("rule_designs", des), ("bearing_metadata", met), ("rul_targets", tgt)):
        print(f"  {name}.csv  {len(d)} rows x {len(d.columns)} columns")
    for line in idx:
        print(f"  {line}")

    print("code, results, supplementary, environment")
    n_scripts, n_library = copy_code(pkg)
    n_tables, n_figures = copy_results(pkg)
    copy_supplementary(pkg)
    environment(pkg)

    counts = dict(scripts=n_scripts, library=n_library, tables=n_tables, figures=n_figures, derived=4 + len(idx),
                  onsets=len(lab), designs=len(des), n_designs=des.design_id.nunique(),
                  trajectories=len(met), targets=len(tgt))
    pkg.joinpath("README.md").write_text(readme(pkg, counts), encoding="utf-8", newline="\n")
    pkg.joinpath("data", "README.md").write_text(data_readme(counts), encoding="utf-8", newline="\n")
    pkg.joinpath("code", "README.md").write_text(code_readme(counts), encoding="utf-8", newline="\n")
    pkg.joinpath("LICENSE-CODE").write_text(MIT, encoding="utf-8", newline="\n")
    cc = (ROOT / "scripts" / "cc_by_4.0.txt").read_text(encoding="utf-8")
    pkg.joinpath("LICENSE-DATA").write_text(DATA_LICENCE_HEADER + cc, encoding="utf-8", newline="\n")

    guard(pkg)
    n_files = manifest(pkg)

    total = sum(p.stat().st_size for p in pkg.rglob("*") if p.is_file())
    print(f"\n{n_files} files, {total / 1e6:.1f} MB in {pkg}")
    if not a.no_zip:
        z = shutil.make_archive(str(a.out / f"FPT-Rule-Audit_v{VERSION}"), "zip", root_dir=a.out, base_dir="FPT-Rule-Audit")
        print(f"zip: {z}  ({Path(z).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
