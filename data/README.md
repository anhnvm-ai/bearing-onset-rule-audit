# Data

## The raw datasets are not redistributed

The original datasets are public, each carries its own terms, and each should be obtained from its provider:

| Dataset | Bearings used | Record | Test stopped when | Source |
|---|---|---|---|---|
| XJTU-SY | 15 of 15 | 1.28 s, 2 axes, 25.6 kHz, every minute | amplitude above ten times its normal-stage maximum, or above 20 g | Wang et al. (2020); Lei et al. (2019) |
| FEMTO | 13 of 17 | 0.1 s, 2 axes, 25.6 kHz, every minute | vibration amplitude exceeds 20 g | Nectoux et al. (2012); NASA PCoE / IEEE PHM 2012 challenge |
| IMS | 4 failed of 12 | 1.024 s, 1 axis, 20 kHz, about every 30 minutes | debris on a magnetic plug exceeds a set level | Lee et al. (2007); Qiu et al. (2006) |
| NSK 6205 | 1 | about 78 s, 2 axes, 25.6 kHz, every hour | bearing temperature exceeds 85 C and vibration exceeds 9 m/s^2 | Jung et al. (2024) |

- **XJTU-SY**: https://biaowang.tech/xjtu-sy-bearing-datasets/ (the tutorial of Lei et al., 2019,
  doi: 10.3901/JME.2019.16.001, describes the tests).
- **FEMTO (PRONOSTIA, IEEE PHM 2012 challenge)**: NASA Prognostics Data Repository,
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- **IMS (NSF I/UCRC Center for Intelligent Maintenance Systems)**: NASA Prognostics Data Repository, as above.
- **NSK 6205**: Mendeley Data, doi: 10.17632/5hcdd3tdvb.6 (version 6), released with Jung et al. (2024).

## What this repository holds

`data/derived/` holds only quantities computed from those datasets: onset indices, targets and metadata. It holds no
raw acquisition.

- `onset_labels.csv` -- one row per trajectory and rule under the retrospective protocol P-retro (330 rows).
  `onset_index` is the first record labeled degraded, empty when the rule never fired (`fired` false, `censored`
  true). Records `onset_index` to `n_records - 1` are degraded, earlier records healthy.
- `rule_designs.csv` -- the 67 designs of the design sweep, per dataset, with the size of change at their alarms.
- `bearing_metadata.csv` -- one row per trajectory (33 rows): length, operating condition, acquisition
  and stopping criterion of its dataset.
- `rul_targets.csv` -- the RUL target of the article for every record of every (trajectory, rule) pair on which the
  rule fires: 1 before the onset, falling linearly to 0 at the last record (153,878 rows).
- `record_index_{xjtu,ims,nsk}.csv` -- the row order of `results/tables/j7_features_*.csv`, one row per record.
  FEMTO carries its own identifiers inside `results/tables/j7c_femto_features.csv`.
- `onsets_by_protocol.csv` -- a copy of `results/tables/j60_onsets.csv` (2,970 rows).
- `onsets_calibrated.csv` -- a copy of `results/tables/j61_matched_onsets.csv` (1,152 rows).
- `rule_configurations.csv` -- a copy of `results/tables/j59_rule_configurations.csv` (77 rows).
- `model_settings.csv` -- a copy of `results/tables/j59_model_settings.csv` (9 rows).

`onsets_by_protocol.csv` gives every onset under each baseline protocol (`arm`), with the baseline window `W`, the
records removed by iterative cleaning, the onset `i_star`, the onset fraction `u` and whether the onset falls inside the
baseline. `onsets_calibrated.csv` gives the onsets of the rules calibrated to a false-alarm rate (`level`), with the
calibrated `multiplier`.

A trajectory on which a rule never fires has no labels under that rule. Do not label it healthy without saying so:
the article shows that the treatment and weighting of such trajectories change the positive rate.

## If you rebuild the raw datasets

The code reads preprocessed acquisitions from `colab_bundle/<dataset>/` next to the scripts (`records.csv`, one row per
record, and `windows.npy`, the windowed vibration samples). `python code/make_workspace.py` writes the `records.csv`
files; the `windows.npy` arrays are rebuilt from the downloaded datasets with steps 01 and 02 of `code/README.md`.
Set the environment variable `BEARING_DATASETS` to the folder that holds the downloaded datasets.
