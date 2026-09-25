# KUPA-KEYS Revision Analysis: Revision-Process Feature Extraction and Ordinal CEFR Modelling

This repository contains the reproducible analysis script used to
extract 13 revision-process features from KUPA-KEYS Task 2 keystroke
data and evaluate their association with CEFR writing proficiency using
ordinal logistic regression.

The repository is anonymised for review.

## Repository contents

-   `K-REV_KUPA_KEYS_CLEAN.py` ---
    End-to-end analysis script.

The script performs Task-2 text reconstruction and validation, obtains
K-REV revision episodes, extracts the 13 writing-level revision
features, constructs the composite CEFR outcome, applies the study's
cross-task quality-control exclusions, and evaluates the final ordinal
model using 10-fold cross-validation.

## Data

The script uses the publicly available KUPA-KEYS dataset:

`https://huggingface.co/datasets/ALTACambridge/KUPA-KEYS`

Required input files:

-   `Task2.csv`
-   `DLLA.csv`

The data are not available in this repository due to their large size.

## Revision-process features

The final feature set contains 13 writing-level measures:

1.  `n_revision_episodes`
2.  `total_revision_time_ratio`
3.  `revision_regularity_sec_sd`
4.  `ratio_backspace_seq_gt3_in_revisions`
5.  `n_backspace_sequences_with_gt1000ms_intervals`
6.  `mean_time_to_edit_ms_in_revisions`
7.  `median_backspaces_per_revision`
8.  `median_chars_deleted_per_revision`
9.  `median_chars_inserted_per_revision`
10. `median_time_from_start_process_sec`
11. `median_pause_before_revision_sec`
12. `n_backspacing_sequences_in_revisions`
13. `mean_backspaces_per_sequence`

The measures are derived from different underlying units of revision
activity. K-REV is used to identify revision episodes. Some measures
summarise revision episodes, whereas the Backspace-sequence measures
summarise physical consecutive Backspace sequences associated with
revision activity.

### Backspace sequences with \>1,000-ms intervals

`n_backspace_sequences_with_gt1000ms_intervals` is a study-specific
temporal extension of Backspace-sequence analysis. It counts physical
consecutive Backspace sequences occurring within K-REV-defined revision
activity for which at least one Backspace press has an associated
preceding key-down (press-to-press) interval greater than 1,000 ms.

The feature is descriptive rather than proficiency-specific: it can be
related to proficiency in the present analysis, but it could also be
used to examine temporal disruption in revision under different tasks or
experimental conditions.

### Unique physical Backspace sequences

Physical Backspace sequences are uniquely identified within each writing
and counted once, even when the same sequence is associated with more
than one K-REV revision episode. This prevents the same physical
sequence from being counted multiple times because of overlapping or
nested episode associations.

## Model

The final model uses:

-   `mord.LogisticIT(alpha=1.0)`
-   median imputation
-   standardization
-   10-fold `StratifiedKFold`
-   shuffling with random state 42

Imputation and standardisation are performed inside the cross-validation
pipeline.

The primary evaluation is based on out-of-fold cross-validation
predictions rather than a separate holdout result.

Reported evaluation measures include:

-   accuracy
-   balanced accuracy
-   mean absolute error (MAE)
-   adjacent accuracy
-   pooled out-of-fold confusion matrix
-   McKelvey--Zavoina pseudo-R² for the full fitted model

## Requirements

The script requires Python and the following packages:

-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `mord`
-   `krev`

Install the required packages in an appropriate Python environment
before running the analysis.

## Running the analysis

From the repository directory, run:

``` bash
python K-REV_KUPA_KEYS_CLEAN.py \
  --task2 /path/to/Task2.csv \
  --dlla /path/to/DLLA.csv \
  --output-dir KUPAKEYS_Revision13_reproduction
```

If `--output-dir` is omitted, the script uses
`KUPAKEYS_Revision13_reproduction`.

## Main outputs

The output directory includes intermediate validation and
feature-extraction files as well as the final model results. Key outputs
include:

-   `00_Task2_reconstruction_validation.csv`
-   `02_KREV_revision_episodes.csv`
-   `05_Revision13_writing_level_features.csv`
-   `07_model_ready_common_population.csv`
-   `09_Model_10fold_CV_fold_results.csv`
-   `10_Model_10fold_CV_summary.csv`
-   `11_Model_OOF_predictions.csv`
-   full-sample standardised coefficients
-   McKelvey--Zavoina pseudo-R² output

The script does not generate figures.

## Reproducibility notes

The analysis begins with raw Task-2 event data and reconstructs the
writing process before revision-feature extraction. Writings that do not
pass the required Task-2 reconstruction validation are excluded. The
script then applies the cross-task quality-control exclusions used in
the analysis and restricts the final ordinal model to the B1--C2
proficiency range.

Missing predictor values are handled through median imputation within
each cross-validation training fold rather than through casewise
deletion.

## Anonymous-review note

This repository is provided for anonymous peer review and therefore does
not include author-identifying information. Author names, affiliations,
and permanent repository metadata can be added after the review process.
