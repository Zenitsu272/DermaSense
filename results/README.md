# Recorded experiment results

These are small, read-only snapshots of the completed local experiments, collected on 9 October 2026. They are evidence for the results described in the main README; they are not inputs to training. New experiments write to the Git-ignored `outputs/` directory.

| Directory | Contents |
|---|---|
| `run_01/` | Original architecture: metrics, split counts, history and charts |
| `improvement_01/seed_*/` | Three-seed validation search histories and validation metrics |
| `improvement_01/final/` | Validation-selected revised model, before development refit |
| `improvement_01/refit/` | 224px model refitted on 575 development images |
| `high_resolution_01/search/` | 320px progressive fine-tuning validation history |
| `high_resolution_01/refit/` | Selected high-resolution configuration refitted on development data |

The `selection.json` files retain the complete candidate comparisons. The published `verification.json` files record checks on the original local artifacts, including predictions and checkpoint hashes; those large checkpoints and patient-level files are not distributed here. Repository portability fixes do not retrain or change these recorded experiments. To rerun integrity checks, generate the corresponding `outputs/` artifacts first.

The 96-image test partition was reused during model development. Scores are exploratory benchmark results, not independent external validation. Neither refit has a separate validation score because the former validation images were included in its training set.
