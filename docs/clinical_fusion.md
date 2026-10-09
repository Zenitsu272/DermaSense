# Clinical fusion and regularized fine-tuning

This follow-up retains the U-Net expert router and ResNet50 image network, and tests a second use of the same six clinical fields at the classification stage. No additional metadata fields, images or labels are introduced.

## Prespecified search

`clinical_fusion.py search` uses only the original 479 training and 96 validation images. It extracts image features on CUDA from the previously selected 224px and 320px training-only checkpoints. For each image it averages pooled features and, separately, class probabilities over four flipped views.

Three candidate families are compared:

1. Existing image model probabilities.
2. An L2-regularized multinomial logistic classifier on clinical variables, blended with image probabilities. Clinical `C` is chosen from 0.1, 1, 10 and its probability weight from 0.1, 0.2, 0.3, 0.4.
3. An L2-regularized multinomial classifier on L2-normalized pooled ResNet features concatenated with scaled clinical features. Image/clinical regularization `C` is chosen from 0.1, 1, 10, 100 and clinical feature scale from 0, 0.1, 0.3, 1, 3. Scale zero is an image-feature control.

There are 66 candidates across the two resolutions. Accuracy, then macro F1, then lower log loss on the validation partition select the configuration. Clinical encoders and head parameters use training data only. Small convex heads fit on CPU; feature extraction uses CUDA. This is frozen-encoder head training, not new end-to-end image training.

## Mixup image fine-tuning

`robust_finetune.py search` also tests actual end-to-end fine-tuning of the image network. It starts from the 320px training-only checkpoint and uses stronger resized crops (scale 0.65–1.0), rotations up to 30 degrees, and color jitter. Half of training batches use mixup with Beta(0.2, 0.2) coefficients. The same mixing coefficient and permutation apply to both images and encoded clinical vectors, and the loss mixes the two corresponding class targets.

The fixed settings are seed 1337, batch size 8, AdamW, new-layer learning rate 1e-4, backbone rate 1e-5, weight decay 0.01, label smoothing 0.03, gradient clipping at 5 and a 20-epoch cosine schedule. Training stops after at least 10 epochs if validation fails to improve for eight epochs. ResNet running batch-normalization statistics stay frozen. Clinical probability weights 0, 0.1 and 0.2 are compared during validation; the clinical classifier uses C=0.1. The best fine-tuned candidate is compared with the earlier fusion-search winner using validation scores only.

## Refit and evaluation

If a fine-tuned candidate wins, its selected epoch count and clinical weight are locked, then the schedule is replayed starting from the previous 575-image refit checkpoint. The clinical head is fitted on those 575 development images. If the earlier fusion candidate wins, only its head is refitted against the corresponding pre-existing development image model. Neither final refit has a separate validation score.

After the fine-tuning search, `ensemble_followup.py` compares the two individual winners and an equal mean of their probabilities. Its three candidate weights (0, 0.5, 1) are recorded before comparison. It refits only the components required by the winner. Only that final selected package is evaluated. Evaluation checks patient, lesion-ID and exact-image separation, saves checkpoint and source hashes, then records metrics, predictions and 1,000 patient-cluster bootstrap replicates. The original test partition has already been inspected in earlier project work, so any new score is still a reused benchmark result, not independent clinical validation. Searching more candidates also increases validation overfitting risk.

## Completed outcome

The clinical blend at 320px, with C=0.1 and a 10% clinical probability contribution, won at **94.79% validation accuracy**. Mixup training stopped after 12 epochs, with its best checkpoint at epoch 4 and **93.75% validation accuracy**. Equal averaging did not improve the validation selection.

After fitting the selected clinical head on all 575 development images, the final system scored **86.46% test accuracy (83/96)**, macro F1 **0.8639**, and macro ROC-AUC **0.9680**. This is below the earlier **87.50% (84/96)** refits. The previous model is preserved, and the above-90% test target remains unmet. This unsuccessful follow-up is retained to make the experiment history transparent.

The higher validation score did not translate to a higher test score. No claim that clinical late fusion improves generalization is supported by this experiment. Neither the 66-head search nor the mixup run justifies presenting 94.79% as final-model test accuracy.

## Commands

For an existing project workspace that contains the earlier completed runs:

```powershell
python clinical_fusion.py search --workspace PATH_TO_WORKSPACE
python clinical_fusion.py refit --workspace PATH_TO_WORKSPACE
python robust_finetune.py search --workspace PATH_TO_WORKSPACE
python ensemble_followup.py search --workspace PATH_TO_WORKSPACE
python ensemble_followup.py refit --workspace PATH_TO_WORKSPACE
python ensemble_followup.py evaluate --workspace PATH_TO_WORKSPACE
python report_followup.py --workspace PATH_TO_WORKSPACE --run robust_ensemble_01
```

The workspace flag is optional when the data and earlier `outputs/` reside beside the scripts. Substitute your own workspace path. Completed results are protected from overwrite.

`predict_clinical_fusion.py` supports packages from either follow-up experiment:

```powershell
python predict_clinical_fusion.py --checkpoint PATH_TO_BEST_SYSTEM.pt --image lesion.png --clinical-json examples/clinical.json
```

The example uses JSON booleans, which the clinical encoder maps to the training categories `True` and `False`. Uppercase strings such as `"TRUE"` are different categorical values and would map to unknown.
