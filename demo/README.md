# Three-cell live evaluation notebook

Open [DermaSense_Live_Demo.ipynb](DermaSense_Live_Demo.ipynb) locally in Jupyter or [in Google Colab](https://colab.research.google.com/github/Zenitsu272/DermaSense/blob/main/demo/DermaSense_Live_Demo.ipynb).

The notebook has exactly **three code cells**:

1. **Load:** load the existing 224px development-refit checkpoint, saved clinical encoder and demo records. No training and no ImageNet download occur. GPU is selected automatically, with CPU fallback. CUDA initialization is performed here so later prediction runs are responsive.
2. **Choose:** upload one or more images. Local Jupyter uses a multiple-file widget; Colab uses its native file-upload dialog.
3. **Predict and score:** show all predicted classes/probabilities, available reference classes, accuracy, macro precision/recall/F1, per-class metrics, a confusion matrix and image thumbnails. Each run saves CSV/JSON results and figures.

Run cells 2 and 3 again for another selection; keep the model loaded in cell 1. The local notebook needs no internet after setup. In Colab, upload `dermasense_assets.zip` once per session before uploading test images; model-file upload time depends on the connection.

## Prepared local evaluation kit

The generated kit is a folder named `DermaSense_Evaluation` in the user's Downloads directory. It contains:

```text
DermaSense_Evaluation/
  Start_Demo.cmd                  # Opens the prepared local Jupyter environment
  START_HERE.txt                  # Local and Colab instructions
  DermaSense_Live_Demo.ipynb       # Exactly three cells
  dermasense_assets.zip           # Portable model/code/record bundle, about 90 MB
  _assets/                       # Already extracted for fast local startup
  images/                        # 20 original PNG images
  clinical_data.csv              # Six clinical inputs per demo image
  ground_truth.csv               # Reference classes, for scoring only
  selection_manifest.json        # Seed, image hashes and source provenance
  verification.json              # Integration-test results
```

The selection uses fixed seed **20261019**, with **7 SCC, 7 SEK and 6 NEV** images sampled randomly from the original test partition. Selection happens before predictions are inspected. Exact duplicate content is excluded. The kit uses the original **87.50% checkpoint**, with unchanged floating-point weights and four-view inference. It does not use the later, lower-scoring clinical-fusion experiment.

Clinical records and reference labels are matched by image SHA-256, so a renamed file still works. The network consumes only the image and the six clinical fields; reference labels are read only after prediction for scoring. Duplicate uploads count once. Unknown images can receive predictions with missing clinical inputs but are not included in accuracy without matching labels. Do not edit a label merely to make a prediction appear correct.

Macro precision/recall/F1 are calculated over the true classes represented in the chosen labelled subset. Absent-class recall/F1 and undefined AUC are displayed as unavailable. A one-class upload therefore does not fabricate a multiclass AUC. The live subset score may be higher or lower than 87.50%; it is not a new independent test result.

## Rebuild from the completed experiment workspace

The model and clinical records are intentionally not distributed through Git. The builder requires the original local data and completed refit outputs:

```powershell
python demo/build_demo.py --workspace PATH_TO_EXPERIMENT_WORKSPACE --destination PATH_TO_DOWNLOADS/DermaSense_Evaluation --python PATH_TO_JUPYTER_PYTHON
```

It verifies the original checkpoint hash, exports a tensor-only inference package loadable with `weights_only=True`, copies the selected original images, writes matching CSVs and builds a checksum-verified assets ZIP. The generated notebook is tied to that ZIP's checksum. Keep the notebook and ZIP from the same build together. The source experiment files are preserved.

For local setup on a new computer, install core requirements and `demo/requirements-demo.txt` into a virtual environment, then run Jupyter from the generated folder. The generated Windows launcher points at the `--python` environment specified when building; use a normal Jupyter launch on other computers or operating systems.

## VS Code setup and kernel selection

Install the Microsoft Python and Jupyter extensions. Use **File > Open Folder** to open the generated evaluation folder; opening only the notebook does not apply that folder's interpreter settings.

Register the prepared environment (the same Python used to build and test the kit):

```powershell
python demo/setup_vscode.py --demo-dir PATH_TO_DOWNLOADS/DermaSense_Evaluation --python PATH_TO_JUPYTER_PYTHON
```

The setup checks dependencies, registers **DermaSense Demo (GPU)**, adds folder-specific interpreter settings and creates `Start_VSCode.cmd`. It preserves the notebook's three cells and any saved outputs. The generated kit also contains a copy of this setup script. The launcher requires the `code` command on PATH; alternatively open the folder from VS Code itself.

In the notebook's top-right kernel menu, select **Select Another Kernel > Jupyter Kernel > DermaSense Demo (GPU)**. If the menu is hidden, use **Ctrl+Shift+P > Notebook: Select Notebook Kernel**. Reload the window if the newly registered kernel has not appeared. Run cell 1 once, then cells 2 and 3 for each image batch. GPU selection is automatic; CPU fallback is supported.

If cell 1 shows `CalledProcessError` from a path containing `spyder-6`, the notebook is using Spyder's separate environment. Select the registered DermaSense kernel instead. Registering a kernel or setting the default interpreter does not override a kernel that VS Code already selected for an open notebook.

The GitHub folder contains the clean notebook, runtime, builder, setup script and tests. Model weights, demo images, clinical records, local environments and generated outputs remain in the local evaluation kit. A fresh clone requires building or supplying that kit before inference.

## Verification

```powershell
python demo/test_demo.py --demo-dir PATH_TO_DEMO_FOLDER --workspace PATH_TO_EXPERIMENT_WORKSPACE
```

Checks include original-checkpoint prediction agreement, live scoring on 10 and 20 images, CPU/GPU agreement, renamed files, duplicate/corrupt input handling, unknown-label exclusion, single-class metrics and a real three-cell Jupyter-kernel execution with the upload widget. A label-isolation check confirms that changing only a scoring label cannot change the prediction. The user-facing notebook is saved without test outputs or preselected images.

The prepared kit passed these checks on the RTX 3050 laptop. A repeated 10-image inference took approximately 0.9 seconds on GPU and 6.1 seconds on CPU; a complete fresh three-cell kernel run took approximately 23 seconds. Timings vary with startup, image sizes and hardware. The local Jupyter server also served the notebook with the prepared Python kernel successfully.

The fresh ZIP extraction and Colab upload branch were exercised using a local simulated Colab interface. Hosted Colab itself was not tested. Installation and internet upload may add time to a fresh hosted session.

Source images: [PAD-UFES-20 v1](https://data.mendeley.com/datasets/zr7vgbcyr2/1), Pacheco et al., **CC BY 4.0**. This is a research evaluation demo.
