"""Publish all completed benchmark results without selecting by test accuracy."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from train import ROOT

def main():
    runs=[('Original',ROOT/'outputs/run_01'),('Revised architecture',ROOT/'outputs/improvement_01/final'),('224px development refit',ROOT/'outputs/improvement_01/refit'),('Progressive resolution refit',ROOT/'outputs/high_resolution_01/refit')]
    available=[(name,path,json.loads((path/'metrics.json').read_text())) for name,path in runs if (path/'metrics.json').exists()];name,out,m=available[-1];target=ROOT/'outputs/improvement_report.html'
    cm=np.asarray(m['confusion_matrix_rows_true_columns_predicted']);fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues');ax.set(xticks=range(3),yticks=range(3),xticklabels=m['class_order'],yticklabels=m['class_order'],xlabel='Predicted class',ylabel='True class',title=name)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(ROOT/'outputs/latest_confusion_matrix.png',dpi=170);plt.close(fig)
    comparison=''.join(f'<tr><td>{n}</td><td>{v["accuracy"]:.2%}</td><td>{v["macro_f1"]:.4f}</td><td>{v["macro_roc_auc_ovr"]:.4f}</td></tr>' for n,p,v in available)
    per=''.join(f'<tr><td>{c}</td><td>{v["support_images"]}</td><td>{v["precision"]:.3f}</td><td>{v["recall_sensitivity"]:.3f}</td><td>{v["specificity"]:.3f}</td><td>{v["f1"]:.3f}</td><td>{v["roc_auc_ovr"]:.3f}</td></tr>' for c,v in m['per_class'].items())
    ci=m['patient_bootstrap_95ci']['accuracy'];rel=out.relative_to(ROOT/'outputs').as_posix();accuracy=m['accuracy'];count=int(np.trace(cm));script='predict_high_resolution.py' if 'high_resolution' in str(out) else 'predict_refit.py'
    high=''
    selection=ROOT/'outputs/high_resolution_01/selection.json'
    if selection.exists():
        s=json.loads(selection.read_text());high=f'<p>The progressive-resolution search compared 224px four-view, 320px single-view, 320px four-view and an equal multiresolution average on the original validation set. Winner: <b>{s["selected"]["name"]}</b>, validation accuracy {s["selected"]["validation"]["accuracy"]:.2%}. This score belongs to the selection model; the final refit has no independent validation score.</p>'
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>SCC / SEK / NEV accuracy experiments</title><style>body{{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#173346}}h1,h2{{color:#14546e}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{border-bottom:1px solid #d6e1e8;padding:10px;text-align:left}}th{{background:#edf4f7}}.score{{font-size:44px;font-weight:750}}img{{max-width:100%}}code{{overflow-wrap:anywhere}}</style></head><body><h1>SCC / SEK / NEV: completed accuracy experiments</h1><div class="score">{accuracy:.2%} test accuracy</div><p>Latest completed configuration: {name}. Correct: {count}/96 images from 68 patients. Accuracy above 90%: <b>{'achieved on this benchmark' if accuracy>.90 else 'not achieved'}</b>.</p>
<h2>What changed</h2><p>The revised architecture retains the full ImageNet-pretrained ResNet50 image path. A U-Net contracting path extracts features; three image experts are weighted by a clinical MLP using age, region, itch, hurt, bleed and elevation. Their output is added to ResNet stage-2 features before classification. This image bypass is an extension of the original method. All training used the local RTX 3050 GPU with mixed precision.</p><p>Three seeds and inference averaging were selected on 479 training / 96 validation images. The selected 224px model was refitted on all 575 development images for the previously selected 27 epochs. Progressive fine-tuning then tested 320px inputs while preserving the patient split and clinical fields. No test image was used for training, preprocessing fitting or checkpoint selection.</p>{high}
<h2>All completed test results</h2><table><tr><th>Experiment</th><th>Accuracy</th><th>Macro F1</th><th>Macro ROC-AUC</th></tr>{comparison}</table><p>Rows report the same 96-image test partition. All checkpoints and per-image predictions are preserved.</p>
<h2>Latest per-class results</h2><table><tr><th>Class</th><th>Images</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>ROC-AUC</th></tr>{per}</table><img src="latest_confusion_matrix.png" alt="Latest confusion matrix">
<h2>Limits of this result</h2><p>Accuracy 95% patient-cluster bootstrap interval: {ci['low']:.2%}–{ci['high']:.2%} (1,000 resamples). This interval does not account for repeated model development. The test partition has been evaluated in earlier experiments, and earlier aggregate scores motivated changes. It is therefore a reused development benchmark, not independent external validation. A fresh cohort is required to confirm generalization. Former validation images enter final refit training, so earlier validation scores cannot be attributed to those refitted models.</p><p>No patient, lesion-ID or exact-image overlap crosses development/test partitions. Labels and the original partition are unchanged. The dataset has only 671 images, and the model only recognizes SCC, SEK and NEV.</p>
<h2>Saved artifacts</h2><p><a href="{rel}/best_system.pt">Latest model package</a> · <a href="{rel}/metrics.json">Full metrics</a> · <a href="{rel}/test_predictions.csv">Per-image predictions</a> · <a href="{rel}/verification.json">Verification</a></p><p>From the project folder: <code>.\\.venv\\Scripts\\python.exe {script} --image IMAGE --clinical-json CLINICAL_JSON</code>.</p><p>Dataset: <a href="https://data.mendeley.com/datasets/zr7vgbcyr2/1">PAD-UFES-20, Pacheco et al., CC BY 4.0</a>.</p></body></html>'''
    target.write_text(html,encoding='utf-8');print(target)

if __name__=='__main__':main()
