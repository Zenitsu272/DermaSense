"""Verify and summarize the selected clinical-fusion follow-up experiment."""
import argparse,csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from clinical_fusion import rows_for
from train import CLASSES,metrics,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--run',required=True);args=ap.parse_args();workspace=args.workspace.resolve();out=workspace/'outputs'/args.run
    m=json.loads((out/'metrics.json').read_text());lock=json.loads((out/'evaluation_lock.json').read_text())
    with (out/'test_predictions.csv').open(newline='') as f:pred=list(csv.DictReader(f))
    rows=rows_for(workspace,'test');dev=rows_for(workspace,'train')+rows_for(workspace,'val')
    assert [r['img_id'] for r in rows]==[r['img_id'] for r in pred]
    assert [r['diagnostic'] for r in rows]==[r['true_class'] for r in pred]
    y=np.array([CLASSES.index(r['true_class']) for r in pred]);p=np.array([[float(r['prob_'+c]) for c in CLASSES] for r in pred],dtype=np.float32)
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=1e-6)
    calculated=metrics(y,p)
    for k,v in calculated.items():
        if isinstance(v,float):assert np.isclose(v,m[k],atol=1e-6),k
    assert calculated['confusion_matrix_rows_true_columns_predicted']==m['confusion_matrix_rows_true_columns_predicted']
    for k in ['patient_id','lesion_id']:assert not ({r[k] for r in rows}&{r[k] for r in dev})
    hashes={r['img_id']:r['sha256'] for r in json.loads((workspace/'data/manifest.json').read_text())};assert not ({hashes[r['img_id']] for r in rows}&{hashes[r['img_id']] for r in dev})
    assert hashlib.sha256((out/'best_system.pt').read_bytes()).hexdigest()==lock['checkpoint_sha256']
    assert hashlib.sha256((workspace/'outputs/run_01/test.csv').read_bytes()).hexdigest()==lock['test_sha256']
    source=lock['source_sha256']
    if isinstance(source,str):source={'clinical_fusion.py':source}
    for name,h in source.items():assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==h,name
    inference=False
    if (out/'example_prediction.json').exists():
        example=json.loads((out/'example_prediction.json').read_text());name=(workspace/'outputs/run_01/example_image.txt').read_text().strip().replace('\\','/').split('/')[-1];row=next(r for r in pred if r['img_id']==name)
        assert row['predicted_class']==example['predicted_class'];assert all(abs(float(row['prob_'+c])-example['probabilities'][c])<.005 for c in CLASSES);inference=True
    save_json(out/'verification.json',{'passed':True,'metrics_recomputed':True,'predictions_and_labels_match':True,'no_patient_lesion_or_exact_image_overlap':True,'checkpoint_and_source_hashes_verified':True,'standalone_inference_agrees':inference,'correct':int((y==p.argmax(1)).sum()),'test_images':len(y)})
    cm=np.array(m['confusion_matrix_rows_true_columns_predicted']);fig,ax=plt.subplots(figsize=(5.4,4.4));ax.imshow(cm,cmap='Blues');ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted class',ylabel='True class',title='Selected follow-up: reused test benchmark')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=170);plt.close(fig)
    selected=lock['selected'];ci=m['patient_bootstrap_95ci']['accuracy']
    per=''.join(f'<tr><td>{c}</td><td>{v["support_images"]}</td><td>{v["precision"]:.3f}</td><td>{v["recall_sensitivity"]:.3f}</td><td>{v["specificity"]:.3f}</td><td>{v["f1"]:.3f}</td></tr>' for c,v in m['per_class'].items())
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>DermaSense follow-up results</title><style>body{{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#183747}}h1,h2{{color:#14546e}}.score{{font-size:46px;font-weight:750}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #cad9e2;text-align:left}}th{{background:#edf4f7}}img{{max-width:100%}}</style></head><body><h1>DermaSense: clinical fusion and regularized training</h1><div class="score">{m['accuracy']:.2%} test accuracy</div><p>{m['correct']} correct / 96 images from 68 patients. Above-90% target met: <b>{m['goal_over_90_percent_met']}</b>. Prior refit: 87.50%.</p>
<h2>Selected method</h2><p>Configuration: <b>{selected['name']}</b>. Its training-only model scored {selected['validation']['accuracy']:.2%} on validation. The final model uses all 575 development images; it has no independent validation score.</p><p>The existing U-Net/ResNet50 image network retains its clinical expert router. A new experiment tested L2-regularized clinical probability fusion and image/clinical feature classifiers using the same six fields. Sixty-six candidates were compared on the original validation partition. A separate GPU fine-tuning run tested stronger crops, rotation, color augmentation and mixup, then compared its best checkpoint with the earlier fusion candidate on validation. Only the winner of this follow-up comparison was evaluated on test.</p><p>Image models use four flipped views. Small logistic heads fit on CPU, image feature extraction and neural-network fine-tuning use CUDA. Final refits use 575 development images while the 96 test images remain excluded from fitting. Model choice, mixture weight, regularization and training duration use validation results, not new test predictions.</p>
<h2>Test metrics</h2><table><tr><th>Accuracy</th><th>Macro F1</th><th>Macro precision</th><th>Macro recall</th><th>Macro ROC-AUC</th></tr><tr><td>{m['accuracy']:.4f}</td><td>{m['macro_f1']:.4f}</td><td>{m['macro_precision']:.4f}</td><td>{m['macro_recall']:.4f}</td><td>{m['macro_roc_auc_ovr']:.4f}</td></tr></table><h2>Class-wise metrics</h2><table><tr><th>Class</th><th>Images</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th></tr>{per}</table><img src="confusion_matrix.png" alt="Confusion matrix">
<h2>Interpretation and limits</h2><p>Patient-cluster bootstrap accuracy interval: {ci['low']:.2%}–{ci['high']:.2%} (95%, 1,000 replicates). This interval does not capture model-selection uncertainty. This is the same small benchmark inspected in previous project experiments. It is not an independent external validation, and any crossing of 90% would not establish clinical reliability. The split and labels were not changed. No patient, lesion-ID or exact-image overlap crosses development/test.</p><h2>Artifacts</h2><p><a href="best_system.pt">Model package</a> · <a href="metrics.json">Full metrics</a> · <a href="test_predictions.csv">Predictions</a> · <a href="verification.json">Verification</a> · <a href="evaluation_lock.json">Selection and integrity lock</a></p><p>Use <code>predict_clinical_fusion.py --checkpoint PACKAGE --image IMAGE --clinical-json CLINICAL_JSON</code> for inference. Data: <a href="https://data.mendeley.com/datasets/zr7vgbcyr2/1">PAD-UFES-20, CC BY 4.0</a>.</p></body></html>'''
    (out/'report.html').write_text(html,encoding='utf-8');print(json.dumps({'accuracy':m['accuracy'],'correct':m['correct'],'verified':True,'report':str(out/'report.html')}))


if __name__=='__main__':main()
