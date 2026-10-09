"""Evaluate the fixed full-development refit on the existing benchmark."""
import csv,hashlib,json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import ROOT,CLASSES,ClinicalEncoder,LesionDataset,metrics,bootstrap,save_json,write_rows
from improve import SEARCH,rows_for,predict
from improved_model import ImprovedClassifier
from refit import OUT

def main():
    if (OUT/'metrics.json').exists():raise RuntimeError('Refit already evaluated; preserving its results')
    torch.set_num_threads(4)
    protocol=json.loads((SEARCH/'protocol.json').read_text())
    for split,expected in protocol['fixed_split_hashes'].items():
        assert hashlib.sha256((ROOT/'outputs/run_01'/f'{split}.csv').read_bytes()).hexdigest()==expected
    cp=torch.load(OUT/'best_system.pt',map_location='cpu',weights_only=False)
    save_json(OUT/'evaluation_lock.json',{'checkpoint_sha256':hashlib.sha256((OUT/'best_system.pt').read_bytes()).hexdigest(),'test_sha256':protocol['fixed_split_hashes']['test'],'refit_protocol':cp['selected']['refit'],'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['improved_model.py','refit.py','evaluate_refit.py','predict_improved.py','predict_refit.py']}})
    rows=rows_for('test');enc=ClinicalEncoder(cp['clinical_encoder'])
    development=rows_for('train')+rows_for('val')
    assert not ({r['patient_id'] for r in development}&{r['patient_id'] for r in rows})
    assert not ({r['lesion_id'] for r in development}&{r['lesion_id'] for r in rows})
    model=ImprovedClassifier(len(enc.state['features']),pretrained=False).to(device='cuda',memory_format=torch.channels_last);model.load_state_dict(cp['model_states'][0])
    loader=DataLoader(LesionDataset(rows,enc),batch_size=16,num_workers=0,pin_memory=True)
    y,p=predict(model,loader,cp['tta']);result=metrics(y,p)
    result.update({'goal_over_90_percent_met':bool(result['accuracy']>.90),'test_images':len(rows),'test_patients':len({r['patient_id'] for r in rows}),'patient_bootstrap_95ci':bootstrap(y,p,rows),'protocol':cp['selected']['refit'],'validation_note':'No validation score for this refit: the former validation set is now training data. The earlier selection model had 92.708% validation accuracy.','test_note':'Existing benchmark reused across experiments; not independent external validation.'})
    save_json(OUT/'metrics.json',result)
    write_rows(OUT/'test_predictions.csv',[{'img_id':r['img_id'],'patient_id':r['patient_id'],'true_class':r['diagnostic'],'predicted_class':CLASSES[int(p[i].argmax())],**{f'prob_{c}':float(p[i,k]) for k,c in enumerate(CLASSES)}} for i,r in enumerate(rows)])
    cm=np.array(result['confusion_matrix_rows_true_columns_predicted']);fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues');ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='True',title='Final refit: existing test benchmark')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(OUT/'confusion_matrix.png',dpi=170);plt.close(fig)
    original=json.loads((ROOT/'outputs/run_01/metrics.json').read_text());selected=json.loads((SEARCH/'final/metrics.json').read_text())
    comparison=''.join(f"<tr><td>{name}</td><td>{original[k]:.3f}</td><td>{selected[k]:.3f}</td><td>{result[k]:.3f}</td></tr>" for name,k in [('Accuracy','accuracy'),('Macro precision','macro_precision'),('Macro recall','macro_recall'),('Macro specificity','macro_specificity'),('Macro F1','macro_f1'),('Macro ROC-AUC','macro_roc_auc_ovr')])
    per=''.join(f"<tr><td>{c}</td><td>{m['support_images']}</td><td>{m['precision']:.3f}</td><td>{m['recall_sensitivity']:.3f}</td><td>{m['specificity']:.3f}</td><td>{m['f1']:.3f}</td><td>{m['roc_auc_ovr']:.3f}</td></tr>" for c,m in result['per_class'].items())
    cis=''.join(f"<li>{k}: {v['low']:.3f}–{v['high']:.3f}</li>" for k,v in result['patient_bootstrap_95ci'].items())
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>Final SCC / SEK / NEV refit</title><style>body{{font:16px/1.6 system-ui;color:#183144;max-width:1050px;margin:40px auto;padding:0 24px}}h1,h2{{color:#124963}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:9px;border-bottom:1px solid #ccd9e0;text-align:left}}th{{background:#edf3f6}}img{{max-width:100%}}.score{{font-size:42px;font-weight:700}}</style></head><body>
<h1>Final clinically guided three-class model</h1><div class="score">{result['accuracy']:.2%} test accuracy</div><p>Correct predictions: {int((y==p.argmax(1)).sum())} / {len(y)}. Target above 90% met: <b>{result['goal_over_90_percent_met']}</b>.</p>
<h2>Changes and final training</h2><p>Preserves the pretrained ResNet50 RGB image path. U-Net's contracting path extracts image features, three experts map them to 512 channels, and a clinical router weights the experts. Their weighted output is added to ResNet stage-2 features, then classified by stages 3/4 and a three-class head. Clinical inputs are age, region, itch, hurt, bleed and elevation. The image bypass is an explicit extension of the original architecture.</p><p>Three seeds and eight inference candidates were compared on the original validation set. Seed 1337 with four-view averaging won at 92.71% validation accuracy; its initial test result was 86.46%. The selected configuration was then refitted from ImageNet initialization on all 575 development images (414 patients), combining original training and validation rows. Training ran for the previously selected 27 epochs, with the original learning-rate schedule fixed in advance. The 96 test images (68 patients) were excluded throughout.</p><p><b>The refitted model has no separate validation score.</b> The earlier 92.71% applies only to the model-selection experiment. Final refit accuracy below is measured on the same existing test partition, which has been inspected in earlier experiments; it is not a new independent estimate.</p>
<h2>Comparison on the same test images</h2><table><tr><th>Metric</th><th>Original model</th><th>Revised, 479 training images</th><th>Final refit, 575 development images</th></tr>{comparison}</table>
<h2>Final per-class metrics</h2><table><tr><th>Class</th><th>Images</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>ROC-AUC</th></tr>{per}</table><img src="confusion_matrix.png" alt="Confusion matrix">
<h2>Uncertainty</h2><p>95% patient-cluster bootstrap intervals (1,000 resamples):</p><ul>{cis}</ul><p>Small cohort, one reused test partition, no external validation. Model development was motivated by earlier aggregate test scores, so a fresh independent cohort is needed to confirm generalization. No patient, lesion-ID or exact-image overlap crosses development/test partitions; no test labels are used in gradient updates, preprocessing fitting, checkpoint selection or class-threshold tuning. The class scope is only SCC, SEK and NEV.</p>
<h2>Reproduce and use</h2><p><a href="best_system.pt">Final model package</a> · <a href="metrics.json">Metrics JSON</a> · <a href="test_predictions.csv">Predictions</a> · <a href="protocol.json">Refit protocol</a> · <a href="evaluation_lock.json">Integrity audit</a> · <a href="../final/report.html">Earlier model-selection report</a></p><p>Inference: <code>predict_refit.py --image IMAGE --clinical-json CLINICAL_JSON</code>. Training source: <code>refit.py</code>. Evaluation source: <code>evaluate_refit.py</code>. Model and feature definitions: <code>improved_model.py</code>, <code>train.py</code>.</p><p>Source dataset: <a href="https://data.mendeley.com/datasets/zr7vgbcyr2/1">PAD-UFES-20 (Pacheco et al., CC BY 4.0)</a>. All three experiments and predictions are preserved.</p></body></html>'''
    (OUT/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['accuracy','macro_f1','macro_roc_auc_ovr','goal_over_90_percent_met']}),flush=True)

if __name__=='__main__':main()
