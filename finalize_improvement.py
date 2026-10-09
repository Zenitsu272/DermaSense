"""Lock validation-selected configuration, then evaluate the existing hold-out once."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import ROOT,CLASSES,ClinicalEncoder,LesionDataset,metrics,bootstrap,save_json,write_rows
from improve import SEARCH,rows_for,predict
from improved_model import ImprovedClassifier

def main():
    torch.set_num_threads(4)
    out=SEARCH/'final';out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():raise RuntimeError('This selected configuration was already evaluated. Do not overwrite its results.')
    protocol=json.loads((SEARCH/'protocol.json').read_text())
    for split,expected in protocol['fixed_split_hashes'].items():
        assert hashlib.sha256((ROOT/'outputs/run_01'/f'{split}.csv').read_bytes()).hexdigest()==expected, f'{split} partition changed'
    selected=json.loads((SEARCH/'selection.json').read_text())['selected']
    checkpoints=[]
    lock={'selection':selected,'selection_sha256':hashlib.sha256((SEARCH/'selection.json').read_bytes()).hexdigest(),'checkpoint_sha256':{str(seed):hashlib.sha256((SEARCH/f'seed_{seed}/best_model.pt').read_bytes()).hexdigest() for seed in selected['seeds']},'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['improved_model.py','improve.py','finalize_improvement.py','predict_improved.py']},'test_policy':'Same test partition previously evaluated for run_01. No test-based selection within this improvement search. This is not fresh external validation.'}
    save_json(out/'locked_selection.json',lock)
    for seed in selected['seeds']:checkpoints.append(torch.load(SEARCH/f'seed_{seed}/best_model.pt',map_location='cpu',weights_only=False))
    encoder=ClinicalEncoder(checkpoints[0]['clinical_encoder']);rows=rows_for('test')
    loader=DataLoader(LesionDataset(rows,encoder),batch_size=16,num_workers=0,pin_memory=True)
    predictions=[];start=time.monotonic()
    for cp in checkpoints:
        model=ImprovedClassifier(len(encoder.state['features']),pretrained=False).to(device='cuda',memory_format=torch.channels_last)
        model.load_state_dict(cp['model_state']);y,p=predict(model,loader,selected['tta']);predictions.append(p)
        del model;torch.cuda.empty_cache()
    p=np.mean(predictions,axis=0);result=metrics(y,p)
    result.update({'test_images':len(rows),'test_patients':len({r['patient_id'] for r in rows}),'selected':selected,'patient_bootstrap_95ci':bootstrap(y,p,rows),'evaluation_seconds':time.monotonic()-start,'test_policy':lock['test_policy'],'goal_over_90_percent_met':bool(result['accuracy']>.90)})
    save_json(out/'metrics.json',result)
    records=[{'img_id':r['img_id'],'patient_id':r['patient_id'],'true_class':r['diagnostic'],'predicted_class':CLASSES[int(p[i].argmax())],**{f'prob_{c}':float(p[i,k]) for k,c in enumerate(CLASSES)}} for i,r in enumerate(rows)]
    write_rows(out/'test_predictions.csv',records)
    torch.save({'architecture':'pretrained_resnet50_with_clinically_routed_unet_residual','class_order':CLASSES,'clinical_encoder':encoder.state,'tta':selected['tta'],'selected':selected,'model_states':[cp['model_state'] for cp in checkpoints]},out/'best_system.pt')
    cm=np.array(result['confusion_matrix_rows_true_columns_predicted']);fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='True',title='Improved model: existing hold-out')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for seed in [42,1337,2026]:
        h=json.loads((SEARCH/f'seed_{seed}/history.json').read_text());ax.plot([x['epoch'] for x in h],[x['val_accuracy'] for x in h],label=f'Seed {seed}')
    ax.axhline(.90,color='black',linestyle='--',alpha=.4,label='90% target');ax.set(xlabel='Epoch',ylabel='Validation accuracy',ylim=(0,1),title='Validation-only training comparison');ax.legend();fig.tight_layout();fig.savefig(out/'validation_curves.png',dpi=170);plt.close(fig)
    original=json.loads((ROOT/'outputs/run_01/metrics.json').read_text())
    candidates=json.loads((SEARCH/'selection.json').read_text())['candidates']
    candidate_rows=''.join(f"<tr><td>{c['name']}</td><td>{c['validation']['accuracy']:.2%}</td><td>{c['validation']['macro_f1']:.3f}</td></tr>" for c in candidates)
    per_rows=''.join(f"<tr><td>{c}</td><td>{v['support_images']}</td><td>{v['precision']:.3f}</td><td>{v['recall_sensitivity']:.3f}</td><td>{v['specificity']:.3f}</td><td>{v['f1']:.3f}</td><td>{v['roc_auc_ovr']:.3f}</td></tr>" for c,v in result['per_class'].items())
    comparison=''.join(f"<tr><td>{name}</td><td>{original[key]:.3f}</td><td>{result[key]:.3f}</td></tr>" for name,key in [('Accuracy','accuracy'),('Macro precision','macro_precision'),('Macro recall','macro_recall'),('Macro F1','macro_f1'),('Macro specificity','macro_specificity'),('Macro ROC-AUC','macro_roc_auc_ovr')])
    intervals=''.join(f"<li>{k}: {v['low']:.3f}–{v['high']:.3f}</li>" for k,v in result['patient_bootstrap_95ci'].items())
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>Improved SCC / SEK / NEV results</title><style>body{{font:16px/1.6 system-ui;color:#183144;max-width:1050px;margin:40px auto;padding:0 24px}}h1,h2{{color:#124963}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:9px;border-bottom:1px solid #ccd9e0;text-align:left}}th{{background:#edf3f6}}img{{max-width:100%}}.score{{font-size:42px;font-weight:700}}</style></head><body><h1>Improved clinically guided skin lesion classification</h1><div class="score">{result['accuracy']:.2%} test accuracy</div><p>Selected configuration: <b>{selected['name']}</b>. Accuracy above 90%: <b>{result['goal_over_90_percent_met']}</b>.</p>
<h2>What changed</h2><p>The original random U-Net-to-ResNet interface discarded ResNet's pretrained RGB stem. This version preserves the entire pretrained ResNet50 image path. A compact U-Net encoder still extracts features; three image expert branches map these to 512 channels and a clinical router weights their contributions. Their weighted sum is added to ResNet stage-2 features before stage 3 and 4 classification. This explicitly adds a pretrained image bypass to the original design. The router uses age, region, itching, pain, bleeding and elevation. No clinical vector is concatenated with image features.</p>
<p>Three predetermined seeds (42, 1337, 2026), up to 45 epochs each, class-weighted cross-entropy with 0.03 label smoothing, AdamW, a three-epoch backbone warmup, and validation-based learning-rate reduction/early stopping. Eight final candidates: each seed and their equal-probability ensemble, each with single-view or four-flip averaging. Checkpoints and candidate selection use validation accuracy, then macro F1, then log loss.</p>
<h2>Comparison on the same 96 test images from 68 patients</h2><table><tr><th>Metric</th><th>Original</th><th>Improved selection</th></tr>{comparison}</table><p>The original 479/96/96 train/validation/test partitions remain unchanged. No patient, lesion-ID or exact-image overlap. All preprocessing statistics are fitted on training data. The test partition was already reported in the earlier experiment, so this comparison is not an untouched external validation. No individual test labels or scores were used to choose among these eight candidates; the selection was locked before evaluation.</p>
<h2>Per-class test results</h2><table><tr><th>Class</th><th>Images</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>ROC-AUC</th></tr>{per_rows}</table><img src="confusion_matrix.png" alt="Confusion matrix">
<h2>Validation search</h2><table><tr><th>Candidate</th><th>Validation accuracy</th><th>Macro F1</th></tr>{candidate_rows}</table><img src="validation_curves.png" alt="Validation curves">
<h2>Uncertainty and limitations</h2><p>95% patient-cluster bootstrap intervals, 1,000 resamples:</p><ul>{intervals}</ul><p>Small dataset and hold-out; validation selection across several runs can overfit validation. Independent patients from an external dataset are needed for a stronger generalization claim. Metadata missingness, demographics and dataset acquisition can still create shortcuts despite excluding the most problematic fields. No unknown/healthy class exists. The model covers only SCC, SEK and NEV.</p>
<h2>Reusable outputs</h2><p><a href="best_system.pt">Selected model package</a> · <a href="metrics.json">Metrics</a> · <a href="test_predictions.csv">Predictions</a> · <a href="locked_selection.json">Selection and source audit</a></p><p>Inference: <code>predict_improved.py --image IMAGE --clinical-json CLINICAL_JSON</code>. Original experiment files remain in <code>outputs/run_01</code>.</p><p>Source: <a href="https://data.mendeley.com/datasets/zr7vgbcyr2/1">PAD-UFES-20, Pacheco et al., CC BY 4.0</a>. Transfer learning reference: <a href="https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html">PyTorch tutorial</a>.</p></body></html>'''
    (out/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['accuracy','macro_f1','macro_roc_auc_ovr','goal_over_90_percent_met']}),flush=True)
    print('DONE: '+str(out/'report.html'),flush=True)

if __name__=='__main__':main()
