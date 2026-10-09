"""Reproducible patient-disjoint training, validation selection and held-out evaluation."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import platform
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image,ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,
                             f1_score,confusion_matrix,roc_auc_score,log_loss,roc_curve,precision_recall_curve,average_precision_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import GuidedClassifier

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME',str(ROOT/'pretrained'))
CLASSES=['SCC','SEK','NEV']
# Publisher metadata availability is strongly associated with diagnosis/biopsy.
# Exclude the highly class-dependent missing fields from the primary experiment.
NUMERIC=['age']
CATEGORICAL=['region','itch','hurt','bleed','elevation']
MISSING={'','UNK','NAN','NONE','NULL'}

def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def load_rows():
    with (ROOT/'data/metadata.csv').open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def write_rows(path,rows,extra=None):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def number(row,key):
    try:
        n=float(row.get(key,''));return n if np.isfinite(n) else np.nan
    except (ValueError,TypeError):return np.nan

def category(row,key):
    v=str(row.get(key,'')).strip()
    return '__MISSING__' if v.upper() in MISSING else v

class ClinicalEncoder:
    def __init__(self, state=None): self.state=state
    def fit(self,rows):
        self.state={'numeric':{},'categorical':{},'features':[]}
        for key in NUMERIC:
            a=np.array([number(r,key) for r in rows]); valid=a[np.isfinite(a)]
            median=float(np.median(valid)) if len(valid) else 0.
            filled=np.where(np.isfinite(a),a,median)
            self.state['numeric'][key]={'median':median,'mean':float(filled.mean()),'std':max(float(filled.std()),1e-6)}
            self.state['features'].extend([key,key+'__missing'])
        for key in CATEGORICAL:
            values=sorted({category(r,key) for r in rows}|{'__MISSING__','__UNKNOWN__'})
            self.state['categorical'][key]=values
            self.state['features'].extend([key+'='+v for v in values])
        return self
    def transform(self,rows):
        result=[]
        for r in rows:
            vector=[]
            for key,stats in self.state['numeric'].items():
                v=number(r,key);missing=not np.isfinite(v)
                vector.extend([((stats['median'] if missing else v)-stats['mean'])/stats['std'],float(missing)])
            for key,values in self.state['categorical'].items():
                v=category(r,key)
                if v not in values:v='__UNKNOWN__'
                vector.extend(float(v==x) for x in values)
            result.append(vector)
        return np.asarray(result,dtype=np.float32)

def create_splits(rows,out):
    # Union identical images and patient IDs before stratified grouping.
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    by_name={m['img_id']:m['sha256'] for m in manifest}
    parent={r['patient_id']:r['patient_id'] for r in rows}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    hashes={}
    for r in rows:
        h=by_name[r['img_id']]
        if h in hashes:parent[find(r['patient_id'])]=find(hashes[h])
        else:hashes[h]=r['patient_id']
    # Some bare lesion IDs recur under different patients (often different labels).
    # Conservatively put every linked patient in one partition rather than assuming
    # these are unrelated local-ID collisions.
    lesions={}
    for r in rows:
        lesion=r['lesion_id']
        if lesion in lesions:parent[find(r['patient_id'])]=find(lesions[lesion])
        else:lesions[lesion]=r['patient_id']
    y=np.array([CLASSES.index(r['diagnostic']) for r in rows])
    groups=np.array([find(r['patient_id']) for r in rows])
    splitter=StratifiedGroupKFold(7,shuffle=True,random_state=42)
    folds=[test for _,test in splitter.split(np.zeros(len(rows)),y,groups)]
    inds={'test':folds[0],'val':folds[1],'train':np.concatenate(folds[2:])}
    stats={}
    for name,idx in inds.items():
        selected=[rows[int(i)] for i in idx]
        write_rows(out/f'{name}.csv',selected)
        stats[name]={'images':len(idx),'patients':len({r['patient_id'] for r in selected}),
                     'classes':{c:sum(r['diagnostic']==c for r in selected) for c in CLASSES}}
        assert all(v>0 for v in stats[name]['classes'].values())
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not (set(groups[inds[a]]) & set(groups[inds[b]]))
        assert not ({by_name[rows[i]['img_id']] for i in inds[a]} & {by_name[rows[i]['img_id']] for i in inds[b]})
        assert not ({rows[i]['lesion_id'] for i in inds[a]} & {rows[i]['lesion_id'] for i in inds[b]})
    stats['method']='Seven stratified patient-group folds, seed=42: first fold test, second validation, five remaining training. Exact byte-identical images and patients sharing a bare lesion ID are conservatively grouped together. No score-based split selection.'
    stats['unique_image_hashes']=len(hashes)
    stats['patient_lesion_and_exact_duplicate_overlap']=False
    save_json(out/'split_summary.json',stats)
    return {name:[rows[int(i)] for i in idx] for name,idx in inds.items()},stats

class LesionDataset(Dataset):
    def __init__(self,rows,encoder,training=False):
        self.rows=rows;self.clinical=torch.from_numpy(encoder.transform(rows))
        self.images=[]
        for r in rows:
            with Image.open(ROOT/'data/images'/r['img_id']) as im:
                self.images.append(ImageOps.pad(im.convert('RGB'),(256,256),method=Image.Resampling.BILINEAR,color=(128,128,128)))
        operations=([transforms.RandomResizedCrop(224,scale=(0.85,1.0),ratio=(0.9,1.1)),
                     transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),
                     transforms.RandomRotation(20,fill=128),transforms.ColorJitter(0.1,0.1,0.1,0.02)] if training else [transforms.Resize((224,224))])
        self.transform=transforms.Compose(operations+[transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.transform(self.images[i]),self.clinical[i],CLASSES.index(self.rows[i]['diagnostic'])

def metrics(y,p):
    prediction=p.argmax(1)
    cm=confusion_matrix(y,prediction,labels=range(3))
    precision,recall,f1,support=precision_recall_fscore_support(y,prediction,labels=range(3),zero_division=0)
    per_class={}
    for k,c in enumerate(CLASSES):
        tp=int(cm[k,k]); fn=int(cm[k].sum()-tp); fp=int(cm[:,k].sum()-tp);tn=int(cm.sum()-tp-fn-fp)
        auc=float(roc_auc_score(y==k,p[:,k])) if 0<int((y==k).sum())<len(y) else None
        per_class[c]={'precision':float(precision[k]),'recall_sensitivity':float(recall[k]),'specificity':tn/(tn+fp) if tn+fp else 0.,'f1':float(f1[k]),'roc_auc_ovr':auc,'support_images':int(support[k]),'TP':tp,'FP':fp,'TN':tn,'FN':fn}
    return {'accuracy':float(accuracy_score(y,prediction)),'balanced_accuracy':float(balanced_accuracy_score(y,prediction)),
            'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),'macro_f1':float(f1.mean()),
            'weighted_f1':float(np.average(f1,weights=support)),
            'macro_specificity':float(np.mean([v['specificity'] for v in per_class.values()])),
            'macro_roc_auc_ovr':float(np.mean([v['roc_auc_ovr'] for v in per_class.values()])) if all(v['roc_auc_ovr'] is not None for v in per_class.values()) else None,
            'log_loss':float(log_loss(y,p,labels=range(3))),
            'multiclass_brier_score':float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),
            'confusion_matrix_rows_true_columns_predicted':cm.tolist(),'class_order':CLASSES,'per_class':per_class}

@torch.inference_mode()
def evaluate(model,loader,device,uniform=False,shuffle=False):
    model.eval();ys=[];ps=[];ws=[]
    shuffled=None
    if shuffle:
        g=torch.Generator().manual_seed(1042)
        shuffled=loader.dataset.clinical[torch.randperm(len(loader.dataset),generator=g)]
    offset=0
    for images,clinical,y in loader:
        if shuffled is not None:clinical=shuffled[offset:offset+len(y)];offset+=len(y)
        with torch.autocast('cuda',dtype=torch.float16):
            logits,w=model(images.to(device),clinical.to(device),uniform=uniform)
        p=torch.softmax(logits.float(),1).cpu().numpy()
        ys.append(y.numpy());ps.append(p);ws.append(w.float().cpu().numpy())
    return np.concatenate(ys),np.concatenate(ps),np.concatenate(ws)

def bootstrap(y,p,rows,n=1000):
    rng=np.random.default_rng(420)
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['patient_id'],[]).append(i)
    group_indices=list(groups.values());scores={'accuracy':[],'macro_f1':[],'macro_roc_auc_ovr':[]}
    for _ in range(n):
        ix=np.concatenate([group_indices[i] for i in rng.integers(len(groups),size=len(groups))])
        yy,pp=y[ix],p[ix]
        scores['accuracy'].append(float(accuracy_score(yy,pp.argmax(1))))
        scores['macro_f1'].append(float(f1_score(yy,pp.argmax(1),labels=range(3),average='macro',zero_division=0)))
        if len(np.unique(yy))==3:scores['macro_roc_auc_ovr'].append(float(roc_auc_score(yy,pp,multi_class='ovr',average='macro')))
    return {k:{'low':float(np.percentile(v,2.5)),'high':float(np.percentile(v,97.5)),'valid_replicates':len(v)} for k,v in scores.items()}

def figures(history,result,y,p,out):
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    epochs=[h['epoch'] for h in history]
    axes[0].plot(epochs,[h['train_loss'] for h in history],label='Training weighted loss')
    axes[0].plot(epochs,[h['val_log_loss'] for h in history],label='Validation unweighted log loss')
    axes[0].legend();axes[0].set_xlabel('Epoch');axes[0].set_title('Loss (different weighting)')
    axes[1].plot(epochs,[h['val_macro_f1'] for h in history],label='Macro F1')
    axes[1].plot(epochs,[h['val_accuracy'] for h in history],label='Accuracy');axes[1].legend();axes[1].set_ylim(0,1);axes[1].set_xlabel('Epoch');axes[1].set_title('Validation only')
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=170);plt.close(fig)
    cm=np.array(result['confusion_matrix_rows_true_columns_predicted'])
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='True',title='Held-out test confusion matrix')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k,c in enumerate(CLASSES):
        fpr,tpr,_=roc_curve(y==k,p[:,k]);axes[0].plot(fpr,tpr,label=f"{c}: AUC {result['per_class'][c]['roc_auc_ovr']:.3f}")
        precision,recall,_=precision_recall_curve(y==k,p[:,k]);axes[1].plot(recall,precision,label=f'{c}: AP {average_precision_score(y==k,p[:,k]):.3f}')
    axes[0].plot([0,1],[0,1],'k--',alpha=.4);axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='Test ROC curves')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Test precision-recall curves')
    for ax in axes:ax.legend();ax.set_xlim(0,1);ax.set_ylim(0,1.02)
    fig.tight_layout();fig.savefig(out/'roc_pr_curves.png',dpi=170);plt.close(fig)

def report(result,stats,config,history,out):
    per=''.join(f"<tr><td>{c}</td><td>{m['support_images']}</td><td>{m['precision']:.3f}</td><td>{m['recall_sensitivity']:.3f}</td><td>{m['specificity']:.3f}</td><td>{m['f1']:.3f}</td><td>{m['roc_auc_ovr']:.3f}</td></tr>" for c,m in result['per_class'].items())
    splits=''.join(f"<tr><td>{s}</td><td>{stats[s]['images']}</td><td>{stats[s]['patients']}</td><td>{stats[s]['classes']}</td></tr>" for s in ['train','val','test'])
    cis=''.join(f"<li>{k}: {v['low']:.3f}–{v['high']:.3f}</li>" for k,v in result['patient_bootstrap_95ci'].items())
    diag=''.join(f"<tr><td>{k}</td><td>{v['accuracy']:.3f}</td><td>{v['macro_f1']:.3f}</td></tr>" for k,v in result['routing_diagnostics'].items())
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>SCC / SEK / NEV training results</title><style>body{{font:16px/1.6 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#182837}}h1,h2{{color:#123e58}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #ccd6df}}th{{background:#e8f1f6}}.cards{{display:flex;gap:20px;flex-wrap:wrap}}.card{{padding:20px;background:#eef5f8;border-radius:12px}}.card b{{display:block;font-size:30px}}img{{max-width:100%}}code{{background:#eef1f4;padding:2px 5px}}</style></head><body>
<h1>Clinically guided three-class skin lesion classification</h1><p>PAD-UFES-20: SCC (192), SEK (235), NEV (244). Research prototype trained on the local NVIDIA GPU.</p>
<div class="cards"><div class="card">Test accuracy<b>{result['accuracy']:.1%}</b></div><div class="card">Macro F1<b>{result['macro_f1']:.3f}</b></div><div class="card">Macro ROC-AUC<b>{result['macro_roc_auc_ovr']:.3f}</b></div></div>
<h2>Data and leakage checks</h2><table><tr><th>Split</th><th>Images</th><th>Patients</th><th>Classes</th></tr>{splits}</table><p>{stats['method']} Patient, lesion and exact-image overlap: zero. Clinical imputation, normalization and categorical vocabulary fitted only on the training split. IDs, diagnosis and biopsy status excluded from clinical inputs. Missing values retained with indicators.</p>
<p>Clinical inputs: age, region, itch, hurt, bleed and elevation. Other fields remain in the saved metadata but are excluded from the router. Gender, diameters, skin type and cancer-history fields were missing in 80% of SEK and 69.3% of NEV images but 0% of SCC images; using their missingness could create a collection-related shortcut. Growth/change were also excluded for unequal missingness. The retained symptom fields have at most about 2% missingness per class; age and region are complete. This reduces, but does not eliminate, shortcut risks.</p>
<h2>Architecture</h2><p>224×224 RGB → U-Net contracting path (16,32,64,128 channels; three pools) → three image experts with dilation 1/2/3 → clinical softmax router → weighted 128×28×28 features → 1×1 adapter to 64 channels → all 16 ResNet50 residual bottleneck blocks → global pooling → dropout → three logits.</p><p>This is a ResNet50-based classifier with its original RGB stem and max-pool replaced by the adapter. It is not an unmodified ResNet50. U-Net has no decoder, segmentation output or segmentation supervision. ImageNet V2 weights initialize residual blocks; the encoder, experts, router, adapter and final head are newly trained.</p>
<h2>Training</h2><p>Seed 42; batch {config['batch_size']}; CUDA mixed precision; AdamW; class-weighted cross-entropy. Backbone frozen for the first {config['warmup_epochs']} epochs, then unfrozen; pretrained batch-normalization running statistics fixed. Augmentation applies only to training images. Best checkpoint selected by validation macro F1, breaking ties with validation log loss. Best epoch: {result['best_epoch']}; completed epochs: {len(history)}. Held-out test used only after selection.</p><p>GPU: {config['gpu']}. Training elapsed: {result['training_seconds']/60:.1f} minutes. Peak allocated GPU memory: {result['peak_gpu_memory_gib']:.2f} GiB.</p>
<img src="training_curves.png" alt="Training and validation curves"><h2>Held-out image-level metrics</h2><table><tr><th>Class</th><th>Images</th><th>Precision</th><th>Recall / sensitivity</th><th>Specificity</th><th>F1</th><th>ROC-AUC (OvR)</th></tr>{per}</table>
<p>Balanced accuracy: {result['balanced_accuracy']:.3f}; macro precision: {result['macro_precision']:.3f}; macro recall: {result['macro_recall']:.3f}; macro specificity: {result['macro_specificity']:.3f}; weighted F1: {result['weighted_f1']:.3f}; log loss: {result['log_loss']:.3f}.</p>
<img src="confusion_matrix.png" alt="Confusion matrix"><img src="roc_pr_curves.png" alt="ROC and precision recall curves">
<h2>Uncertainty</h2><p>95% percentile intervals from 1,000 patient-cluster bootstrap resamples of the held-out set (image-level metrics; within-patient correlation preserved):</p><ul>{cis}</ul>
<h2>Routing diagnostics</h2><table><tr><th>Inference condition</th><th>Accuracy</th><th>Macro F1</th></tr>{diag}</table><p>These are inference perturbations of the same trained model, not separately trained image-only baselines. They do not establish a causal clinical benefit. No hyperparameters or checkpoint selections were changed based on these test diagnostics.</p>
<h2>Limitations and scope</h2><p>One small patient-disjoint split, one seed, no external validation. The three labels cover only SCC, SEK and NEV; no healthy, melanoma, BCC, ACK or unknown category is recognized. SCC cases are biopsy-proven; many benign labels come from clinical consensus. Patient demographics and missingness can act as shortcuts. Routing weights describe mixture selection, not a validated explanation of diagnosis. This experiment is a research prototype, not a clinically validated diagnostic system.</p>
<h2>Files and reproducibility</h2><p><a href="metrics.json">Metrics JSON</a> · <a href="test_predictions.csv">Test predictions and routing weights</a> · <a href="history.csv">Epoch history</a> · <a href="config.json">Configuration</a> · <a href="split_summary.json">Split audit</a></p><p>Checkpoint: <code>best_model.pt</code>; preprocessing: <code>clinical_encoder.json</code>. See project <code>README.md</code> for commands.</p>
<h2>Source</h2><p>Pacheco et al. (2020), PAD-UFES-20, Data in Brief 32:106221. <a href="https://data.mendeley.com/datasets/zr7vgbcyr2/1">Official dataset</a> · <a href="https://doi.org/10.1016/j.dib.2020.106221">Dataset paper</a>. Data license: CC BY 4.0.</p></body></html>'''
    (out/'report.html').write_text(html,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=35);parser.add_argument('--batch-size',type=int,default=8);parser.add_argument('--patience',type=int,default=10);parser.add_argument('--run',default='run_01');parser.add_argument('--smoke',action='store_true');parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is required; refusing silent CPU fallback')
    torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda');out=ROOT/'outputs'/args.run;out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists() and not args.evaluate_only and not args.smoke:raise RuntimeError('Completed run exists; choose a new --run name')
    if args.evaluate_only:
        splits={}
        for name in ('train','val','test'):
            with (out/f'{name}.csv').open(newline='',encoding='utf-8') as f:splits[name]=list(csv.DictReader(f))
        stats=json.loads((out/'split_summary.json').read_text())
        encoder=ClinicalEncoder(json.loads((out/'clinical_encoder.json').read_text()))
    else:
        splits,stats=create_splits(load_rows(),out)
        encoder=ClinicalEncoder().fit(splits['train']);save_json(out/'clinical_encoder.json',encoder.state)
    dim=len(encoder.state['features'])
    trainset=LesionDataset(splits['train'],encoder,True)
    valset=LesionDataset(splits['val'],encoder)
    trainloader=DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(42))
    valloader=DataLoader(valset,batch_size=args.batch_size,num_workers=0,pin_memory=True)
    model=GuidedClassifier(dim,pretrained=not args.evaluate_only).to(device)
    config={'class_order':CLASSES,'seed':42,'epochs_max':args.epochs,'batch_size':args.batch_size,'patience':args.patience,'warmup_epochs':4,'clinical_dim':dim,'clinical_inputs':NUMERIC+CATEGORICAL,'new_layers_lr':0.0003,'backbone_lr':0.00002,'weight_decay':0.0001,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),'parameters':sum(p.numel() for p in model.parameters()),'selection':'validation macro F1; lower validation log loss breaks ties','split_method':stats['method'],'test_policy':'Untouched until checkpoint selection; no post-test tuning','pretrained':'torchvision ResNet50 IMAGENET1K_V2 residual blocks only','model':'U-Net encoder + three metadata-weighted experts + adapted ResNet50 residual body'}
    print(json.dumps({'gpu':config['gpu'],'parameters':config['parameters'],'clinical_dim':dim,'splits':stats}),flush=True)
    if args.smoke:
        model.train();model.fix_pretrained_bn()
        images,clinical,y=next(iter(trainloader))
        with torch.autocast('cuda',dtype=torch.float16): logits,w=model(images.to(device),clinical.to(device));loss=nn.functional.cross_entropy(logits,y.to(device))
        loss.backward()
        assert logits.shape==(len(y),3) and torch.isfinite(loss)
        assert torch.allclose(w.sum(1).float(),torch.ones(len(y),device=device),atol=.002)
        for name,module in [('encoder',model.encoder),('experts',model.experts),('router',model.router),('resnet',model.resnet)]:
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters()),name+' has no gradient'
        print(f'SMOKE PASS: forward/backward on CUDA; all branches receive gradients; peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB',flush=True)
        return
    if args.evaluate_only:
        checkpoint=torch.load(out/'best_model.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model_state']);config=json.loads((out/'config.json').read_text());history=json.loads((out/'history.json').read_text());elapsed=history[-1]['elapsed_seconds']
    else:
        save_json(out/'config.json',config)
        backbone=[p for n,p in model.named_parameters() if n.startswith('resnet.') and not n.startswith('resnet.fc.')]
        rest=[p for n,p in model.named_parameters() if not n.startswith('resnet.') or n.startswith('resnet.fc.')]
        optimizer=torch.optim.AdamW([{'params':rest,'lr':config['new_layers_lr']},{'params':backbone,'lr':config['backbone_lr']}],weight_decay=config['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4,min_lr=1e-7)
        scaler=torch.amp.GradScaler('cuda')
        counts=np.array([stats['train']['classes'][c] for c in CLASSES]);weight=torch.tensor(counts.sum()/(3*counts),dtype=torch.float32,device=device)
        criterion=nn.CrossEntropyLoss(weight=weight)
        best=-1.;best_loss=float('inf');stale=0;history=[];start=time.monotonic()
        for epoch in range(1,args.epochs+1):
            model.freeze_residual_backbone(epoch<=config['warmup_epochs'])
            model.train();model.fix_pretrained_bn();loss_sum=0.;seen=0;correct=0;t=time.monotonic()
            for images,clinical,y in trainloader:
                images,clinical,y=images.to(device,non_blocking=True),clinical.to(device,non_blocking=True),y.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits,w=model(images,clinical);loss=criterion(logits,y)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5.0);scaler.step(optimizer);scaler.update()
                loss_sum+=loss.item()*len(y);seen+=len(y);correct+=(logits.argmax(1)==y).sum().item()
            yy,pp,ww=evaluate(model,valloader,device);m=metrics(yy,pp)
            row={'epoch':epoch,'train_loss':loss_sum/seen,'train_accuracy_augmented':correct/seen,'val_accuracy':m['accuracy'],'val_macro_f1':m['macro_f1'],'val_log_loss':m['log_loss'],'epoch_seconds':time.monotonic()-t,'elapsed_seconds':time.monotonic()-start,'lr_new':optimizer.param_groups[0]['lr'],'lr_backbone':optimizer.param_groups[1]['lr']}
            history.append(row);save_json(out/'history.json',history);write_rows(out/'history.csv',history)
            improved=m['macro_f1']>best+1e-6 or (abs(m['macro_f1']-best)<1e-6 and m['log_loss']<best_loss)
            if improved:
                best=m['macro_f1'];best_loss=m['log_loss'];stale=0
                state={'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'clinical_encoder':encoder.state,'config':config,'epoch':epoch,'validation_metrics':m}
                torch.save(state,out/'best_model.pt')
            else:stale+=1
            scheduler.step(m['macro_f1'])
            print(f"Epoch {epoch:02d}/{args.epochs}: train loss={row['train_loss']:.4f}, val accuracy={m['accuracy']:.3f}, val macro F1={m['macro_f1']:.3f}, {row['epoch_seconds']:.1f}s"+(' BEST' if improved else ''),flush=True)
            if epoch>=14 and stale>=args.patience:
                print('Early stopping based only on validation macro F1.',flush=True);break
        elapsed=time.monotonic()-start
        checkpoint=torch.load(out/'best_model.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint selected. Evaluating held-out test once; no further training.',flush=True)
    testset=LesionDataset(splits['test'],encoder);testloader=DataLoader(testset,batch_size=args.batch_size,num_workers=0,pin_memory=True)
    y,p,w=evaluate(model,testloader,device);result=metrics(y,p)
    result.update({'best_epoch':checkpoint['epoch'],'training_seconds':elapsed,'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'test_images':len(y),'test_patients':stats['test']['patients'],'patient_bootstrap_95ci':bootstrap(y,p,splits['test'])})
    predictions=[]
    for i,r in enumerate(splits['test']):
        predictions.append({'img_id':r['img_id'],'patient_id':r['patient_id'],'lesion_id':r['lesion_id'],'true_class':r['diagnostic'],'predicted_class':CLASSES[int(p[i].argmax())],**{f'prob_{c}':float(p[i,k]) for k,c in enumerate(CLASSES)},**{f'expert_{k+1}_weight':float(w[i,k]) for k in range(3)}})
    write_rows(out/'test_predictions.csv',predictions)
    diagnostics={'clinical_routing':metrics(y,p)}
    for name,kwargs in [('uniform_routing',{'uniform':True}),('shuffled_clinical_data',{'shuffle':True})]:
        yy,pp,_=evaluate(model,testloader,device,**kwargs);diagnostics[name]=metrics(yy,pp)
    result['routing_diagnostics']=diagnostics
    result['routing_mean_weights']=w.mean(0).tolist()
    save_json(out/'metrics.json',result)
    figures(history,result,y,p,out);report(result,stats,config,history,out)
    print('FINAL TEST: '+json.dumps({k:result[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','macro_specificity','macro_roc_auc_ovr','best_epoch','training_seconds']}),flush=True)
    print('COMPLETE: '+str(out/'report.html'),flush=True)

if __name__=='__main__':main()
