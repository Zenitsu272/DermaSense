"""Validation-selected clinical late fusion and regularized feature-head training.

Image features are computed on CUDA; small convex classification heads are fitted
on CPU. No test predictions are computed until the candidate and refit are locked.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from improved_model import ImprovedClassifier
from train import CLASSES, ClinicalEncoder, bootstrap, metrics, save_json, write_rows


class FeatureModel(ImprovedClassifier):
    def encode(self, images, clinical):
        shared = self.encoder(images)
        weights = self.router(clinical).softmax(1)
        guided = sum(e(shared) * weights[:, i, None, None, None] for i, e in enumerate(self.experts))
        r = self.resnet
        x = r.maxpool(r.relu(r.bn1(r.conv1(images))))
        x = r.layer2(r.layer1(x)) + guided
        return r.avgpool(r.layer4(r.layer3(x))).flatten(1)


class Images(Dataset):
    def __init__(self, workspace, rows, encoder, size):
        self.clinical = torch.from_numpy(encoder.transform(rows))
        self.labels = [CLASSES.index(r['diagnostic']) for r in rows]
        self.images = []
        for row in rows:
            with Image.open(workspace / 'data/images' / row['img_id']) as im:
                self.images.append(ImageOps.pad(im.convert('RGB'), (round(size*8/7),)*2,
                                               method=Image.Resampling.BILINEAR, color=(128,)*3))
        self.transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                             transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.clinical[i], self.labels[i]


def rows_for(workspace, split):
    with (workspace / 'outputs/run_01' / f'{split}.csv').open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def base_checkpoint(workspace, size, refit):
    if size == 224:
        path = workspace / ('outputs/improvement_01/refit/best_system.pt' if refit else
                            'outputs/improvement_01/seed_1337/best_model.pt')
    else:
        path = workspace / ('outputs/high_resolution_01/refit/best_model.pt' if refit else
                            'outputs/high_resolution_01/search/best_model.pt')
    cp = torch.load(path, map_location='cpu', weights_only=False)
    state = cp['model_states'][0] if 'model_states' in cp else cp['model_state']
    return path, state, ClinicalEncoder(cp['clinical_encoder'])


@torch.inference_mode()
def extract(workspace, rows, state, encoder, size):
    model = FeatureModel(len(encoder.state['features']), pretrained=False).to(device='cuda', memory_format=torch.channels_last)
    model.load_state_dict(state)
    model.eval()
    loader = DataLoader(Images(workspace, rows, encoder, size), batch_size=8, pin_memory=True)
    fs, ps = [], []
    for x, c, _ in loader:
        x = x.cuda(memory_format=torch.channels_last); c = c.cuda()
        views, probs = [], []
        for axes in [(), (-1,), (-2,), (-2,-1)]:
            with torch.autocast('cuda', dtype=torch.float16):
                features = model.encode(x.flip(axes) if axes else x, c)
                logits = model.resnet.fc(features)
            views.append(features.float()); probs.append(logits.float().softmax(1))
        fs.append(torch.stack(views).mean(0).cpu().numpy())
        ps.append(torch.stack(probs).mean(0).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(fs), np.concatenate(ps)


def feature_matrix(image, clinical, scale):
    image = image / np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-8)
    return np.concatenate([image, clinical * scale], axis=1)


def fit_head(x, y, c):
    with threadpool_limits(limits=4):
        head = LogisticRegression(C=c, solver='lbfgs', max_iter=2000, tol=1e-6, random_state=1337).fit(x, y)
    assert list(head.classes_) == [0,1,2]
    if int(head.n_iter_.max()) >= 2000:
        raise RuntimeError('Head optimization did not converge')
    return head


def key(candidate):
    m = candidate['validation']
    return m['accuracy'], m['macro_f1'], -m['log_loss']


def search(workspace, out):
    if (out/'selection.json').exists():
        raise RuntimeError('Completed search preserved')
    protocol = {
        'clinical_inputs': ['age','region','itch','hurt','bleed','elevation'],
        'image_sizes': [224,320], 'seed': 1337,
        'feature_head_C': [.1,1.,10.,100.], 'clinical_feature_scales': [0.,.1,.3,1.,3.],
        'clinical_head_C': [.1,1.,10.], 'clinical_probability_blend_weights': [.1,.2,.3,.4],
        'image_views': ['original','horizontal flip','vertical flip','both flips'],
        'criterion': 'validation accuracy, macro F1, lower log loss',
        'fixed_split_sha256': {s: hashlib.sha256((workspace/'outputs/run_01'/f'{s}.csv').read_bytes()).hexdigest() for s in ['train','val','test']},
        'test_policy': 'No test rows loaded by search. Same previously inspected benchmark; not independent external validation.',
        'fit_policy': 'Frozen previously trained U-Net/ResNet encoders, CUDA feature extraction, CPU L2-regularized multinomial logistic heads; no test-based thresholds.'
    }
    save_json(out/'protocol.json', protocol)
    train_rows, val_rows = rows_for(workspace,'train'), rows_for(workspace,'val')
    ytr = np.array([CLASSES.index(r['diagnostic']) for r in train_rows])
    yv = np.array([CLASSES.index(r['diagnostic']) for r in val_rows])
    candidates = []
    for size in protocol['image_sizes']:
        path, state, enc = base_checkpoint(workspace,size,False)
        save_json(out/f'base_{size}.json', {'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'clinical_encoder':enc.state})
        cache = out/f'features_{size}.npz'
        if cache.exists():
            a = np.load(cache); ftr,p_tr,fv,pv = [a[k] for k in ['train_features','train_probs','val_features','val_probs']]
        else:
            print(f'CUDA feature extraction: {size}px, training and validation only', flush=True)
            f,p = extract(workspace,train_rows+val_rows,state,enc,size)
            ftr,fv = f[:len(ytr)],f[len(ytr):]; p_tr,pv = p[:len(ytr)],p[len(ytr):]
            np.savez(cache,train_features=ftr,train_probs=p_tr,val_features=fv,val_probs=pv)
        ctr,cv = enc.transform(train_rows),enc.transform(val_rows)
        candidates.append({'name':f'image_{size}','kind':'image','size':size,'validation':metrics(yv,pv)})
        for c in protocol['clinical_head_C']:
            head = fit_head(ctr,ytr,c); pc = head.predict_proba(cv)
            for alpha in protocol['clinical_probability_blend_weights']:
                candidates.append({'name':f'blend_{size}_C{c}_a{alpha}','kind':'blend','size':size,'C':c,'alpha':alpha,'validation':metrics(yv,(1-alpha)*pv+alpha*pc)})
        for scale in protocol['clinical_feature_scales']:
            for c in protocol['feature_head_C']:
                head = fit_head(feature_matrix(ftr,ctr,scale),ytr,c)
                p = head.predict_proba(feature_matrix(fv,cv,scale))
                candidates.append({'name':f'feature_{size}_C{c}_scale{scale}','kind':'feature','size':size,'C':c,'scale':scale,'validation':metrics(yv,p)})
        print(f'{size}px candidates complete; best validation accuracy so far: {max(c["validation"]["accuracy"] for c in candidates):.4f}', flush=True)
    candidates.sort(key=key,reverse=True)
    save_json(out/'selection.json',{'selected':candidates[0],'candidates':candidates,'test_accessed':False})
    print('SELECTED '+json.dumps(candidates[0]),flush=True)


def refit(workspace,out):
    if (out/'best_system.pt').exists():
        raise RuntimeError('Completed refit preserved')
    chosen = json.loads((out/'selection.json').read_text())['selected']
    rows = rows_for(workspace,'train') + rows_for(workspace,'val')
    path,state,enc = base_checkpoint(workspace,chosen['size'],True)
    head_state = None
    if chosen['kind'] != 'image':
        c = enc.transform(rows)
        if chosen['kind'] == 'feature':
            f,_ = extract(workspace,rows,state,enc,chosen['size'])
            x = feature_matrix(f,c,chosen['scale'])
        else:
            x = c
        head = fit_head(x,np.array([CLASSES.index(r['diagnostic']) for r in rows]),chosen['C'])
        head_state = {'weight':torch.from_numpy(head.coef_.astype(np.float32)), 'bias':torch.from_numpy(head.intercept_.astype(np.float32))}
    cp = {'class_order':CLASSES,'selected':chosen,'clinical_encoder':enc.state,'image_model':state,'head':head_state,
          'refit':{'images':len(rows),'patients':len({r['patient_id'] for r in rows}),'base_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'validation_score':'None for refit; original validation rows are now training rows'}}
    torch.save(cp,out/'best_system.pt')
    save_json(out/'refit.json',cp['refit'])
    print('REFIT COMPLETE '+chosen['name'],flush=True)


def combined_probabilities(cp,features,clinical,image_probs):
    chosen=cp['selected']
    if chosen['kind']=='image':return image_probs
    x=feature_matrix(features,clinical,chosen['scale']) if chosen['kind']=='feature' else clinical
    head=cp['head'];z=x@head['weight'].numpy().T+head['bias'].numpy();z=z-z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
    return (1-chosen['alpha'])*image_probs+chosen['alpha']*p if chosen['kind']=='blend' else p


def predict_package(workspace,rows,cp):
    predictions=[]
    for component in cp.get('components',[cp]):
        enc=ClinicalEncoder(component['clinical_encoder'])
        f,ip=extract(workspace,rows,component['image_model'],enc,component['selected']['size'])
        predictions.append(combined_probabilities(component,f,enc.transform(rows),ip))
    return np.mean(predictions,axis=0)


def evaluate(workspace,out):
    if (out/'metrics.json').exists():raise RuntimeError('Completed evaluation preserved')
    protocol=json.loads((out/'protocol.json').read_text())
    for split,h in protocol['fixed_split_sha256'].items():
        assert hashlib.sha256((workspace/'outputs/run_01'/f'{split}.csv').read_bytes()).hexdigest()==h
    cp=torch.load(out/'best_system.pt',map_location='cpu',weights_only=False)
    save_json(out/'evaluation_lock.json',{'checkpoint_sha256':hashlib.sha256((out/'best_system.pt').read_bytes()).hexdigest(),'source_sha256':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['clinical_fusion.py','predict_clinical_fusion.py','robust_finetune.py','ensemble_followup.py']},'selected':cp['selected'],'test_sha256':protocol['fixed_split_sha256']['test']})
    rows=rows_for(workspace,'test');dev=rows_for(workspace,'train')+rows_for(workspace,'val')
    for k in ['patient_id','lesion_id']:assert not ({r[k] for r in rows}&{r[k] for r in dev})
    hashes={r['img_id']:r['sha256'] for r in json.loads((workspace/'data/manifest.json').read_text())}
    assert not ({hashes[r['img_id']] for r in rows}&{hashes[r['img_id']] for r in dev})
    p=predict_package(workspace,rows,cp)
    y=np.array([CLASSES.index(r['diagnostic']) for r in rows]);m=metrics(y,p)
    m.update(test_images=len(rows),test_patients=len({r['patient_id'] for r in rows}),correct=int((y==p.argmax(1)).sum()),goal_over_90_percent_met=bool(m['accuracy']>.90),patient_bootstrap_95ci=bootstrap(y,p,rows),selected=cp['selected']['name'],test_note='Previously inspected test benchmark; not independent external validation.',validation_note='Selection score belongs to train-only model, not development refit.')
    save_json(out/'metrics.json',m)
    write_rows(out/'test_predictions.csv',[{'img_id':r['img_id'],'patient_id':r['patient_id'],'true_class':r['diagnostic'],'predicted_class':CLASSES[int(p[i].argmax())],**{f'prob_{c}':float(p[i,k]) for k,c in enumerate(CLASSES)}} for i,r in enumerate(rows)])
    print('FINAL '+json.dumps(m),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['search','refit','evaluate']);ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--run',default='clinical_fusion_01');args=ap.parse_args()
    assert torch.cuda.is_available(),'CUDA required';torch.set_num_threads(4);torch.manual_seed(1337);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    workspace=args.workspace.resolve();out=workspace/'outputs'/args.run;out.mkdir(parents=True,exist_ok=True)
    {'search':search,'refit':refit,'evaluate':evaluate}[args.mode](workspace,out)


if __name__=='__main__':main()
