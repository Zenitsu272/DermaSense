"""Portable inference and evaluation UI for the three-cell DermaSense notebook.

Reference labels are used only after prediction, to score uploaded demo images.
The network receives RGB tensors and the six saved clinical input fields only.
"""
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import html
import io
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

CLASSES = ['SCC', 'SEK', 'NEV']
FULL_NAMES = {'SCC': 'Squamous cell carcinoma', 'SEK': 'Seborrheic keratosis', 'NEV': 'Nevus'}
CLINICAL_FIELDS = ['age', 'region', 'itch', 'hurt', 'bleed', 'elevation']


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DemoModel(nn.Module):
    def __init__(self, clinical_dim):
        super().__init__()
        self.encoder = nn.Sequential(DoubleConv(3,16),nn.MaxPool2d(2),DoubleConv(16,32),nn.MaxPool2d(2),
                                     DoubleConv(32,64),nn.MaxPool2d(2),DoubleConv(64,128))
        self.experts = nn.ModuleList([nn.Sequential(nn.Conv2d(128,64,3,padding=d,dilation=d,bias=False),
                                                  nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,512,1)) for d in (1,2,3)])
        self.router = nn.Sequential(nn.Linear(clinical_dim,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,3))
        self.resnet = resnet50(weights=None)  # Never downloads or trains a model.
        self.resnet.fc = nn.Sequential(nn.Dropout(.3),nn.Linear(2048,3))

    def forward(self, x, clinical):
        shared = self.encoder(x)
        weights = self.router(clinical).softmax(1)
        guided = sum(expert(shared)*weights[:,i,None,None,None] for i,expert in enumerate(self.experts))
        r = self.resnet
        x = r.maxpool(r.relu(r.bn1(r.conv1(x))))
        x = r.layer2(r.layer1(x)) + guided
        x = r.layer4(r.layer3(x))
        return r.fc(r.avgpool(x).flatten(1))


def encode_clinical(rows, state):
    vectors = []
    for row in rows:
        vector = []
        for key, stats in state['numeric'].items():
            try:
                value = float(row.get(key, ''))
            except (TypeError, ValueError):
                value = float('nan')
            missing = not np.isfinite(value)
            vector.extend([((stats['median'] if missing else value)-stats['mean'])/stats['std'], float(missing)])
        for key, choices in state['categorical'].items():
            value = str(row.get(key, '')).strip()
            if value.upper() in {'', 'UNK', 'NAN', 'NONE', 'NULL'}:
                value = '__MISSING__'
            if value not in choices:
                value = '__UNKNOWN__'
            vector.extend(float(value == choice) for choice in choices)
        vectors.append(vector)
    return np.asarray(vectors, dtype=np.float32)


def upload_items(selection):
    """Normalize Colab dicts and ipywidgets 7/8 upload values."""
    value = getattr(selection, 'value', selection)
    if isinstance(value, dict):
        return [(str(name), bytes(item['content'] if isinstance(item, dict) else item)) for name,item in value.items()]
    return [(str(item['name']), bytes(item['content'])) for item in value]


def pick_images():
    from IPython.display import display, HTML
    display(HTML('<h3>2 · Choose images</h3><p>Select any number of images from the supplied <b>images</b> folder, then run cell 3. '
                 'Demo clinical data and labels are matched automatically, including renamed files.</p>'))
    try:
        from google.colab import files
    except ImportError:
        import ipywidgets as widgets
        picker = widgets.FileUpload(accept='.png,.jpg,.jpeg,.webp,.bmp', multiple=True, description='Choose images',
                                    layout=widgets.Layout(width='210px'))
        status = widgets.HTML('No images selected.')
        def changed(change):
            count = len(upload_items(change['new']))
            status.value = f'<b>{count}</b> image(s) selected. Run cell 3 to predict.'
        picker.observe(changed, names='value')
        display(widgets.VBox([picker, status]))
        return picker
    else:
        return files.upload()


def score_predictions(records):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
    labelled = [r for r in records if r['true_class'] in CLASSES]
    if not labelled:
        return None
    y = np.array([CLASSES.index(r['true_class']) for r in labelled])
    yp = np.array([CLASSES.index(r['predicted_class']) for r in labelled])
    p = np.array([[r['prob_'+c] for c in CLASSES] for r in labelled], dtype=np.float32)
    present = np.unique(y)
    precision, recall, f1, _ = precision_recall_fscore_support(y, yp, labels=present, zero_division=0)
    cm = confusion_matrix(y, yp, labels=range(3))
    per_class = {}
    aucs = []
    for k,c in enumerate(CLASSES):
        tp = int(cm[k,k]); support = int(cm[k].sum()); predicted = int(cm[:,k].sum())
        fn = support-tp; fp = predicted-tp; tn = len(y)-tp-fn-fp
        auc = float(roc_auc_score(y == k, p[:,k])) if 0 < support < len(y) else None
        if auc is not None:
            aucs.append(auc)
        per_class[c] = {'support':support, 'precision':tp/predicted if predicted else None,
                        'recall':tp/support if support else None,
                        'f1':2*tp/(2*tp+fp+fn) if support else None,
                        'specificity':tn/(tn+fp) if tn+fp else None, 'roc_auc':auc}
    return {'accuracy':float((y==yp).mean()), 'correct':int((y==yp).sum()), 'labelled_images':len(y),
            'predicted_images':len(records), 'unlabelled_images':len(records)-len(y),
            'macro_precision':float(precision.mean()), 'macro_recall':float(recall.mean()),
            'macro_f1':float(f1.mean()), 'balanced_accuracy':float(recall.mean()),
            'macro_roc_auc':float(np.mean(aucs)) if aucs else None,
            'macro_classes':[CLASSES[int(i)] for i in present], 'class_order':CLASSES,
            'confusion_matrix':cm.tolist(), 'per_class':per_class,
            'scope':'Metrics describe this uploaded labelled subset, not a new independent test cohort.'}


class Demo:
    def __init__(self, asset_dir, device=None):
        started = time.perf_counter()
        self.asset_dir = Path(asset_dir)
        manifest = json.loads((self.asset_dir/'asset_manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['sha256'].items():
            actual = hashlib.sha256((self.asset_dir/name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Asset verification failed: {name}. Recopy the original assets ZIP.')
        checkpoint = torch.load(self.asset_dir/'demo_model.pt', map_location='cpu', weights_only=True)
        if checkpoint['class_order'] != CLASSES:
            raise ValueError('Unexpected checkpoint class order')
        self.encoder = checkpoint['clinical_encoder']
        self.tta = checkpoint['tta']
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.model = DemoModel(len(self.encoder['features']))
        self.model.load_state_dict(checkpoint['model_state'], strict=True)
        self.model.to(device=self.device, memory_format=torch.channels_last).eval()
        self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                              transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        records = json.loads((self.asset_dir/'demo_records.json').read_text(encoding='utf-8'))['records']
        self.clinical_by_hash = {r['sha256']:{k:r['clinical'].get(k,'') for k in CLINICAL_FIELDS} for r in records}
        self.labels_by_hash = {r['sha256']:r['label'] for r in records}
        # Initialize CUDA libraries in the load cell, before the live upload test.
        if self.device.type == 'cuda':
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
                warm_image=torch.zeros(8,3,224,224,device=self.device).to(memory_format=torch.channels_last)
                warm_clinical=torch.zeros(8,len(self.encoder['features']),device=self.device)
                self.model(warm_image,warm_clinical)
            torch.cuda.synchronize()
        self.model_load_seconds = time.perf_counter()-started
        self.last_result = None

    @torch.inference_mode()
    def predict(self, files):
        from PIL import ImageOps
        started = time.perf_counter()
        samples, skipped, seen = [], [], set()
        for filename, payload in files:
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen:
                skipped.append({'filename':filename,'reason':'Duplicate image content; counted once.'})
                continue
            try:
                with Image.open(io.BytesIO(payload)) as im:
                    raw = im.convert('RGB')
                    thumbnail = raw.copy(); thumbnail.thumbnail((240,180))
                    padded = ImageOps.pad(raw,(256,256),method=Image.Resampling.BILINEAR,color=(128,128,128))
                    tensor = self.transform(padded)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                skipped.append({'filename':filename,'reason':f'Unreadable image ({type(exc).__name__}).'})
                continue
            seen.add(digest)
            samples.append({'filename':filename,'sha256':digest,'tensor':tensor,'thumbnail':thumbnail,
                            'clinical':self.clinical_by_hash.get(digest,{})})
        records = []
        for start in range(0,len(samples),8):
            batch = samples[start:start+8]
            x = torch.stack([s['tensor'] for s in batch]).to(device=self.device,memory_format=torch.channels_last)
            c = torch.from_numpy(encode_clinical([s['clinical'] for s in batch],self.encoder)).to(self.device)
            views = []
            for axes in ([(),(-1,),(-2,),(-2,-1)] if self.tta else [()]):
                context = torch.autocast('cuda',dtype=torch.float16) if self.device.type=='cuda' else nullcontext()
                with context:
                    logits = self.model(x.flip(axes) if axes else x,c)
                views.append(logits.float().softmax(1))
            probabilities = torch.stack(views).mean(0).cpu().numpy()
            for sample,p in zip(batch,probabilities):
                # Reference labels enter scoring here, AFTER the network has predicted.
                label = self.labels_by_hash.get(sample['sha256'])
                predicted = CLASSES[int(p.argmax())]
                records.append({'filename':sample['filename'],'true_class':label,'predicted_class':predicted,
                                'confidence':float(p.max()),'correct':predicted==label if label else None,
                                'clinical_available':bool(sample['clinical']),
                                **{'prob_'+c:float(p[k]) for k,c in enumerate(CLASSES)}})
        return {'predictions':records,'metrics':score_predictions(records),'skipped':skipped,
                'seconds':time.perf_counter()-started,'thumbnails':[s['thumbnail'] for s in samples]}


def load_demo(asset_dir):
    from IPython.display import display,HTML
    demo = Demo(asset_dir)
    device = torch.cuda.get_device_name(0) if demo.device.type=='cuda' else 'CPU (no GPU required)'
    display(HTML('<h2>DermaSense · Live evaluation</h2>'
                 '<p><b>SCC</b> — squamous cell carcinoma &nbsp; · &nbsp; <b>SEK</b> — seborrheic keratosis '
                 '&nbsp; · &nbsp; <b>NEV</b> — nevus</p>'
                 f'<p>Model ready on <b>{html.escape(device)}</b>. Loading: {demo.model_load_seconds:.1f} seconds. No training.</p>'
                 '<p>Reference result: <b>87.50% on the original 96-image test set</b>. '
                 'The next cells measure only the images you choose.</p>'))
    return demo


def run_demo(demo, selection, output_root='demo_outputs'):
    from IPython.display import display,HTML,FileLink
    import pandas as pd
    import matplotlib.pyplot as plt
    files = upload_items(selection)
    if not files:
        print('Choose at least one image in cell 2, then rerun this cell.')
        return None
    result = demo.predict(files)
    rows, m = result['predictions'], result['metrics']
    if not rows:
        print('No readable images were selected. Choose PNG or JPEG images and try again.')
        for item in result['skipped']:
            print(item['filename']+': '+item['reason'])
        return result
    out = Path(output_root)/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    exported={k:v for k,v in result.items() if k!='thumbnails'}
    (out/'metrics.json').write_text(json.dumps(exported,indent=2,allow_nan=False),encoding='utf-8')
    display(HTML(f'<h3>3 · Results for {len(rows)} image(s)</h3><p>Inference and scoring: {result["seconds"]:.2f} seconds.</p>'))
    if m:
        cards = [('Accuracy',m['accuracy']),('Macro precision',m['macro_precision']),('Macro recall',m['macro_recall']),('Macro F1',m['macro_f1'])]
        display(HTML('<div style="display:flex;flex-wrap:wrap;gap:12px">'+''.join(
            f'<div style="padding:12px 20px;border:1px solid #7795a7;border-radius:8px"><div>{name}</div>'
            f'<b style="font-size:25px">{value:.2%}</b></div>' for name,value in cards)+'</div>'
            f'<p><b>{m["correct"]}/{m["labelled_images"]}</b> labelled images correct. '
            f'Macro metrics use the true classes present in this upload: {", ".join(m["macro_classes"])}. '
            'This small demo subset is not a new independent test score.</p>'))
    else:
        display(HTML('<p><b>Accuracy unavailable:</b> these images have no matching reference labels. '
                     'Predictions are shown, but labels cannot be inferred from the predictions.</p>'))
    if any(not r['clinical_available'] for r in rows):
        display(HTML('<p>Unrecognized images use missing clinical inputs and are excluded from accuracy unless a reference label is available. '
                     'Use the supplied demo images for the complete image + clinical evaluation.</p>'))
    table = pd.DataFrame([{'Image':r['filename'],'Reference':r['true_class'] or 'Unknown','Prediction':r['predicted_class'],
                           'Confidence':f'{r["confidence"]:.1%}','Result':('Correct' if r['correct'] else 'Incorrect') if r['correct'] is not None else 'Not scored',
                           **{f'P({c})':f'{r["prob_"+c]:.3f}' for c in CLASSES}} for r in rows])
    display(table.style.hide(axis='index'))
    if m:
        per = pd.DataFrame([{'Class':c,**v} for c,v in m['per_class'].items()])
        display(HTML('<h4>Per-class metrics</h4>'))
        display(per.style.hide(axis='index').format(precision=3,na_rep='—'))
        cm=np.array(m['confusion_matrix']);fig,ax=plt.subplots(figsize=(4.7,4))
        ax.imshow(cm,cmap='Blues');ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Reference',title='Uploaded labelled images')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=150);display(fig);plt.close(fig)
    count=min(len(rows),40);cols=min(count,5);fig,axes=plt.subplots(math.ceil(count/cols),cols,figsize=(3*cols,2.7*math.ceil(count/cols)),squeeze=False)
    for ax in axes.flat:ax.axis('off')
    for ax,row,thumbnail in zip(axes.flat,rows[:count],result['thumbnails'][:count]):
        ax.imshow(thumbnail);color='#176344' if row['correct'] else '#a33e28' if row['correct'] is not None else '#345'
        ax.set_title(f'{row["predicted_class"]} · {row["confidence"]:.0%}\nReference: {row["true_class"] or "unknown"}',color=color,fontsize=10)
    fig.tight_layout();fig.savefig(out/'prediction_gallery.png',dpi=130);display(fig);plt.close(fig)
    if result['skipped']:
        print('Skipped:', '; '.join(x['filename']+' — '+x['reason'] for x in result['skipped']))
    print('Saved this run:',out)
    display(FileLink(str(out/'predictions.csv')));display(FileLink(str(out/'metrics.json')))
    result['output_dir']=str(out);demo.last_result=result
    return result
