"""Progressive 320px fine-tuning; validation selection precedes refit and evaluation."""
import argparse,csv,hashlib,json,random,time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image,ImageOps
from train import ROOT,CLASSES,ClinicalEncoder,metrics,bootstrap,save_json,write_rows
from improve import SEARCH,rows_for,predict,select_key
from improved_model import ImprovedClassifier

OUT=ROOT/'outputs/high_resolution_01'

def transform(size,training=False):
    ops=[transforms.RandomResizedCrop(size,scale=(.85,1),ratio=(.9,1.1)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(20,fill=128),transforms.ColorJitter(.1,.1,.1,.02)] if training else [transforms.Resize((size,size))]
    return transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

class Images(Dataset):
    def __init__(self,rows,encoder,training=False,size=320):
        self.rows=rows;self.c=torch.from_numpy(encoder.transform(rows));self.images=[];self.transform=transform(size,training)
        for r in rows:
            with Image.open(ROOT/'data/images'/r['img_id']) as im:self.images.append(ImageOps.pad(im.convert('RGB'),(round(size*8/7),)*2,method=Image.Resampling.BILINEAR,color=(128,)*3))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.transform(self.images[i]),self.c[i],CLASSES.index(self.rows[i]['diagnostic'])

def setup():
    random.seed(1337);np.random.seed(1337);torch.manual_seed(1337);torch.cuda.manual_seed_all(1337);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    OUT.mkdir(exist_ok=True)
    original=json.loads((SEARCH/'protocol.json').read_text())
    for s,h in original['fixed_split_hashes'].items():assert hashlib.sha256((ROOT/'outputs/run_01'/f'{s}.csv').read_bytes()).hexdigest()==h

def fit(refit=False):
    setup();out=OUT/('refit' if refit else 'search');out.mkdir(exist_ok=True)
    if (out/'completed.json').exists():raise RuntimeError('Completed run preserved')
    if refit:
        selected=json.loads((OUT/'selection.json').read_text())['selected']
        if selected['high_weight']==0:raise RuntimeError('Higher resolution did not win validation')
        cp=torch.load(SEARCH/'refit/best_system.pt',map_location='cpu',weights_only=False);state=cp['model_states'][0];enc=ClinicalEncoder(cp['clinical_encoder']);epochs=json.loads((OUT/'search/completed.json').read_text())['best_epoch'];rows=rows_for('train')+rows_for('val')
    else:
        cp=torch.load(SEARCH/'seed_1337/best_model.pt',map_location='cpu',weights_only=False);state=cp['model_state'];enc=ClinicalEncoder(cp['clinical_encoder']);epochs=20;rows=rows_for('train')
    config={'size':320,'batch_size':8,'seed':1337,'max_epochs':epochs,'backbone_lr':1e-5,'new_lr':1e-4,'weight_decay':1e-3,'label_smoothing':.03,'training_images':len(rows),'starting_checkpoint':'575-image refit' if refit else '479-image validation-selected seed 1337','selection':'validation accuracy, macro F1, lower log loss; single/four-view/50-50 multiresolution; final refit duration fixed from validation','test_accessed':False}
    save_json(out/'protocol.json',config);save_json(out/'clinical_encoder.json',enc.state)
    data=Images(rows,enc,True);loader=DataLoader(data,batch_size=8,shuffle=True,pin_memory=True,generator=torch.Generator().manual_seed(1337))
    if not refit:valloader=DataLoader(Images(rows_for('val'),enc),batch_size=8,pin_memory=True)
    model=ImprovedClassifier(len(enc.state['features']),False).to(device='cuda',memory_format=torch.channels_last);model.load_state_dict(state)
    new=[p for n,p in model.named_parameters() if not n.startswith('resnet.') or n.startswith('resnet.fc.')];back=[p for n,p in model.named_parameters() if n.startswith('resnet.') and not n.startswith('resnet.fc.')]
    opt=torch.optim.AdamW([{'params':new,'lr':1e-4},{'params':back,'lr':1e-5}],weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=1e-7)
    counts=np.bincount([CLASSES.index(r['diagnostic']) for r in rows],minlength=3);lossfn=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(3*counts),dtype=torch.float32,device='cuda'),label_smoothing=.03);scaler=torch.amp.GradScaler('cuda');history=[];best=(-1,-1,-float('inf'));best_epoch=0;stale=0
    for epoch in range(1,epochs+1):
        t=time.monotonic();model.train();model.fix_bn();total=0;correct=0
        for x,c,y in loader:
            x=x.cuda(memory_format=torch.channels_last);c=c.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):z,_=model(x,c);loss=lossfn(z,y)
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();total+=loss.item()*len(y);correct+=(z.argmax(1)==y).sum().item()
        row={'epoch':epoch,'train_loss':total/len(rows),'train_accuracy':correct/len(rows)}
        if not refit:
            yy,pp=predict(model,valloader);m=metrics(yy,pp);row.update(val_accuracy=m['accuracy'],val_macro_f1=m['macro_f1'],val_log_loss=m['log_loss']);better=select_key(m)>best
            if better:best=select_key(m);best_epoch=epoch;stale=0
            else:stale+=1
        else:better=epoch==epochs;best_epoch=epoch
        if better:torch.save({'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'clinical_encoder':enc.state,'epoch':epoch,'config':config},out/'best_model.pt')
        scheduler.step();row['seconds']=time.monotonic()-t;history.append(row);write_rows(out/'history.csv',history);print(('REFIT ' if refit else 'SEARCH ')+json.dumps(row),flush=True)
        if not refit and epoch>=10 and stale>=8:break
    if not refit:
        model.load_state_dict(torch.load(out/'best_model.pt',map_location='cpu',weights_only=False)['model_state']);yy,p0=predict(model,valloader);_,p4=predict(model,valloader,True);low=np.load(SEARCH/'seed_1337/val_predictions.npz')['tta'];np.savez(out/'val_predictions.npz',y=yy,single=p0,tta=p4)
        candidates=[]
        for name,p,weight,tta in [('224_tta',low,0,True),('320_single',p0,1,False),('320_tta',p4,1,True),('224_320_tta',(low+p4)/2,.5,True)]:candidates.append({'name':name,'high_weight':weight,'tta':tta,'validation':metrics(yy,p)})
        candidates.sort(key=lambda v:select_key(v['validation']),reverse=True);save_json(OUT/'selection.json',{'selected':candidates[0],'candidates':candidates,'test_accessed':False});print('VALIDATION SELECTION '+json.dumps(candidates[0]),flush=True)
    save_json(out/'completed.json',{'best_epoch':best_epoch,'epochs_run':len(history)})

def evaluate():
    setup();out=OUT/'refit'
    if (out/'metrics.json').exists():raise RuntimeError('Completed evaluation preserved')
    chosen=json.loads((OUT/'selection.json').read_text())['selected'];high=torch.load(out/'best_model.pt',map_location='cpu',weights_only=False);low=torch.load(SEARCH/'refit/best_system.pt',map_location='cpu',weights_only=False)
    cp={'class_order':CLASSES,'clinical_encoder':high['clinical_encoder'],'selected':chosen,'tta':chosen['tta'],'models':[{'size':320,'weight':chosen['high_weight'],'state':high['model_state']}]}
    if chosen['high_weight']<1:cp['models'].append({'size':224,'weight':1-chosen['high_weight'],'state':low['model_states'][0]})
    torch.save(cp,out/'best_system.pt');save_json(out/'evaluation_lock.json',{'model_sha256':hashlib.sha256((out/'best_system.pt').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((ROOT/'high_resolution.py').read_bytes()).hexdigest(),'selection':chosen,'test_sha256':hashlib.sha256((ROOT/'outputs/run_01/test.csv').read_bytes()).hexdigest()})
    rows=rows_for('test');dev=rows_for('train')+rows_for('val');enc=ClinicalEncoder(cp['clinical_encoder']);by_hash={r['img_id']:r['sha256'] for r in json.loads((ROOT/'data/manifest.json').read_text())}
    for key in ['patient_id','lesion_id']:assert not ({r[key] for r in rows}&{r[key] for r in dev})
    assert not ({by_hash[r['img_id']] for r in rows}&{by_hash[r['img_id']] for r in dev})
    p=0
    for spec in cp['models']:
        model=ImprovedClassifier(len(enc.state['features']),False).to(device='cuda',memory_format=torch.channels_last);model.load_state_dict(spec['state']);yy,pp=predict(model,DataLoader(Images(rows,enc,size=spec['size']),batch_size=8,pin_memory=True),cp['tta']);p=p+spec['weight']*pp;del model
    m=metrics(yy,p);m.update(test_images=len(rows),test_patients=len({r['patient_id'] for r in rows}),goal_over_90_percent_met=bool(m['accuracy']>.90),patient_bootstrap_95ci=bootstrap(yy,p,rows),test_note='Same 96-image benchmark reused; model development motivated by earlier aggregate test feedback. Not an independent external estimate.',validation_note='Selection validation score belongs to the 479-image training experiment; final models use all 575 development images and have no independent validation score.')
    save_json(out/'metrics.json',m);write_rows(out/'test_predictions.csv',[{'img_id':r['img_id'],'patient_id':r['patient_id'],'true_class':r['diagnostic'],'predicted_class':CLASSES[int(p[i].argmax())],**{f'prob_{c}':float(p[i,k]) for k,c in enumerate(CLASSES)}} for i,r in enumerate(rows)]);print(json.dumps(m),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['search','refit','evaluate']);args=ap.parse_args()
    if args.mode=='evaluate':evaluate()
    else:fit(args.mode=='refit')
