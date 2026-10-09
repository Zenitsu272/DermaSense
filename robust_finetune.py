"""Predefined mixup fine-tuning, validation selection, then fixed development refit."""
import argparse,csv,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from clinical_fusion import (Images,base_checkpoint,combined_probabilities,extract,fit_head,
                             rows_for)
from improved_model import ImprovedClassifier
from train import CLASSES,ClinicalEncoder,bootstrap,metrics,save_json,write_rows


@torch.inference_mode()
def infer(model,loader):
    model.eval();probabilities=[]
    for x,c,_ in loader:
        x=x.cuda(memory_format=torch.channels_last);c=c.cuda();views=[]
        for axes in [(),(-1,),(-2,),(-2,-1)]:
            with torch.autocast('cuda',dtype=torch.float16):z,_=model(x.flip(axes) if axes else x,c)
            views.append(z.float().softmax(1))
        probabilities.append(torch.stack(views).mean(0).cpu().numpy())
    return np.concatenate(probabilities)


def train(workspace,out,refit=False):
    dest=out/('refit' if refit else 'search');dest.mkdir(exist_ok=True)
    if (dest/'completed.json').exists():raise RuntimeError('Completed training preserved')
    seed=1337;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    selected=json.loads((out/'selection.json').read_text())['selected'] if refit else None
    if refit and selected['kind']=='previous':raise RuntimeError('Earlier fusion won; no new refit needed')
    path,state,enc=base_checkpoint(workspace,320,refit)
    rows=rows_for(workspace,'train')+(rows_for(workspace,'val') if refit else [])
    y=np.array([CLASSES.index(r['diagnostic']) for r in rows]);c=enc.transform(rows)
    clinical=fit_head(c,y,.1)
    epochs=selected['epoch'] if refit else 20
    config={'seed':seed,'epochs':epochs,'size':320,'batch':8,'mixup_alpha':.2,'mixup_probability':.5,'new_lr':1e-4,'backbone_lr':1e-5,'weight_decay':.01,'label_smoothing':.03,'cosine_T_max':20,'crop_scale':[.65,1.0],'rotation':30,'clinical_probability_weight_candidates':[0.,.1,.2],'selection':'four-view validation accuracy, macro F1, lower log loss','training_images':len(rows),'base_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'test_accessed':False}
    save_json(dest/'protocol.json',config)
    data=Images(workspace,rows,enc,320)
    data.transform=transforms.Compose([transforms.RandomResizedCrop(320,scale=(.65,1),ratio=(.85,1.15)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(30,fill=128),transforms.ColorJitter(.15,.15,.15,.025),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    loader=DataLoader(data,batch_size=8,shuffle=True,pin_memory=True,generator=torch.Generator().manual_seed(seed))
    if not refit:
        val=rows_for(workspace,'val');yv=np.array([CLASSES.index(r['diagnostic']) for r in val]);pc=clinical.predict_proba(enc.transform(val));vl=DataLoader(Images(workspace,val,enc,320),batch_size=8,pin_memory=True)
    model=ImprovedClassifier(len(enc.state['features']),False).to(device='cuda',memory_format=torch.channels_last);model.load_state_dict(state)
    new=[p for n,p in model.named_parameters() if not n.startswith('resnet.') or n.startswith('resnet.fc.')];back=[p for n,p in model.named_parameters() if n.startswith('resnet.') and not n.startswith('resnet.fc.')]
    opt=torch.optim.AdamW([{'params':new,'lr':1e-4},{'params':back,'lr':1e-5}],weight_decay=.01);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=1e-7)
    counts=np.bincount(y,minlength=3);lossfn=nn.CrossEntropyLoss(weight=torch.tensor(len(y)/(3*counts),dtype=torch.float32,device='cuda'),label_smoothing=.03);scaler=torch.amp.GradScaler('cuda')
    best=(-1,-1,-float('inf'));history=[];stale=0;winner=None
    for epoch in range(1,epochs+1):
        start=time.monotonic();model.train();model.fix_bn();total=0;mixes=0
        for x,c,batch_y in loader:
            x=x.cuda(memory_format=torch.channels_last);c=c.cuda();batch_y=batch_y.cuda();opt.zero_grad(set_to_none=True)
            lam=float(np.random.beta(.2,.2)) if np.random.random()<.5 else 1.
            perm=torch.randperm(len(batch_y),device='cuda')
            if lam<1:x=lam*x+(1-lam)*x[perm];c=lam*c+(1-lam)*c[perm];mixes+=1
            with torch.autocast('cuda',dtype=torch.float16):
                z,_=model(x,c);loss=lam*lossfn(z,batch_y)+(1-lam)*lossfn(z,batch_y[perm])
            assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();total+=loss.item()*len(batch_y)
        row={'epoch':epoch,'train_loss':total/len(rows),'mixed_batches':mixes}
        if not refit:
            ip=infer(model,vl);choices=[]
            for alpha in config['clinical_probability_weight_candidates']:
                m=metrics(yv,(1-alpha)*ip+alpha*pc);choices.append({'name':f'robust_320_epoch{epoch}_alpha{alpha}','kind':'robust','epoch':epoch,'alpha':alpha,'size':320,'C':.1,'validation':m})
            candidate=max(choices,key=lambda q:(q['validation']['accuracy'],q['validation']['macro_f1'],-q['validation']['log_loss']));m=candidate['validation'];key=(m['accuracy'],m['macro_f1'],-m['log_loss']);better=key>best
            row.update(val_accuracy=m['accuracy'],val_macro_f1=m['macro_f1'],val_log_loss=m['log_loss'],clinical_alpha=candidate['alpha'])
            if better:best=key;winner=candidate;stale=0
            else:stale+=1
        else:better=epoch==epochs
        if better:
            torch.save({'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'clinical_encoder':enc.state,'epoch':epoch,'clinical_head':{'weight':torch.from_numpy(clinical.coef_.astype(np.float32)),'bias':torch.from_numpy(clinical.intercept_.astype(np.float32))}},dest/'best_model.pt')
            if not refit:np.savez(dest/'val_predictions.npz',image_probs=ip,clinical_probs=pc,y=yv)
        schedule.step();row['seconds']=time.monotonic()-start;history.append(row);write_rows(dest/'history.csv',history);print(('REFIT ' if refit else 'SEARCH ')+json.dumps(row),flush=True)
        if not refit and epoch>=10 and stale>=8:break
    if not refit:
        previous=json.loads((workspace/'outputs/clinical_fusion_01/selection.json').read_text())['selected'];previous={**previous,'kind':'previous','source_kind':previous['kind']}
        selected=max([winner,previous],key=lambda q:(q['validation']['accuracy'],q['validation']['macro_f1'],-q['validation']['log_loss']))
        save_json(out/'selection.json',{'selected':selected,'candidates':[winner,previous],'test_accessed':False});print('SELECTED '+json.dumps(selected),flush=True)
    else:
        cp=torch.load(dest/'best_model.pt',map_location='cpu',weights_only=False)
        package={'class_order':CLASSES,'selected':{**selected,'kind':'blend'},'clinical_encoder':cp['clinical_encoder'],'image_model':cp['model_state'],'head':cp['clinical_head'],'refit':config}
        torch.save(package,out/'best_system.pt')
    save_json(dest/'completed.json',{'epochs_run':len(history),'selected_epoch':selected['epoch'] if refit else winner['epoch']})


def evaluate(workspace,out):
    if (out/'metrics.json').exists():raise RuntimeError('Completed evaluation preserved')
    cp=torch.load(out/'best_system.pt',map_location='cpu',weights_only=False)
    save_json(out/'evaluation_lock.json',{'checkpoint_sha256':hashlib.sha256((out/'best_system.pt').read_bytes()).hexdigest(),'source_sha256':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['robust_finetune.py','clinical_fusion.py','predict_clinical_fusion.py']},'selected':cp['selected'],'test_sha256':hashlib.sha256((workspace/'outputs/run_01/test.csv').read_bytes()).hexdigest()})
    rows=rows_for(workspace,'test');dev=rows_for(workspace,'train')+rows_for(workspace,'val')
    for k in ['patient_id','lesion_id']:assert not ({r[k] for r in rows}&{r[k] for r in dev})
    hashes={r['img_id']:r['sha256'] for r in json.loads((workspace/'data/manifest.json').read_text())};assert not ({hashes[r['img_id']] for r in rows}&{hashes[r['img_id']] for r in dev})
    enc=ClinicalEncoder(cp['clinical_encoder']);f,ip=extract(workspace,rows,cp['image_model'],enc,320);p=combined_probabilities(cp,f,enc.transform(rows),ip);y=np.array([CLASSES.index(r['diagnostic']) for r in rows]);m=metrics(y,p)
    m.update(test_images=len(rows),test_patients=len({r['patient_id'] for r in rows}),correct=int((y==p.argmax(1)).sum()),goal_over_90_percent_met=bool(m['accuracy']>.90),patient_bootstrap_95ci=bootstrap(y,p,rows),selected=cp['selected']['name'],test_note='Previously inspected benchmark; not independent external validation.',validation_note='Selection score belongs to training-only model, not development refit.')
    save_json(out/'metrics.json',m);write_rows(out/'test_predictions.csv',[{'img_id':r['img_id'],'patient_id':r['patient_id'],'true_class':r['diagnostic'],'predicted_class':CLASSES[int(p[i].argmax())],**{f'prob_{c}':float(p[i,k]) for k,c in enumerate(CLASSES)}} for i,r in enumerate(rows)]);print('FINAL '+json.dumps(m),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['search','refit','evaluate']);ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parent);args=ap.parse_args();workspace=args.workspace.resolve();out=workspace/'outputs/robust_finetune_01';out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    original=json.loads((workspace/'outputs/clinical_fusion_01/protocol.json').read_text())
    for s,h in original['fixed_split_sha256'].items():assert hashlib.sha256((workspace/'outputs/run_01'/f'{s}.csv').read_bytes()).hexdigest()==h
    if args.mode=='evaluate':evaluate(workspace,out)
    else:train(workspace,out,args.mode=='refit')


if __name__=='__main__':main()
