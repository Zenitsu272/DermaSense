"""Validation-only candidate search. Test evaluation is a separate explicit step."""
import argparse,copy,csv,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from train import ROOT,CLASSES,ClinicalEncoder,LesionDataset,metrics,save_json,write_rows,bootstrap
from improved_model import ImprovedClassifier

SEARCH=ROOT/'outputs/improvement_01'
SEEDS=[42,1337,2026]
def rows_for(split):
    with (ROOT/'outputs/run_01'/f'{split}.csv').open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

@torch.inference_mode()
def predict(model,loader,tta=False):
    model.eval();ys=[];ps=[]
    for x,c,y in loader:
        x=x.cuda(memory_format=torch.channels_last);c=c.cuda()
        probs=[]
        for axes in ([(),(-1,),(-2,),(-2,-1)] if tta else [()]):
            xx=x.flip(axes) if axes else x
            with torch.autocast('cuda',dtype=torch.float16):z,_=model(xx,c)
            probs.append(z.float().softmax(1))
        ps.append(torch.stack(probs).mean(0).cpu().numpy());ys.append(y.numpy())
    return np.concatenate(ys),np.concatenate(ps)

def select_key(m):return (m['accuracy'],m['macro_f1'],-m['log_loss'])

def train_seed(seed,args,trainset,valset,encoder):
    out=SEARCH/f'seed_{seed}';out.mkdir(exist_ok=True)
    if (out/'validation.json').exists():return
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=ImprovedClassifier(len(encoder.state['features'])).to(device='cuda',memory_format=torch.channels_last)
    trainloader=DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(seed))
    valloader=DataLoader(valset,batch_size=args.batch_size,num_workers=0,pin_memory=True)
    backbone=[p for n,p in model.named_parameters() if n.startswith('resnet.') and not n.startswith('resnet.fc.')]
    new=[p for n,p in model.named_parameters() if not n.startswith('resnet.') or n.startswith('resnet.fc.')]
    optimizer=torch.optim.AdamW([{'params':new,'lr':5e-4},{'params':backbone,'lr':3e-5}],weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4,min_lr=1e-7)
    counts=np.bincount([CLASSES.index(r['diagnostic']) for r in trainset.rows],minlength=3)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(3*counts),device='cuda',dtype=torch.float32),label_smoothing=.03)
    scaler=torch.amp.GradScaler('cuda');history=[];best=(-1.,-1.,-float('inf'));stale=0;start=time.monotonic()
    config={'seed':seed,'epochs_max':args.epochs,'batch_size':args.batch_size,'pretrained':'ResNet50 ImageNet1K V2, complete RGB path retained','architecture':'U-Net encoder -> metadata-weighted 512-channel experts added to ResNet50 layer2 -> layer3/layer4 -> head','backbone_lr':3e-5,'new_lr':5e-4,'weight_decay':1e-3,'label_smoothing':.03,'warmup_epochs':3,'selection':'validation accuracy, then macro F1, then lower log loss','clinical_encoder':encoder.state,'gpu':torch.cuda.get_device_name(0),'test_accessed':False}
    save_json(out/'config.json',config)
    for epoch in range(1,args.epochs+1):
        t=time.monotonic();model.freeze_backbone(epoch<=3);model.train();model.fix_bn();loss_sum=0;n=0;correct=0
        for x,c,y in trainloader:
            x=x.cuda(memory_format=torch.channels_last);c=c.cuda();y=y.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):z,w=model(x,c);loss=criterion(z,y)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optimizer);scaler.update()
            loss_sum+=loss.item()*len(y);n+=len(y);correct+=(z.argmax(1)==y).sum().item()
        yy,pp=predict(model,valloader);m=metrics(yy,pp)
        row={'epoch':epoch,'train_loss':loss_sum/n,'train_accuracy':correct/n,'val_accuracy':m['accuracy'],'val_macro_f1':m['macro_f1'],'val_log_loss':m['log_loss'],'epoch_seconds':time.monotonic()-t}
        history.append(row);write_rows(out/'history.csv',history);save_json(out/'history.json',history)
        key=select_key(m);improved=key>best
        if improved:
            best=key;stale=0
            torch.save({'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'clinical_encoder':encoder.state,'config':config,'epoch':epoch,'validation':m},out/'best_model.pt')
        else:stale+=1
        scheduler.step(m['accuracy'])
        print(f"Seed {seed} epoch {epoch:02d}: loss={loss_sum/n:.3f}, val accuracy={m['accuracy']:.3f}, F1={m['macro_f1']:.3f}, {row['epoch_seconds']:.1f}s"+(' BEST' if improved else ''),flush=True)
        if epoch>=18 and stale>=12:break
    checkpoint=torch.load(out/'best_model.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model_state'])
    yy,p0=predict(model,valloader);_,p4=predict(model,valloader,True)
    np.savez(out/'val_predictions.npz',y=yy,single=p0,tta=p4)
    save_json(out/'validation.json',{'single':metrics(yy,p0),'tta':metrics(yy,p4),'best_epoch':checkpoint['epoch'],'training_seconds':time.monotonic()-start,'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30})
    print(f"Seed {seed} finished: validation single={metrics(yy,p0)['accuracy']:.3f}, four-view={metrics(yy,p4)['accuracy']:.3f}",flush=True)
    del model,optimizer;torch.cuda.empty_cache()

def select_candidates():
    candidates=[];arrays={}
    for seed in SEEDS:
        data=np.load(SEARCH/f'seed_{seed}/val_predictions.npz');arrays[seed]=data;y=data['y']
        for mode in ['single','tta']:
            candidates.append({'name':f'seed_{seed}_{mode}','seeds':[seed],'tta':mode=='tta','validation':metrics(y,data[mode])})
    for mode in ['single','tta']:
        p=np.mean([arrays[s][mode] for s in SEEDS],axis=0)
        candidates.append({'name':f'ensemble_{mode}','seeds':SEEDS,'tta':mode=='tta','validation':metrics(y,p)})
    candidates.sort(key=lambda c:select_key(c['validation']),reverse=True)
    save_json(SEARCH/'selection.json',{'selected':candidates[0],'candidates':candidates,'selected_using':'validation only','test_accessed':False})
    print('SELECTED ON VALIDATION: '+json.dumps(candidates[0]),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=45);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    assert torch.cuda.is_available(),'CUDA required'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    SEARCH.mkdir(exist_ok=True)
    if not (SEARCH/'protocol.json').exists():
        save_json(SEARCH/'protocol.json',{'seeds':SEEDS,'epochs_max':args.epochs,'batch_size':args.batch_size,'selection_candidates':'each of three seeds and equal average of all three, each with single-view or four-flip averaging: 8 candidates','criterion':'validation accuracy, then macro F1, then negative log loss','fixed_split_hashes':{s:hashlib.sha256((ROOT/'outputs/run_01'/f'{s}.csv').read_bytes()).hexdigest() for s in ['train','val','test']},'test_policy':'Original test has been reported previously. This search reads only training and validation rows. Final selected candidate is evaluated separately; no test-score model selection.'})
    encoder=ClinicalEncoder().fit(rows_for('train'))
    save_json(SEARCH/'clinical_encoder.json',encoder.state)
    trainset=LesionDataset(rows_for('train'),encoder,True);valset=LesionDataset(rows_for('val'),encoder)
    if args.smoke:
        model=ImprovedClassifier(len(encoder.state['features'])).to(device='cuda',memory_format=torch.channels_last);model.train();model.fix_bn()
        x,c,y=next(iter(DataLoader(trainset,batch_size=args.batch_size)))
        with torch.autocast('cuda',dtype=torch.float16):z,w=model(x.cuda(memory_format=torch.channels_last),c.cuda());loss=nn.functional.cross_entropy(z,y.cuda())
        loss.backward()
        for name in ['encoder','experts','router','resnet']:
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in getattr(model,name).parameters()),name
        assert z.shape==(args.batch_size,3) and torch.isfinite(loss)
        print('IMPROVED GPU SMOKE PASS, peak GiB',torch.cuda.max_memory_allocated()/2**30,flush=True);return
    for seed in SEEDS:train_seed(seed,args,trainset,valset,encoder)
    select_candidates()

if __name__=='__main__':main()
