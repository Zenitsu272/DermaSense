"""Refit locked validation-selected settings on train+validation; never train on test."""
import csv,hashlib,json,random,time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from train import ROOT,CLASSES,ClinicalEncoder,LesionDataset,save_json,write_rows
from improve import SEARCH,rows_for
from improved_model import ImprovedClassifier

OUT=SEARCH/'refit'
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'best_system.pt').exists():raise RuntimeError('Completed refit already exists')
    selected=json.loads((SEARCH/'selection.json').read_text())['selected']
    assert selected['seeds']==[1337], 'This refit protocol locks the validation winner, seed 1337'
    original=torch.load(SEARCH/'seed_1337/best_model.pt',map_location='cpu',weights_only=False)
    epochs=original['epoch'];seed=1337;batch=16
    history=json.loads((SEARCH/'seed_1337/history.json').read_text())
    rows=rows_for('train')+rows_for('val')
    # Split integrity verified against immutable files; test rows are not loaded.
    protocol=json.loads((SEARCH/'protocol.json').read_text())
    for split,expected in protocol['fixed_split_hashes'].items():
        assert hashlib.sha256((ROOT/'outputs/run_01'/f'{split}.csv').read_bytes()).hexdigest()==expected
    encoder=ClinicalEncoder().fit(rows)
    config={'seed':seed,'epochs':epochs,'batch_size':batch,'training_images':len(rows),'training_patients':len({r['patient_id'] for r in rows}),'data':'original train + validation, original test excluded','epoch_selection':'best epoch from original validation-selected seed 1337; no new early stopping','lr_schedule':'replay fixed learning-rate schedule from the completed validation run','inference_tta':selected['tta'],'test_policy':'Previously evaluated 96-image benchmark; no test training or per-example tuning. Not independent external validation.'}
    save_json(OUT/'protocol.json',config);save_json(OUT/'clinical_encoder.json',encoder.state);write_rows(OUT/'development.csv',rows)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=LesionDataset(rows,encoder,True);loader=DataLoader(data,batch_size=batch,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(seed))
    model=ImprovedClassifier(len(encoder.state['features'])).to(device='cuda',memory_format=torch.channels_last)
    backbone=[p for n,p in model.named_parameters() if n.startswith('resnet.') and not n.startswith('resnet.fc.')]
    new=[p for n,p in model.named_parameters() if not n.startswith('resnet.') or n.startswith('resnet.fc.')]
    opt=torch.optim.AdamW([{'params':new,'lr':5e-4},{'params':backbone,'lr':3e-5}],weight_decay=1e-3)
    schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=4,min_lr=1e-7)
    counts=np.bincount([CLASSES.index(r['diagnostic']) for r in rows],minlength=3)
    lossfn=nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(3*counts),dtype=torch.float32,device='cuda'),label_smoothing=.03)
    scaler=torch.amp.GradScaler('cuda');log=[];start=time.monotonic()
    print('LOCKED REFIT PROTOCOL: '+json.dumps(config),flush=True)
    for epoch in range(1,epochs+1):
        model.freeze_backbone(epoch<=3);model.train();model.fix_bn();loss_sum=0;seen=0;correct=0;t=time.monotonic()
        for x,c,y in loader:
            x=x.cuda(memory_format=torch.channels_last);c=c.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):z,_=model(x,c);loss=lossfn(z,y)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update()
            seen+=len(y);loss_sum+=loss.item()*len(y);correct+=(z.argmax(1)==y).sum().item()
        log.append({'epoch':epoch,'training_loss':loss_sum/seen,'training_accuracy_augmented':correct/seen,'lr_new':opt.param_groups[0]['lr'],'lr_backbone':opt.param_groups[1]['lr'],'seconds':time.monotonic()-t})
        # Fixed historical schedule; no validation or test inference during refit.
        schedule.step(history[epoch-1]['val_accuracy'])
        write_rows(OUT/'history.csv',log)
        print(f'Refit {epoch:02d}/{epochs}: loss={loss_sum/seen:.3f}, training accuracy={correct/seen:.3f}, {time.monotonic()-t:.1f}s',flush=True)
    config['training_seconds']=time.monotonic()-start
    torch.save({'architecture':'pretrained_resnet50_with_clinically_routed_unet_residual','class_order':CLASSES,'clinical_encoder':encoder.state,'tta':selected['tta'],'selected':{'name':'seed_1337_refit_development_tta','original_validation_selection':selected,'refit':config},'model_states':[{k:v.detach().cpu() for k,v in model.state_dict().items()}]},OUT/'best_system.pt')
    save_json(OUT/'completed.json',config)
    print('REFIT COMPLETE',flush=True)

if __name__=='__main__':main()
