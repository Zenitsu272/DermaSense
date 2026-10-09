"""One validation-only comparison of the previous and mixup models and their mean."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from clinical_fusion import (base_checkpoint,evaluate,fit_head,rows_for)
from robust_finetune import train
from train import CLASSES,metrics,save_json


def search(workspace,out):
    if (out/'selection.json').exists():raise RuntimeError('Completed selection preserved')
    previous=json.loads((workspace/'outputs/clinical_fusion_01/selection.json').read_text())['selected']
    robust_results=json.loads((workspace/'outputs/robust_finetune_01/selection.json').read_text())
    robust=next(c for c in robust_results['candidates'] if c['kind']=='robust')
    assert previous['kind']=='blend' and previous['size']==320,'This comparison is scoped to the recorded fusion winner'
    train_rows=rows_for(workspace,'train');val_rows=rows_for(workspace,'val');_,_,enc=base_checkpoint(workspace,320,False)
    clinical=fit_head(enc.transform(train_rows),np.array([CLASSES.index(r['diagnostic']) for r in train_rows]),previous['C'])
    old=np.load(workspace/'outputs/clinical_fusion_01/features_320.npz')['val_probs'];old=(1-previous['alpha'])*old+previous['alpha']*clinical.predict_proba(enc.transform(val_rows))
    data=np.load(workspace/'outputs/robust_finetune_01/search/val_predictions.npz');new=(1-robust['alpha'])*data['image_probs']+robust['alpha']*data['clinical_probs'];y=data['y']
    candidates=[]
    for kind,p in [('previous',old),('robust',new),('ensemble',(old+new)/2)]:
        candidates.append({'name':f'followup_{kind}','kind':kind,'validation':metrics(y,p),'previous':previous,'robust':robust})
    candidates.sort(key=lambda c:(c['validation']['accuracy'],c['validation']['macro_f1'],-c['validation']['log_loss']),reverse=True)
    save_json(out/'selection.json',{'selected':candidates[0],'candidates':candidates,'test_accessed':False})
    protocol=json.loads((out/'protocol.json').read_text());protocol['fixed_split_sha256']=json.loads((workspace/'outputs/clinical_fusion_01/protocol.json').read_text())['fixed_split_sha256'];save_json(out/'protocol.json',protocol)
    print('SELECTED '+json.dumps(candidates[0]),flush=True)


def refit(workspace,out):
    if (out/'best_system.pt').exists():raise RuntimeError('Completed refit preserved')
    selected=json.loads((out/'selection.json').read_text())['selected'];components=[]
    if selected['kind'] in ['previous','ensemble']:
        components.append(torch.load(workspace/'outputs/clinical_fusion_01/best_system.pt',map_location='cpu',weights_only=False))
    if selected['kind'] in ['robust','ensemble']:
        component=out/'component';component.mkdir(exist_ok=True)
        save_json(component/'selection.json',{'selected':selected['robust'],'purpose':'Fixed component refit required by the validation-selected follow-up system'})
        if not (component/'best_system.pt').exists():train(workspace,component,True)
        components.append(torch.load(component/'best_system.pt',map_location='cpu',weights_only=False))
    package={'class_order':CLASSES,'selected':selected,'components':components,'refit':{'images':575,'patients':414,'independent_validation':False}}
    torch.save(package,out/'best_system.pt');print('FINAL PACKAGE READY '+selected['name'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['search','refit','evaluate']);ap.add_argument('--workspace',type=Path,required=True);args=ap.parse_args();workspace=args.workspace.resolve();out=workspace/'outputs/robust_ensemble_01'
    out.mkdir(parents=True,exist_ok=True)
    if args.mode=='search' and not (out/'protocol.json').exists():
        save_json(out/'protocol.json',{'candidates':['clinical fusion winner','mixup checkpoint winner','equal mean of their probabilities'],'selection':'validation accuracy, macro F1, lower log loss','weights':[0,.5,1],'test_accessed':False,'reason':'Fixed equal averaging of earlier and mixup-trained image models; no test-score selection'})
    assert (out/'protocol.json').exists(),'Run the validation search first'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    {'search':search,'refit':refit,'evaluate':evaluate}[args.mode](workspace,out)


if __name__=='__main__':main()
