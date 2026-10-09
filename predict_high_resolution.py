"""Inference with the validation-selected progressive-resolution model package."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image,ImageOps
from train import ClinicalEncoder
from high_resolution import OUT,transform
from improved_model import ImprovedClassifier

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',required=True,type=Path);ap.add_argument('--clinical-json',required=True,type=Path);ap.add_argument('--checkpoint',type=Path,default=OUT/'refit/best_system.pt');ap.add_argument('--output',type=Path);args=ap.parse_args();torch.set_num_threads(4)
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False);enc=ClinicalEncoder(cp['clinical_encoder']);c=torch.from_numpy(enc.transform([json.loads(args.clinical_json.read_text())])).cuda();p=0
    with torch.inference_mode():
        for spec in cp['models']:
            size=spec['size']
            with Image.open(args.image) as im:im=ImageOps.pad(im.convert('RGB'),(round(size*8/7),)*2,method=Image.Resampling.BILINEAR,color=(128,)*3)
            x=transform(size)(im).unsqueeze(0).cuda(memory_format=torch.channels_last);model=ImprovedClassifier(len(enc.state['features']),False).to(device='cuda',memory_format=torch.channels_last);model.load_state_dict(spec['state']);model.eval();views=[]
            for axes in ([(),(-1,),(-2,),(-2,-1)] if cp['tta'] else [()]):
                with torch.autocast('cuda',dtype=torch.float16):z,_=model(x.flip(axes) if axes else x,c)
                views.append(z.float().softmax(1))
            p=p+spec['weight']*torch.stack(views).mean(0)[0].cpu();del model
    result={'predicted_class':cp['class_order'][int(p.argmax())],'probabilities':dict(zip(cp['class_order'],p.tolist())),'configuration':cp['selected']['name']};body=json.dumps(result,indent=2)
    if args.output:args.output.write_text(body)
    print(body)

if __name__=='__main__':main()
