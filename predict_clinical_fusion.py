"""Predict from the validation-selected clinical-fusion package."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image,ImageOps
from torchvision import transforms

from clinical_fusion import FeatureModel,combined_probabilities
from train import ClinicalEncoder


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--image',type=Path,required=True)
    ap.add_argument('--clinical-json',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    clinical=json.loads(args.clinical_json.read_text(encoding='utf-8'))
    if not isinstance(clinical,dict):raise ValueError('Clinical input must be a JSON object')
    predictions=[]
    for component in cp.get('components',[cp]):
        enc=ClinicalEncoder(component['clinical_encoder']);c=enc.transform([clinical]);size=component['selected']['size']
        with Image.open(args.image) as im:
            im=ImageOps.pad(im.convert('RGB'),(round(size*8/7),)*2,method=Image.Resampling.BILINEAR,color=(128,)*3)
        transform=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        x=transform(im).unsqueeze(0).cuda(memory_format=torch.channels_last);cc=torch.from_numpy(c).cuda()
        model=FeatureModel(len(enc.state['features']),False).to(device='cuda',memory_format=torch.channels_last);model.load_state_dict(component['image_model']);model.eval()
        fs,ps=[],[]
        with torch.inference_mode():
            for axes in [(),(-1,),(-2,),(-2,-1)]:
                with torch.autocast('cuda',dtype=torch.float16):
                    f=model.encode(x.flip(axes) if axes else x,cc);z=model.resnet.fc(f)
                fs.append(f.float());ps.append(z.float().softmax(1))
        f=torch.stack(fs).mean(0).cpu().numpy();ip=torch.stack(ps).mean(0).cpu().numpy()
        predictions.append(combined_probabilities(component,f,c,ip)[0]);del model
    p=sum(predictions)/len(predictions)
    result={'predicted_class':cp['class_order'][int(p.argmax())],'probabilities':dict(zip(cp['class_order'],map(float,p))),'configuration':cp['selected']['name']}
    body=json.dumps(result,indent=2)
    if args.output:args.output.write_text(body,encoding='utf-8')
    print(body)


if __name__=='__main__':main()
