"""Predict SCC/SEK/NEV from an image and a JSON object of clinical fields."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image,ImageOps
from torchvision import transforms
from model import GuidedClassifier
from train import ClinicalEncoder,CLASSES,ROOT

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image',type=Path,required=True)
    ap.add_argument('--clinical-json',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/run_01/best_model.pt')
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; pass --device cpu explicitly for CPU inference')
    # Load only trusted checkpoints, such as those produced by this project.
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    encoder=ClinicalEncoder(checkpoint['clinical_encoder'])
    row=json.loads(args.clinical_json.read_text(encoding='utf-8'))
    if not isinstance(row,dict):raise ValueError('Clinical JSON must be one object with metadata field names')
    model=GuidedClassifier(len(encoder.state['features']),pretrained=False).to(args.device)
    model.load_state_dict(checkpoint['model_state']);model.eval()
    with Image.open(args.image) as im:im=ImageOps.pad(im.convert('RGB'),(256,256),method=Image.Resampling.BILINEAR,color=(128,128,128))
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    x=transform(im).unsqueeze(0).to(args.device)
    c=torch.from_numpy(encoder.transform([row])).to(args.device)
    with torch.inference_mode():
        with torch.autocast(device_type=args.device,dtype=torch.float16,enabled=args.device=='cuda'):logits,w=model(x,c)
        p=logits.float().softmax(1)[0].cpu().tolist()
    result={'predicted_class':CLASSES[max(range(3),key=lambda i:p[i])],
            'probabilities':dict(zip(CLASSES,p)),'expert_weights':w[0].float().cpu().tolist(),
            'class_scope':CLASSES,'checkpoint_epoch':checkpoint['epoch']}
    if args.output:args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
