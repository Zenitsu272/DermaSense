"""Inference for the validation-selected improved model or ensemble."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image,ImageOps
from torchvision import transforms
from train import ROOT,ClinicalEncoder
from improved_model import ImprovedClassifier

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',type=Path,required=True);ap.add_argument('--clinical-json',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/improvement_01/final/best_system.pt');ap.add_argument('--output',type=Path);args=ap.parse_args()
    torch.set_num_threads(4)
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False);enc=ClinicalEncoder(cp['clinical_encoder'])
    clinical=json.loads(args.clinical_json.read_text());c=torch.from_numpy(enc.transform([clinical])).cuda()
    with Image.open(args.image) as im:im=ImageOps.pad(im.convert('RGB'),(256,256),method=Image.Resampling.BILINEAR,color=(128,128,128))
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);x=transform(im).unsqueeze(0).cuda()
    probabilities=[]
    with torch.inference_mode():
        for state in cp['model_states']:
            model=ImprovedClassifier(len(enc.state['features']),pretrained=False).cuda();model.load_state_dict(state);model.eval()
            views=[]
            for axes in ([(),(-1,),(-2,),(-2,-1)] if cp['tta'] else [()]):
                with torch.autocast('cuda',dtype=torch.float16):z,_=model(x.flip(axes) if axes else x,c)
                views.append(z.float().softmax(1))
            probabilities.append(torch.stack(views).mean(0));del model
        p=torch.stack(probabilities).mean(0)[0].cpu().tolist()
    result={'predicted_class':cp['class_order'][max(range(3),key=lambda i:p[i])],'probabilities':dict(zip(cp['class_order'],p)),'configuration':cp['selected']['name']}
    if args.output:args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
