"""Build a portable, three-cell evaluation notebook and 20 unbiased demo samples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import textwrap
import zipfile

import torch

HERE = Path(__file__).resolve().parent
FIELDS = ['age','region','itch','hurt','bleed','elevation']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def notebook(bundle_sha):
    first = textwrap.dedent('''\
        # CELL 1 — Load the saved model (run once; no training).
        # Local: keep this notebook inside the supplied evaluation folder.
        # Colab: when prompted, upload dermasense_assets.zip from that folder.
        from pathlib import Path
        import hashlib, importlib.util, subprocess, sys, zipfile

        packages = {"torch":"torch", "torchvision":"torchvision", "numpy":"numpy",
                    "PIL":"Pillow", "sklearn":"scikit-learn", "pandas":"pandas",
                    "matplotlib":"matplotlib", "ipywidgets":"ipywidgets"}
        missing = [p for m,p in packages.items() if importlib.util.find_spec(m) is None]
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        assets = Path("_assets")
        if not (assets / "dermasense_demo.py").exists():
            bundle = Path("dermasense_assets.zip")
            if not bundle.exists():
                try:
                    from google.colab import files
                except ImportError:
                    raise FileNotFoundError("Open this notebook from the supplied evaluation folder, beside dermasense_assets.zip.")
                print("Upload dermasense_assets.zip once. Upload test images in cell 2.")
                uploaded = files.upload()
                matches = [data for name,data in uploaded.items() if name.lower().endswith(".zip")]
                if len(matches) != 1:
                    raise ValueError("Select exactly one file: dermasense_assets.zip")
                bundle.write_bytes(matches[0])
            if hashlib.sha256(bundle.read_bytes()).hexdigest() != "BUNDLE_SHA_PLACEHOLDER":
                raise ValueError("Wrong or incomplete assets ZIP. Use the ZIP supplied with this notebook.")
            allowed = {"dermasense_demo.py", "demo_model.pt", "demo_records.json", "asset_manifest.json"}
            assets.mkdir(exist_ok=True)
            with zipfile.ZipFile(bundle) as archive:
                if set(archive.namelist()) != allowed:
                    raise ValueError("Unexpected asset contents")
                for name in allowed:
                    (assets / name).write_bytes(archive.read(name))
        spec = importlib.util.spec_from_file_location("dermasense_live_runtime", assets / "dermasense_demo.py")
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        demo = runtime.load_demo(assets)
    ''').replace('BUNDLE_SHA_PLACEHOLDER',bundle_sha)
    second = '# CELL 2 — Upload any number of the 20 supplied images.\n# Rerun this cell to replace your selection.\nselection = runtime.pick_images()\n'
    third = '# CELL 3 — Predict and calculate metrics for the selected images.\n# Repeat cells 2 and 3 for another live test; the model remains loaded.\nresults = runtime.run_demo(demo, selection)\n'
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3 (DermaSense Demo)','language':'python','name':'python3'},
             'language_info':{'name':'python','version':'3.12'},'colab':{'name':'DermaSense_Live_Demo.ipynb','provenance':[]}},
            'cells':[{'cell_type':'code','id':cid,'metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}
                     for cid,code in zip(['load-model','choose-images','predict-and-score'],[first,second,third])]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--python',type=Path,default=Path(sys.executable));ap.add_argument('--seed',type=int,default=20261019);args=ap.parse_args()
    workspace=args.workspace.resolve();dest=args.destination.resolve()
    if dest.exists() and any(dest.iterdir()) and not (dest/'selection_manifest.json').exists():
        raise RuntimeError('Destination is not an existing generated demo folder; choose an empty folder.')
    dest.mkdir(parents=True,exist_ok=True);assets=dest/'_assets';assets.mkdir(exist_ok=True);images=dest/'images';images.mkdir(exist_ok=True)
    checkpoint_path=workspace/'outputs/improvement_01/refit/best_system.pt'
    lock=json.loads((workspace/'outputs/improvement_01/refit/evaluation_lock.json').read_text())
    assert sha(checkpoint_path)==lock['checkpoint_sha256'],'Source checkpoint integrity failed'
    test_path=workspace/'outputs/run_01/test.csv'
    with test_path.open(newline='',encoding='utf-8') as f:test=list(csv.DictReader(f))
    with (workspace/'outputs/improvement_01/refit/development.csv').open(newline='',encoding='utf-8') as f:development=list(csv.DictReader(f))
    assert not ({r['patient_id'] for r in test}&{r['patient_id'] for r in development})
    rng=random.Random(args.seed);selected=[];seen=set()
    for label,count in [('SCC',7),('SEK',7),('NEV',6)]:
        candidates=sorted([r for r in test if r['diagnostic']==label],key=lambda r:r['img_id']);rng.shuffle(candidates)
        accepted=0
        for row in candidates:
            digest=sha(workspace/'data/images'/row['img_id'])
            if digest in seen:continue
            selected.append({**row,'sha256':digest});seen.add(digest);accepted+=1
            if accepted==count:break
        assert accepted==count
    rng.shuffle(selected)
    # Selection is complete before any model predictions are inspected.
    record_list=[]
    for row in selected:
        src=workspace/'data/images'/row['img_id'];shutil.copy2(src,images/src.name)
        assert sha(images/src.name)==row['sha256']
        record_list.append({'filename':row['img_id'],'sha256':row['sha256'],'clinical':{k:row[k] for k in FIELDS},'label':row['diagnostic']})
    manifest={'seed':args.seed,'method':'Fixed-seed random selection within classes, without using predictions; 7 SCC, 7 SEK, 6 NEV. Exact duplicate content excluded.',
              'source_partition':'Original 96-image test partition; all excluded from model training','test_csv_sha256':sha(test_path),
              'source_checkpoint_sha256':sha(checkpoint_path),'class_counts':{'SCC':7,'SEK':7,'NEV':6},
              'records':[{'filename':r['filename'],'sha256':r['sha256'],'label':r['label']} for r in record_list],
              'dataset':'PAD-UFES-20 v1, CC BY 4.0','source':'https://data.mendeley.com/datasets/zr7vgbcyr2/1'}
    (dest/'selection_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    write_csv(dest/'clinical_data.csv',[{'img_id':r['filename'],**r['clinical']} for r in record_list])
    write_csv(dest/'ground_truth.csv',[{'img_id':r['filename'],'diagnostic':r['label']} for r in record_list])
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    assert len(checkpoint['model_states'])==1
    export={'model_state':checkpoint['model_states'][0],'clinical_encoder':checkpoint['clinical_encoder'],'class_order':checkpoint['class_order'],
            'tta':checkpoint['tta'],'image_size':224,'reference_test_accuracy':.875,'source_checkpoint_sha256':sha(checkpoint_path)}
    torch.save(export,assets/'demo_model.pt')
    (assets/'demo_records.json').write_text(json.dumps({'records':record_list},indent=2),encoding='utf-8')
    shutil.copy2(HERE/'dermasense_demo.py',assets/'dermasense_demo.py')
    (assets/'asset_manifest.json').write_text(json.dumps({'sha256':{name:sha(assets/name) for name in ['demo_model.pt','demo_records.json','dermasense_demo.py']}},indent=2),encoding='utf-8')
    bundle=dest/'dermasense_assets.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in ['demo_model.pt','demo_records.json','dermasense_demo.py','asset_manifest.json']:z.write(assets/name,name)
    nb=notebook(sha(bundle));body=json.dumps(nb,indent=1,ensure_ascii=False)
    (dest/'DermaSense_Live_Demo.ipynb').write_text(body,encoding='utf-8');(HERE/'DermaSense_Live_Demo.ipynb').write_text(body,encoding='utf-8')
    launcher=f'@echo off\r\ncd /d "%~dp0"\r\n"{args.python.resolve()}" -m notebook DermaSense_Live_Demo.ipynb --ServerApp.ip=127.0.0.1\r\nif errorlevel 1 pause\r\n'
    (dest/'Start_Demo.cmd').write_bytes(launcher.encode('utf-8'))
    instructions='''DERMASENSE — LIVE EVALUATION (3 CELLS)

FASTEST OPTION: LOCAL JUPYTER, ALREADY PREPARED ON THIS COMPUTER
1. Double-click Start_Demo.cmd. Leave its terminal open during the demonstration.
2. In the opened notebook, run cell 1 (Shift+Enter). It loads the saved model; no training.
3. Run cell 2 and click Choose images. Open this folder's images subfolder.
   Select any number of images (Ctrl-click individual files or Ctrl+A for all 20).
4. Run cell 3. It shows every prediction, accuracy, precision, recall, F1,
   per-class metrics, a confusion matrix and an image gallery.
5. For another test, rerun cells 2 and 3. Cell 1 need not be rerun.

GOOGLE COLAB OPTION
1. Open https://colab.research.google.com and upload DermaSense_Live_Demo.ipynb.
2. Optional: Runtime > Change runtime type > T4 GPU. CPU is also supported.
3. Run cell 1 and upload dermasense_assets.zip when prompted (one upload per session).
   The ZIP contains model weights, helper code and matching demo records. Network
   upload time depends on your connection; no training or dataset download occurs.
4. Run cell 2 and upload any images from the local images folder, then run cell 3.

WHAT IS INCLUDED
- images/: 20 randomly selected original test images (7 SCC, 7 SEK, 6 NEV).
- clinical_data.csv: age, region, itch, hurt, bleed, elevation for those images.
- ground_truth.csv: reference diagnoses used only to calculate metrics.
- selection_manifest.json: sampling seed, provenance and image hashes.
- _assets/: ready-to-load local model and demo records. Keep this folder intact.
- dermasense_assets.zip: portable assets for Colab or another notebook installation.

Images are matched by file content, so renaming a demo image is supported. Duplicate
copies are counted once. Predictions use the image and its clinical record, not
the reference label. Unknown external images receive predictions with missing
clinical inputs but cannot receive an accuracy score without reference labels.

The original model scored 87.50% on 96 test images. The live score is calculated
only from your selected labelled images and may be higher or lower. The 20 images
were not chosen based on prediction correctness. This is a research demonstration.
Metrics and figures from each upload are saved in demo_outputs/ with a timestamp.

DATA ATTRIBUTION
PAD-UFES-20, Pacheco et al., version 1, CC BY 4.0.
https://data.mendeley.com/datasets/zr7vgbcyr2/1
Dataset paper: https://doi.org/10.1016/j.dib.2020.106221
'''
    (dest/'START_HERE.txt').write_text(instructions,encoding='utf-8')
    print(json.dumps({'folder':str(dest),'images':len(record_list),'counts':manifest['class_counts'],'asset_zip_mb':round(bundle.stat().st_size/1e6,2),'notebook_code_cells':len(nb['cells'])},indent=2))


if __name__=='__main__':main()
