"""Integration checks for portable inference, subset scoring and notebook execution."""
import argparse
import csv
import importlib.util
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--demo-dir',type=Path,required=True);ap.add_argument('--workspace',type=Path,required=True)
    args=ap.parse_args();folder=args.demo_dir.resolve();test_out=Path(__file__).parent/'test_outputs';test_out.mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('demo_test_runtime',folder/'_assets/dermasense_demo.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    manifest=json.loads((folder/'selection_manifest.json').read_text());files=[(r['filename'],(folder/'images'/r['filename']).read_bytes()) for r in manifest['records']]
    assert len(files)==20 and len(list((folder/'images').glob('*.png')))==20
    with (args.workspace/'outputs/improvement_01/refit/test_predictions.csv').open(newline='') as f:reference={r['img_id']:r for r in csv.DictReader(f)}
    demo=runtime.Demo(folder/'_assets')
    twenty=demo.predict(files);ten=demo.predict(files[:10])
    assert len(twenty['predictions'])==20 and len(ten['predictions'])==10
    max_delta=0.
    for row in twenty['predictions']:
        expected=reference[row['filename']]
        assert row['predicted_class']==expected['predicted_class'],row['filename']
        for c in runtime.CLASSES:max_delta=max(max_delta,abs(row['prob_'+c]-float(expected['prob_'+c])))
    assert max_delta<.005,max_delta
    assert abs(twenty['metrics']['accuracy']-sum(r['correct'] for r in twenty['predictions'])/20)<1e-12
    assert abs(ten['metrics']['accuracy']-sum(r['correct'] for r in ten['predictions'])/10)<1e-12
    # Renaming does not lose metadata, and duplicate byte-identical files count once.
    renamed=demo.predict([('renamed.png',files[0][1]),('duplicate.png',files[0][1]),('broken.png',b'not an image')])
    assert len(renamed['predictions'])==1 and len(renamed['skipped'])==2
    assert renamed['predictions'][0]['clinical_available']
    assert renamed['metrics']['labelled_images']==1 and renamed['metrics']['macro_roc_auc'] is None
    # Changing a scoring label cannot change network predictions/probabilities.
    digest=runtime.hashlib.sha256(files[0][1]).hexdigest();old=demo.labels_by_hash[digest]
    demo.labels_by_hash[digest]=next(c for c in runtime.CLASSES if c!=old)
    relabelled=demo.predict([files[0]])['predictions'][0];demo.labels_by_hash[digest]=old
    assert relabelled['predicted_class']==renamed['predictions'][0]['predicted_class']
    for c in runtime.CLASSES:assert abs(relabelled['prob_'+c]-renamed['predictions'][0]['prob_'+c])<1e-6
    unknown=io.BytesIO();Image.new('RGB',(80,80),(123,99,78)).save(unknown,format='PNG')
    external=demo.predict([('external.png',unknown.getvalue())]);assert external['metrics'] is None and external['predictions'][0]['true_class'] is None
    assert not external['predictions'][0]['clinical_available']
    assert demo.predict([])['metrics'] is None
    cpu=runtime.Demo(folder/'_assets',device='cpu');cpu_result=cpu.predict(files[:10])
    for row,gpu in zip(cpu_result['predictions'],ten['predictions']):
        assert row['predicted_class']==gpu['predicted_class']
        for c in runtime.CLASSES:assert abs(row['prob_'+c]-gpu['prob_'+c])<.005
    print('Inference, metadata matching, scoring boundaries and CPU checks passed.',flush=True)
    import nbformat
    from nbclient import NotebookClient
    nb=nbformat.read(folder/'DermaSense_Live_Demo.ipynb',as_version=4);nbformat.validate(nb)
    assert len(nb.cells)==3 and all(c.cell_type=='code' for c in nb.cells)
    # Only the QA copy receives an automatic 10-image selection; the shipped notebook remains blank.
    nb.cells[1].source+='''\nimport datetime, json
_records=json.loads(Path("selection_manifest.json").read_text())["records"][:10]
selection.value=tuple({"name":r["filename"],"type":"image/png","size":(Path("images")/r["filename"]).stat().st_size,
                      "content":memoryview((Path("images")/r["filename"]).read_bytes()),
                      "last_modified":datetime.datetime.now(datetime.timezone.utc)} for r in _records)
'''
    nb.cells[2].source+='\nassert len(results["predictions"])==10\nassert results["metrics"]["labelled_images"]==10\n'
    started=time.perf_counter()
    client=NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(folder)}},store_widget_state=False)
    executed=client.execute();nbformat.write(executed,test_out/'live_demo.executed.ipynb')
    assert not any(o.output_type=='error' for c in executed.cells for o in c.get('outputs',[]))
    notebook_seconds=time.perf_counter()-started
    # Exercise fresh ZIP bootstrap and native-upload return values without a Google login.
    # This is a local Colab API simulation, not a claim of testing hosted Colab itself.
    colab_folder=test_out/'colab_bootstrap';colab_folder.mkdir(exist_ok=True)
    colab_nb=nbformat.read(folder/'DermaSense_Live_Demo.ipynb',as_version=4)
    prelude=f'''import sys, types, json
from pathlib import Path
_demo_folder=Path({str(folder)!r})
_image_rows=json.loads((_demo_folder/"selection_manifest.json").read_text())["records"][:3]
_uploads=[{{"dermasense_assets.zip":(_demo_folder/"dermasense_assets.zip").read_bytes()}},
          {{r["filename"]:(_demo_folder/"images"/r["filename"]).read_bytes() for r in _image_rows}}]
_colab=types.ModuleType("google.colab")
_colab.files=types.SimpleNamespace(upload=lambda:_uploads.pop(0))
sys.modules["google.colab"]=_colab
'''
    # Force ZIP validation/extraction on every QA execution, using a fresh child folder.
    import uuid
    colab_folder=colab_folder/uuid.uuid4().hex;colab_folder.mkdir()
    colab_nb.cells[0].source=prelude+colab_nb.cells[0].source
    colab_nb.cells[2].source+='\nassert len(results["predictions"])==3\nassert results["metrics"]["labelled_images"]==3\n'
    colab_client=NotebookClient(colab_nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(colab_folder)}},store_widget_state=False)
    colab_executed=colab_client.execute()
    assert not any(o.output_type=='error' for c in colab_executed.cells for o in c.get('outputs',[]))
    summary={'passed':True,'three_code_cells':True,'real_notebook_kernel_executed':True,
             'model_load_seconds':demo.model_load_seconds,'gpu_or_auto_device':str(demo.device),
             'twenty_images':{'seconds':twenty['seconds'],'accuracy':twenty['metrics']['accuracy'],'correct':twenty['metrics']['correct']},
             'ten_images':{'seconds':ten['seconds'],'accuracy':ten['metrics']['accuracy'],'correct':ten['metrics']['correct']},
             'cpu_ten_seconds':cpu_result['seconds'],'notebook_ten_run_wall_seconds':notebook_seconds,
             'colab_zip_and_upload_route_simulated_locally':True,'hosted_colab_tested':False,
             'max_probability_difference_from_original':max_delta,'ground_truth_not_used_for_predictions':True,
             'renamed_images_match':True,'duplicate_and_corrupt_images_handled':True,'unknown_images_not_scored':True,
             'single_class_metrics_valid':True,'random_selection_seed':manifest['seed']}
    (folder/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (test_out/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
