"""Independent saved-prediction, partition and checkpoint integrity checks."""
import argparse,csv,hashlib,json
import numpy as np
from train import ROOT,CLASSES,metrics,save_json
from improve import rows_for

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args();out=ROOT/args.directory
    with (out/'test_predictions.csv').open(newline='') as f:pred=list(csv.DictReader(f))
    rows=rows_for('test');dev=rows_for('train')+rows_for('val');assert [r['img_id'] for r in rows]==[r['img_id'] for r in pred]
    assert [r['diagnostic'] for r in rows]==[r['true_class'] for r in pred]
    y=np.array([CLASSES.index(r['true_class']) for r in pred]);p=np.array([[float(r['prob_'+c]) for c in CLASSES] for r in pred]);assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all();assert np.allclose(p.sum(1),1,atol=1e-6)
    calculated=metrics(y,p);saved=json.loads((out/'metrics.json').read_text())
    for k,v in calculated.items():
        if isinstance(v,float):assert np.isclose(v,saved[k],atol=1e-6),k
    assert calculated['confusion_matrix_rows_true_columns_predicted']==saved['confusion_matrix_rows_true_columns_predicted']
    for k in ['patient_id','lesion_id']:assert not ({r[k] for r in rows}&{r[k] for r in dev})
    hashes={r['img_id']:r['sha256'] for r in json.loads((ROOT/'data/manifest.json').read_text())};assert not ({hashes[r['img_id']] for r in rows}&{hashes[r['img_id']] for r in dev})
    lock=json.loads((out/'evaluation_lock.json').read_text());expected=lock.get('checkpoint_sha256',lock.get('model_sha256'));assert hashlib.sha256((out/'best_system.pt').read_bytes()).hexdigest()==expected
    assert hashlib.sha256((ROOT/'outputs/run_01/test.csv').read_bytes()).hexdigest()==lock['test_sha256']
    if isinstance(lock['source_sha256'],dict):
        for name,h in lock['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    else:assert hashlib.sha256((ROOT/'high_resolution.py').read_bytes()).hexdigest()==lock['source_sha256']
    example=False
    if (out/'example_prediction.json').exists():
        example=json.loads((out/'example_prediction.json').read_text());name=(ROOT/'outputs/run_01/example_image.txt').read_text().strip().replace('\\','/').split('/')[-1];row=next(r for r in pred if r['img_id']==name)
        assert example['predicted_class']==row['predicted_class'];assert all(abs(example['probabilities'][c]-float(row['prob_'+c]))<.005 for c in CLASSES);example=True
    checks={'passed':True,'saved_metrics_recomputed':True,'probabilities_valid':True,'test_rows_and_labels_unchanged':True,'development_test_patient_lesion_and_exact_image_overlap':False,'model_and_source_hashes_verified':True,'standalone_inference_agrees':example,'correct_test_predictions':int((y==p.argmax(1)).sum()),'test_images':len(y)};save_json(out/'verification.json',checks);print(json.dumps(checks))

if __name__=='__main__':main()
