"""Download the official PAD-UFES-20 release and retain SCC/SEK/NEV only."""
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DOWNLOADS = ROOT / 'downloads'
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
CLASSES = {'SCC': 192, 'SEK': 235, 'NEV': 244}

def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def download(item):
    dest = DOWNLOADS / item['filename']
    info = item['content_details']
    if dest.exists() and dest.stat().st_size == info['size'] and sha256(dest) == info['sha256_hash']:
        return dest
    partial = dest.with_suffix(dest.suffix + '.part')
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = dict(HEADERS)
            if offset:
                headers['Range'] = f'bytes={offset}-'
            req = urllib.request.Request(info['download_url'], headers=headers)
            with urllib.request.urlopen(req, timeout=90) as response:
                mode = 'ab' if offset and response.status == 206 else 'wb'
                if mode == 'wb': offset = 0
                last = time.monotonic()
                with partial.open(mode) as output:
                    while chunk := response.read(2 * 1024 * 1024):
                        output.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last > 20:
                            print(f"{dest.name}: {offset / info['size']:.1%} ({offset / 1e6:.0f} MB)", flush=True)
                            last = time.monotonic()
            assert partial.stat().st_size == info['size'], 'Download size mismatch'
            assert sha256(partial) == info['sha256_hash'], 'Download checksum mismatch'
            partial.replace(dest)
            print(f'{dest.name}: downloaded and SHA-256 verified', flush=True)
            return dest
        except Exception as exc:
            print(f'{dest.name}: attempt {attempt+1}: {exc}', flush=True)
            if attempt == 4: raise
            time.sleep(2 * (attempt + 1))

def main():
    DATA.mkdir(exist_ok=True)
    (DATA / 'images').mkdir(exist_ok=True)
    DOWNLOADS.mkdir(exist_ok=True)
    with fetch('https://data.mendeley.com/public-api/datasets/zr7vgbcyr2') as response:
        release = json.load(response)
    files = release['files']
    meta_item = next(f for f in files if f['filename'] == 'metadata.csv')
    meta_path = download(meta_item)
    with meta_path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if r['diagnostic'] in CLASSES]
    counts = {c: sum(r['diagnostic'] == c for r in rows) for c in CLASSES}
    assert counts == CLASSES, counts
    wanted = {r['img_id'] for r in rows}
    assert len(wanted) == len(rows) == 671
    with (DATA / 'metadata.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Subset metadata: {counts}, {len(set(r["patient_id"] for r in rows))} patients', flush=True)
    archives = [f for f in files if f['filename'].endswith('.zip')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(download, archives))
    for path in paths:
        count = 0
        with zipfile.ZipFile(path) as z:
            for entry in z.infolist():
                name = Path(entry.filename).name
                if name in wanted and not entry.is_dir():
                    target = DATA / 'images' / name
                    with z.open(entry) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    count += 1
        print(f'{path.name}: extracted {count} selected images', flush=True)
    assert {p.name for p in (DATA / 'images').glob('*.png')} == wanted
    manifest = []
    for row in rows:
        p = DATA / 'images' / row['img_id']
        with Image.open(p) as img:
            size = img.size
            img.verify()
        manifest.append({'img_id': p.name, 'diagnostic': row['diagnostic'], 'sha256': sha256(p), 'width': size[0], 'height': size[1], 'bytes': p.stat().st_size})
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    provenance = {'source': 'https://data.mendeley.com/datasets/zr7vgbcyr2/1', 'doi': '10.17632/zr7vgbcyr2.1', 'license': 'CC BY 4.0', 'citation': 'Pacheco et al. (2020). PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. Data in Brief 32, 106221.', 'counts': counts, 'patients': len(set(r['patient_id'] for r in rows)), 'lesions': len(set(r['lesion_id'] for r in rows)), 'files': files, 'selected_images_verified': len(manifest), 'cleanup_pending': True}
    (DATA / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print('READY: all 671 images verified; archives can now be deleted.', flush=True)

if __name__ == '__main__':
    main()
