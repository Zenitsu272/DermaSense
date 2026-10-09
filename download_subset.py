"""Selectively extract official ZIP entries over HTTP Range requests."""
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
import time
import urllib.request
import zipfile
import zlib
from PIL import Image
from prepare_data import ROOT, DATA, DOWNLOADS, HEADERS, CLASSES, download, fetch, sha256

def range_read(url, start, end):
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={**HEADERS, 'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=60) as r:
                assert r.status == 206, f'Server did not honor range: {r.status}'
                assert r.headers['Content-Range'].startswith(f'bytes {start}-'), 'Wrong range returned'
                result = r.read()
                assert len(result) == end-start+1
                return result
        except Exception:
            if attempt == 5: raise
            time.sleep(attempt+1)

class RemoteArchive(io.RawIOBase):
    def __init__(self, url, size): self.url, self.size, self.pos = url, size, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos
    def read(self, n=-1):
        n = min(self.size-self.pos, n if n >= 0 else self.size-self.pos)
        if n <= 0: return b''
        data = range_read(self.url, self.pos, self.pos+n-1)
        self.pos += len(data)
        return data

def extract(job):
    url, archive_size, entry = job
    target = DATA / 'images' / Path(entry.filename).name
    if target.exists() and target.stat().st_size == entry.file_size:
        payload = target.read_bytes()
        if zlib.crc32(payload) & 0xffffffff == entry.CRC: return target
    offset = entry.header_offset
    end = min(archive_size-1, offset + 30 + len(entry.filename.encode('utf-8')) + len(entry.extra) + entry.compress_size + 256)
    payload = range_read(url, offset, end)
    assert payload[:4] == b'PK\x03\x04'
    name_len, extra_len = struct.unpack_from('<HH', payload, 26)
    begin = 30 + name_len + extra_len
    if len(payload) < begin + entry.compress_size:
        payload = range_read(url, offset, offset + begin + entry.compress_size-1)
    compressed = payload[begin:begin+entry.compress_size]
    if entry.compress_type == zipfile.ZIP_DEFLATED: content = zlib.decompress(compressed, -15)
    elif entry.compress_type == zipfile.ZIP_STORED: content = compressed
    else: raise ValueError(f'Unsupported ZIP compression {entry.compress_type}')
    assert len(content) == entry.file_size
    assert zlib.crc32(content) & 0xffffffff == entry.CRC, f'CRC mismatch: {entry.filename}'
    with Image.open(io.BytesIO(content)) as image: image.verify()
    target.write_bytes(content)
    return target

def main():
    (DATA / 'images').mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    with fetch('https://data.mendeley.com/public-api/datasets/zr7vgbcyr2') as r: release = json.load(r)
    files = release['files']
    meta = download(next(f for f in files if f['filename'] == 'metadata.csv'))
    with meta.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames
        rows = [r for r in reader if r['diagnostic'] in CLASSES]
    assert {c:sum(r['diagnostic']==c for r in rows) for c in CLASSES} == CLASSES
    wanted = {r['img_id'] for r in rows}
    with (DATA/'metadata.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields); writer.writeheader(); writer.writerows(rows)
    jobs=[]
    for f in files:
        if not f['filename'].endswith('.zip'): continue
        info=f['content_details']
        req=urllib.request.Request(info['download_url'],headers=HEADERS,method='HEAD')
        with urllib.request.urlopen(req,timeout=40) as r: url=r.geturl()
        with zipfile.ZipFile(RemoteArchive(url,info['size'])) as z:
            selected=[e for e in z.infolist() if Path(e.filename).name in wanted]
        jobs.extend((url,info['size'],e) for e in selected)
        print(f"{f['filename']}: {len(selected)} selected entries",flush=True)
    assert len(jobs)==len(wanted)==671
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for i, path in enumerate(pool.map(extract,jobs),1):
            if i % 25 == 0 or i == 671:
                print(f'Extracted and CRC-verified {i}/671 images ({time.monotonic()-start:.0f}s)',flush=True)
    assert {p.name for p in (DATA/'images').glob('*.png')} == wanted
    manifest=[]
    for r in rows:
        p=DATA/'images'/r['img_id']
        with Image.open(p) as im: w,h=im.size; im.verify()
        manifest.append({'img_id':p.name,'diagnostic':r['diagnostic'],'sha256':sha256(p),'width':w,'height':h,'bytes':p.stat().st_size})
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2))
    provenance={'source':'https://data.mendeley.com/datasets/zr7vgbcyr2/1','doi':'10.17632/zr7vgbcyr2.1','license':'CC BY 4.0','citation':'Pacheco et al. (2020), PAD-UFES-20, Data in Brief 32:106221. DOI:10.1016/j.dib.2020.106221','method':'HTTP byte ranges of original ZIP files; selected entries checked against ZIP CRC32; metadata checked against publisher SHA256; local image SHA256 manifest recorded. Full archives were not downloaded completely.','files':files,'counts':CLASSES,'patients':len(set(r['patient_id'] for r in rows)),'lesions':len(set(r['lesion_id'] for r in rows)),'selected_images_verified':len(manifest),'bytes_retained':sum(x['bytes'] for x in manifest),'cleanup_pending':True}
    (DATA/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('READY: all 671 selected images verified.',flush=True)

if __name__=='__main__': main()
