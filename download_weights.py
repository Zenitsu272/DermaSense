"""Fetch official pretrained weights in parallel, verifying the published hash prefix."""
import concurrent.futures
import hashlib
import math
from pathlib import Path
from download_subset import range_read

ROOT=Path(__file__).resolve().parent
URL='https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
SIZE=102540417
CHUNK=4*1024*1024

def main():
    target=ROOT/'pretrained/hub/checkpoints/resnet50-11ad3fa6.pth'
    target.parent.mkdir(parents=True,exist_ok=True)
    def get(i):
        data=range_read(URL,i*CHUNK,min(SIZE-1,(i+1)*CHUNK-1))
        print(f'Weight chunk {i+1}/{math.ceil(SIZE/CHUNK)}',flush=True)
        return data
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        payload=b''.join(pool.map(get,range(math.ceil(SIZE/CHUNK))))
    digest=hashlib.sha256(payload).hexdigest()
    assert len(payload)==SIZE and digest.startswith('11ad3fa6'),digest
    target.write_bytes(payload)
    print('PRETRAINED READY; SHA256='+digest,flush=True)

if __name__=='__main__':main()
