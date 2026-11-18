import json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
w=Path(sys.argv[1]);rev='0919fb684102ef115126612f3fbb84d1027e7317'
files=json.loads((w/'bundle/model-files.json').read_text())
def download(name):
 for attempt in range(4):
  try:
   path=Path(hf_hub_download('MiniMaxAI/MiniMax-H3',filename=name,revision=rev))
   print('DOWNLOADED',name,path.stat().st_size,flush=True);return name,str(path)
  except Exception as exc:
   response=getattr(exc,'response',None)
   if response is None or response.status_code not in (429,500,502,503,504) or attempt==3:raise
   delay=int(response.headers.get('Retry-After','60'));print('RETRY',name,response.status_code,delay,flush=True);time.sleep(delay)
with ThreadPoolExecutor(max_workers=4) as pool:paths=dict(pool.map(download,files))
models={name:str(Path(paths[f'{name}/model_index.json']).parent) for name in ['FL2VA','Ref2VA']}
(w/'evidence/models.json').write_text(json.dumps(dict(repo='MiniMaxAI/MiniMax-H3',revision=rev,paths=models),indent=2)+'\n')
print('NATIVE_MODEL_PARTITIONS_READY',flush=True)
