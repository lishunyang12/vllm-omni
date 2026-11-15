"""Download only the selected serving model and its pinned H3 encoder dependency."""
import json
from pathlib import Path
import sys
from huggingface_hub import snapshot_download
work=Path(sys.argv[1]).resolve()
key=sys.argv[2]
spec=json.loads((work/'matrix.json').read_text())
if key not in ('h3_image','h3_video','qwen_image','wan22'): raise ValueError(key)
keys=[key]+(['h3_encoder'] if key.startswith('h3_') else [])
records={}
for name in keys:
    model=spec['models'][name]
    assert len(model['revision'])==40
    options={'ignore_patterns':['assets/**','*.md','.gitattributes']}
    if name=='h3_encoder': options={'allow_patterns':['text_encoder/**']}
    if name=='h3_video': options={'allow_patterns':['FL2VA/**','text_encoder/**','*.json']}
    path=snapshot_download(model['model'],revision=model['revision'],max_workers=8,**options)
    if name=='h3_video': path+='/FL2VA'
    records[name]=dict(model,path=path)
if 'h3_encoder' in records:
    ref=work/'hf-cache/hub/models--MiniMaxAI--MiniMax-H3/refs/main'
    ref.parent.mkdir(parents=True,exist_ok=True)
    ref.write_text(records['h3_encoder']['revision'])
(work/'evidence/serving-models.json').write_text(json.dumps(records,indent=2)+'\n')
