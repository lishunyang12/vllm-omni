"""Download the immutable model revisions recorded alongside the quality results."""
import json
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

work = Path(sys.argv[1]).resolve()
spec = json.loads((work / 'matrix.json').read_text())
records = {}
for key, model in spec['models'].items():
    revision = model['revision']
    if not revision or len(revision) != 40:
        raise ValueError(f'Model {key} must have an immutable revision')
    options = {'ignore_patterns': ['assets/**', '*.md', '.gitattributes']}
    if key == 'h3_encoder':
        options = {'allow_patterns': ['text_encoder/**']}
    if key == 'h3_video':
        options = {'allow_patterns': ['FL2VA/**', 'text_encoder/**', '*.json']}
    path = snapshot_download(model['model'], revision=revision, max_workers=8, **options)
    if key == 'h3_video':
        path += '/FL2VA'
    records[key] = dict(model, path=path)
    (work / 'evidence/models.json').write_text(json.dumps(records, indent=2) + '\n')
ref = work / 'hf-cache/hub/models--MiniMaxAI--MiniMax-H3/refs/main'
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(records['h3_encoder']['revision'])
(work / 'evidence/h3-video-model.json').write_text(json.dumps({'h3_video': records['h3_video']}, indent=2) + '\n')
(work / 'evidence/h3-video-download.exit').write_text('0\n')
