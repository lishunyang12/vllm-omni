"""Download the immutable model revisions recorded alongside the quality results."""
import json
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

work = Path(sys.argv[1]).resolve()
spec = json.loads((work / 'matrix.json').read_text())
records = {}
for key in ('qwen_image', 'wan22'):
    model = spec['models'][key]
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
