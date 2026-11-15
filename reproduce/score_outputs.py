"""Strict full-resolution RGB SSIM, pooled PSNR and framewise AlexNet LPIPS.

Metrics use saved lossless PNGs, never resized previews or lossy MP4s. PSNR is
computed from pooled pixel MSE; SSIM and LPIPS are means over matched frames.
No automatic pass threshold: these metrics measure difference, not preference.
"""
import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def encode_video(paths, target, fps):
    import av
    with av.open(str(target), 'w') as container:
        size = Image.open(paths[0]).size
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast'}
        for path in paths:
            with Image.open(path) as image:
                frame = av.VideoFrame.from_ndarray(np.asarray(image.convert('RGB')), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)



def make_pair(a_path, b_path, reference, candidate, thumbnail=False):
    with Image.open(a_path) as image:
        a = image.convert('RGB')
    with Image.open(b_path) as image:
        b = image.convert('RGB')
    if a.size != b.size:
        raise ValueError('Paired viewing outputs require matching dimensions')
    if thumbnail:
        a.thumbnail((768, 768))
        b.thumbnail((768, 768))
    w, h = a.size
    canvas = Image.new('RGB', (w * 2, h + 40), '#18212f')
    canvas.paste(a, (0, 40))
    canvas.paste(b, (w, 40))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20)
    except OSError:
        font = ImageFont.load_default(size=20)
    draw.text((12, 8), reference, fill='white', font=font)
    draw.text((w + 12, 8), candidate, fill='white', font=font)
    return canvas


def encode_pair(refs, candidates, target, reference, candidate, fps):
    import av
    with av.open(str(target), 'w') as container:
        w, h = Image.open(refs[0]).size
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = w * 2, h + 40
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast'}
        for a_path, b_path in zip(refs, candidates, strict=True):
            pair = make_pair(a_path, b_path, reference, candidate)
            frame = av.VideoFrame.from_ndarray(np.asarray(pair), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def compare(refs, candidates, metric, device):
    import torch
    from skimage.metrics import structural_similarity
    if len(refs) != len(candidates) or not refs:
        raise ValueError('Frame counts must match and be nonzero')
    rows = []
    squared_error_sum, pixel_count = 0.0, 0
    with torch.inference_mode():
        for index, (ref, candidate) in enumerate(zip(refs, candidates, strict=True)):
            with Image.open(ref) as image:
                a = np.asarray(image.convert('RGB'))
            with Image.open(candidate) as image:
                b = np.asarray(image.convert('RGB'))
            if a.shape != b.shape:
                raise ValueError(f'Frame {index}: shape mismatch {a.shape} != {b.shape}')
            diff = a.astype(np.float64) - b.astype(np.float64)
            squared_error_sum += np.square(diff).sum()
            pixel_count += a.size
            mse = np.square(diff).mean()
            ssim = structural_similarity(a, b, data_range=255, channel_axis=-1,
                                         gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            def tensor(array):
                return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device).float().div(127.5).sub(1)
            distance = float(metric(tensor(a), tensor(b)).item())
            rows.append(dict(frame=index, ssim=float(ssim),
                             psnr_db=10 * math.log10(255**2 / mse) if mse else 'inf',
                             lpips=distance, exact_pixels=bool(np.array_equal(a, b))))
    pooled_mse = squared_error_sum / pixel_count
    return dict(frame_count=len(rows), ssim=float(np.mean([r['ssim'] for r in rows])),
                psnr_db=10 * math.log10(255**2 / pooled_mse) if pooled_mse else 'inf',
                lpips=float(np.mean([r['lpips'] for r in rows])),
                exact_pixels=all(r['exact_pixels'] for r in rows)), rows


def main():
    import lpips
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    work = args.work.resolve()
    spec = json.loads((work / 'matrix.json').read_text())
    metric = lpips.LPIPS(net='alex', version='0.1').to(args.device).eval()
    rows, failures, gallery = [], [], []
    output = work / 'outputs'
    for key, case in spec['cases'].items():
        galleries = []
        for label in spec['variants']:
            directory = output / key / label
            metadata_path = directory / 'metadata.json'
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text())
            if metadata['status'] != 'completed':
                continue
            paths = sorted((directory / 'frames').glob('*.png'))
            if len(paths) != case['frames']:
                raise ValueError(f'Wrong number of frames in {directory}')
            relative = directory.relative_to(work).as_posix()
            if case['frames'] > 1:
                encode_video(paths, directory / 'video.mp4', case['fps'])
                media = f'<video controls muted loop preload="metadata" src="{relative}/video.mp4"></video>'
            else:
                with Image.open(paths[0]) as image:
                    image.thumbnail((768, 768))
                    image.save(directory / 'preview.png')
                media = f'<a href="{relative}/frames/00000.png"><img src="{relative}/preview.png"></a>'
            galleries.append(f'<figure><figcaption>{label}</figcaption>{media}</figure>')
        gallery.append(f'<h2>{key}</h2><p>{html.escape(case["prompt"])}</p><div class="gallery">' + ''.join(galleries) + '</div>')
        # The BF16 control isolates the PR; quantization is compared to that same PR's BF16.
        for reference, candidate in [('base_bf16', 'pr_bf16'), ('pr_bf16', 'pr_mxfp8'), ('pr_bf16', 'pr_fp8_block')]:
            try:
                for label in (reference, candidate):
                    metadata = json.loads((output / key / label / 'metadata.json').read_text())
                    if metadata['status'] != 'completed':
                        raise ValueError(f'{label} generation did not complete')
                    if metadata['case'] != case:
                        raise ValueError(f'{label} sampling parameters differ from the manifest')
                a = sorted((output / key / reference / 'frames').glob('*.png'))
                b = sorted((output / key / candidate / 'frames').glob('*.png'))
                if len(a) != case['frames'] or len(b) != case['frames']:
                    raise ValueError('Unexpected frame count')
                aggregate, frames = compare(a, b, metric, args.device)
                record = dict(model=key, reference=reference, candidate=candidate, **aggregate)
                rows.append(record)
                target_dir = output / key / candidate
                if case['frames'] > 1:
                    encode_pair(a, b, target_dir / 'comparison.mp4', reference, candidate, case['fps'])
                else:
                    make_pair(a[0], b[0], reference, candidate, thumbnail=True).save(target_dir / 'comparison.png')
                (output / key / candidate / 'quality.json').write_text(json.dumps(dict(**record, frames=frames), indent=2) + '\n')
                print(json.dumps(record), flush=True)
            except (FileNotFoundError, ValueError) as error:
                failures.append(dict(model=key, reference=reference, candidate=candidate, error=str(error)))
    paired_gallery = []
    for key, case in spec['cases'].items():
        media = []
        for candidate in ('pr_bf16', 'pr_mxfp8', 'pr_fp8_block'):
            suffix = 'mp4' if case['frames'] > 1 else 'png'
            path = output / key / candidate / f'comparison.{suffix}'
            if not path.exists():
                continue
            relative = path.relative_to(work).as_posix()
            tag = f'<video controls muted loop src="{relative}"></video>' if suffix == 'mp4' else f'<img src="{relative}">'
            media.append(f'<figure><figcaption>{candidate}</figcaption>{tag}</figure>')
        paired_gallery.append(f'<h2>{key}: synchronized comparisons</h2>' + ''.join(media))
    evidence = dict(definitions=dict(ssim='Native RGB; gaussian sigma=1.5; population covariance; data_range=255; mean over frames',
                                    psnr_db='10 log10(255^2 / pooled RGB pixel MSE); inf means identical',
                                    lpips='AlexNet LPIPS v0.1; native resolution; RGB [-1,1]; mean over frames',
                                    pixels='Lossless PNG exports before video encoding; no resizing or frame truncation'),
                    results=rows, unavailable=failures)
    (work / 'metrics.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if rows:
        with (work / 'metrics.csv').open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    table = '<table><tr><th>Model</th><th>Reference → candidate</th><th>SSIM ↑</th><th>PSNR dB ↑</th><th>LPIPS ↓</th></tr>'
    for row in rows:
        psnr = row['psnr_db'] if isinstance(row['psnr_db'], str) else f"{row['psnr_db']:.4f}"
        table += f"<tr><td>{row['model']}</td><td>{row['reference']} → {row['candidate']}</td><td>{row['ssim']:.6f}</td><td>{psnr}</td><td>{row['lpips']:.6f}</td></tr>"
    report = '<!doctype html><meta charset="utf-8"><title>PR 10390 quality comparisons</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#f6f7f9;color:#18212f}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0;padding:12px;background:white}img,video{width:100%}figcaption{font-weight:600;margin-bottom:10px}td,th{padding:10px;border-bottom:1px solid #bbb;text-align:left}</style><h1>PR 10390: generated outputs and quality</h1><p>Matched model revisions, prompts, seeds and sampling settings. One example per model. Scores measure full-resolution pixel differences; they do not establish perceptual preference or broad quality equivalence. MP4s are viewing copies; metrics use lossless source frames.</p>'
    report += table + '</table>' + ''.join(paired_gallery) + '<h2>Individual outputs</h2>' + ''.join(gallery) + '<h2>Unavailable comparisons</h2><pre>' + html.escape(json.dumps(failures, indent=2)) + '</pre>'
    (work / 'report.html').write_text(report)


if __name__ == '__main__':
    main()
