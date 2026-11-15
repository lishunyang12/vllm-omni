"""Verify exported pixels, score aggregation and complete MP4 decoding."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import statistics

import av
from PIL import Image


def verify_frame(item):
    path, expected_hash, size = item
    with Image.open(path) as image:
        assert image.size == size, path
        actual_hash = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
    assert actual_hash == expected_hash, path


def verify_video(item):
    path, width, height, frames, fps, work = item
    with av.open(str(path)) as container:
        assert len(container.streams.video) == 1 and not container.streams.audio, path
        stream = container.streams.video[0]
        assert stream.average_rate == fps and stream.codec_context.name == 'h264', path
        stream.codec_context.thread_count = 2
        count = 0
        for frame in container.decode(video=0):
            assert (frame.width, frame.height) == (width, height), path
            count += 1
        assert count == frames, (path, count, frames)
    return dict(path=path.relative_to(work).as_posix(), frames=count, width=width, height=height,
                fps=fps, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('work', type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    spec = json.loads((work / 'matrix.json').read_text())
    metrics = json.loads((work / 'metrics.json').read_text())
    assert not metrics['unavailable'], metrics['unavailable']
    assert len(metrics['results']) == len(spec['cases']) * 3
    frames, videos = [], []
    for key, case in spec['cases'].items():
        for label, variant in spec['variants'].items():
            directory = work / 'outputs' / key / label
            metadata = json.loads((directory / 'metadata.json').read_text())
            assert metadata['status'] == 'completed' and metadata['case'] == case
            assert metadata['variant'] == variant
            paths = sorted((directory / 'frames').glob('*.png'))
            assert len(paths) == len(metadata['pixel_sha256']) == case['frames']
            frames.extend((path, digest, (case['width'], case['height']))
                          for path, digest in zip(paths, metadata['pixel_sha256'], strict=True))
            if case['frames'] > 1:
                videos.append((directory / 'video.mp4', case['width'], case['height'],
                               case['frames'], case['fps'], work))
                if label != 'base_bf16':
                    videos.append((directory / 'comparison.mp4', case['width'] * 2, case['height'] + 40,
                                   case['frames'], case['fps'], work))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify_frame, frames))
    print('Verified lossless pixel hashes:', len(frames), flush=True)
    for row in metrics['results']:
        quality = json.loads((work / 'outputs' / row['model'] / row['candidate'] / 'quality.json').read_text())
        per_frame = quality['frames']
        assert len(per_frame) == row['frame_count'] == spec['cases'][row['model']]['frames']
        for key in ['ssim', 'lpips']:
            assert math.isfinite(row[key])
            assert math.isclose(row[key], statistics.mean(frame[key] for frame in per_frame), abs_tol=1e-12)
        assert -1 <= row['ssim'] <= 1
        assert row['exact_pixels'] == all(frame['exact_pixels'] for frame in per_frame)
        mse = statistics.mean(0.0 if frame['psnr_db'] == 'inf' else 255**2 * 10**(-frame['psnr_db'] / 10)
                              for frame in per_frame)
        if mse == 0:
            assert row['psnr_db'] == 'inf' and row['ssim'] == 1 and abs(row['lpips']) < 1e-8
        else:
            assert math.isclose(row['psnr_db'], 10 * math.log10(255**2 / mse), abs_tol=1e-10)
    with ThreadPoolExecutor(max_workers=2) as pool:
        verified_videos = list(pool.map(verify_video, videos))
    result = dict(status='passed', lossless_png_count=len(frames), quality_pairs=len(metrics['results']),
                  mp4_count=len(verified_videos), integrity_decoder_pyav_version=av.__version__,
                  videos=verified_videos)
    (work / 'artifact-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Verified MP4s:', len(verified_videos), flush=True)


if __name__ == '__main__':
    main()
