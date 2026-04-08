"""Debug script: inspect VoxCPM2Model._generate_with_prompt_cache internals."""
import inspect
from voxcpm.model.voxcpm2 import VoxCPM2Model

print("=== _generate_with_prompt_cache signature ===")
print(inspect.signature(VoxCPM2Model._generate_with_prompt_cache))

print("\n=== Lines with audio_vae / decode / latent / yield ===")
src = inspect.getsource(VoxCPM2Model._generate_with_prompt_cache)
for i, line in enumerate(src.split('\n')):
    if any(kw in line.lower() for kw in ['audio_vae', 'decode', 'latent', 'yield', 'pred_audio']):
        print(f"{i:4d}: {line}")

print("\n=== _inference signature ===")
if hasattr(VoxCPM2Model, '_inference'):
    print(inspect.signature(VoxCPM2Model._inference))
else:
    print("No _inference method found")

print("\n=== Lines in _inference with yield / latent / decode ===")
if hasattr(VoxCPM2Model, '_inference'):
    src2 = inspect.getsource(VoxCPM2Model._inference)
    for i, line in enumerate(src2.split('\n')):
        if any(kw in line.lower() for kw in ['audio_vae', 'decode', 'latent', 'yield', 'pred_audio']):
            print(f"{i:4d}: {line}")

print("\n=== generate_with_prompt_cache (public) source ===")
if hasattr(VoxCPM2Model, 'generate_with_prompt_cache'):
    src3 = inspect.getsource(VoxCPM2Model.generate_with_prompt_cache)
    print(src3[:500])
