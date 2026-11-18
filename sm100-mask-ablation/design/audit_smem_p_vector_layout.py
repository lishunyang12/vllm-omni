#!/usr/bin/env python3
"""CPU-only finite audit for the serial shared-P 16-byte vector-store candidate.

No CUDA imports/execution. Layout comes from the exported fixed-image helpers.
This proves the modeled address/packing invariants, not GPU compilation or
hardware correctness. The full existing byte/oracle gate is still required.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path

root = Path(__file__).resolve().parent
rel = Path('techniques/sparse_backends/sol_attn/sm100/mainloop.py')
serial = (root / 'smem-p-serial' / rel).read_text()
vector = (root / 'smem-p-vector' / rel).read_text()
helpers = (root / 'blackwell_helpers.py').read_text()
atom = (root / 'tcgen05-layout-source.py').read_text()
assert 'major_mode_size_bits % sw128_num_contiguous_bits == 0' in helpers
assert 'return SmemLayoutAtomKind.K_SW128' in helpers
assert 'order=(0, 1, 2) if is_k_major else (1, 0, 2)' in helpers
assert 'sw = core.make_swizzle(3, 4, 3)' in atom
assert '(8, num_contiguous_elems), stride=(num_contiguous_elems, 1)' in atom
assert 'num_contiguous_elems = num_contiguous_bits // element_type.width' in atom
assert 'num_contiguous_bits = 1024' in atom

# K-major BF16 chooses K_SW128: atom (8,64):(64,1). Tiling to
# (64,128,1), order=(0,1,2), adds 8 M tiles before the second K tile.
# Its coalesced outer map is (64,(64,2),1):(64,(1,4096),8192).
# The iterator swizzle operates on BYTE addresses after this element offset.
def outer(row, column):
    return row * 64 + column % 64 + (column // 64) * 4096

def swizzle_byte(address):
    return address ^ ((address & (7 << 7)) >> 3)

def physical(base, row, column):
    return swizzle_byte(base + 2 * outer(row, column))

offsets = [outer(r, c) for r in range(64) for c in range(128)]
assert sorted(offsets) == list(range(8192))
segments = 0
for base in range(0, 1024, 128):
    # All possible 128B allocation-alignment phases of this 1KiB-period
    # swizzle. Higher address bits do not enter the XOR and are unchanged.
    written = []
    for row in range(64):
        for column in range(0, 128, 8):
            raw = base + 2 * outer(row, column)
            addresses = [physical(base, row, column + j) for j in range(8)]
            assert raw % 16 == 0
            assert addresses[0] % 16 == 0
            assert addresses == [addresses[0] + 2*j for j in range(8)]
            assert outer(row, column) % 8 == 0
            written.extend(byte for a in addresses for byte in (a, a+1))
            segments += 1
    assert sorted(written) == list(range(base, base + 16384))

# Ld16x64b.x64, 4 owner warps. The instruction's per-thread destination
# layout in bits is ((2,2,8),(32,64)):((32768,32,4096),(1,64)).
# Thus bit1 of lane chooses column parity; bit0 chooses row+8; lane//4
# chooses row0..7. The existing score/softmax helper already uses xor2.
def owner(tid):
    lane = tid % 32
    row = (16 * (tid // 32) + lane // 4 + 8*(lane % 2) + 16) % 64
    parity = (lane // 2) % 2
    return row, parity

scalar_coverage = Counter()
vector_coverage = Counter()
for tid in range(128):
    row, parity = owner(tid)
    peer_row, peer_parity = owner(tid ^ 2)
    assert row == peer_row and parity != peer_parity
    for i in range(64):
        scalar_coverage[row, 2*i + parity] += 1
    if parity == 0:
        for chunk in range(16):
            for j in range(8):
                vector_coverage[row, 8*chunk + j] += 1
assert scalar_coverage == vector_coverage
assert len(vector_coverage) == 8192
assert set(vector_coverage.values()) == {1}

# Symbolic bytes prove PRMT preserves any BF16 bit pattern: no numeric
# values or special-value assumptions are needed after the unchanged CVT.
def prmt(a, b, selector):
    source = a + b
    return tuple(source[(selector >> (4*i)) & 7] for i in range(4))

symbols = tuple((c, byte) for c in range(8) for byte in range(2))
packed = ()
for pair in range(2):
    c = pair * 4
    own = symbols[2*c:2*c+2] + symbols[2*(c+2):2*(c+2)+2]
    peer = symbols[2*(c+1):2*(c+1)+2] + symbols[2*(c+3):2*(c+3)+2]
    packed += prmt(own, peer, 0x5410) + prmt(own, peer, 0x7632)
assert packed == symbols

# Remove exactly the replaced helper and its added owner_tidx argument.
# All synchronization, route math, K/V, MMA ordering, and epilogue source
# must otherwise equal the already GPU-validated scalar serial candidate.
def without_helper(source):
    start = source.index('@cute.jit\ndef _store_pair_probability_smem(')
    end = source.index('@cute.jit\ndef _load_pack_k_half(', start)
    return source[:start] + source[end:]
normalized = vector.replace('probabilities, pair_coords, sExactP, owner_tidx',
                            'probabilities, pair_coords, sExactP')
assert without_helper(serial) == without_helper(normalized)
assert serial.count('cute.arch.fence_view_async_shared()') == vector.count('cute.arch.fence_view_async_shared()')

print('CPU-only audit; GPU correctness and performance are not established.')
print('Layout: K_SW128, atom (8,64):(64,1), byte Swizzle(3,4,3).')
print(f'PASS: {segments} segments across all 8 allocation phases; each physically contiguous and 16B aligned.')
print('PASS: each phase covers all 16384 allocation bytes exactly once, no out-of-bounds stores.')
print('PASS: 128 owner threads, xor2 row/parity pairing, 64 writers cover 8192 BF16 elements exactly once.')
print('PASS: symbolic PRMT bytes equal the original eight BF16 bit patterns in column order.')
print('PASS: outside the store helper and its argument, source equals scalar serial byte-for-byte.')
for filename in ('blackwell_helpers.py', 'tcgen05-layout-source.py', 'smem-p-serial.patch', 'smem-p-vector.patch'):
    print(f'SHA256 {filename}: {sha256((root/filename).read_bytes()).hexdigest()}')
