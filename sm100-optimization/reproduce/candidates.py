"""Isolated SM100 candidates; production package files and dispatch stay intact."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import textwrap

from sol_attn import interface
import sol_attn.sm100.mainloop as baseline_mainloop
import sol_attn.sm100.softmax as baseline_softmax


def load_module(name, path, text):
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_candidate(name, directory):
    if name not in {'skip_identity', 'wide32', 'skip_identity_wide32', 'fullpair', 'skip_identity_wide32_fullpair'}:
        raise ValueError(name)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    softmax = Path(baseline_softmax.__file__).read_text()
    if 'wide32' in name:
        begin = softmax.index('def _rescale_m64_partial_o(')
        end = softmax.index('\n\n@cute.jit', begin)
        part = softmax[begin:end]
        assert part.count('correction_width = 16') == 1
        assert part.count('tcgen05.copy.Repetition(8)') == 2
        part = part.replace('correction_width = 16', 'correction_width = 32')
        part = part.replace('tcgen05.copy.Repetition(8)', 'tcgen05.copy.Repetition(16)')
        softmax = softmax[:begin] + part + softmax[end:]
    if name.startswith('skip_identity'):
        begin = softmax.index('    relative_o = _zero_based_tmem_tensor', softmax.index('def _rescale_m64_partial_o('))
        end = softmax.index('\n\n@cute.jit', begin)
        body = softmax[begin:end]
        guard = '''    # A per-lane branch would violate tcgen05 .sync.aligned. The ballot
    # makes this decision identical for every lane of the participating warp.
    # An all-one scale leaves its accumulator elements unchanged. Producer
    # completion and the later probability-ready barrier remain in the caller.
    needs_rescale = Int32(cute.arch.vote_ballot_sync(alpha != Float32(1.0)))
    if needs_rescale != Int32(0):
'''
        softmax = softmax[:begin] + guard + textwrap.indent(body, '    ') + softmax[end:]
    soft_name = f'_candidate_softmax_{name}'
    load_module(f'sol_attn.sm100.{soft_name}', directory / f'{soft_name}.py', softmax)
    mainloop = Path(baseline_mainloop.__file__).read_text()
    assert mainloop.count('from .softmax import (') == 1
    mainloop = mainloop.replace('from .softmax import (', f'from .{soft_name} import (')
    if 'fullpair' in name:
        needle = '                if exact_owner_warp == Int32(0):'
        assert mainloop.count(needle) == 1
        mainloop = mainloop.replace(needle, '''                needs_pair_mask = (
                    valid0 != Int32(N_MEMBER)
                    or valid1 != Int32(N_MEMBER)
                    or q_len != Int32(M)
                )
                if exact_owner_warp == Int32(0) and needs_pair_mask:''')
        begin = mainloop.index('                semantic_row = (', mainloop.index('# Site 1: owner warp'))
        end = mainloop.index('                probabilities, next_max, next_sum, row_alpha = (', begin)
        mainloop = (mainloop[:begin] + '                if needs_pair_mask:\n'
                    + textwrap.indent(mainloop[begin:end], '    ') + mainloop[end:])
    main_name = f'_candidate_mainloop_{name}'
    load_module(f'sol_attn.sm100.{main_name}', directory / f'{main_name}.py', mainloop)
    api = Path(interface.__file__).read_text()
    assert api.count('from .sm100 import forward') == 1
    api = api.replace('from .sm100 import forward', f'from .sm100.{main_name} import forward')
    loaded = load_module(f'sol_attn._candidate_interface_{name}', directory / f'interface_{name}.py', api)
    provenance = dict(name=name, softmax_sha256=hashlib.sha256(softmax.encode()).hexdigest(),
                      mainloop_sha256=hashlib.sha256(mainloop.encode()).hexdigest(),
                      interface_sha256=hashlib.sha256(api.encode()).hexdigest())
    return loaded.sol_attn, provenance
