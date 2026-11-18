"""Diagnostic only: force the public cuDNN non-CLC scheduler, then install the probe."""
from functools import wraps


def install():
    import cudnn.block_sparse_attention as bsa
    from forward_probe import install as install_measurement
    original = bsa.block_sparse_attention_forward
    if not getattr(original, '_h3_no_clc_diagnostic', False):
        @wraps(original)
        def no_clc(*args, **kwargs):
            kwargs['use_clc'] = False
            return original(*args, **kwargs)
        no_clc._h3_no_clc_diagnostic = True
        bsa.block_sparse_attention_forward = no_clc
    install_measurement()
