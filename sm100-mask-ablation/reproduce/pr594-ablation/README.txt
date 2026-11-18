PR594 regression isolation after the independent-process oracle failure
=====================================================================
All sources are experimental copies. No pinned source, PR, or previous
experiment was edited. No GPU work was performed by the design agent.

Reference pins:
  63 = 63b4e65e9db1dfea45cc7d394902145c2f520821 (PR593)
  150 = 150114b6f0079bcd479c6c87bd0b15174581d984 (PR594)

Name              Base  alpha==1 skip  width32/x16  skip full-tile mask
63-alpha-only      63    on            off         off
63-wide-only       63    off           on          off
63-mask-only       63    off           off         on
150-no-alpha       150   off           on          on
150-no-wide        150   on            off         on
150-no-mask        150   on            on          off

Each directory contains the original-name sol_attn package under
techniques/sparse_backends. Each corresponding .patch is relative to its
specified pinned base and is separately reversible. manifest.json records
source/patch hashes and exact flags. The build script in the parent directory
constructs each factor from the pinned package snapshots; it also verifies
that mainloop.py and softmax.py are the only differing Python package files.

Minimum localization path
  1. Freeze the failing T8193/H1/seed71/tau1/diag/no-mask QKV and confirm
     its digest in every run. Use a fresh process, one original-name sol_attn
     import and CUTE_DSL_DISABLE_FILE_CACHING=1 for every source.
  2. Run the three 63-only variants. Compare independently against the PR593
     frozen output AND the independent hybrid oracle; save floats, byte
     mismatch, per-Q-block differences, repeats and PTX/cubin.
  3. If one factor alone fails, run its matching 150-no variant to check
     whether reverting it repairs the combined PR594 as well.
  4. If all three single factors pass, run all three 150-no variants, which
     are the pairwise factor combinations. This distinguishes interactions
     from a single-factor regression. Do not select a repair from timing.
  5. A passing minimal repair still needs fresh-process independent-oracle
     small/partial-tail/all/mixed-mask and graph-mutation gates, then frozen
     H3 validation. Earlier same-process byte passes against PR594 are not
     sufficient evidence after the new oracle failure.

All six variants parsed as Python, and factor separation was checked on CPU.
No pass/fail or performance result for these variants is claimed here.
