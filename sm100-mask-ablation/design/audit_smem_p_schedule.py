#!/usr/bin/env python3
"""Persisted copy of the previously executed CPU-only SMEM-P dependency audit.

The assertions and model match the interactive audit. Running this file records
another CPU audit execution, not GPU validation or an expanded protocol proof.
"""
from collections import defaultdict, deque
from pathlib import Path

root=Path('/work/design-agent')
rel=Path('techniques/sparse_backends/sol_attn/sm100/mainloop.py')
base=(Path('/baseline-source')/rel).read_text()
serial=(root/'smem-p-serial'/rel).read_text()
overlap=(root/'smem-p-overlap'/rel).read_text()
for source in (serial,overlap):
 for start,end in [
  ('            # Route generation:', '        if warp_idx == Int32(0) and has_pair_exact:'),
  ('@cute.jit\ndef _store_pair_probability_chunked_tmemp(', '@cute.jit\ndef _store_pair_probability_smem('),
 ]:
  if end in base:
   assert base[base.index(start):base.index(end)]==source[source.index(start):source.index(end)]
  else:
   base_end=base.index('@cute.jit\ndef _load_pack_k_half(')
   assert base[base.index(start):base_end]==source[source.index(start):source.index(end)]
# Dependency-graph audit of a group with one exact shared-P slot and one K/V
# stage. Issue and completion are distinct; the issuer orders QK(i+1) before
# PV(i). This checks for circular waits rather than estimating execution time.
for n in range(1,130):
 deps={}
 def add(name,*requirements):
  deps[name]=set(requirements)
 def e(tag,i):return f'{tag}:{i}'
 for i in range(n):
  add(e('Kissue',i),*([e('Vissue',i-1),e('QKdone',i-1)] if i else []))
  add(e('Kdone',i),e('Kissue',i))
  add(e('Vissue',i),e('Kissue',i),*([e('PVdone',i-1)] if i else []))
  add(e('Vdone',i),e('Vissue',i))
  add(e('QKissue',i),e('Kdone',i),*([e('Sfree',i-1)] if i else []),*([e('PVissue',i-2)] if i>=2 else []))
  add(e('QKdone',i),e('QKissue',i),*([e('QKdone',i-1)] if i==1 else []),*([e('PVdone',i-2)] if i>=2 else []))
  add(e('Sload',i),e('QKdone',i),*([e('Pready',i-1)] if i else []))
  add(e('Sfree',i),e('Sload',i))
  add(e('softmax',i),e('Sload',i))
  add(e('Pwrite',i),e('softmax',i),*([e('PVack',i-1)] if i else []))
  add(e('Orescale',i),e('softmax',i),*([e('PVack',i-1)] if i else []))
  add(e('Pready',i),e('Pwrite',i),e('Orescale',i),e('Vdone',i),*([e('QKissue',i+1)] if i+1<n else [e('QKissue',i)]))
  add(e('PVissue',i),e('Pready',i),*([e('PVack',i-1)] if i else []))
  add(e('PVdone',i),e('PVissue',i),e('QKdone',i+1 if i+1<n else i))
  add(e('PVack',i),e('PVdone',i),e('softmax',i+1) if i+1<n else e('Pready',i))
 edges=defaultdict(set)
 degree={key:len(values) for key,values in deps.items()}
 for key,values in deps.items():
  for value in values:
   assert value in deps,(key,value)
   edges[value].add(key)
 todo=deque(key for key,value in degree.items() if not value)
 order=[]
 while todo:
  key=todo.popleft();order.append(key)
  for nxt in edges[key]:
   degree[nxt]-=1
   if degree[nxt]==0:todo.append(nxt)
 assert len(order)==len(deps),(n,[key for key,value in degree.items() if value])
 position={key:i for i,key in enumerate(order)}
 for i in range(n-1):
  assert position[e('Sfree',i)]<position[e('QKissue',i+1)]
  assert position[e('PVdone',i)]<position[e('Pwrite',i+1)]
  assert position[e('PVdone',i)]<position[e('Orescale',i+1)]
  assert position[e('QKdone',i)]<position[e('Kissue',i+1)]
  assert position[e('PVdone',i)]<position[e('Vissue',i+1)]
print('CPU audit passed: approximate route and exact K/V producer unchanged; event graph acyclic for 1..129 pairs, with S/P/O/K/V lifetime ordering.')
