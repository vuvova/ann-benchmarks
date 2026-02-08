#!/usr/bin/env python

import h5py
import sys

val=tag=None

if '=' in sys.argv[1]:
    key,val=sys.argv[1].split('=',1)
    assert val
elif '@' in sys.argv[1]:
    key,tag=sys.argv[1].split('@',1)
    assert tag
    tag='_'+tag+'_'
else:
    key=sys.argv[1]
    assert key

for f in sys.argv[2:]:
    try:
        h=h5py.File(f, 'r+')
    except:
        continue
    if tag:
        val=f.partition(tag)[-1].partition('_')[0]
        assert val
    if val:
        if h.attrs[key] != val:
            print(f"{f}\n  {key}={h.attrs[key]} -> {val}")
            h.attrs[key]=val
    else:
        print(f"{f}\n  {key}={h.attrs[key]}")
