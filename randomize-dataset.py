#!/usr/bin/env python

import h5py
import sys
import scipy

if len(sys.argv) < 2:
    print("Usage: randomize-dataset.py from.hdf5 to.hdf5")
    exit()

fnfrom=sys.argv[1]
fnto=sys.argv[2]

f=h5py.File(fnfrom)
fto=h5py.File(fnto, 'x')

rrot=scipy.stats.ortho_group.rvs(dim=len(f["train"][0]))

print("Copying everything from the source file")
fto.attrs["distance"]=f.attrs["distance"]
fto.create_dataset_like("distances", f["distances"], data=f["distances"])
fto.create_dataset_like("neighbors", f["neighbors"], data=f["neighbors"])

def randomize(dset):
    src=f[dset]
    dst=fto.create_dataset_like(dset, src)
    for i in range(len(src)):
        dst[i]=rrot.dot(src[i])
        if i % 10000 == 0: print(f"{dset}: {i} of {len(src)}")

randomize("test")
randomize("train")
