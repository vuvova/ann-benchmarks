#!/usr/bin/env python

import h5py
import sys
import random

if len(sys.argv) < 4:
    print("Usage: shrink-dataset.py from.hdf5 to.hdf5 rows [neighbors]")
    exit()

fnfrom=sys.argv[1]
fnto=sys.argv[2]
rows=int(sys.argv[3])
if len(sys.argv) < 5:
    neighbors=rows
else:
    neighbors=int(sys.argv[4])

f=h5py.File(fnfrom)
fto=h5py.File(fnto, 'x')

neighbors = min(neighbors, len(f['neighbors'][0]))

# assuming one attribute "distance" on /, no groups,
# and four datasets "distances", "neighbors", "test", "train"

print("Copying attributes, distances, and test")
fto.attrs["distance"]=f.attrs["distance"]
fto.create_dataset_like("test", f["test"], data=f["test"])
fto.create_dataset_like("distances", f["distances"], data=f["distances"][:,:neighbors],
                        shape=(len(f["distances"]), neighbors))

d=fto.create_dataset_like("train", f["train"], shape=(rows, len(f["train"][0])))
n=fto.create_dataset_like("neighbors", f["neighbors"],
                          shape=(len(f["neighbors"]), neighbors))

print("Collecting neighbors")
s=set()
for i in f["neighbors"]:
    for j in i[:neighbors]:
        s.add(j)

if rows < len(s):
    print(f"Cannot shrink {fnfrom} to less than {len(s)} rows")
    exit()

print(f"Got {len(s)}, padding up to {rows}")
while len(s) < rows:
    s.add(random.randint(0, len(f["train"])))

s=list(s)
random.shuffle(s)
reverse=[None] * len(f["train"])

print("Copying train")
for i in range(len(s)):
    d[i]=f["train"][s[i]]
    reverse[s[i]]=i

print("Copying neighbors")
for i in range(len(f["neighbors"])):
    n[i] = [reverse[x] for x in f["neighbors"][i][:neighbors]]
