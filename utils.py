import itertools
import os
from random import shuffle

def calc_hparams(hparams, shuffle=False):
    arr =[]
    for row in itertools.product(*hparams.values()):
        arr.append({k:v for k,v in zip(hparams.keys(), row)})
    if shuffle:
        shuffle(arr)
    for i in arr:
        print(i)
    return arr