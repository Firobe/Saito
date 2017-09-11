#!/bin/python
import numpy as np
import json, re, os
import MeCab
from config import *
from nico import createWAV
from joblib import Parallel, delayed

"""
    Return list of comments from a given comment file
    The comments are processed by MECAB and are sorted
    by time in the video
"""
def parseComments(filename, limit=np.inf):
    print("Opening " + filename + "...")
    with open(filename, 'r') as myfile:
        data=myfile.readlines()
    raw = [json.loads(j) for j in data]
    for r in raw:
        r.pop('date'); r.pop('command');
        r['content'] = mecab.parse(r['content'])
    return sorted(raw, key = lambda r: r['vpos'])

mecab = MeCab.Tagger("-Ochasen")

