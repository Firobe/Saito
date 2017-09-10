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

"""
    Return list of paths to comment files
"""
def retrieveFiles(directory = COMMENT_FOLDER):
    return [f for f in os.listdir(directory) if ('jsonl' in f)]

"""
    Create a WAV file containing the audio of a video
    for every given ID, in parllel
"""
def createAudio(files):
    Parallel(n_jobs=2)(delayed(oneFile)(f) for f in files)

def oneFile(f):
    sm = re.search('([a-z]{2}[0-9]+)', f)
    if sm: id = sm.group(1)
    else: raise NameError(f + " is not a Nico ID")
    try:
        createWAV(id)
    except Exception as e:
        print(str(e))

        

#files = retrieve_files(COMMENTS_DIR)
#P = parse_comments(COMMENTS_DIR + files[0])
#print(P)
createAudio(retrieveFiles())
