#!/bin/python
import vamp
import librosa
import matplotlib.pyplot as plt
import numpy as np
import json
from nicotools.nicodown_async import VideoDmc, VideoSmile
from os import listdir
import MeCab
mecab = MeCab.Tagger("-Ochasen")
print("Import OK")

def extract_melody(filename, position, around=5):
    data, rate = librosa.load(filename, offset = position - around,\
            duration = 2 * around)
    result = vamp.collect(data, rate, "mtg-melodia:melodia");
    frameDur, melody = result['vector']
    frameDur = frameDur.to_frame(rate)
    rich_melody = []
    cur = melody[0]
    save = 0
    for i in range(len(melody)):
        if melody[i] <= 0: melody[i] = 0
        if melody[i] != cur:
            rich_melody.append((cur, (i - save) / rate))
            cur = melody[i]
            save = i
    return rich_melody

def retrieve_files(directory='.'):
    return [f for f in listdir(directory) if ('jsonl' in f)]

def parse_comments(filename, limit=np.inf):
    print("Opening " + filename + "...")
    with open(filename, 'r') as myfile:
        data=myfile.readlines()
    raw = [json.loads(j) for j in data]
    for r in raw:
        r.pop('date'); r.pop('command');
        r['content'] = mecab.parse(r['content'])
    return sorted(raw, key = lambda r: r['vpos'])

COMMENTS_DIR = 'comments/'

files = retrieve_files(COMMENTS_DIR)
P = parse_comments(COMMENTS_DIR + files[0])
print(P)
