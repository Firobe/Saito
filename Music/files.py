#!/bin/python

import json, urllib.request, re, zipfile, os
from config import *

"""
    Return list of paths to comment files
"""
def retrieveFiles(directory = COMMENT_FOLDER):
    return [f for f in os.listdir(directory) if ('jsonl' in f)]

def getFilteredIDs(category, criterion):
    f = open(METADATA_FOLDER + category + ".jsonl", "r") 
    data = f.readlines()
    raw = [json.loads(j) for j in data]
    return [cat + "/" + v['video_id'] + ".jsonl"
            for v in raw if criterion(v)]

def isMusicAndPopular(v):
    return v['category'] == '音楽' and v['comment_num'] > 500

def retrieveCategories():
    page = urllib.request.urlopen(DATASET_URL + "video/video.html").read()
    s = str(page).replace('\\n', '\n')
    return [m.group(1) for m in re.finditer('href="(.*).zip"', s)]

def unzip(file, toDir, members = None):
    z = zipfile.ZipFile(file, 'r')
    z.extractall(toDir, members = members)
    z.close()
    os.remove(file)

def download(url, file):
    with open(file, 'wb') as f:
        data = urllib.request.urlopen(url).read()
        f.write(data)

"""
    This script will download metadata from the NicoNico set
    and filter only videos using which are music and have at least 500
    comments. It will then download only the comments of these videos.

    Uses config.py
"""
# Get every "category" existing
categories = retrieveCategories()
for cat in categories:
    print("Category " + cat + "/" + categories[-1])
    filename = cat + ".zip"
    if os.path.isfile(METADATA_FOLDER + cat + ".jsonl"):
        print("> Skipped !")
    else:
        # Download and unzip metadata file
        url = DATASET_URL + "video/" + filename
        download(url, METADATA_FOLDER + filename)
        unzip(METADATA_FOLDER + filename, METADATA_FOLDER)
        # Extract ids of interesting videos
        ids = getFilteredIDs(cat, isMusicAndPopular)
        # Download comments file and extract only filtered files
        url = DATASET_URL + "comment/" + filename
        download(url, COMMENT_FOLDER + filename)
        unzip(COMMENT_FOLDER + filename, COMMENT_FOLDER, ids)
        print("> Got " + str(len(ids)) + " musics !")
