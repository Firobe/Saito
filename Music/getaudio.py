import nicopy
import subprocess, os
from config import *

def login():
    if login.logged:
        return login.cookie
    login.cookie = nicopy.login(NICO_EMAIL, NICO_PASSWORD)
    login.logged = True
    return login.cookie
login.logged = False

def videoType(vidId):
    return nicopy.get_video_info(vidId)['movie_type']

def getVid(vidId):
    cookie = login()
    print("Downloading ", vidId)
    try:
        vid = nicopy.get_flv(vidId, cookie)
    except Exception as e:
        raise NameError("NicoNico is not responding !")
    if vid == b'403 Forbidden':
        raise NameError(vidId + " is not downloadable on NicoVideo")
    vidType = videoType(vidId)
    return vid, vidType

def toFile(data, filename):
    with open(filename, 'wb') as f:
        f.write(data)
    print(filename, " written to disk")

"""
    Extract audio from video by calling ffmpeg
    (WAV format)
"""
def getAudio(vid):
    command = "ffmpeg -i pipe:0 -f wav -vn -"
    process = subprocess.Popen(command, stdin = subprocess.PIPE,
            stdout = subprocess.PIPE, shell = True)
    audio, _ = process.communicate(vid)
    return audio

def audioFromID(vidId):
    vid, _ = getVid(vidId)
    return getAudio(vid)

"""
    Create the file {WAV_FOLDER}/{vidId}.wav
    containing the audio of the NNV video with given ID
"""
def createWAV(vidId):
    filename = WAV_FOLDER + vidId + ".wav"
    if os.path.isfile(filename):
        print(filename + " already on disk !")
        return
    audio = audioFromID(vidId)
    toFile(audio, filename)

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

"""
    Extract the Nico ID of a filename and download the
    corresponding audio file (if available)
"""
def oneFile(f):
    sm = re.search('([a-z]{2}[0-9]+)', f)
    if sm: id = sm.group(1)
    else: raise NameError(f + " is not a Nico ID")
    try:
        createWAV(id)
    except Exception as e:
        print(str(e))

"""
    This will try to download the audio file corresponding to each
    file stored in COMMENT_FOLDER

    !! FFMPEG have to be installed on the system
    !! The nicopy library have to be installed (available in pip)

    NicoNico servers are very slow and often fail, so you will probably
    have to launch the script multiple times (audio files that are
    already downloaded will not be downloaded again)
"""
if __name__ == "__main__":
    createAudio(retrieveFiles())
