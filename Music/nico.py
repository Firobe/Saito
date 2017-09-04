import nicopy
import subprocess
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
    vid = nicopy.get_flv(vidId, cookie)
    if vid == b'403 Forbidden':
        raise NameError(vidId + " is not available on NicoVideo")
    vidType = videoType(vidId)
    return vid, vidType

def toFile(data, filename):
    with open(filename, 'wb') as f:
        f.write(data)
    print(filename, " written to disk")

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
    audio = audioFromID(vidId)
    toFile(audio, WAV_FOLDER + vidId + ".wav")
