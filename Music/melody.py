import vamp
import librosa
import numpy as np

"""
    The MELODIA VAMP plugin should be installed on the system
    Melodia is provided in the 'melodia' subfolder
    To install it : http://www.vamp-plugins.org/download.html#install
"""

"""
    Returns a list of frequencies corresponding to the identified melody
    of the music 'filename' in the interval
    [position - around, position + around]

    When the values are negative, it means that there is probably no melody
    at this moment but that if there is, abs(value) is the estimate of its
    frequency

    Also returns the sample rate in Hz.
"""
def extractRawMelody(filename, position, around):
    data, rate = librosa.load(filename, offset = position - around,\
            duration = 2 * around)
    result = vamp.collect(data, rate, "mtg-melodia:melodia");
    frameDur, melody = result['vector']
    #frameDur = frameDur.to_frame(rate)
    return melody, rate

"""
    Takes a list of frequencies and the sample rate, and returns the
    aggregated duration of each note in seconds.

    For exemple, if the list is [3, 3, 3, 4, 5, 5, 6, 6, 6] and the sample
    rate is 1 Hz, the result will be [(3, 3), (4, 1), (5, 2), (6, 3)]
"""
def richMelody(melody, rate):
    rich_melody = []
    cur = melody[0]
    save = 0
    for i in range(len(melody)):
        if melody[i] <= 0: melody[i] = 0
        if melody[i] != cur or i == len(melody) - 1:
            rich_melody.append((cur, (i - save) / rate))
            cur = melody[i]
            save = i
    return rich_melody

"""
    Transform list of frequencies to list of integer number of
    semitones relative to A4 (440 Hz).
"""
def nearestSemiTone(melody):
    f0 = 440 # 440 Hz : base frequency, note 0 (A4)
    ratio = 2 ** (1/12)
    for i in range(len(melody)):
        f = abs(melody[i])
        # We have f = f0 * ratio^n where n is the number
        # of semitones between A4 and f
        n = (np.log(f) - np.log(f0)) / np.log(ratio)
        melody[i] = round(n)
    return melody

def extractMelody(filename, position, around=5):
    M, rate = extractRawMelody(filename, position, around)
    M = nearestSemiTone(M)
    return richMelody(M, rate)
