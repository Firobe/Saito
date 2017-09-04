import vamp
import librosa

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

