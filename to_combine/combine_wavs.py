'''
    concatenate wavs into a single one
'''

import os
import random
import argparse
import random

def rename(path, name):
    for count, filename in enumerate(os.listdir(path)):
        dst = name + "_" + str(count) + ".wav"
        src = path + filename
        dst = path + dst
        os.rename(src, dst)

# rename("backward/", "backward")
# rename("forward/", "forward")
# rename("up/", "up")
# rename("down/", "down")
# rename("left/", "left")
# rename("right/", "right")
# rename("marvin/", "marvin")
# rename("noise/", "bg_noise")

_, _, filenames = next(os.walk("words/"), (None, None, []))

filenames.sort()

random.shuffle(filenames)

f = open('wav_list.txt', 'a')

for filename in enumerate(filenames):
    f.write(str(filename) + ' ' + '\n')

f.close()

from pydub import AudioSegment

combined_sounds = None
for i, filename in enumerate(filenames):
    if combined_sounds is None:
        combined_sounds = AudioSegment.from_wav("words/" + filename)
    else:
        combined_sounds = combined_sounds + AudioSegment.from_wav("words/" + filename)

combined_sounds.export("../combined_sounds/combined_sounds.wav", format="wav")

print('done combining!')



# parser = argparse.ArgumentParser()
# parser.add_argument('--path',   type=str)
# args = parser.parse_args()
#
# path = args.path
# _, _, filenames = next(walk(path), (None, None, []))
#
# from pydub import AudioSegment
#
# combined_sounds = None
# for i, filename in enumerate(filenames):
#     # if i == 1000:
#     #     break
#     if combined_sounds is None:
#         combined_sounds = AudioSegment.from_wav(path + "/" + filename)
#     else:
#         combined_sounds = combined_sounds + AudioSegment.from_wav(path + "/" + filename)
#
# combined_sounds.export("../combined_sounds/combined_sounds.wav", format="wav")
#
# print('done combining!')
