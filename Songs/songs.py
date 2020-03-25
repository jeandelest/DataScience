# MP3 to MIDI
import os
import ffmpeg
import subprocess

# convert wav to mp3 
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        dst = os.path.join(dirname, filename.replace("mp3", "wav"))
        stream = ffmpeg.input(os.path.join(dirname, filename))
        stream = ffmpeg.hflip(stream)
        print("New name : ", dst)
        stream = ffmpeg.output(stream, dst)