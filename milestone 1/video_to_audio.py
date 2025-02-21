# -*- coding: utf-8 -*-
"""video to audio.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JYNoR6QpWzqjmVv4mZAOc3XpHtIcFchx
"""

from google.colab import drive
drive.mount("/content/gdrive")

#Specific to google colab ignore if you are not using google colab.

import subprocess
import os

def extract_audio_from_video(video_file, output_audio_file):
    command = [
        'ffmpeg',
        '-i', video_file, #video input
        '-q:a', '0', #audio quality setting
        '-map', 'a', #audio extraction
        output_audio_file #audio output
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Audio extracted Successfully")
    except subprocess.CalledProcessError:
        print("Error: Audio extraction failed.")
    except FileNotFoundError:
        print("Error: FFmpeg not found.")

if __name__ == "__main__":
    video_file = input("Enter video file path with video name : ") #eg: /content/gdrive/My Drive/ytdl/forest.mp4
    audio_name = input("Enter output audio file name (only filename, don't include extension): ") #eg: forestaud
    audio_format = input("Enter audio format : ") #eg: mp3,m4a etc
    output_location = input("Enter the folder path where the audio file should be saved : ") #eg: /content/gdrive/My Drive/ytdl/

    # Saving the audio file in desired folder (/content/gdrive/My Drive/ytdl/forestaud.mp3)
    output_audio_file = os.path.join(output_location, f"{audio_name}.{audio_format}")

    extract_audio_from_video(video_file, output_audio_file)