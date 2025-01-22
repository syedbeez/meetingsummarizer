# -*- coding: utf-8 -*-
"""summarization

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1orFlByUsPD1rBJ1wE2eJWXFG5wZTnviv
"""

from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize the text based on given ratio
def summarize_text(text, summary_ratio):
    # Calculate the maximum length based on the summary ratio
    max_length = int(len(text.split()) * summary_ratio / 100)

    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Load the translated text
translated_file_path = r'/content/gdrive/My Drive/meet/transcription_translated.txt'

try:
    with open(translated_file_path, 'r', encoding='utf-8') as file:
        translated_content = file.read()

    # Summarize the translated content
    summary_ratio = 30  # Summary ratio in percentage
    summarized_content = summarize_text(translated_content, summary_ratio)

    print("Summarized Text:")
    print(summarized_content)

except FileNotFoundError:
    print(f"Error: The file {translated_file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

with open( r'/content/gdrive/My Drive/meet/transcription_summary.txt', 'w', encoding='utf-8') as file:
        file.write(summarized_content)