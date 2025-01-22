import os
import nltk
import sys
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from pydub import AudioSegment
import openai
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import whisper
from googletrans import Translator
from pyannote.audio import Pipeline
import numpy as np
import re
from datetime import date
from email.mime.text import MIMEText
import smtplib

# Download NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')


openai.api_key = 'yourapikey'
EMAIL_SENDER = 'yourmail@gmail.com'
EMAIL_PASSWORD = 'pass'


genre_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sia = SentimentIntensityAnalyzer()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
whisper_model = whisper.load_model("large")
translator = Translator()
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="your_huggingface_token")

output_dir = r"C:\Users\91950\Downloads\info"
os.makedirs(output_dir, exist_ok=True)

def save_output(filename, content):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)
    return filepath

def extract_audio(file_path, file_type):
    if file_type == 'video':
        audio_file = file_path.replace('.mp4', '.wav')
        sound = AudioSegment.from_file(file_path, format="mp4")
        sound.export(audio_file, format="wav")
        return audio_file
    return file_path

def transcribe_audio(audio_file):
    result = whisper_model.transcribe(audio_file)
    transcription = result["text"]
    save_output("transcription.txt", transcription)
    return transcription

def translate_non_english(text):
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?]) +', text)
    translated_text = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        lang = translator.detect(sentence).lang
        translated = translator.translate(sentence, dest='en').text if lang != 'en' else sentence
        translated_text.append(translated)
    translated_text = ' '.join(translated_text)
    save_output("translated_text.txt", translated_text)
    return translated_text
    
def get_industry_from_chatgpt(transcript):
    prompt = f"Based on the following transcript of a meeting, identify the most relevant industry:\n\n{transcript}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    industry = response['choices'][0]['message']['content'].strip()
    save_output("industry.txt", industry)
    return industry

def analyze_industry(transcript):
    # Create a list of all topics from the industries
    possible_industries = [
        "software development", "AI advancements", "IT support", "cybersecurity",
        "investment strategy", "market analysis", "financial planning", "stock performance",
        "patient care", "medical advancements", "health policy", "pharmaceuticals",
        "curriculum development", "teaching methods", "student performance", "e-learning",
        "brand strategy", "customer segmentation", "ad campaigns", "market trends",
        "property analysis", "real estate investments", "mortgages",
        "contract law", "corporate law", "legal compliance", "case analysis"
    ]
    
    # Use genre_model for analyzing the transcript
    result = genre_model(transcript, candidate_labels=possible_industries)
    
    # Extract the industry/topic with the highest score
    industry_topic = result['labels'][0]  # Get the label with the highest score
    
    # Define industries associated with the topics
    if industry_topic in ["software development", "AI advancements", "IT support", "cybersecurity"]:
        detected_industry = "Technology"
    elif industry_topic in ["investment strategy", "market analysis", "financial planning", "stock performance"]:
        detected_industry = "Finance"
    elif industry_topic in ["patient care", "medical advancements", "health policy", "pharmaceuticals"]:
        detected_industry = "Healthcare"
    elif industry_topic in ["curriculum development", "teaching methods", "student performance", "e-learning"]:
        detected_industry = "Education"
    elif industry_topic in ["brand strategy", "customer segmentation", "ad campaigns", "market trends"]:
        detected_industry = "Marketing"
    elif industry_topic in ["property analysis", "real estate investments", "mortgages"]:
        detected_industry = "Real Estate"
    elif industry_topic in ["contract law", "corporate law", "legal compliance", "case analysis"]:
        detected_industry = "Legal"
    else:
        detected_industry = "Unknown"
    
    # If industry is unknown, use ChatGPT to determine the industry
    if detected_industry == "Unknown":
        detected_industry = get_industry_from_chatgpt(transcript)

    # Return the detected industry
    return detected_industry

    
def analyze_genre_sentiment(transcript):
    possible_genres = ["business meeting", "conference", "discussion", "presentation", "interview", "team meeting", "project review", "sales pitch", "workshop"]
    genre_result = genre_model(transcript, possible_genres)
    genre = genre_result['labels'][0]
    
    sentences = nltk.sent_tokenize(transcript)
    sentiment_results = []
    sentiment_scores = []

    for sentence in sentences:
        score = sia.polarity_scores(sentence)
        sentiment_scores.append(score['compound'])
        sentiment_label = "POSITIVE" if score['compound'] > 0 else "NEGATIVE" if score['compound'] < 0 else "NEUTRAL"
        sentiment_results.append(f"Sentence: {sentence}\nSentiment: {sentiment_label} (Score: {score['compound']:.2f})\n")

    average_sentiment = np.mean(sentiment_scores)
    average_sentiment_label = "POSITIVE" if average_sentiment > 0 else "NEGATIVE" if average_sentiment < 0 else "NEUTRAL"
    sentiment_summary = f"Average Sentiment: {average_sentiment_label} (Score: {average_sentiment:.2f})\n"
    sentiment_results.insert(0, sentiment_summary)
    save_output("sentiment_analysis.txt", sentiment_summary + '\n'.join(sentiment_results))
    return genre, sentiment_results, sentiment_summary

def summarize_text(transcript):
    summary = summarizer(transcript, max_length=150, min_length=60, do_sample=False)[0]['summary_text']
    save_output("summary.txt", summary)
    return summary

def generate_plan_of_action(transcript):
    prompt = (
        "Based on the following transcript of a meeting, generate a clear and actionable plan:\n\n"
        f"{transcript}\n\n"
        "Please provide the plan of action in bullet points."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    plan_of_action = response['choices'][0]['message']['content']
    save_output("plan_of_action.txt", plan_of_action)
    return plan_of_action

def perform_diarization(audio_file):
    diarization = pipeline(audio_file)
    diarization_text = ""
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_text += f"Speaker {speaker}: from {turn.start:.1f}s to {turn.end:.1f}s\n"
    save_output("diarization.txt", diarization_text)
    return diarization_text
    
def count_speakers(audio_file):
    diarization = pipeline(audio_file)
    speakers = set()  # Set to track unique speakers

    for _, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)  # Add speaker to the set

    # Return the number of unique speakers
    return len(speakers)

def send_email(subject, body, recipient):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    recipients = [email.strip() for email in recipient.split(",")]
    msg['To'] = ", ".join(recipients)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp_server.sendmail(EMAIL_SENDER, recipients, msg.as_string())

def process_file():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    email = email_entry.get()
    file_type = 'video' if file_path.endswith('.mp4') else 'audio'
    audio_file = extract_audio(file_path, file_type)

    transcription = transcribe_audio(audio_file)
    translated_text = translate_non_english(transcription)
    genre, sentiment_results, sentiment_summary = analyze_genre_sentiment(translated_text)
    summary = summarize_text(translated_text)
    plan_of_action = generate_plan_of_action(translated_text)
    diarization_text = perform_diarization(audio_file)
    speakers_count= count_speakers(audio_file)
    detected_industry = analyze_industry(translated_text)

    output_text = (
        #f"Transcription:\n{transcription}\n\n"
        #f"Translation:\n{translated_text}\n\n"
        f"Genre: {genre}\n\n"
        f"Industry/topic:{detected_industry}\n\n"
        f"{sentiment_summary}\n\n"
        #f"Sentiment Analysis:\n{''.join(sentiment_results)}\n\n"
        f"Summary:\n{summary}\n\n"
        f"Plan of Action:\n{plan_of_action}\n\n"
        #f"Diarization:\n{diarization_text}\n\n"
        f"Number of speakers: {speakers_count}\n\n"
    )

    output_textbox.delete(1.0, tk.END)
    output_textbox.insert(tk.END, output_text)

def send_email_wrapper():
    edited_body = output_textbox.get("1.0", tk.END)
    today_date = date.today().strftime("%B %d, %Y")
    email = email_entry.get()
    
    try:
        send_email(f"Meeting Summary for {today_date}", edited_body, email)
        messagebox.showinfo("Success", "Email sent successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while sending email: {e}")

root = tk.Tk()
root.title("Meeting Summarizer")

tk.Label(root, text="Enter the recipient's email address:").pack()
email_entry = tk.Entry(root, width=50)
email_entry.pack()

tk.Button(root, text="Select File and Process", command=process_file).pack()
output_textbox = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20)
output_textbox.pack()

tk.Button(root, text="Send Email", command=send_email_wrapper).pack()
tk.Button(root, text="Exit", command=root.quit).pack()

root.mainloop()
