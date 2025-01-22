import os
import shutil
import requests
import base64
import time
from datetime import datetime
import nltk
import PySimpleGUI as sg
from pydub import AudioSegment
import openai
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import whisper
from googletrans import Translator
from pyannote.audio import Pipeline
import numpy as np
import re
from email.mime.text import MIMEText
import smtplib

# Initialize NLTK
nltk.download('punkt')
nltk.download('vader_lexicon')

# Set OpenAI API Key and email credentials directly in the code
openai.api_key = ''
EMAIL_SENDER = ''
EMAIL_PASSWORD = '' #email app password
CLIENT_ID = "" #zoom client id
CLIENT_SECRET = "" #zoom client secret
ACCOUNT_ID = "" #zoom account id

# Initialize models
genre_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sia = SentimentIntensityAnalyzer()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
whisper_model = whisper.load_model("large")
translator = Translator()
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")


# Directory to save outputs
output_dir = r"C:\Users\91950\Downloads\info"
os.makedirs(output_dir, exist_ok=True)

def save_output(filename, content):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)
    return filepath

# Function to obtain the access token
def get_access_token():
    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={ACCOUNT_ID}"
    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    headers = {"Authorization": f"Basic {encoded_credentials}"}
    response = requests.post(url, headers=headers)
    response_data = response.json()
    
    access_token = response_data.get("access_token")
    if access_token:
        return access_token
    else:
        print("Error obtaining access token:", response_data)
        return None

# Function to create an instant Zoom meeting
def create_meeting(access_token, topic):
    url = "https://api.zoom.us/v2/users/me/meetings"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    meeting_details = {
        "topic": topic,
        "type": 1,  # Instant meeting
        "settings": {
            "auto_recording": "local",  # Enable local recording
            "host_video": True,
            "participant_video": True,
            "mute_upon_entry": True
        }
    }
    response = requests.post(url, headers=headers, json=meeting_details)
    if response.status_code == 201:
        meeting_info = response.json()
        return meeting_info.get("join_url")  # Return the meeting link
    else:
        print("Error creating meeting:", response.json())
        return None

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
    transcriptionNT = result["text"]
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

def summarize_text(transcript, summary_ratio=30):
    target_length = int(len(transcript) * (summary_ratio / 100))
    summary = summarizer(transcript, max_length=target_length, min_length=target_length // 2, do_sample=False)[0]['summary_text']
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
    diarization = diarization_pipeline(audio_file)
    diarization_text = ""
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_text += f"Speaker {speaker}: from {turn.start:.1f}s to {turn.end:.1f}s\n"
    save_output("diarization.txt", diarization_text)
    return diarization_text
    
def count_speakers(audio_file):
    diarization = diarization_pipeline(audio_file)
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

def copy_meeting_recording():
    zoom_recordings_dir = r"C:\Users\91950\Documents\Zoom"
    
    # Get the most recent folder
    recent_folder = max([f for f in os.listdir(zoom_recordings_dir) if os.path.isdir(os.path.join(zoom_recordings_dir, f))], key=lambda x: os.path.getmtime(os.path.join(zoom_recordings_dir, x)))
    
    # Get the most recent MP4 file in that folder
    recent_folder_path = os.path.join(zoom_recordings_dir, recent_folder)
    mp4_files = [f for f in os.listdir(recent_folder_path) if f.endswith('.mp4')]
    if mp4_files:
        recent_mp4_file = max(mp4_files, key=lambda x: os.path.getmtime(os.path.join(recent_folder_path, x)))
        recent_mp4_path = os.path.join(recent_folder_path, recent_mp4_file)

        # Define the new path for copied and renamed file
        timestamp = datetime.now().strftime("%I-%M%p_%d-%m-%Y")
        new_folder_name = f"gmeet"
        new_folder_path = os.path.join(r"C:\Users\91950\Downloads\info", new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        # Create a unique file name if it already exists
        new_file_name = f"meeting_{timestamp}.mp4"
        new_file_path = os.path.join(new_folder_path, new_file_name)

        # Ensure the new file path is unique
        counter = 1
        while os.path.exists(new_file_path):
            new_file_name = f"meeting_{timestamp}({counter}).mp4"
            new_file_path = os.path.join(new_folder_path, new_file_name)
            counter += 1

        # Copy and rename the file
        shutil.copy(recent_mp4_path, new_file_path)
        return new_file_path
    return None


# GUI layout
sg.theme('DarkGrey2')
layout = [
    [sg.Text('Meeting Topic'), sg.InputText(key='meeting_topic')],
    [sg.Text('Recipient Emails (comma-separated)'), sg.InputText(key='recipients')],
    [sg.Text('Recording Path'), sg.InputText(key='recording_path', size=(50, 1))],
    [sg.Text('Meeting Link'), sg.InputText(key='meeting_link', size=(50, 1), disabled=True)],  # New field for the meeting link
    [sg.Multiline(size=(80, 20), key='output', disabled=False)],  # Make it editable
    [sg.Slider(range=(10, 100), default_value=30, orientation='h', key='summary_ratio', enable_events=True)],
    [sg.Button('Create Meeting'), sg.Button('Copy Meeting Recording'), sg.Button('Transcribe & Analyze'), sg.Button('Send Email')]
]
window = sg.Window('Meeting Summarizer - Infosys Springboard 5.0', layout)      
# Event loop
recording_path = None  # Initialize recording_path at the start

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break
    if event == 'Create Meeting':
        access_token = get_access_token()
        meeting_topic = values['meeting_topic']
        if access_token:
            meeting_url = create_meeting(access_token, meeting_topic)
            if meeting_url:
                window['meeting_link'].update(meeting_url)  # Update the meeting link input field
                sg.popup(f'Meeting created! Join here: {meeting_url}')
    
    if event == 'Copy Meeting Recording':
        recording_path = copy_meeting_recording()
        if recording_path:
            window['recording_path'].update(recording_path)

    if event == 'Transcribe & Analyze':
        if recording_path:  # Check if recording_path is not None
            # Step 1: Extract audio from the recording
            audio_file = extract_audio(recording_path, 'video')
            
            # Step 2: Transcribe audio to text
            transcription = transcribe_audio(audio_file)
            transcriptionnt = transcription
            
            # Step 3: Translate non-English text to English
            translated_text = translate_non_english(transcription)
            
            # Step 4: Analyze genre and sentiment based on translated text
            genre, sentiment_results, sentiment_summary = analyze_genre_sentiment(translated_text)
            
            # Step 5: Summarize the translated text
            summary = summarize_text(translated_text, summary_ratio=values['summary_ratio'])
            
            # Step 6: Perform diarization on the audio file
            diarization_result = perform_diarization(audio_file)

            # Step 7: Generate the plan of action based on the translated text
            plan_of_action = generate_plan_of_action(translated_text)

            speakers_count= count_speakers(audio_file)
            
            detected_industry = analyze_industry(translated_text)

            # Prepare the final output for the GUI
            today_date = datetime.now().strftime("%d-%m-%y")
            output_content = (
                f"Greetings,\n This mail contains the details of meeting held on {today_date}\n\n"
                f"\nSummary:\n{summary}\n\n"
                f"Plan of Action:\n{plan_of_action}\n\n"
                #f"Transcription:\n{transcriptionnt}\n\n"
                #f"Translated Text:\n{translated_text}\n\n"
                f"Genre: {genre}\n\n"
                f"Industry/Topic: {detected_industry}\n\n"
                f"Sentiment Analysis:\n{sentiment_summary}\n\n" #+
                #''.join(sentiment_results) + 
                #f"Diarization:\n{diarization_result}\n\n"
                f"Number of Speakers/Participants: {speakers_count}\n\n"
                "Thanks and Regards,\n Meeting Summarizer"
            )
            
            # Update the output in the GUI
            window['output'].update(output_content)
        else:
            sg.popup("No recording found. Please copy the meeting recording first.")

    if event == 'Send Email':
        output_content = values['output']
        recipients = values['recipients']
        if recipients:  # Check if recipients field is not empty
            today_date = datetime.now().strftime("%d-%m-%y")
            email_subject = f"Meeting Summary - {today_date}"
            send_email(email_subject, output_content, values['recipients'])
            sg.popup("Email sent successfully!")
            save_output(f"{today_date}_full_meeting_summary.txt", output_content)
            

# Close the window
window.close()
