# -*- coding: utf-8 -*-
"""mail.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fq2zhj7MhkungM8NXxbpdtQXoiGQCKrF
"""

from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)

import smtplib
from email.mime.text import MIMEText
from datetime import date

# File paths of the text files
file_paths = [r'/content/gdrive/My Drive/meet/transcription_translated.txt', r'/content/gdrive/My Drive/meet/transcription_summary.txt', r'/content/gdrive/My Drive/meet/transcription_analysis.txt', r'/content/gdrive/My Drive/meet/plan_of_action.txt']

# Read content of text files
def read_file_content(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Generate the email subject and body
today_date = date.today().strftime("%B %d, %Y")
subject = f"Meeting Summary and Plan of Action for {today_date}"

# Create the email with contents of text files.
body = f"Hello, meeting summary and plan of action of {today_date}\n\n"
for file_path in file_paths:
    file_name = file_path.split("/")[-1]  # Get file name
    content = read_file_content(file_path)
    body += f"{file_name}\n\n{content}\n\n"
body += "Thanks and regards,\n\nSyed Abdallah Albeez"

sender = "albeezsyedabdallah@gmail.com" #from mail
recipients = ["syed7862001@gmail.com"] #to mail
password = "xtdj lqvu jbkn smrj" #fake place holder password #USE 16 CHARACTER APP PASSWORD (two factor auth should be enabled to generate)

# Function to send email
def send_email(subject, body, sender, recipients, password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(sender, password)
        smtp_server.sendmail(sender, recipients, msg.as_string())
    print("Message sent!")
send_email(subject, body, sender, recipients, password)