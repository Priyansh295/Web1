# pip install SpeechRecognition gTTS pyserial  - Run this command on your VS code terminal 

import speech_recognition as sr
from gtts import gTTS
import os
import serial
import time

# Set up serial communication with Arduino
arduino = serial.Serial(port='COM9', baudrate=9600, timeout=1)  # Adjust 'COM12' to your Arduino's port

# Initialize the recognizer
recognizer = sr.Recognizer()

def listen_command():
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening for command...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"Command received: {command}")
            return command.upper()
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Could not request results; check your network connection.")
            return None

def execute_command(command):
    if "SET" in command:
        try:
            angle = int(command.split()[-1])
            angle = round(angle / 5) * 5  # Round to nearest multiple of 5
            if 0 <= angle <= 180:
                arduino.write(f"{angle}\n".encode())
                speak(f"Turning to {angle} degrees")
            else:
                speak("Angle must be between 0 and 180 degrees")
        except ValueError:
            speak("Please specify a valid angle")
    else:
        speak("Sorry, I didn't understand the command.")

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("start response.mp3")  # For Windows, use 'start response.mp3'; for Mac use 'afplay response.mp3'

while True:
    command = listen_command()
    if command:
        execute_command(command)
    time.sleep(2)  # Short delay to avoid multiple rapid commands