import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
//import Img1 from './imgs/image1.png';
// import Img2 from './imgs/image2.png';
// import Img3 from './imgs/image3.png';
// import Img4 from './imgs/image4.png';
// import Img5 from './imgs/image5.png';
import './lab14.css';

hljs.registerLanguage('python', python);

const codeSnippetPython = `
import speech_recognition as sr
from gtts import gTTS
import os
import serial
import time

# Set up serial communication with Arduino
arduino = serial.Serial(port='COM12', baudrate=9600, timeout=1)  # Adjust 'COM12' to your Arduino's port

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
    if command == "SWITCH ON":
        arduino.write(b"SWITCH ON\\n")
        speak("Switching on the light")
    elif command == "SWITCH OFF":
        arduino.write(b"SWITCH OFF\\n")
        speak("Switching off the light")
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
`;

const codeSnippetArduino = `
const int ledPin = 7;  // Pin to which the LED is connected

void setup() {
  pinMode(ledPin, OUTPUT);  // Set the LED pin as an output
  digitalWrite(ledPin, LOW);  // Ensure the LED is off at startup
  Serial.begin(9600);  // Start serial communication at 9600 baud
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\\n');  // Read the command from serial input
    command.trim();  // Remove any leading or trailing whitespace
    if (command == "SWITCH ON") {
      digitalWrite(ledPin, HIGH);  // Switch on the LED
    } else if (command == "SWITCH OFF") {
      digitalWrite(ledPin, LOW);  // Switch off the LED
    }
  }
}
`;

const Lab2 = () => {
  const [highlightedCodeSnippet, setHighlightedCodeSnippet] = useState("");

  useEffect(() => {
    hljs.highlightAll();
  }, []);

  const ParticleCanvas = () => {
    const canvasRef = useRef(null);

    useEffect(() => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      let particles = [];

      // Function to create a particle
      function Particle(x, y) {
        this.x = x;
        this.y = y;
        this.size = Math.random() * 0.4 * 5 + 0.85; // 15% of the size
        this.speedX = Math.random() * 3 - 1.5;
        this.speedY = Math.random() * 3 - 1.5;
      }

      // Function to draw particles and connect them with lines
      function drawParticles() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (let i = 0; i < particles.length; i++) {
          ctx.fillStyle = 'orangered'; // Change particle color to orangered
          ctx.beginPath();
          ctx.arc(particles[i].x, particles[i].y, particles[i].size, 0, Math.PI * 2);
          ctx.fill();

          particles[i].x += particles[i].speedX;
          particles[i].y += particles[i].speedY;

          // Wrap particles around the screen
          if (particles[i].x > canvas.width) particles[i].x = 0;
          if (particles[i].x < 0) particles[i].x = canvas.width;
          if (particles[i].y > canvas.height) particles[i].y = 0;
          if (particles[i].y < 0) particles[i].y = canvas.height;

          // Draw lines between neighboring particles
          for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const opacity = 1 - distance / 100; // Opacity based on distance

            if (opacity > 0) {
              ctx.strokeStyle = `rgba(0, 0, 0, ${opacity})`; // Set line opacity
              ctx.lineWidth = 0.5; // Set line thickness
              ctx.beginPath();
              ctx.moveTo(particles[i].x, particles[i].y);
              ctx.lineTo(particles[j].x, particles[j].y);
              ctx.stroke();
            }
          }
        }

        requestAnimationFrame(drawParticles);
      }

      for (let i = 0; i < 120; i++) {
        particles.push(new Particle(Math.random() * canvas.width, Math.random() * canvas.height));
      }

      drawParticles();

      return () => {
        particles = [];
      };
    }, []);

    return <canvas ref={canvasRef} style={{ position: 'fixed', zIndex: -1, top: 0, left: 0, width: '100vw', height: '100vh' }} />;
  };

  const handleHeadingClick = (section) => {
    const snippet = section === "Python" ? codeSnippetPython : codeSnippetArduino;
    setHighlightedCodeSnippet(snippet);
  };

  return (
    <div className="dashboard">
      <ParticleCanvas />
      <div className="Layout" style={{ display: "flex", justifyContent: "space-around", color: '#09F' }}>
        <div className="box3">
          <h2>Lab Experiment: Voice-Controlled LED with Arduino</h2>
          <br />
          <p><b>Introduction:</b> Create a voice-controlled system to switch an LED on and off using Python and Arduino.</p>
          <br />
          <p><b>Objective:</b> Develop a system that listens to voice commands and controls an LED connected to an Arduino board.</p>
          <br />
          <p><b>Components Needed:</b></p>
          <ul>
            <li>Arduino Board</li>
            <li>LED</li>
            <li>220-ohm Resistor</li>
            <li>Jumper Wires</li>
            <li>Microphone</li>
            <li>Computer</li>
          </ul>
          <br />
          <p><b>Steps:</b></p>
          <ol>
            <li><b>Set Up Arduino IDE:</b>
              <ul>
                <li>Download and install the Arduino IDE from <a href="https://www.arduino.cc/en/software" target="_blank" rel="noopener noreferrer">here</a>.</li>
                <li>Connect your Arduino board to your computer using a USB cable.</li>
                <li>Select the correct board and port in the Arduino IDE (`Tools &gt Board` and `Tools &gt Port`).</li>
              </ul>
            </li>
            <li><b>Connect the LED:</b>
              <ul>
                <li>Connect the anode (longer leg) of the LED to a 220-ohm resistor.</li>
                <li>Connect the other end of the resistor to digital pin 7 on the Arduino.</li>
                <li>Connect the cathode (shorter leg) of the LED to the ground (GND) pin on the Arduino.</li>
              </ul>
            </li>
            <li><b>Upload Arduino Code:</b>
              <ul>
                <li>Open the Arduino IDE and paste the following code:</li>
                <button onClick={() => handleHeadingClick("Arduino")}>Show Arduino Code</button>
                <li>Click the upload button (right arrow icon) in the Arduino IDE to upload the code to the Arduino board.</li>
              </ul>
            </li>
            <li><b>Install Python Libraries:</b>
              <ul>
                <li>Open a terminal or command prompt and run the following commands:</li>
                <code>pip install speechrecognition gtts pyserial</code>
              </ul>
            </li>
            <li><b>Write Python Code:</b>
              <ul>
                <li>Create a new Python file and paste the following code:</li>
                <button onClick={() => handleHeadingClick("Python")}>Show Python Code</button>
              </ul>
            </li>
            <li><b>Run the Python Script:</b>
              <ul>
                <li>Ensure the Arduino is connected to the computer and the code has been successfully uploaded.</li>
                <li>Open a terminal or command prompt.</li>
                <li>Navigate to the directory where your Python script is saved.</li>
                <li>Run the Python script by typing <code>python your_script_name.py</code> and pressing Enter.</li>
                <li>The Python script will start listening for voice commands and control the LED based on the recognized commands.</li>
              </ul>
            </li>
          </ol>
          <br />
          <p><b>Conclusion:</b> This experiment demonstrates how to create a voice-controlled system using Python and Arduino to switch an LED on and off.</p>
        </div>
        <div className="box4">
          <div className="code-container">
            <pre className="code-snippet">
              <code className="python">
                {highlightedCodeSnippet ? highlightedCodeSnippet.trim() : codeSnippetPython.trim()}
              </code>
            </pre>
          </div>
          <div>
            <button onClick={() => handleHeadingClick("Python")}>Show Python Code</button>
            <button onClick={() => handleHeadingClick("Arduino")}>Show Arduino Code</button>
          </div>
        </div>
      </div>
      <div>
        <button className="button">
          <a href="https://www.kaggle.com/code/percival224/unit-4-lab-3" target="_blank" rel="noopener noreferrer"> View Runnable Code</a>
        </button>
      </div>
    </div>
  );
};

export default Lab2;
