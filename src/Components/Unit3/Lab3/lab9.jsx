import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import './lab9.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `
import cv2
import numpy as np import datetime import csv
from sklearn.ensemble import IsolationForest import joblib
import os

# Initialize video capture cap = cv2.VideoCapture(0)

# Create directory to save anomaly frames if not os.path.exists('anomalies'):
os.makedirs('anomalies')

# Open CSV file for writing anomaly log
with open('anomaly_log.csv', 'w', newline='') as anomaly_log_file: anomaly_log_writer = csv.writer(anomaly_log_file) anomaly_log_writer.writerow(["Timestamp", "ImageFile"])

# Open CSV file for writing data log
with open('data_log.csv', 'w', newline='') as data_log_file: data_log_writer = csv.writer(data_log_file) data_log_writer.writerow(["Timestamp", "Motion"])

ret, frame1 = cap.read() ret, frame2 = cap.read() data_log = []
model_ready = False while cap.isOpened():
# Motion detection
diff = cv2.absdiff(frame1, frame2)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) dilated = cv2.dilate(thresh, None, iterations=3)
contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

motion = 0
for contour in contours:
if cv2.contourArea(contour) < 900: continue
x, y, w, h = cv2.boundingRect(contour)
cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
motion += 1

frame1 = frame2
ret, frame2 = cap.read()

# Save data with timestamp timestamp = datetime.datetime.now()
data_log_writer.writerow([timestamp, motion]) data_log.append([motion])

# Initial model training if len(data_log) == 100:
model = IsolationForest(contamination=0.01) model.fit(data_log)
joblib.dump(model, 'isolation_forest_model.pkl') print("Initial model training complete. Model is now ready to
detect anomalies.")
print("Select feed window and press q to quit") model_ready = True

# Periodic model retraining
if len(data_log) > 100 and len(data_log) % 50 == 0: # Retrain every 50 new frames
model = IsolationForest(contamination=0.01) model.fit(data_log)
joblib.dump(model, 'isolation_forest_model.pkl') print("Model retrained and updated.")

# Anomaly detection if model_ready:
feature_vector = np.array([[motion]]) anomaly = model.predict(feature_vector) if anomaly == -1:
print(f"Anomaly detected at {timestamp}") # Save the frame to file anomaly_filename =
f"anomalies/anomaly_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
cv2.imwrite(anomaly_filename, frame1) # Log anomaly to CSV file
anomaly_log_writer.writerow([timestamp, anomaly_filename])

# Display video cv2.imshow("feed", frame1)
if cv2.waitKey(1) & 0xFF == ord('q'): break

# Release resources cap.release()
 
cv2.destroyAllWindows()
print("Video capture released and windows destroyed. Exiting program.")

`;

const codeSections = {
  Section1: `
# Import Libraries
import cv2
import numpy as np
import datetime
import csv
from sklearn.ensemble import IsolationForest
import joblib
import os

# Initialize video capture
cap = cv2.VideoCapture(0)
`,

  Section2: `
# Create directory to save anomaly frames
if not os.path.exists('anomalies'):
    os.makedirs('anomalies')

# Open CSV file for writing anomaly log
anomaly_log_file = open('anomaly_log.csv', 'w', newline='')
anomaly_log_writer = csv.writer(anomaly_log_file)
anomaly_log_writer.writerow(["Timestamp", "ImageFile"])

# Open CSV file for writing data log
data_log_file = open('data_log.csv', 'w', newline='')
data_log_writer = csv.writer(data_log_file)
data_log_writer.writerow(["Timestamp", "Motion"])
`,

  Section3: `
# Read initial frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()
data_log = []

# Flag to check if model is ready for anomaly detection
model_ready = False
`,

  Section4: `
while cap.isOpened():
    # Motion detection
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion = 0
    for contour in contours:
        if cv2.contourArea(contour) < 900:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion += 1

    frame1 = frame2
    ret, frame2 = cap.read()

    # Save data with timestamp
    timestamp = datetime.datetime.now()
    data_log_writer.writerow([timestamp, motion])
    data_log.append([motion])
`,

  Section5: `
    # Initial model training
    if len(data_log) == 100:
        model = IsolationForest(contamination=0.01)
        model.fit(data_log)
        joblib.dump(model, 'isolation_forest_model.pkl')
        print("Initial model training complete. Model is now ready to detect anomalies.")
        print("Select feed window and press q to quit")
        model_ready = True

    # Periodic model retraining
    if len(data_log) > 100 and len(data_log) % 50 == 0:  # Retrain every 50 new frames
        model = IsolationForest(contamination=0.01)
        model.fit(data_log)
        joblib.dump(model, 'isolation_forest_model.pkl')
        print("Model retrained and updated.")
`,

  Section6: `
    # Anomaly detection
    if model_ready:
        feature_vector = np.array([[motion]])
        anomaly = model.predict(feature_vector)
        if anomaly == -1:
            print(f"Anomaly detected at {timestamp}")
            # Save the frame to file
            anomaly_filename = f"anomalies/anomaly_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(anomaly_filename, frame1)
            # Log anomaly to CSV file
            anomaly_log_writer.writerow([timestamp, anomaly_filename])
`,

  Section7: `
    # Display video
    cv2.imshow("feed", frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
anomaly_log_file.close()
data_log_file.close()
print("Video capture released and windows destroyed. Exiting program.")
`
};


const Lab2 = () => {
  const [highlightedCodeSnippet, setHighlightedCodeSnippet] = useState("");


  useEffect(() => {

    hljs.highlightAll();
  }
  )

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
    const snippet = codeSections[section];
    setHighlightedCodeSnippet(snippet);
  };

  return (
    <div className="dashboard">
      <ParticleCanvas />
      <div className="Layout" style={{ display: "flex", justifyContent: "space-around", color: '#09F' }}>
      <div className="box3">
  <h2>Anomaly Detection and Real-Time Monitoring on a Raspberry Pi</h2> <br />

<strong onClick={() => handleHeadingClick("Section1")}>Section 1: Import Libraries and Initialize Video Capture</strong> <br />
<p>Import necessary libraries for video processing, anomaly detection, and model handling. Initialize video capture to use the webcam.</p><br/>

<strong onClick={() => handleHeadingClick("Section2")}>Section 2: Create Directories and Open CSV Files</strong> <br />
<p>Create a directory to save frames where anomalies are detected. Open CSV files for logging anomalies and motion data.</p><br/>

<strong onClick={() => handleHeadingClick("Section3")}>Section 3: Initialize Frames and Data Log</strong> <br />
<p>Read the initial frames for motion detection. Initialize the list to store motion data. Set a flag for model readiness.</p><br/>

<strong onClick={() => handleHeadingClick("Section4")}>Section 4: Motion Detection and Data Logging</strong> <br />
<p>Detect motion between consecutive frames. Convert the difference to grayscale and apply Gaussian blur. Threshold and dilate the image to find contours representing motion. Draw rectangles around detected motion areas and log the motion data.</p><br/>

<strong onClick={() => handleHeadingClick("Section5")}>Section 5: Initial Model Training and Retraining</strong> <br />
<p>Train the initial Isolation Forest model after collecting 100 data points. Periodically retrain the model with new data every 50 frames.</p><br/>

<strong onClick={() => handleHeadingClick("Section6")}>Section 6: Anomaly Detection and Logging</strong> <br />
<p>Detect anomalies using the trained model. If an anomaly is detected, save the frame and log the anomaly.</p><br/>

<strong onClick={() => handleHeadingClick("Section7")}>Section 7: Display Video and Clean Up</strong> <br />
<p>Display the video feed with detected motion. Release video capture and close windows when 'q' is pressed.</p><br/>


  <strong onClick={() => handleHeadingClick("UnsupervisedLearning")}>Unsupervised Learning and Isolation Forests</strong> <br /> <br />
  <ul>
    <li><strong>Unsupervised Learning</strong></li>
    <p>Unsupervised Learning is a type of machine learning that deals with data without predefined labels. The goal is to infer the natural structure within a dataset. In this project, we focus on anomaly detection using:</p>
  </ul> <br />
  <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />
  <ul>
    <li><strong>Isolation Forests:</strong> A popular and effective method for unsupervised anomaly detection. They work by isolating anomalies instead of profiling normal data points.</li>
  </ul> <br />
  <p>Below are diagrams illustrating the concept of Isolation Trees and Isolation Forests, showing how multiple Isolation Trees are combined to detect anomalies.</p><br/>
  <b>Isolation Trees:</b>
  <img style={{width: '100%'}} src={Img3} alt="image3" /> <br /> <br />

  <b>Isolation Forests:</b>
  <img style={{width: '100%'}} src={Img2} alt="image2" /> <br /> <br />
  <strong onClick={() => handleHeadingClick("Implementation")}>Implementation on Raspberry Pi</strong>
  <p>This could also be implemented using a raspberry pi and a camera module instead of your web cam. That can be connected to the internet and be used for real time monitoring of any environment.</p>
</div>

        <div className="box4">
          <div className="code-container">
            <pre className="code-snippet">
              <code className="python" >
              {highlightedCodeSnippet ? highlightedCodeSnippet.trim() : codeSnippet2.trim()}
              </code>
            </pre>
          </div>
        </div>
      </div>
      <div> 
          <button className="button">
          <a href="https://www.kaggle.com/code/adityahr700/unit-3-lab-3?scriptVersionId=184231769" target="_blank"> View Runable code</a>
          </button>
        </div>
      </div>
  );
};
export default Lab2;
