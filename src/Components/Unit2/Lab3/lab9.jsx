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
  Step1: `
# Install libraries if not already installed


!pip install tensorflow scikit-learn


# Import libraries import numpy as np import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
 
from tensorflow.keras.applications import VGG16 from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.metrics.pairwise import cosine_similarity from sklearn.neighbors import KNeighborsClassifier from sklearn.cluster import KMeans
from sklearn.metrics import classification_report import matplotlib.pyplot as plt

# Define the path to your dataset
dataset_path = '/kaggle/input/real-life-industrial-dataset-of-casting-product'


# Load dataset (assuming data is organized in directories by class) datagen = ImageDataGenerator(rescale=1./255)
dataset = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary')


# Function to load and prepare the data def load_data(dataset):
features = [] labels = []
for batch in dataset:
X_batch, y_batch = batch features.extend(X_batch) labels.extend(y_batch)
if len(features) >= dataset.samples: break
return np.array(features), np.array(labels)


X, y = load_data(dataset)
 
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Normalize and reshape the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 224 * 224 * 3))
X_test = scaler.transform(X_test.reshape(-1, 224 * 224 * 3))


# Load pre-trained VGG16 model for feature extraction
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)


# Function to extract features
def extract_features(model, data): features = model.predict(data)
return features.reshape(features.shape[0], -1)


X_train_features = extract_features(model, X_train.reshape(-1, 224, 224, 3))
X_test_features = extract_features(model, X_test.reshape(-1, 224, 224, 3))

`,
  SplitData: `
from sklearn.model_selection import train_test_split

x_train , x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

import matplotlib.pyplot as plt 

plt.imshow(x_train[1])
print(y_train[1])
`,
  DeepLearningModel: `
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential([
    Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='valid', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  
])

model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Use SparseCategoricalCrossentropy for multi-class classification
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
`,

  TrainModel: `
  model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=1, validation_data=(x_val, y_val))

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
  <p> <b>Anomaly detection </b>in the context of real-time monitoring involves identifying unusual patterns that do not conform to expected behavior. These deviations from the norm can indicate potential issues or noteworthy events, making anomaly detection a crucial aspect of various applications, including environmental monitoring, security, and predictive maintenance.</p><br/>

  <strong onClick={() => handleHeadingClick("AnomalyDetection")}>Anomaly Detection</strong>
  <p>Anomalies, also known as outliers, are data points that differ significantly from other observations. Anomaly detection can be broadly categorized into:</p>
  <ul>
    <li><strong>1. Supervised Anomaly Detection:</strong> Requires a labeled dataset with normal and abnormal instances.</li>
    <li><strong>2. Unsupervised Anomaly Detection:</strong> Does not require labeled data, instead identifies anomalies based on their deviation from the majority of data points.</li>
  </ul>
  <p>For this lab, we focus on unsupervised anomaly detection using Isolation Forests, implemented on a Raspberry Pi with a camera module for real-time environmental monitoring.</p><br/>

  <strong onClick={() => handleHeadingClick("RealTimeMonitoring")}>Real-Time Monitoring by Recursively Training a Model</strong> <br />
  <p>Real-time monitoring involves continuously collecting data, processing it, and analyzing it to detect anomalies. The system we implement will:</p> <br />
  <ol style={{marginLeft: '25px'}}>
    <li><b>Capture data:</b> Use the Raspberry Pi camera module to capture video frames as input data.</li>
    ib-divider
    <li><b>Process data:</b> Extract relevant features such as motion intensity from video frames.</li>
    <li><b>Train the model: </b> Use the captured data to train an Isolation Forest model periodically.</li>
    <li><b>Detect anomalies:</b> Use the trained model to detect anomalies in real-time.</li>
    <li><b>Log anomalies:</b> Save anomaly data (timestamp, image) for further analysis.</li>
  </ol><br/>

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
  <p>Hardware and Software Setup includes:</p>
  <ul>
    <li>Raspberry Pi with a camera module and a USB microphone.</li>
    <li>Required packages installation commands for Linux on Raspberry Pi.</li>
  </ul>
  <p>Python Script for Data Collection and Anomaly Detection involves capturing video frames, detecting motion, training the model periodically, and detecting and logging anomalies.</p>
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
          <a href="https://www.kaggle.com/code/priyansh2904/lab-5?scriptVersionId=182441640" target="_blank"> View Runable code</a>
          </button>
        </div>
      </div>
  );
};
export default Lab2;
