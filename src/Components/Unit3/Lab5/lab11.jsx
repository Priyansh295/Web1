import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.gif';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import './lab11.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Paths to the dataset
TRAIN_PATH = '/kaggle/input/dataset/dev_gearbox/gearbox/train'
TEST_PATH = '/kaggle/input/dataset/dev_gearbox/gearbox/test'

def load_audio_files(path):
    audio_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)
    return audio_files

def extract_features(file_list, n_fft=1024, hop_length=512):
    features = []
    for file_path in file_list:
        y, sr = librosa.load(file_path, sr=None)
        spectrogram = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=False)
        magnitude, _ = librosa.magphase(spectrogram)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=1e-6)
        features.append(magnitude_db.flatten())
    return np.array(features)

# Load audio files
train_files = load_audio_files(TRAIN_PATH)
test_files = load_audio_files(TEST_PATH)

# Extract features
X_train = extract_features(train_files)
X_test = extract_features(test_files)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduce the dimensionality of the data using PCA for visualization purposes
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Visualize the data distribution using PCA components
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], s=5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Training Data')
plt.show()

# Select an eps value from the plot and apply DBSCAN
eps_value = 480  # Chosen based on the k-distance plot elbow
dbscan = DBSCAN(eps=eps_value, min_samples=20)
dbscan.fit(X_train_scaled)

# Predict the cluster for each sample in the test set
test_clusters = dbscan.fit_predict(X_test_scaled)

print(f"Cluster assignments for the test set: {test_clusters}")

# Visualize the clustering using PCA components
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_clusters, cmap='viridis', s=5)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'DBSCAN Clustering with eps={eps_value}')
plt.show()
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
        <h2>Implementation of an Unsupervised Algorithm to cluster machine sound for Anomaly Detection</h2> <br />
        <h4>DBSCAN: A Pictorial Representation and Explanation</h4> <br />

        <p>DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a popular clustering algorithm that groups together points that are closely packed and marks points that lie alone in low-density regions as outliers.</p> <br />
        
        <h3>Key Concepts</h3> <br />
        <ul>
          <li><strong>Epsilon (eps):</strong> The maximum distance between two points for them to be considered as in the same neighborhood.</li> <br />
          <li><strong>MinPts (min_samples):</strong> The minimum number of points required to form a dense region (a cluster).</li> <br />
        </ul> <br />
        <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />

        <h3>Working of DBSCAN</h3> <br />
        <ul>
          <li><strong>Core Points:</strong> A point is a core point if it has at least min_samples points (including itself) within eps.</li> <br />
          <li><strong>Border Points:</strong> A point that has fewer than min_samples points within eps, but is in the neighborhood of a core point.</li> <br />
          <li><strong>Noise Points:</strong> A point that is neither a core point nor a border point.</li> <br />
        </ul> <br />

        <h3>Algorithm Steps</h3> <br />
        <ol style={{marginLeft: '25px'}}>
          <li><strong>Label Points:</strong>
            <ul>
              <li>If a point has at least min_samples points within eps, mark it as a core point.</li>
              <li>If a point is within eps distance of a core point, mark it as a border point.</li>
              <li>If a point is neither, mark it as noise.</li>
            </ul>
          </li> <br />
          <li><strong>Cluster Formation:</strong>
            <ul>
              <li>Start with an arbitrary point and retrieve its eps-neighborhood.</li>
              <li>If it’s a core point, form a cluster. Add all points within eps as part of this cluster.</li>
              <li>Recursively visit each point within this cluster and include all reachable points within eps.</li>
              <li>Continue until all points are visited.</li>
            </ul>
          </li> <br />
        </ol> <br />

        <h3>Advantages and Disadvantages</h3> <br />
        <ul>
          <li><strong>Advantages:</strong> DBSCAN can find arbitrarily shaped clusters and is robust to noise and outliers. It does not require specifying the number of clusters beforehand.</li> <br />
          <img style={{width: '100%'}} src={Img2} alt="image3" /> <br /> <br />
          <li><strong>Disadvantages:</strong> DBSCAN's performance depends on the choice of eps and min_samples. It may struggle with varying densities and high-dimensional data.</li> <br />
          <img style={{width: '100%'}} src={Img3} alt="image3" /> <br /> <br />
        </ul> <br />
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
          <a href="https://www.kaggle.com/code/priyanshsurana/unit-3-lab-5?scriptVersionId=184229319" target="_blank"> View Runable code</a>
          </button>
        </div>
      </div>
  );
};
export default Lab2;
