import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.gif';
import './lab8.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# loading the data from csv file to a Pandas DataFrame
customer_data = pd.read_csv('Mall_Customers.csv')

# first 5 rows in the dataframe
customer_data.head()

# finding the number of rows and columns
customer_data.shape

# getting some informations about the dataset
customer_data.info()

# checking for missing values
customer_data.isnull().sum()

relevant_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
X = customer_data.iloc[:,[3,4]].values

print(X)

# Normalizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# finding wcss value for different number of clusters

wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)
  
  # plot an elbow graph

sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

# Adding the cluster labels to the original DataFrame
customer_data['Cluster'] = Y

print(Y)


# Select only relevant columns for cluster analysis
cluster_summary = customer_data.groupby('Cluster')[relevant_columns].mean()
print(cluster_summary)

# Count of males and females in each cluster
gender_count = customer_data.groupby(['Cluster', 'Genre']).size().unstack(fill_value=0)
print(gender_count)

# Visualizing the number of customers in each cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=customer_data)
plt.title('Number of Customers in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.show()

# Visualizing the clusters (Original Data)
plt.figure(figsize=(16, 8))
colors = ['green', 'red', 'yellow', 'violet', 'blue']

# Original data visualization
plt.subplot(1, 2, 1)
for i in range(5):
    plt.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
plt.title('Customer Groups (Original Data)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()


# PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X)

# Visualize the PCA-reduced clusters
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=Y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Customer Segmentation using PCA')
plt.legend()
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
      <h2>Unsupervised Learning for Customer Segmentation</h2>

      <p><strong onClick={() => handleHeadingClick("Step1")}>Data Loading and Preprocessing</strong></p> <br />
      <ol style={{marginLeft: '25px'}}>
        <li><b>Loading Data:</b> The code loads customer data from a CSV file named 
        'Mall_Customers.csv' into a Pandas DataFrame called customer_data.</li> <br />

        <li><b>Data Inspection:</b> The code checks the first five rows of the data using
          <ul>
            <li><b>customer_data.head()</b> the number of rows and columns using</li>
            <li><b>customer_data.shape</b>  and the data types of each column using</li>
            <li><b>customer_data.info()</b>  and the data types of each column using</li>
            <li><b>customer_data.isnull().sum()</b></li> <br />
          </ul>
        </li>
      </ol> <br />

      <p><strong onClick={() => handleHeadingClick("Step2")}>Clustering: </strong></p> <br />
      <ol style={{marginLeft: '25px'}}>
        <li> <strong><b>K-Means Clustering: </b></strong> The code performs K-Means clustering on the selected features 
        to identify customer segments. It calculates the Within Cluster Sum of Squares (WCSS) for 
        different numbers of clusters (from 1 to 10) and plots an elbow graph to determine the optimal 
        number of clusters.</li> <br />

        <img style={{width: '100%'}} src={Img2} alt="image2" /> <br /> <br />

        <li><strong onClick={() => handleHeadingClick("elbow")}><b>Optimal Number of Clusters:</b></strong> The code determines the optimal number of clusters to be 5 based on the elbow graph.</li> <br />
        <p style={{marginLeft: '25px'}}> <b>Elbow Plot:</b> </p>
        <p style={{marginLeft: '25px'}}> The elbow method is a technique used to determine the optimal 
          number of clusters in K-Means clustering. It involves plotting the WCSS against the number 
          of clusters and identifying the point where the curve starts to flatten, indicating that 
          adding more clusters does not significantly improve the clustering.</p> <br />

          <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />
        <li><b>Clustering:</b> The code applies K-Means clustering with 5 clusters to the data and assigns each customer to a cluster.</li> <br />
      </ol> <br />

      <p><strong onClick={() => handleHeadingClick("Visual")}>Visualization</strong></p>
      <ol style={{marginLeft: '25px'}}>
        <li><strong onClick={() => handleHeadingClick("ClusterSummary")}><b>Cluster Summary:</b></strong> The code calculates the mean annual income and spending score for each cluster and prints the results.</li> <br />
        <li><strong onClick={() => handleHeadingClick("Gender")}><b>Gender Distribution:</b></strong> The code counts the number of males and females in each cluster and prints the results.</li> <br />
        <li><strong onClick={() => handleHeadingClick("NoCustomer")}><b>Number of Customers in Each Cluster: </b></strong>  The code visualizes the number of customers in each cluster using a count plot.</li> <br />
        <li><strong onClick={() => handleHeadingClick("OrgVisual")}><b>Original Data Visualization: </b> </strong> The code visualizes the original data with each cluster represented by a different color.</li> <br />
        <li><strong onClick={() => handleHeadingClick("PcaK")}><b>PCA and K-Means Clustering:</b></strong> The code applies Principal Component Analysis (PCA) to 
        reduce the dimensionality of the data and then performs K-Means clustering on the transformed 
        data. It visualizes the clusters in the PCA space.</li>
      </ol>
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
          <a href="https://www.kaggle.com/code/priyansh2904/unit3lab2/notebook?scriptVersionId=183932297" target="_blank"> View Runable code</a>
          </button>
        </div>
      </div>
  );
};
export default Lab2;
