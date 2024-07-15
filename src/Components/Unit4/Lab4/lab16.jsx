import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
// import Img1 from './imgs/image1.png';
// import Img2 from './imgs/image2.png';
// import Img3 from './imgs/image3.png';
// import Img4 from './imgs/image4.png';
// import Img5 from './imgs/image5.png';
import './lab16.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `
# Install libraries if not already installed
!pip install tensorflow scikit-learn

# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Define the path to your dataset
dataset_path = '/kaggle/input/real-life-industrial-dataset-of-casting-product'

# Load dataset (assuming data is organized in directories by class)
datagen = ImageDataGenerator(rescale=1./255)
dataset = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary')

# Function to load and prepare the data
def load_data(dataset):
    features = []
    labels = []
    for batch in dataset:
        X_batch, y_batch = batch
        features.extend(X_batch)
        labels.extend(y_batch)
        if len(features) >= dataset.samples:
            break
    return np.array(features), np.array(labels)

X, y = load_data(dataset)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize and reshape the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 224 * 224 * 3))
X_test = scaler.transform(X_test.reshape(-1, 224 * 224 * 3))

# Load pre-trained VGG16 model for feature extraction
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

# Function to extract features
def extract_features(model, data):
    features = model.predict(data)
    return features.reshape(features.shape[0], -1)

X_train_features = extract_features(model, X_train.reshape(-1, 224, 224, 3))
X_test_features = extract_features(model, X_test.reshape(-1, 224, 224, 3))

# Compute cosine similarity between test samples and train samples
cos_sim = cosine_similarity(X_test_features, X_train_features)
predicted_labels_cos_sim = y_train[np.argmax(cos_sim, axis=1)]

# Evaluate the cosine similarity classification
print("Cosine Similarity Classification Report")
print(classification_report(y_test, predicted_labels_cos_sim))

# Function to preprocess the given image path
def preprocess_image(image_path):
    print(f"Preprocessing image: {image_path}")
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to extract features of the given image
def extract_image_features(model, img_array):
    print("Extracting image features")
    features = model.predict(img_array)
    return features.reshape(1, -1)

# Function to classify the given image
def classify_image(image_path, model, X_train_features, y_train):
    img_array = preprocess_image(image_path)
    img_features = extract_image_features(model, img_array)
    cos_sim = cosine_similarity(img_features, X_train_features)
    predicted_label = y_train[np.argmax(cos_sim, axis=1)]
    return predicted_label[0]

# Provide the path to the image you want to classify
image_path = '/kaggle/input/testimage1/cast_def_0_138.jpeg'

# Classify the given image
predicted_quality = classify_image(image_path, model, X_train_features, y_train)
print(f"The predicted quality for the image is: {'High' if predicted_quality == 1 else 'Low'}")

# Display the given image
img = load_img(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()
`;

const codeSections = {
  Step1: `
# Install libraries if not already installed


!pip install tensorflow scikit-learn
`,
  ImportLibs: `
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
`,
  DefinePath: `
  dataset_path = '/kaggle/input/real-life-industrial-dataset-of-casting-product'
`,

  LoadDataset: `
  datagen = ImageDataGenerator(rescale=1./255)
  dataset = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary')
  `,
  LoadPrepData: `
  X, y = load_data(dataset)
  `,
  SplitData: `
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  `,
  NormaliseReshape: `
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train.reshape(-1, 224 * 224 * 3))
  X_test = scaler.transform(X_test.reshape(-1, 224 * 224 * 3))
  `,
  LoadVGG16: `
  base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
  model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
  `,
  ExtractFeatures:`
  X_train_features = extract_features(model, X_train.reshape(-1, 224, 224, 3))
  X_test_features = extract_features(model, X_test.reshape(-1, 224, 224, 3))
  `,
  ComputeCosSim: `
  cos_sim = cosine_similarity(X_test_features, X_train_features)
  predicted_labels_cos_sim = y_train[np.argmax(cos_sim, axis=1)]
  `,
  EvaluateClass: `
  print("Cosine Similarity Classification Report")
  print(classification_report(y_test, predicted_labels_cos_sim))
  `,
  PreprocessingImage: `
  def preprocess_image(image_path):
    print(f"Preprocessing image: {image_path}")
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

  def extract_image_features(model, img_array):
      print("Extracting image features")
      features = model.predict(img_array)
      return features.reshape(1, -1)
  `,
  ClassifyImage: `
  image_path = '/kaggle/input/testimage1/cast_def_0_138.jpeg'
  predicted_quality = classify_image(image_path, model, X_train_features, y_train)
  print(f"The predicted quality for the image is: {'High' if predicted_quality == 1 else 'Low'}")
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
        <h2>Image Classification Using Pre-trained VGG16 and Cosine Similarity</h2> <br />

        <p><strong onClick={() => handleHeadingClick("Step1")}>1. Installing Libraries</strong></p>
        <p>The code starts by ensuring that all the necessary libraries, such as TensorFlow and scikit-learn, are installed to enable machine learning functionalities.</p> <br />

        <p><strong onClick={() => handleHeadingClick("ImportLibs")}>2. Importing Libraries</strong></p>
        <p>Essential libraries like NumPy for handling data, TensorFlow for machine learning, and various scikit-learn modules for preprocessing and model evaluation are imported.</p> <br />

        <p><strong onClick={() => handleHeadingClick("DefinePath")}>3. Defining Dataset Path</strong></p>
        <p>The path where the dataset is stored is defined, guiding the code to locate and process the dataset correctly.</p> <br />

        <p><strong onClick={() => handleHeadingClick("LoadDataset")}>4. Loading Dataset</strong></p>
        <p>The dataset is loaded and prepared using TensorFlow's ImageDataGenerator, which also handles the image resizing and normalization.</p> <br />

        <p><strong onClick={() => handleHeadingClick("LoadPrepData")}>5. Loading and Preparing Data</strong></p>
        <p>A function is set up to load the data, extract images and labels, and ensure that all the data needed for training and testing is ready and accessible.</p> <br />

        <p><strong onClick={() => handleHeadingClick("SplitData")}>6. Splitting Data into Training and Testing Sets</strong></p>
        <p>The data is divided into training and testing sets to provide a robust evaluation of the model's performance, maintaining an unbiased approach towards model validation.</p> <br />

        <p><strong onClick={() => handleHeadingClick("NormalizeReshape")}>7. Normalizing and Reshaping Features</strong></p>
        <p>Data normalization and reshaping are critical for preparing the data to fit the input requirements of the pre-trained VGG16 model, ensuring that each input is treated equally during model training.</p> <br />

        <p><strong onClick={() => handleHeadingClick("LoadVGG16")}>8. Loading Pre-trained VGG16 Model</strong></p>
        <p>The pre-trained VGG16 model is loaded, set to extract deep features from the images. This model, trained on extensive datasets like ImageNet, provides a rich feature extraction capability.</p> <br />

        <p><strong onClick={() => handleHeadingClick("ExtractFeatures")}>9. Extracting Features</strong></p>
        <p>A function is defined to pass images through the VGG16 model and extract significant features necessary for classifying the images effectively using machine learning techniques.</p> <br />

        <p><strong onClick={() => handleHeadingClick("ComputeCosSim")}>10. Computing Cosine Similarity</strong></p>
        <p>Cosine similarity measures are computed between features of the training set and the test set to determine how similar the images are, aiding in the classification process based on the most similar training examples.</p> <br />

        <p><strong onClick={() => handleHeadingClick("EvaluateClass")}>11. Evaluating Cosine Similarity Classification</strong></p>
        <p>The effectiveness of using cosine similarity for classification is evaluated, providing insights into the accuracy of this method in categorizing images based on learned patterns and similarities.</p> <br />

        <p><strong onClick={() => handleHeadingClick("PreprocessImage")}>12. Preprocessing and Classifying New Images</strong></p>
        <p>Functions are prepared to preprocess new images, extract their features using the pre-trained model, and classify them by comparing these features to those of the images in the training set through cosine similarity.</p> <br />

        <p><strong onClick={() => handleHeadingClick("ClassifyImage")}>13. Displaying Classification Results</strong></p>
        <p>The classification results are displayed, showing the predicted labels for new images, which helps in understanding the model's performance and its practical application in real-world scenarios.</p> <br />

        <p>The Key aspects of this code are:-</p>
        <ul>
        <li><b>Feature Extraction:</b> Feature extraction involves identifying and extracting important characteristics or patterns from industrial equipment images.  In the context of image classification, these features can include edges, textures, shapes, and more complex structures.</li> <br />
        {/* <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />
        <img style={{width: '100%'}} src={Img2} alt="image2" /> <br /> <br /> */}
        <li><b>Cosine Similarity: </b> Cosine similarity measures the similarity between two vectors by comparing the angle between them. It is often used for image classification by comparing the feature vectors of images to determine how similar they are.</li>
        
        </ul>
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
            <a href="https://www.kaggle.com/code/pushkarns/unit-3-lab-1-final" target="_blank"> View Runable code</a>
            </button>
    </div>
    </div>
    );
};
export default Lab2;
