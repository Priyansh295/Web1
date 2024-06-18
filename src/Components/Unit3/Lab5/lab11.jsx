import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.png';
import './lab11.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `
# 1. Import Libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.metrics import classification_report, accuracy_score

# 2. Load Data

train = pd.read_csv("./datasets/fashion-mnist_train.csv")
test = pd.read_csv("./datasets/fashion-mnist_test.csv")

train.head()

# The catagories of clothing were in dataset
class_labels= ["T-shirt/top","Trouser","Pullover" ,"Dress","Coat" ,"Sandal" ,"Shirt" ,"Sneaker" ,"Bag" ,"Ankle boot"]

test.head()

# store data as an array 
train_data = np.array(train, dtype= "float32")
test_data = np.array(test, dtype= "float32")

x_train = train_data[:, 1:]
y_train = train_data[:, 0]

x_test = test_data[:, 1:]
y_test = test_data[:, 0]

# arrange pixel values between 0 to 1 
x_train = x_train/255
x_test = x_test/255

# 3. Split Data

x_train , x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

plt.imshow(x_train[1])
print(y_train[1])

# 4. Build a Deep Learning Model and Train the Model

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


model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=1, validation_data=(x_val, y_val))

# 5. Test Model

y_pred = model.predict(x_test)

model.evaluate(x_test, y_test)

# Plot the training and validation accuracy and loss
import matplotlib.pyplot as plt

plt.figure(figsize= (16,30))
j=1

for i in np.random.randint(0, 1000,60):
    plt.subplot(10,6, j)
    j+=1
    plt.imshow(x_test[i].reshape(28,28), cmap='Greys')
    plt.title('Actual = {} / {} \nPredicted = {} / {}'.format(class_labels[int(y_test[i])], int(y_test[i]), class_labels[np.argmax(y_pred[i])], np.argmax(y_pred[i])))
    plt.axis('off')


# 6. Evaluate 

# Convert one-hot encoded labels to class indices if needed
if y_test.ndim == 2 and y_test.shape[1] > 1:
    y_test_indices = np.argmax(y_test, axis=1)
else:
    y_test_indices = y_test  # If y_test is already in integer form

# Convert y_pred to class indices
y_pred_indices = np.argmax(y_pred, axis=1)

# Generate the classification report
cr = classification_report(y_test_indices, y_pred_indices, target_names=class_labels)
print(cr)

# Calculate and print the overall accuracy
accuracy = accuracy_score(y_test_indices, y_pred_indices)
print(f"Overall Accuracy: {accuracy}")

# 7. Save Model

model.save('fashion_mnist_cnn_model.h5')

#Load Model

fashion_model = tf.keras.models.load_model('fashion_mnist_cnn_model.h5')
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
      <h2>XGBoost: Extreme Gradient Boosting Explained</h2> <br />

      <p>XGBoost, short for "Extreme Gradient Boosting," is a scalable and highly efficient machine 
        learning algorithm used for supervised learning tasks, including classification and regression. 
        Developed by Tianqi Chen, XGBoost has become one of the most popular algorithms in the field 
        due to its speed, performance, and versatility. Here's a detailed look into XGBoost, supported by 
        theoretical explanations and illustrative images.</p> <br />

      <h3>Theory Behind XGBoost</h3> <br />
      <ul>
        <li><strong>Gradient Boosting Framework:</strong> XGBoost is based on the gradient boosting framework, 
          which builds models sequentially. Each new model attempts to correct the errors of the 
          previous models. The overall model is a weighted sum of all previous models.</li> <br />
        <li><strong>Decision Trees:</strong> XGBoost primarily uses decision trees as its base learners. These are 
          simple models that split the data into branches based on feature values to make 
          predictions.</li> <br />
        <li><strong>Boosting:</strong> Boosting is an ensemble technique that combines the outputs of several weak 
          learners to create a strong learner. Each new model is trained to correct the errors made 
          by the previous models, focusing more on the difficult-to-predict instances.</li> <br />
        <li><strong>Gradient Descent:</strong> In XGBoost, the new models are fit on the negative gradient of the 
          loss function. This means each new model aims to reduce the errors (residuals) of the 
          previous models by moving in the direction of the steepest descent (gradient) of the loss 
          function.</li> <br />
      </ul> <br />
      <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />

      <h3>Mathematical Formulation and Visual Illustration</h3> <br />
      <img style={{width: '100%'}} src={Img2} alt="image2" /> <br /> <br />
      <p><strong>Gradient Boosting Process:</strong></p>
      <ol>
        <li><strong>Step 1:</strong> Start with an initial prediction (e.g., the mean of the target values).</li> <br />
        <li><strong>Step 2:</strong> Compute the residuals (errors) between the actual and predicted values.</li> <br />
        <li><strong>Step 3:</strong> Fit a new model to the residuals.</li> <br />
        <li><strong>Step 4:</strong> Update the predictions by adding the new model's predictions, scaled by a 
          learning rate.</li> <br />
        <li><strong>Step 5:</strong> Repeat steps 2-4 for a specified number of iterations or until the residuals are 
          minimized.</li> <br />
      </ol>

      <p><strong>Decision Tree Example:</strong></p> <br />
      <img style={{width: '100%'}} src={Img3} alt="image3" /> <br /> <br />
      <p>XGBoost uses decision trees as base learners. Each tree splits the data based on 
        feature values to make predictions.</p> <br />
        <img style={{width: '100%'}} src={Img4} alt="image4" /> <br /> <br />
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
          <a href="https://www.kaggle.com/code/priyanshsurana/notebook7ea446b9bf?scriptVersionId=183970208" target="_blank"> View Runable code</a>
          </button>
        </div>
      </div>
  );
};
export default Lab2;
