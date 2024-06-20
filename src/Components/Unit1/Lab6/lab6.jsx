import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver';
import Img1 from './imgs/image1.gif';
import Img2 from './imgs/image2.jpg';
import Img3 from './imgs/image3.gif';
import Img4 from './imgs/image4.jpg';
import Img5 from './imgs/image5.png';
import Img6 from './imgs/image6.webp';
import './lab6.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `
# Import Libraries 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.linear_model import LogisticRegression
import time
import matplotlib.pyplot as plt
from IPython.display import display
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Data Preparations 

train_img_path = '/kaggle/input/car-damage-severity-dataset/data3a/training'
test_img_path = '/kaggle/input/car-damage-severity-dataset/data3a/validation'

batch_size = 32
img_height = 224
img_width = 224

train_data_gen = ImageDataGenerator(rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        validation_split=0.20,) 

# Use flow_from_directory for the training dataset
train_ds = train_data_gen.flow_from_directory(
    train_img_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  
    subset='training', 
    seed=123,
    shuffle=True  
)
valid_ds = train_data_gen.flow_from_directory(
    train_img_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  
    subset='validation', 
    seed=123,
    shuffle=True  
)

test_data_gen = ImageDataGenerator(rescale=1./255,)  # You may adjust other parameters as needed

# Use flow_from_directory for the test dataset
test_ds = test_data_gen.flow_from_directory(
    test_img_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Assuming label_mode='int' in your original code
    shuffle=False  # Set to True if you want to shuffle the data
)

cl=test_ds.class_indices
print(cl)

def plot_images_from_dataset(dataset, num_images=9):
    # Fetch a batch of images and labels from the dataset
    images, labels = next(iter(dataset))

    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(images))):  
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        
        # Map the label index back to the original class name
        label_index = labels[i].argmax()  # Assumes one-hot encoding
        class_name = next(key for key, value in cl.items() if value == label_index)
        
        plt.title(f"Class: {class_name}")
        plt.axis("off")
    plt.show()


# Assuming test_ds is your dataset
plot_images_from_dataset(test_ds)


# CNN USING PRETRAINED EFF NET**

from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow_addons.metrics import F1Score

img_size = (224, 224)
lr = 0.001
class_count = 3

img_shape = (img_size[0], img_size[1], 3)

base_model = DenseNet169(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
base_model.trainable = True
x = base_model.output
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), 
          activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
x = Dropout(rate=.4, seed=123)(x)
output = Dense(class_count, activation='softmax')(x)
model_eff = Model(inputs=base_model.input, outputs=output)
model_eff.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', 
              metrics=['accuracy','AUC'])


epochs=50

# Train the model
history_eff = model_eff.fit(
    train_ds,
    epochs=epochs,
    validation_data=valid_ds,  
    verbose=1,
)

# Save training and validation histories for later analysis
all_train_histories = [history_eff.history['accuracy']]
all_val_histories = [history_eff.history['val_accuracy']]

model_eff.save('model_eff.h5')

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Function to preprocess image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict and display the image with its predicted class
def predict_and_display_image(image_path, model):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make predictions
    predictions = model.predict(processed_image)
    
    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    class_name = next(key for key, value in cl.items() if value == predicted_class_index)
    
    # Display the image with its predicted class
    img = load_img(image_path)
    plt.imshow(img)
    plt.title(f"Predicted Class: {class_name}")
    plt.axis("off")
    plt.show()

# Path to your own image
your_image_path = '/kaggle/input/test-2-minor/0015.JPEG'  # Change this to your image path

# Predict and display the image
predict_and_display_image(your_image_path, model_eff)

# Assuming train_ds, valid_ds, test_ds, and base_model are already defined
def extract_features(dataset, model, max_batches=35):
    features = []
    labels = []
    for idx, (images, label_batch) in enumerate(dataset):
        start_time = time.time()
        features_batch = model.predict(images)
        features.append(features_batch)
        labels.append(label_batch)
        end_time = time.time()
        print(f"Batch {idx + 1}/{len(dataset)} processed in {end_time - start_time:.2f} seconds")
        if idx >= max_batches:
            print("Breaking loop to prevent infinite iteration.")
            break  # Break the loop after processing max_batches batches
    return np.concatenate(features), np.concatenate(labels)

start_time = time.time()
train_features, train_labels = extract_features(train_ds, base_model)
end_time = time.time()
print(f"Feature extraction for training set completed in {end_time - start_time:.2f} seconds")

start_time = time.time()
valid_features, valid_labels = extract_features(valid_ds, base_model)
end_time = time.time()
print(f"Feature extraction for validation set completed in {end_time - start_time:.2f} seconds")

start_time = time.time()
test_features, test_labels = extract_features(test_ds, base_model)
end_time = time.time()
print(f"Feature extraction for test set completed in {end_time - start_time:.2f} seconds")

# Flatten the labels for logistic regression
train_labels = np.argmax(train_labels, axis=1)
valid_labels = np.argmax(valid_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=10)
start_time = time.time()
log_reg.fit(train_features, train_labels)
end_time = time.time()
print(f"Logistic Regression training completed in {end_time - start_time:.2f} seconds")

# Evaluate Logistic Regression
valid_pred = log_reg.predict(valid_features)
test_pred = log_reg.predict(test_features)

print("Validation Accuracy:", accuracy_score(valid_labels, valid_pred))
print("Test Accuracy:", accuracy_score(test_labels, test_pred))
print("Classification Report:\n", classification_report(test_labels, test_pred))
print("Confusion Matrix:\n", confusion_matrix(test_labels, test_pred))

# Prediction function using logistic regression
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_and_display_image(image_path, base_model, log_reg_model):
    processed_image = preprocess_image(image_path)
    features = base_model.predict(processed_image)
    prediction = log_reg_model.predict(features)
    predicted_class_index = prediction[0]
    class_name = next(key for key, value in cl.items() if value == predicted_class_index)
    img = load_img(image_path)
    plt.imshow(img)
    plt.title(f"Predicted Class: {class_name}")
    plt.axis("off")
    plt.show()

upload_btn = widgets.FileUpload(accept='image/*', multiple=False)
output = widgets.Output()

def on_upload_change(change):
    for filename, file_info in change['new'].items():
        with open('uploaded_image.jpg', 'wb') as f:
            f.write(file_info['content'])
        with output:
            output.clear_output()
            predict_and_display_image('uploaded_image.jpg', base_model, log_reg)

upload_btn.observe(on_upload_change, names='value')
display(upload_btn, output)

# Assuming train_ds, valid_ds, test_ds, and base_model are already defined
def extract_features(dataset, model, max_batches=35):
    features = []
    labels = []
    for idx, (images, label_batch) in enumerate(dataset):
        start_time = time.time()
        features_batch = model.predict(images)
        features.append(features_batch)
        labels.append(label_batch)
        end_time = time.time()
        print(f"Batch {idx + 1}/{len(dataset)} processed in {end_time - start_time:.2f} seconds")
        if idx >= max_batches:
            print("Breaking loop to prevent infinite iteration.")
            break  # Break the loop after processing max_batches batches
    return np.concatenate(features), np.concatenate(labels)

start_time = time.time()
train_features, train_labels = extract_features(train_ds, base_model)
end_time = time.time()
print(f"Feature extraction for training set completed in {end_time - start_time:.2f} seconds")

start_time = time.time()
valid_features, valid_labels = extract_features(valid_ds, base_model)
end_time = time.time()
print(f"Feature extraction for validation set completed in {end_time - start_time:.2f} seconds")

start_time = time.time()
test_features, test_labels = extract_features(test_ds, base_model)
end_time = time.time()
print(f"Feature extraction for test set completed in {end_time - start_time:.2f} seconds")

# Flatten the labels for random forest classifier
train_labels = np.argmax(train_labels, axis=1)
valid_labels = np.argmax(valid_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Random Forest Model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
rf_clf.fit(train_features, train_labels)
end_time = time.time()
print(f"Random Forest training completed in {end_time - start_time:.2f} seconds")

# Evaluate Random Forest
valid_pred = rf_clf.predict(valid_features)
test_pred = rf_clf.predict(test_features)

print("Validation Accuracy:", accuracy_score(valid_labels, valid_pred))
print("Test Accuracy:", accuracy_score(test_labels, test_pred))
print("Classification Report:\n", classification_report(test_labels, test_pred))
print("Confusion Matrix:\n", confusion_matrix(test_labels, test_pred))

# Prediction function using Random Forest
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_and_display_image(image_path, base_model, rf_model):
    processed_image = preprocess_image(image_path)
    features = base_model.predict(processed_image)
    prediction = rf_model.predict(features)
    predicted_class_index = prediction[0]
    class_name = next(key for key, value in cl.items() if value == predicted_class_index)
    img = load_img(image_path)
    plt.imshow(img)
    plt.title(f"Predicted Class: {class_name}")
    plt.axis("off")
    plt.show()

upload_btn = widgets.FileUpload(accept='image/*', multiple=False)
output = widgets.Output()

def on_upload_change(change):
    for filename, file_info in change['new'].items():
        with open('uploaded_image.jpg', 'wb') as f:
            f.write(file_info['content'])
        with output:
            output.clear_output()
            predict_and_display_image('uploaded_image.jpg', base_model, rf_clf)

upload_btn.observe(on_upload_change, names='value')
display(upload_btn, output)

`;

const codeSections = {
  logisticRegression: `
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Assuming train_ds, valid_ds, test_ds, and base_model are already defined
def extract_features(dataset, model, max_batches=35):
    features = []
    labels = []
    for idx, (images, label_batch) in enumerate(dataset):
        start_time = time.time()
        features_batch = model.predict(images)
        features.append(features_batch)
        labels.append(label_batch)
        end_time = time.time()
        print(f"Batch {idx + 1}/{len(dataset)} processed in {end_time - start_time:.2f} seconds")
        if idx >= max_batches:
            print("Breaking loop to prevent infinite iteration.")
            break  # Break the loop after processing max_batches batches
    return np.concatenate(features), np.concatenate(labels)

start_time = time.time()
train_features, train_labels = extract_features(train_ds, base_model)
end_time = time.time()
print(f"Feature extraction for training set completed in {end_time - start_time:.2f} seconds")

start_time = time.time()
valid_features, valid_labels = extract_features(valid_ds, base_model)
end_time = time.time()
print(f"Feature extraction for validation set completed in {end_time - start_time:.2f} seconds")

start_time = time.time()
test_features, test_labels = extract_features(test_ds, base_model)
end_time = time.time()
print(f"Feature extraction for test set completed in {end_time - start_time:.2f} seconds")

# Flatten the labels for logistic regression
train_labels = np.argmax(train_labels, axis=1)
valid_labels = np.argmax(valid_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=10)
start_time = time.time()
log_reg.fit(train_features, train_labels)
end_time = time.time()
print(f"Logistic Regression training completed in {end_time - start_time:.2f} seconds")

# Evaluate Logistic Regression
valid_pred = log_reg.predict(valid_features)
test_pred = log_reg.predict(test_features)

print("Validation Accuracy:", accuracy_score(valid_labels, valid_pred))
print("Test Accuracy:", accuracy_score(test_labels, test_pred))
print("Classification Report:\n", classification_report(test_labels, test_pred))
print("Confusion Matrix:\n", confusion_matrix(test_labels, test_pred))

# Prediction function using logistic regression
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_and_display_image(image_path, base_model, log_reg_model):
    processed_image = preprocess_image(image_path)
    features = base_model.predict(processed_image)
    prediction = log_reg_model.predict(features)
    predicted_class_index = prediction[0]
    class_name = next(key for key, value in cl.items() if value == predicted_class_index)
    img = load_img(image_path)
    plt.imshow(img)
    plt.title(f"Predicted Class: {class_name}")
    plt.axis("off")
    plt.show()

upload_btn = widgets.FileUpload(accept='image/*', multiple=False)
output = widgets.Output()

def on_upload_change(change):
    for filename, file_info in change['new'].items():
        with open('uploaded_image.jpg', 'wb') as f:
            f.write(file_info['content'])
        with output:
            output.clear_output()
            predict_and_display_image('uploaded_image.jpg', base_model, log_reg)

upload_btn.observe(on_upload_change, names='value')
display(upload_btn, output)
`,
  randomForest: `
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Assuming train_ds, valid_ds, test_ds, and base_model are already defined
def extract_features(dataset, model, max_batches=35):
    features = []
    labels = []
    for idx, (images, label_batch) in enumerate(dataset):
        start_time = time.time()
        features_batch = model.predict(images)
        features.append(features_batch)
        labels.append(label_batch)
        end_time = time.time()
        print(f"Batch {idx + 1}/{len(dataset)} processed in {end_time - start_time:.2f} seconds")
        if idx >= max_batches:
            print("Breaking loop to prevent infinite iteration.")
            break  # Break the loop after processing max_batches batches
    return np.concatenate(features), np.concatenate(labels)

start_time = time.time()
train_features, train_labels = extract_features(train_ds, base_model)
end_time = time.time()
print(f"Feature extraction for training set completed in {end_time - start_time:.2f} seconds")

start_time = time.time()
valid_features, valid_labels = extract_features(valid_ds, base_model)
end_time = time.time()
print(f"Feature extraction for validation set completed in {end_time - start_time:.2f} seconds")

start_time = time.time()
test_features, test_labels = extract_features(test_ds, base_model)
end_time = time.time()
print(f"Feature extraction for test set completed in {end_time - start_time:.2f} seconds")

# Flatten the labels for random forest classifier
train_labels = np.argmax(train_labels, axis=1)
valid_labels = np.argmax(valid_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Random Forest Model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
rf_clf.fit(train_features, train_labels)
end_time = time.time()
print(f"Random Forest training completed in {end_time - start_time:.2f} seconds")

# Evaluate Random Forest
valid_pred = rf_clf.predict(valid_features)
test_pred = rf_clf.predict(test_features)

print("Validation Accuracy:", accuracy_score(valid_labels, valid_pred))
print("Test Accuracy:", accuracy_score(test_labels, test_pred))
print("Classification Report:\n", classification_report(test_labels, test_pred))
print("Confusion Matrix:\n", confusion_matrix(test_labels, test_pred))

# Prediction function using Random Forest
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_and_display_image(image_path, base_model, rf_model):
    processed_image = preprocess_image(image_path)
    features = base_model.predict(processed_image)
    prediction = rf_model.predict(features)
    predicted_class_index = prediction[0]
    class_name = next(key for key, value in cl.items() if value == predicted_class_index)
    img = load_img(image_path)
    plt.imshow(img)
    plt.title(f"Predicted Class: {class_name}")
    plt.axis("off")
    plt.show()

upload_btn = widgets.FileUpload(accept='image/*', multiple=False)
output = widgets.Output()

def on_upload_change(change):
    for filename, file_info in change['new'].items():
        with open('uploaded_image.jpg', 'wb') as f:
            f.write(file_info['content'])
        with output:
            output.clear_output()
            predict_and_display_image('uploaded_image.jpg', base_model, rf_clf)

upload_btn.observe(on_upload_change, names='value')
display(upload_btn, output)
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
          <h2>Machine Learning Model Theories Explained</h2> <br />

          <p><strong onClick={() => handleHeadingClick("logisticRegression")}>Logistic Regression:</strong></p> <br />
          <ol>
            <li><b>Purpose:</b>  Logistic regression is used to classify things into one of two groups.</li> <br />
            <li><b>How it works:</b>
              <ul>
                <li><b>input:</b> Takes various features (input variables).</li>
                <li><b>Output:</b> Predicts the probability that an instance belongs to a certain class (between 0 and 1).</li>
                <li><b>Function:</b> Uses a special function called the logistic (or sigmoid) function to convert inputs into a probability.</li>
              </ul>
            </li>
            <li><b>Key Points:</b>
              <ul>
                <li>Uses examples with known outcomes to learn from.</li>
                <li>It is a linear model, meaning the relationship between the input features and the output is linear</li>
              </ul>
            </li>
          </ol> <br />
          <img style={{ width: '100%' }} src={Img2} alt="image2" /> <br /> <br />
          <img style={{ width: '100%' }} src={Img3} alt="image3" /> <br /> <br />
          <img style={{ width: '100%' }} src={Img6} alt="image6" /> <br /> <br />

          <p><strong onClick={() => handleHeadingClick("randomForest")}>Random Forest:</strong></p> <br />
          <ol>
            <li><b>Purpose:</b> Random Forest is used for both classification (grouping things) and regression (predicting numbers).</li> <br />
            <li><b>How it works:</b>
              <ul>
                <li><b>Trees:</b> Builds many decision trees to make predictions.</li>
                <li><b>Combination:</b> Combines results from all trees for a final prediction.</li>
                <li><b>Classification:</b>  For classification, it chooses the most common class from all trees.</li>
                <li><b>Regression:</b> For regression, it takes the average prediction from all trees.</li>
              </ul>
            </li> <br />
            <li><b>Key Points:</b>
              <ul>
                <li> Each tree is trained on a random subset of the data and uses a random subset of features.</li>
                <li> Makes the model less likely to overfit and better at handling complex data. Hence provides beter accuracy</li>
              </ul>
            </li>
          </ol>
          <img style={{ width: '100%' }} src={Img1} alt="image1" /> <br /> <br />
          <img style={{ width: '100%' }} src={Img4} alt="imag41" /> <br /> <br />
          <img style={{ width: '100%' }} src={Img5} alt="image5" /> <br /> <br />
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
          <a href="https://www.kaggle.com/code/pushkarns/lab6new" target="_blank"> View Runable code</a>
        </button>
      </div>
    </div>
  );
};
export default Lab2;
