import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.png';
import Img5 from './imgs/image5.png';
import Img6 from './imgs/image6.gif';
import Img7 from './imgs/image7.png';
import Img8 from './imgs/image8.png';
import Img9 from './imgs/image9.png';
import Img10 from './imgs/image10.png';
import './lab4.css';

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
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow_addons.metrics import F1Score
import ipywidgets as widgets
from IPython.display import display
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# Data Preparation

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

# Define the upload button and output widget
upload_btn = widgets.FileUpload(accept='image/*', multiple=False)
output = widgets.Output()

# Function to handle the image upload and classification
def on_upload_change(change):
    for filename, file_info in change['new'].items():
        # Save the uploaded image to a temporary path
        with open('uploaded_image.jpg', 'wb') as f:
            f.write(file_info['content'])
        
        # Display the image and prediction
        with output:
            output.clear_output()  # Clear previous outputs
            predict_and_display_image('uploaded_image.jpg', model_eff)

# Attach the function to the upload button
upload_btn.observe(on_upload_change, names='value')

# Display the upload button and output widget
display(upload_btn, output)


plt.plot(history_eff.history['accuracy'], label='Training Accuracy')
plt.plot(history_eff.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy over Epochs')
plt.show()

# Testing code
test_accuracy = model_eff.evaluate(test_ds)

# Confusion matrix
true_labels = test_ds.classes
predictions = model_eff.predict(test_ds)
predicted_labels = np.argmax(predictions, axis=1)

sns.heatmap(confusion_matrix(true_labels, predicted_labels), annot=True)

# Print classification report
print(classification_report(true_labels, predicted_labels))

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual Classes')
plt.show()
`;

const codeSections = {
  step1: `
# Data Preparation

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

`,
  step2: `

  # CNN USING PRETRAINED EFF NET**

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
`,
  step3: `
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

# Define the upload button and output widget
upload_btn = widgets.FileUpload(accept='image/*', multiple=False)
output = widgets.Output()

# Function to handle the image upload and classification
def on_upload_change(change):
    for filename, file_info in change['new'].items():
        # Save the uploaded image to a temporary path
        with open('uploaded_image.jpg', 'wb') as f:
            f.write(file_info['content'])
        
        # Display the image and prediction
        with output:
            output.clear_output()  # Clear previous outputs
            predict_and_display_image('uploaded_image.jpg', model_eff)

# Attach the function to the upload button
upload_btn.observe(on_upload_change, names='value')

# Display the upload button and output widget
display(upload_btn, output)


plt.plot(history_eff.history['accuracy'], label='Training Accuracy')
plt.plot(history_eff.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy over Epochs')
plt.show()

# Testing code
test_accuracy = model_eff.evaluate(test_ds)

# Confusion matrix
true_labels = test_ds.classes
predictions = model_eff.predict(test_ds)
predicted_labels = np.argmax(predictions, axis=1)

sns.heatmap(confusion_matrix(true_labels, predicted_labels), annot=True)

# Print classification report
print(classification_report(true_labels, predicted_labels))

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual Classes')
plt.show()
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
        <h2>checking severity of damage using CNN</h2> <br />
        
        <p><strong onClick={() => handleHeadingClick("step1")}>STEP 1: DEFINING PARAMETERS FOR TRAINING AND TESTING PHASE</strong></p> <br />
        <img style={{width: '100%'}} src={Img3} alt="image3" /> <br /> <br />
        <img style={{width: '100%'}} src={Img4} alt="image4" /> <br /> <br />

        <p><strong onClick={() => handleHeadingClick("step2")}>STEP 2: DEFINING THE CNN THE CRUX OF OUR MODEL TO PREDICT THE SEVERITY OF DAMAGE ON VEHICLES</strong></p> <br />
        <p><b>Convolution layer:</b> The core building block of a CNN is the convolutional layer. It applies filters to small regions of the input data, known as receptive fields. Each filter is a small matrix of weights that slides across the input data, performing a dot product with the input pixels. This process is known as convolution.</p> <br />
        <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />
        <img style={{width: '100%'}} src={Img5} alt="image5" /> <br /> <br />
        <p>The output of the convolutional layer is a feature map, which is a two-dimensional representation of the input data. This feature map captures the presence of specific features in the input data, such as edges or lines</p> <br />
        <img style={{width: '100%'}} src={Img6} alt="image6" /> <br /> <br />
        <p><b>Pooling Layer:</b>The pooling layer simplifies the output of the convolutional layer by performing nonlinear downsampling. This reduces the number of parameters that the network needs to learn, making the network more efficient.</p>
        <img style={{width: '100%'}} src={Img7} alt="image7" /> <br /> <br />
        <p><b>Flatten Layer:</b></p>
        <p>The flatten layer is a component of the convolutional neural networks (CNN's). A complete convolutional neural network can be broken down into two parts:</p> <br />
        <p><b>CNN:</b> The convolutional neural network that comprises the convolutional layers.</p> <br />
        <p><b>ANN:</b>The artificial neural network that comprises dense layers.</p> <br />
        <img style={{width: '100%'}} src={Img8} alt="image8" /> <br /> <br />
        <img style={{width: '100%'}} src={Img9} alt="image9" /> <br /> <br />

        <p><strong onClick={() => handleHeadingClick("step3")}>Step 3: USING A PRETRAINED MODEL</strong></p> <br />
        <p>This code uses a pre-trained DenseNet169 model, which means it’s a model that has already been trained on a large dataset (ImageNet) and knows how to recognize many common features in images. We exclude the final layers of DenseNet169 to add our own custom layers for our specific task. This allows us to take advantage of the pre-trained model's knowledge and adapt it to classify images into the three categories.</p> <br />
        <p>DenseNet (Densely Connected Convolutional Networks) connects each layer to every other layer in a feed-forward fashion. This creates a dense connection pattern where the input to a layer includes the feature maps of all preceding layers.DenseNet-169 specifically refers to a DenseNet model with 169 layers.</p> <br />
        <p>Using this pre-trained model helps our new model learn faster and perform better with less training data and time. This approach is known as transfer learning.</p> <br />
        <img style={{width: '100%'}} src={Img10} alt="image10" /> <br /> <br />

        <p><strong>Model Prediction:</strong></p> <br />
        <p><strong>Making Predictions:</strong> The predict_and_display_image function preprocesses the input image and feeds it to the model to obtain predictions. The model returns a set of probabilities for each class.</p> <br />
        <p><strong>Class Label Extraction:</strong> The predicted class is identified by finding the index with the highest probability using np.argmax(predictions). This index is then mapped to the corresponding class name using a dictionary that contains class indices as values and class names as keys.</p><br />

        <p><strong>Displaying Predictions:</strong></p> <br />
        <p><strong>Visualization:</strong> The original image is displayed using matplotlib.pyplot, and the predicted class is shown as the title of the image plot. This helps in visually verifying the model's prediction.</p><br />
      </div>

        <div className="box4">
          <div className="code-container">
            <pre className="code-snippet">
              <code className="python" style={{color:'#2f3130'}}>
              {highlightedCodeSnippet ? highlightedCodeSnippet.trim() : codeSnippet2.trim()}
              </code>
            </pre>
          </div>
        </div>
      </div>
      <div> 
          <button className="button">
          <a href="https://www.kaggle.com/code/pushkarns/lab4fhafkahahslfha" target="_blank"> View Runable code</a>
          </button>
        </div>
      </div>
  );
};
export default Lab2;
