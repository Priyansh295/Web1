import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.png';
import Img6 from './imgs/image6.png';
import Img8 from './imgs/image8.gif';
import Img9 from './imgs/image9.png';
import Img10 from './imgs/image10.png';
import Img11 from './imgs/image11.png';
import Img12 from './imgs/image12.gif';
import Img13 from './imgs/image13.png';
import Img14 from './imgs/image14.gif';
import Img15 from './imgs/image15.jpg';
import './lab5.css';

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
  importLibraries: `
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.metrics import classification_report, accuracy_score
`,
  LoadData: `
ttrain = pd.read_csv("./archive (3)/fashion-mnist_train.csv")
test = pd.read_csv("./archive (3)/fashion-mnist_test.csv")
`,
explore: `
train.head()

test.head()
`,

labels: `
# The catagories of clothing were in dataset
class_labels= ["T-shirt/top","Trouser","Pullover" ,"Dress","Coat" ,"Sandal" ,"Shirt" ,"Sneaker" ,"Bag" ,"Ankle boot"]
`,

preprocess: `
x_train = train_data[:, 1:]
y_train = train_data[:, 0]

x_test = test_data[:, 1:]
y_test = test_data[:, 0]

# arrange pixel values between 0 to 1 
x_train = x_train/255
x_test = x_test/255
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

  `,

  TestModel: `
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

  `,
  
  Evaluate: `
from sklearn.metrics import classification_report, accuracy_score

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
`,

  SaveModel: `
  # Save Model
model.save('fashion_mnist_cnn_model.h5')

#Load Model

fashion_model = tf.keras.models.load_model('fashion_mnist_cnn_model.h5')
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
      <h2>CNN Model for Image Classification Explained</h2>

      <p><strong onClick={() => handleHeadingClick("importLibraries")}>1. Import Libraries:</strong></p>
      <p>To start, we need to import several essential libraries:</p> <br />
      <ul>
        <li><b>NumPy:</b> Used for numerical computations.</li>
        <li><b>Pandas:</b> Used for data manipulation and analysis</li>
        <li><b>TensorFlow and Keras:</b> Used for building and training neural networks.</li>
        <li><b>Scikit-learn (sklearn):</b> It provides many algorithms for tasks like classification, regression, and clustering.</li>
        <li><b>Matplotlib:</b> Matplotlib is a library in Python that helps create visualizations like graphs and charts.</li>
      </ul> <br />

      <p><strong onClick={() => handleHeadingClick("LoadData")}>2. Load Data:</strong></p>
      <p>We load our training and test datasets from CSV files using Pandas. The training data is 
      stored in a variable called <b><i>train</i></b>, and the test data is stored in <b><i>test</i></b>.</p> <br />

      <p><strong onClick={() => handleHeadingClick("explore")}>3. Explore Data</strong></p>
      <p>To understand the structure and content of our data, we display the first 5 rows of the training 
      dataset.</p> <br />
      <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />
      <img style={{width: '100%'}} src={Img2} alt="image2" /> <br /> <br />

      <p><strong onClick={() => handleHeadingClick("labels")}>4. Define Class Labels</strong></p>
      <p>We define the class labels for the clothing items in our dataset. These labels represent the 
      categories we aim to classify.</p> <br />

      <p><strong onClick={() => handleHeadingClick("preprocess")}>5. Preprocess Data</strong></p>
      <p>The data needs to be reshaped and normalized to be suitable for training our CNN model. 
      Normalization ensures that all pixel values are between 0 and 1.</p> <br />
      <img style={{width: '100%'}} src={Img9} alt="image9" /> <br /> <br />

      <p><strong onClick={() => handleHeadingClick("SplitData")}>6. : Split Data into Training and Validation Sets</strong></p>
      <p>We split the training data into training and validation sets to evaluate the model's 
      performance during training.</p> < br/>

      <p><strong onClick={() => handleHeadingClick("DeepLearningModel")}>7. Build CNN Model:</strong></p>
      <p>We construct a Convolutional Neural Network (CNN) using Keras. The model consists of 
      several layers, including convolutional layers, pooling layers, and dense layers.</p> < br/>
      <p> <b>Convolutional Neural Networks (CNNs):</b> CNNs are specialized neural networks 
      for processing data with a grid-like topology, such as images. They automatically 
      and adaptively learn spatial hierarchies of features through backpropagation</p> <br />
      <img style={{width: '100%'}} src={Img10} alt="image10" /> <br /> <br />

      <p style={{marginLeft: '25px'}}><b>Convolutional layers:</b></p>
      <p style={{marginLeft: '25px'}}> The core building block of a CNN is the convolutional 
      layer. It applies filters to small regions of the input data, known as receptive fields. 
      Each filter is a small matrix of weights that slides across the input data, performing a 
      dot product with the input pixels. This process is known as convolution.</p> <br />
      <img style={{width: '100%'}} src={Img11} alt="image11" /> <br /> <br />
      <p>The output of the convolutional layer is a feature map, which is a two-dimensional 
        representation of the input data. This feature map captures the presence of specific 
        features in the input data, such as edges or lines</p> <br />
        <img style={{width: '100%'}} src={Img12} alt="image12" /> <br /> <br />

      <p style={{marginLeft: '25px'}}><b>Pooling layers:</b></p>
      <p style={{marginLeft: '25px'}}> A pooling layer in a neural network helps simplify 
        and reduce the size of the data from the convolutional layer. By doing this, 
        it decreases the number of details the network needs to handle, which makes the 
        network faster and more efficient.</p> <br />
      <img style={{width: '100%'}} src={Img6} alt="image6" /> <br /> <br />

      <p style={{marginLeft: '25px'}}><b>Activation and Classification Layers:</b></p>
      <p style={{marginLeft: '25px'}}> The output of the pooling layer is fed into an activation function, such as the Rectified Linear Unit (ReLU), which helps the network learn more complex features.</p> <br />
      <p style={{marginLeft: '25px'}}>The final layer of the CNN is typically a fully connected layer that outputs a probability distribution over all classes, allowing the network to classify the input data.</p> <br />
      <img style={{width: '100%'}} src={Img8} alt="image8" /> <br /> <br />
      <img style={{width: '100%'}} src={Img13} alt="image13" /> <br /> <br />

      <p><strong onClick={() => handleHeadingClick("TrainModel")}>8. Train Model:</strong></p> <br />
      <ul>
        <li> Feed the training images and their labels into the CNN.</li>
        <li> The CNN learns to recognize patterns and features in the images through a process called backpropagation.</li>
        <li> During training, the model's weights are adjusted to minimize the difference between its predictions and the true labels.</li>
      </ul> <br />
      <img style={{width: '100%'}} src={Img14} alt="image14" /> <br /> <br />

      <p><strong onClick={() => handleHeadingClick("TestModel")}>9. Test Model</strong></p> <br />
      <ul>
        <li> Feed the test images into the trained CNN model. </li>
        <li> Compare the model's predictions to the true labels to calculate metrics like accuracy, precision, and recall</li>
      </ul> <br />
      <img style={{width: '100%'}} src={Img3} alt="image3" /> <br /> <br />

      <p><strong onClick={() => handleHeadingClick("Evaluate")}>10. Evaluate the Model:</strong></p>
      <p>After training, we evaluate the model on the validation set to see how well it performs.</p> < br/>
      <img style={{width: '100%'}} src={Img15} alt="image15" /> <br /> <br />
      <img style={{width: '100%'}} src={Img4} alt="image4" /> <br /> <br />

      <p><strong onClick={() => handleHeadingClick("SaveModel")}>7. Save and Load the Model:</strong></p>
      <p>The code saves the trained model to a file for future use. It can be loaded back into memory later to make predictions on new data.</p> <br />
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
