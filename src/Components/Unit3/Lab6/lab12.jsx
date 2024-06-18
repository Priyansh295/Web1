import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import './lab12.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `
import pandas as pd
import numpy as np

# Parameters
n_samples = 1000

# Generate data
np.random.seed(42)
temperature = np.random.normal(loc=25, scale=5, size=n_samples)
pressure = np.random.normal(loc=1, scale=0.2, size=n_samples)
vibration = np.random.normal(loc=0.01, scale=0.005, size=n_samples)

# Classify data as healthy or unhealthy
def classify(temperature, pressure, vibration):
    if temperature > 30 or pressure > 1.2 or vibration > 0.02:
        return 'unhealthy'
    else:
        return 'healthy'

data = pd.DataFrame({
    'temperature': temperature,
    'pressure': pressure,
    'vibration': vibration
})
data['status'] = data.apply(lambda row: classify(row['temperature'], row['pressure'], row['vibration']), axis=1)

# Save training data
data.to_csv('training_data.csv', index=False)
print("Training data generated and saved to 'training_data.csv'")

data = pd.read_csv('training_data.csv')
data.head()

# Extract features and labels
X = data[['temperature', 'pressure', 'vibration']]
y = data['status']

# Convert labels to numerical values
y = y.map({'healthy': 0, 'unhealthy': 1})

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

import numpy as np

def real_time_data_generator():
    while True:
        temperature = np.random.normal(loc=25, scale=5)
        pressure = np.random.normal(loc=1, scale=0.2)
        vibration = np.random.normal(loc=0.01, scale=0.005)
        yield {'temperature': temperature, 'pressure': pressure, 'vibration': vibration}

import joblib

# Save the model to a file
joblib.dump(model, 'equipment_monitoring_model.pkl')
print("Model saved to 'equipment_monitoring_model.pkl'")

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from collections import deque
import pandas as pd
import numpy as np
import joblib
import time

# Load the trained model
model = joblib.load('equipment_monitoring_model.pkl')

# Initialize the app
app = dash.Dash(__name__)

# Create a deque to hold the real-time data
window_size = 30  # Define the size of the sliding window (30 seconds)
data_deque = deque(maxlen=window_size)
error_timestamps = deque(maxlen=1000)  # Keep track of error timestamps

# Function to update the deque with new data
def update_data():
    generator = real_time_data_generator()
    while True:
        new_data = next(generator)
        data_deque.append(new_data)
        yield

# Create a data generator
data_gen = update_data()

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Real-Time Industrial Equipment Monitoring Dashboard'),

    dcc.Graph(id='temperature-graph'),
    dcc.Graph(id='pressure-graph'),
    dcc.Graph(id='vibration-graph'),

    html.Div(id='status-container', style={'textAlign': 'center', 'marginTop': '20px', 'width': '100%', 'padding': '0'}),
    
    html.Div(id='additional-info', style={'display': 'flex', 'justifyContent': 'space-around', 'marginTop': '20px', 'width': '100%'}),
    
    dcc.Interval(
        id='interval-component',
        interval=1000,  # in milliseconds (1 second)
        n_intervals=0
    )
])

def get_status_style(status):
    if status == 0:
        return {
            'backgroundColor': '#98FB98',  # pastel green
            'color': '#006400',  # dark green text
            'padding': '20px',
            'borderRadius': '10px',
            'width': '96.3%',
            'textAlign': 'center',
            'margin': '0'
        }
    else:
        return {
            'backgroundColor': '#FFCCCB',  # pastel red
            'color': '#8B0000',  # dark red text
            'padding': '20px',
            'borderRadius': '10px',
            'width': '96.3%',
            'textAlign': 'center',
            'margin': '0'
        }

def get_status_color_and_text(status):
    if status == 0:
        return 'Healthy'
    else:
        return 'Unhealthy'

def get_time_since_last_error():
    if error_timestamps:
        return time.time() - error_timestamps[-1]
    return float('inf')

def get_error_count_last_10_seconds():
    current_time = time.time()
    count = sum(1 for timestamp in error_timestamps if current_time - timestamp <= 10)
    return count

# Update temperature graph
@app.callback(
    Output('temperature-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_temperature_graph(n):
    next(data_gen)  # Update the deque with new data
    df = pd.DataFrame(data_deque)
    if df.empty:
        return go.Figure()  # Return an empty figure if no data

    z = df['temperature']
    figure = {
        'data': [
            go.Scatter(
                x=list(range(n-len(df), n)),
                y=df['temperature'],
                mode='lines+markers',
                marker=dict(
                    size=10,
                    color=z,
                    colorscale='RdBu_r',  # Inverted RdBu colorscale
                    showscale=True,
                    colorbar=dict(title='Temperature')
                ),
                line=dict(
                    color='black',
                    width=2
                ),
                name='Temperature'
            )
        ],
        'layout': go.Layout(
            title='Temperature Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Temperature (°C)'),
            xaxis_range=[max(0, n-window_size), n]  # Sliding window
        )
    }
    return figure

# Update pressure graph
@app.callback(
    Output('pressure-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_pressure_graph(n):
    df = pd.DataFrame(data_deque)
    if df.empty:
        return go.Figure()  # Return an empty figure if no data

    z = df['pressure']
    figure = {
        'data': [
            go.Scatter(
                x=list(range(n-len(df), n)),
                y=df['pressure'],
                mode='lines+markers',
                marker=dict(
                    size=10,
                    color=z,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Pressure')
                ),
                line=dict(
                    color='black',
                    width=2
                ),
                name='Pressure'
            )
        ],
        'layout': go.Layout(
            title='Pressure Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Pressure (atm)'),
            xaxis_range=[max(0, n-window_size), n]  # Sliding window
        )
    }
    return figure

# Update vibration graph
@app.callback(
    Output('vibration-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_vibration_graph(n):
    df = pd.DataFrame(data_deque)
    if df.empty:
        return go.Figure()  # Return an empty figure if no data

    z = df['vibration']
    figure = {
        'data': [
            go.Scatter(
                x=list(range(n-len(df), n)),
                y=df['vibration'],
                mode='lines+markers',
                marker=dict(
                    size=10,
                    color=z,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Vibration')
                ),
                line=dict(
                    color='black',
                    width=2
                ),
                name='Vibration'
            )
        ],
        'layout': go.Layout(
            title='Vibration Over Time',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Vibration (m/s^2)'),
            xaxis_range=[max(0, n-window_size), n]  # Sliding window
        )
    }
    return figure

# Update status indicator
@app.callback(
    Output('status-container', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_status_indicator(n):
    df = pd.DataFrame(data_deque)
    if not df.empty:
        # Make sure to use a DataFrame with the same feature names as used during training
        latest_data = df[['temperature', 'pressure', 'vibration']].iloc[-1].to_frame().T
        current_status = model.predict(latest_data)[0]
        status_text = get_status_color_and_text(current_status)
        status_style = get_status_style(current_status)
        
        if current_status == 1:
            error_timestamps.append(time.time())
        
        return html.Div(f'Current Status: {status_text}', style=status_style)
    else:
        return html.Div('No Data', style={'backgroundColor': 'black', 'color': 'white', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'width': '100%', 'margin': '0'})

# Update additional info
@app.callback(
    Output('additional-info', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_additional_info(n):
    time_since_last_error = get_time_since_last_error()
    error_count_last_10_seconds = get_error_count_last_10_seconds()
    
    time_since_last_error_text = f'Time Since Last Error: {time_since_last_error:.2f} seconds'
    error_count_text = f'Number of Errors in Last 10 Seconds: {error_count_last_10_seconds}'
    
    return [
        html.Div(time_since_last_error_text, style={'backgroundColor': '#E0E0E0', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'width': '45%', 'color': 'black', 'margin': '0'}),
        html.Div(error_count_text, style={'backgroundColor': '#E0E0E0', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'width': '45%', 'color': 'black', 'margin': '0'})
    ]

# Real-time data generator function
def real_time_data_generator():
    while True:
        temperature = np.random.normal(loc=25, scale=5)
        pressure = np.random.normal(loc=1, scale=0.2)
        vibration = np.random.normal(loc=0.01, scale=0.005)
        yield {'temperature': temperature, 'pressure': pressure, 'vibration': vibration}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

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
        <h2>Industrial Equipment Monitoring System</h2> <br />

        <p><strong>Step 1: Generate Training Data</strong><br/>
        Synthetic sensor readings are generated to simulate temperature, pressure, and vibration from industrial equipment. Data points are classified as 'healthy' or 'unhealthy' based on defined conditions.</p> <br />

        <p><strong>Step 2: Save Training Data</strong><br/>
        Data is saved to a CSV file named 'training_data.csv'.</p> <br />

        <p><strong>Step 3: Load and Preprocess Data</strong><br/>
        The code loads the training data, converting the 'status' column into numerical values.</p> <br />

        <p><strong>Step 4: Split Data into Training and Testing Sets</strong><br/>
        The data is split using the train_test_split function, setting aside 20% for testing.</p> <br />

        <p><strong>Step 5: Train the Model</strong><br/>
        A Random Forest classifier is trained with parameters set for 100 trees and a random state of 42.</p> <br />

        <p><strong>Step 6: Evaluate the Model</strong><br/>
        Model performance is assessed on the testing data with accuracy metrics.</p> <br />

        <p><strong>Step 7: Save the Model</strong><br/>
        The trained model is saved to 'equipment_monitoring_model.pkl' using joblib.</p> <br />

        <p><strong>Step 8: Create the Dashboard</strong><br/>
        A real-time dashboard is set up for monitoring, featuring graphs and a status indicator that updates based on the model's predictions.</p> <br />

        <p><strong>Step 9: Update Data</strong><br/>
        Data is continuously updated in real-time simulating sensor readings.</p> <br />

        <p><strong>Step 10: Update Graphs</strong><br/>
        Graphs for temperature, pressure, and vibration are updated every second.</p> <br />

        <p><strong>Step 11: Update Status Indicator</strong><br/>
        A status indicator updates in real-time to reflect the equipment's condition as 'healthy' or 'unhealthy'.</p> <br />

        <p><strong>Step 12: Run the Dashboard</strong><br/>
        The dashboard runs, displaying real-time updates of the equipment's status.</p> <br />
        <p>Here are some Pictures of the graph when you run them </p> <br />
        <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />
        <img style={{width: '100%'}} src={Img2} alt="image2" /> <br /> <br />
        <img style={{width: '100%'}} src={Img3} alt="image3" /> <br /> <br />
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
          <a href="https://www.kaggle.com/code/priyansh2904/unit3lab6?scriptVersionId=184180277" target="_blank"> View Runable code</a>
          </button>
        </div>
      </div>
  );
};
export default Lab2;
