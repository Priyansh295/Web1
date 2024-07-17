import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.png';
// import Img2 from './imgs/image2.png';
// import Img3 from './imgs/image3.png';
// import Img4 from './imgs/image4.png';
// import Img5 from './imgs/image5.png';
import './lab15.css';

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
# Install necessary library
!pip install google-cloud-dialogflow

# Import libraries
from google.cloud import dialogflow_v2 as dialogflow
import os
import pandas as pd
import json

print("done")

`,
  Step2: `
json_key_path = '/kaggle/input/keysfile/peschatbot45-obnl-7909ed2abbff.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path

# Verify that the environment variable is set correctly
print(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
`,
  Step3: `
  client = dialogflow.IntentsClient()
  project_id = 'peschatbot45-obnl'
  parent = f"projects/{project_id}/agent"
  print(parent)
`,

  Step4: `
  def get_existing_intents():
    intents = client.list_intents(request={"parent": parent})
    return {intent.display_name: intent for intent in intents}

  # Get and print existing intents
  existing_intents = get_existing_intents()
  print("Existing Intents:", list(existing_intents.keys()))

  `,
  Step5: `
  # FAQs data
faqs = [
    {
        "question": "What are your opening hours?",
        "answers": [
            "We are open from 9 AM to 5 PM, Monday to Friday.",
            "Our business hours are from 9 AM to 5 PM, Monday to Friday."
        ]
    },
    {
        "question": "Where are you located?",
        "answers": [
            "We are located at 20 Ingram Street, Gotham opposite the Daily Planet.",
            "You will find us opposite the Daily Planet at 120 Ingram Street, Gotham."
        ]
    },
    {
        "question": "How can I contact customer service?",
        "answers": [
            "You can contact customer service at (123) 456-7890 or email us at support@example.com.",
            "For customer service call (123) 456-7890",
            "Please email us at support@example.com."
        ]
    },
    {
        "question": "What is your return policy?",
        "answers": [
            "Our return policy allows returns within 30 days of purchase with a receipt.",
            "If you have a receipt, you can return the items within 30 days of purchase as long they have not been used or damaged."
        ]
    }
]

# Get existing intents
existing_intents = get_existing_intents()

# Create or update intents
for faq in faqs:
    create_or_update_intent(faq["question"], [faq["question"]], faq["answers"], existing_intents)

  `,
  Step6: `
    def detect_intent_texts(project_id, session_id, texts, language_code):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)
    
    for text in texts:
        text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.types.QueryInput(text=text_input)
        
        response = session_client.detect_intent(session=session, query_input=query_input)
        print(f"\nQuery text: {response.query_result.query_text}")
        print(f"\nDetected intent: {response.query_result.intent.display_name}")
        print(f"\nResponse text: {response.query_result.fulfillment_text}\n")
        print(f"----------------------------------------------------------------")
  `,
  Step7: `
  # Test queries
test_queries = [
    "Hi", # This should trigger Welcome intent
    "What are your opening hours?",
    "Where are you located?",
    "How can I contact customer service?",
    "What is your return policy?",
    "What is your email address?",  # This should trigger the fallback intent
    "Do you offer discounts?"       # This should also trigger the fallback intent
]

detect_intent_texts(project_id, "unique_session_id", test_queries, "en")
  `,
  Step8: `
  # Example: Add training phrases to "What are your opening hours?" intent

intent_name = "What are your opening hours?"
additional_training_phrases = [
    "When do you open?",
    "What time do you start business?",
    "Tell me your business hours.",
]

# Get existing intents
existing_intents = get_existing_intents()

# Add new training phrases
create_or_update_intent(intent_name, additional_training_phrases, [], existing_intents)

# Test the updated intent with a query
test_query = [
    "What are your opening hours?",
    "When do you open?",
    "Tell me your business hours."
]

detect_intent_texts(project_id, "unique_session_id", test_query, "en")
  `,
  Step9:`
  def delete_all_intents():
    intents = client.list_intents(request={"parent": parent})
    for intent in intents:
        client.delete_intent(request={"name": intent.name})
    print("Deleted all intents.")

# Delete all intents
delete_all_intents()

# Verify deletion
existing_intents = get_existing_intents()
print("Existing Intents After Deletion:", existing_intents)
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
    <h2>Lab Experiment 3: Developing a Chatbot for Website FAQ</h2> <br />

    <p><b>Objective:</b> To create a chatbot that automatically answers frequently asked questions on a website.</p> <br />

    <p><b>Step 1: Collect FAQs</b></p>
    <p>Collect a list of common FAQs and their answers for the website. These will form the basis of the chatbot's knowledge base.</p> <br />

    <p><b>Step 2: Use a Chatbot Development Platform</b></p>
    <p>Use a platform like Dialogflow or Microsoft Bot Framework to create the chatbot. These platforms provide tools for building, training, and deploying chatbots.</p> <br />

    <p><b>Step 3: Train the Chatbot</b></p>
    <p>Train the chatbot on the collected FAQs to recognize and respond to user queries accurately. This involves setting up intents, entities, and responses in the chatbot platform.</p> <br />

    <p><b>Step 4: Integrate the Chatbot</b></p>
    <p>Integrate the chatbot into the website using the platform's web integration tools. This typically involves adding a script to the website's HTML code.</p> <br />

    <p><b>Step 5: Test the Chatbot</b></p>
    <p>Test the chatbot with various queries to ensure it responds accurately and helpfully. This is important for verifying the chatbot's performance and making necessary adjustments.</p> <br />

    <p><b onClick={() => handleHeadingClick("WhatIsChatbot")}>What is a Chatbot?</b></p>
    <p>A chatbot is a software application designed to simulate human conversation. It uses natural language processing (NLP) to understand and respond to user inputs. Chatbots can be rule-based, following predefined paths, or AI-driven, using machine learning to improve over time.</p> <br />

    <p><b onClick={() => handleHeadingClick("TypesOfChatbots")}>Types of Chatbots:</b></p>
    <ul>
        <li><b>Rule-Based Chatbots:</b>
            <ul>
                <li>Follow predefined rules and scripts.</li>
                <li>Good for simple, linear tasks.</li>
                <li>Limited flexibility.</li>
            </ul>
        </li>
        <li><b>AI-Driven Chatbots:</b>
            <ul>
                <li>Use machine learning and NLP.</li>
                <li>Handle more complex interactions.</li>
                <li>Improve over time with more data.</li>
            </ul>
        </li>
        <li><b>Hybrid Chatbots:</b>
            <ul>
                <li>Mix of Rules and AI: Uses simple rules and smart AI to answer questions.</li>
                <li>Personal Replies: Gives personalized responses to users.</li>
                <li>Switch to Human: Passes the conversation to a human if things get too complicated.</li>
            </ul>
        </li>
    </ul> <br />

    <p><b onClick={() => handleHeadingClick("OurChatbot")}>What will our chatbot be?</b></p>
    <p>Our chatbot will be a rule-based system designed to answer common questions by recognizing user queries and providing predefined responses. It will use natural language understanding (NLU) to determine the intent behind a question and then deliver one of several preset answers to create a more dynamic conversation. This will make it useful for handling frequently asked questions, such as business hours or contact information, in an efficient and user-friendly manner.</p> <br />

    <p><strong onClick={() => handleHeadingClick("Step1")}>Step 1: Install the Necessary Library</strong></p>
    <p>We need to install the google-cloud-dialogflow library to interact with Dialogflow.
    Next, we need to import the required libraries and set up authentication using our JSON key file.
    </p> <br />

    <p><strong onClick={() => handleHeadingClick("Step2")}>Step 2: Set Up Authentication</strong></p>
    <p>We need to set an environment variable to point to our JSON key file. This file contains credentials for our Google Cloud project. A google account has already been created and its json key has been placed in the keyfile.</p> <br />

    <p><strong onClick={() => handleHeadingClick("Step3")}>Step 3: Initialize Dialogflow Client</strong></p>
    <p>We need to initialize the Dialogflow client to interact with Dialogflow and set our Project ID and Parent Path. Both have already been done in the code above.</p> <br />

    <p><strong onClick={() => handleHeadingClick("ChatbotComponents")}>Components of our Chatbot:</strong></p>
    <ul>
        <li><strong>Natural Language Understanding (NLU):</strong> Helps the chatbot understand what the user is saying.
            <ul>
                <li>Intent Recognition: Figures out what the user wants (e.g., "What are your opening hours?").</li>
                <li>Entity Extraction: Picks out specific pieces of information (e.g., "Monday").</li>
            </ul>
        </li>
        <li><strong>Dialog Management:</strong> Manages the flow of the conversation.
            <ul>
                <li>Intent Matching: Matches user input to a predefined response.</li>
                <li>Response Selection: Chooses the appropriate response.</li>
            </ul>
        </li>
        <li><strong>Natural Language Generation (NLG):</strong> Turns the chatbot's decision into a readable message.
            <ul>
                <li>Response Templates: Predefined messages for common questions.</li>
            </ul>
        </li>
    </ul> <br />
    <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />

    <p><strong onClick={() => handleHeadingClick("Step4")}>Step 4: Define Helper Functions</strong></p>
    <p>We need two helper functions: one to get existing intents and another to create or update intents.</p>
    <ul>
        <li><strong>Get Existing Intents:</strong> This function fetches all existing intents in the Dialogflow agent and returns them as a dictionary for quick lookup.
        </li>
        <li><strong>Create or Update an Intent:</strong> This function creates a new intent or updates an existing one with training phrases and response messages.
        </li>
    </ul> <br />

    <p><strong onClick={() => handleHeadingClick("Step5")}>Step 5: Prepare FAQs Data</strong></p>
    <p>We need to prepare the data for our FAQs. This data will be used to create or update intents in Dialogflow when we call our helper functions. Use <code>get_existing_intents()</code> to fetch already existing intents and <code>create_or_update_intent()</code> to either create or update the intent.</p>
    
    <p><strong onClick={() => handleHeadingClick("Step6")}>Step 6: Define Function to Detect Intents</strong></p>
    <p>We need a function to detect intents based on user queries. This function sends queries to Dialogflow and prints the responses.</p>
    <p><strong onClick={() => handleHeadingClick("Step7")}>Step 7: Test the Chatbot</strong></p>
    <p>We define a list of test queries that correspond to the FAQs and call <code>detect_intent_texts()</code> to verify that the chatbot responds correctly. If you re-run this block you will get different versions of responses, which helps in improving user experience. Note--Rerun this block if detected intent is Default Fallback Intents.</p>
    <p><strong onClick={() => handleHeadingClick("Step8")}>Step 8: Training Phrases</strong></p>
    <p>Training phrases are examples of how people ask questions to a chatbot. They teach the chatbot to understand and answer questions correctly. For example, if someone asks "When do you open?" or "What time do you start?", these examples help the chatbot learn what to say. Having lots of examples helps the chatbot get better at talking with people and giving the right answers, improving how well it can help users and handle different conversations.</p>
    
    <p>This code snippet will update the "What are your opening hours?" intent with additional training phrases and then test it with sample queries. Note--Rerun this block if detected intent is Default Fallback Intents.</p> <br />

    <p><strong onClick={() => handleHeadingClick("Step9")}>Step 9: Deletion</strong></p>
    <p>We can run this block if we intend to make new intents and remove the previous ones. In addition, the <code>delete_all_intents()</code> function ensures that you can start from a clean slate by deleting all existing intents in the Dialogflow agent. This is useful for resetting the agent during development and testing.</p>
    
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
            <a href="https://www.kaggle.com/code/percival224/unit-4-lab-3" target="_blank"> View Runable code</a>
            </button>
    </div>
    </div>
    );
};
export default Lab2;
