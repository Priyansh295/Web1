import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.png';
import './lab16.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `

!pip install spacy
!python -m spacy download en_core_web_sm
!pip install Flask

import spacy
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Define the set of training data
queries = [
    "What are your business hours?",
    "How do I reset my password?",
    "Where is my order?",
    "Can I return an item?",
    "What is the status of my ticket?",
    "How do I contact support?"
]

responses = [
    "Our business hours are 9 AM to 5 PM, Monday to Friday.",
    "To reset your password, click on 'Forgot Password' on the login page.",
    "You can check your order status by logging into your account and visiting the 'Orders' section.",
    "To return an item, please visit our returns page and follow the instructions.",
    "Your ticket status can be checked in the 'Support' section after logging in.",
    "You can contact support by emailing support@example.com or calling 123-456-7890."
]

# Create and train a simple model using scikit-learn
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(queries, responses)

# Initialize Flask app
app = Flask(__name__)

# Define the route for the chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get the user's query from the request
    query = request.json.get('query')

    # Predict the response using the trained model
    response = model.predict([query])[0]

    # Return the response as JSON
    return jsonify({"response": response})

# Function to run the app in a separate thread
def run_app():
    try:
        app.run(debug=True, port=5002, use_reloader=False)
    except Exception as e:
        print(f"Error: {e}")

# Run the Flask app in a separate thread
from threading import Thread
flask_thread = Thread(target=run_app)
flask_thread.start()

from threading import Thread

def run_app():
    app.run(debug=True, port=5002, use_reloader=False)

flask_thread = Thread(target=run_app)
flask_thread.start()

import requests

def get_chatbot_response(query):
    # Define the URL for the chatbot
    url = "http://127.0.0.1:5002/chatbot"

    # Send the POST request with the user query
    response = requests.post(url, json={"query": query})

    # Get the JSON response and extract the message
    return response.json().get("response")

# Function to test chatbot with a given query
def test_chatbot(query):
    # Get the chatbot response
    response = get_chatbot_response(query)

    # Print the response
    print("User query:", query)
    print("Chatbot response:", response)

# Example query for testing
test_query = "What are your business hours?"
test_chatbot(test_query)

test_chatbot("How do I reset my password?")
test_chatbot("Where is my order?")
`;

const codeSections = {
  SpaCy: `
import spacy
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Define the set of training data
queries = [
    "What are your business hours?",
    "How do I reset my password?",
    "Where is my order?",
    "Can I return an item?",
    "What is the status of my ticket?",
    "How do I contact support?"
]

responses = [
    "Our business hours are 9 AM to 5 PM, Monday to Friday.",
    "To reset your password, click on 'Forgot Password' on the login page.",
    "You can check your order status by logging into your account and visiting the 'Orders' section.",
    "To return an item, please visit our returns page and follow the instructions.",
    "Your ticket status can be checked in the 'Support' section after logging in.",
    "You can contact support by emailing support@example.com or calling 123-456-7890."
]

# Create and train a simple model using scikit-learn
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(queries, responses)
`,
  Flask: `
app = Flask(__name__)

# Define the route for the chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get the user's query from the request
    query = request.json.get('query')
    
    # Predict the response using the trained model
    response = model.predict([query])[0]

    # Return the response as JSON
    return jsonify({"response": response})
# Function to run the app in a separate thread
def run_app():
    try:
        app.run(debug=True, port=5002, use_reloader=False)
    except Exception as e:
        print(f"Error: {e}")

# Run the Flask app in a separate thread
from threading import Thread
flask_thread = Thread(target=run_app)
flask_thread.start()

`,
  RESTful: `
  def get_chatbot_response(query):
    # Define the URL for the chatbot
    url = "http://127.0.0.1:5002/chatbot"
    
    # Send the POST request with the user query
    response = requests.post(url, json={"query": query})
    
    # Get the JSON response and extract the message
    return response.json().get("response")
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
    <h2>Chatbot Using NLP Techniques for Customer Service or Operational Queries</h2> <br />

    <p><b>Overview of spaCy</b></p>
    <p>spaCy is a robust library for Natural Language Processing (NLP) in Python. It is designed for high performance and large-scale text processing. The library is well-suited for real-world applications, providing tools for diverse NLP tasks.</p> <br />

    <p><b>Overview of Flask</b></p>
    <p>Flask is a minimalistic and flexible web framework for Python developers. It's easy to start with Flask for small projects and can also be expanded with various extensions for building complex applications.</p> <br />

    <p><b>Overview of RESTful APIs</b></p>
    <p>RESTful API is a design pattern for APIs. It stands for Representational State Transfer and uses HTTP methods for web services. It's widely adopted due to its simplicity and effectiveness in allowing various applications to communicate over the internet.</p> <br />

    <p><h3><strong  onClick={() => handleHeadingClick("SpaCy")}>Spacy</strong></h3></p>
    <p><b>Key Features of spaCy</b></p>
    <ul>
        <li>Tokenization: Splits text into individual words and punctuation.</li>
        <li>Part-of-Speech Tagging: Identifies parts of speech for each word, like nouns and verbs.</li>
        <li>Named Entity Recognition (NER): Recognizes and classifies names, dates, and companies within text.</li>
        <li>Dependency Parsing: Establishes relationships between words, showing how sentences are structured.</li>
        <li>Text Classification: Trains models to categorize text into predefined labels.</li>
        <li>Pre-trained Models: Offers models trained on large datasets, ready to use for various languages.</li>
        <li>Customization: Allows customization of models for specific tasks and datasets.</li>
    </ul> <br />
    <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />

    <p><h3><strong onClick={() => handleHeadingClick("Flask")}>Flask</strong></h3></p>
    <p><b>Key Features of Flask</b></p>
    <ul>
        <li>Micro-framework: Provides core functionality with options to include additional features as needed.</li>
        <li>Routing: Maps URLs to Python function handlers, making URL management easy.</li>
        <li>Templates: Integrates with Jinja2 for dynamic HTML rendering.</li>
        <li>Request Handling: Simplifies the management of incoming data and responses.</li>
        <li>Development Server: Includes a server for local testing and development.</li>
        <li>Extensions: Supports a wide range of plugins for added functionalities like ORM, form validation, and more.</li>
        <li>RESTful Support: Well-suited for creating APIs that can handle RESTful requests.</li>
    </ul> <br />
    <img style={{width: '100%'}} src={Img2} alt="image2" /> <br /> <br />

    <p><h3><strong onClick={() => handleHeadingClick("RESTful")}>RESTful API</strong></h3></p>
    <p><b>Key Characteristics of RESTful APIs</b></p>
    <ul>
        <li>Stateless: Each request must have all necessary information; the server does not remember past requests.</li>
        <li>Client-Server Structure: Allows clients and servers to evolve separately without depending on each other.</li>
        <li>Cacheable: Clients can cache responses to improve performance and reduce server load.</li>
        <li>Uniform Interface: Makes the system simpler and more modular, allowing separate components to evolve.</li>
        <li>Hypermedia Driven: Clients interact with the server via hyperlinks provided dynamically by server responses.</li>
    </ul> <br />
    <img style={{width: '100%'}} src={Img3} alt="image3" /> <br /> <br />

    <p><b>HTTP Methods in RESTful APIs</b></p>
    <ul>
        <li>GET: Retrieves information from the server.</li>
        <li>POST: Sends new information to the server.</li>
        <li>PUT: Updates existing information on the server.</li>
        <li>DELETE: Removes existing information from the server.</li>
        <li>PATCH: Makes partial updates to existing information on the server.</li>
    </ul> <br />
    <img style={{width: '100%'}} src={Img4} alt="image4" /> <br /> <br />

    <p><b>Example Use Case of RESTful APIs</b></p>
    <p>A simple API for a book collection might include actions like retrieving all books, getting details of a specific book, adding a new book, updating an existing book, and deleting a book from the collection.</p> <br />

    <p>The Key aspects of integrating these technologies are:-</p>
    <ul>
        <li><b>NLP with spaCy:</b> Utilizes spaCy for efficient text analysis and processing in web applications.</li>
        <li><b>Web Development with Flask:</b> Employs Flask's features to build user interfaces and manage web requests, facilitating interaction with NLP applications.</li>
        <li><b>RESTful API Design:</b> Develops APIs that are easy to use, maintain, and scale, enhancing communication between different software components.</li>
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
            <a href="https://www.kaggle.com/code/priyanshsurana/chatbot/notebook  " target="_blank"> View Runable code</a>
            </button>
    </div>
    </div>
    );
};
export default Lab2;
