import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.png';
import './lab13.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `
from flask import Flask, request, render_template_string

# Initialize the Flask application
app = Flask(__name__)

# Define the expert system class
class BajajExpertSystem:
    def __init__(self):
        # Initialize rules with conditions and conclusions
        self.rules = [
            {"conditions": ["not_starting", "battery_low"], "conclusion": "Charge the battery."},
            {"conditions": ["not_starting", "battery_ok"], "conclusion": "Check the starter motor."},
            {"conditions": ["starting", "stalls_frequently"], "conclusion": "Check the fuel supply."},
            {"conditions": ["starting", "poor_acceleration"], "conclusion": "Check the air filter."},
            {"conditions": ["starting", "unusual_noises"], "conclusion": "Check the engine."},
        ]

    # Method to diagnose based on symptoms
    def diagnose(self, symptoms):
        # Loop through each rule to find a match with the provided symptoms
        for rule in self.rules:
            if all(condition in symptoms for condition in rule["conditions"]):
                return rule["conclusion"]
        return "No diagnosis found. Please consult a professional mechanic."

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    # Create an instance of the expert system
    system = BajajExpertSystem()
    if request.method == 'POST':
        # Get the list of symptoms from the form submission
        symptoms = request.form.getlist('symptoms')
        # Use the expert system to diagnose based on the symptoms
        diagnosis = system.diagnose(symptoms)
        # Render the diagnosis result
        return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f4f4f4;
                        text-align: center;
                        padding: 50px;
                    }
                    h1 {
                        color: #333;
                    }
                    p {
                        color: #666;
                    }
                    .button {
                        background-color: #4CAF50;
                        color: white;
                        padding: 14px 20px;
                        margin: 8px 0;
                        border: none;
                        cursor: pointer;
                        transition: all 0.3s ease 0s;
                    }
                    .button:hover {
                        background-color: #45a049;
                    }
                    form {
                        background-color: #ffffff;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 0 10px 0 rgba(0,0,0,0.1);
                    }
                    label {
                        margin-right: 10px;
                    }
                </style>
            </head>
            <body>
                <h1>Diagnosis Result</h1>
                <p>{{ diagnosis }}</p>
                <a href="/" class="button">Back</a>
            </body>
            </html>
        """, diagnosis=diagnosis)
    # Render the initial form for users to input symptoms
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background-color: #f4f4f4;
                    text-align: center;
                    padding: 50px;
                }
                h1 {
                    color: #333;
                }
                form {
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px 0 rgba(0,0,0,0.1);
                }
                .button {
                    background-color: #008CBA;
                    color: white;
                    padding: 15px 20px;
                    margin: 10px 0;
                    border: none;
                    cursor: pointer;
                    transition: opacity 0.3s ease-in-out;
                }
                .button:hover {
                    opacity: 0.7;
                }
                label {
                    margin-right: 10px;
                }
            </style>
        </head>
        <body>
            <h1>Welcome to the Bajaj Expert System</h1>
            <form method="post">
                <label><input type="checkbox" name="symptoms" value="not_starting"> Not Starting</label><br>
                <label><input type="checkbox" name="symptoms" value="battery_low"> Battery Low</label><br>
                <label><input type="checkbox" name="symptoms" value="battery_ok"> Battery OK</label><br>
                <label><input type="checkbox" name="symptoms" value="starting"> Starting</label><br>
                <label><input type="checkbox" name="symptoms" value="stalls_frequently"> Stalls Frequently</label><br>
                <label><input type="checkbox" name="symptoms" value="poor_acceleration"> Poor Acceleration</label><br>
                <label><input type="checkbox" name="symptoms" value="unusual_noises"> Unusual Noises</label><br>
                <input type="submit" value="Diagnose" class="button">
            </form>
        </body>
        </html>
    """)

# Run the Flask application in debug mode
if __name__ == "__main__":
    app.run(debug=True)

`;

const codeSections = {
  Knowledge: `
class BajajExpertSystem:
    def __init__(self):
        # Initialize rules with conditions and conclusions
        self.rules = [
            {"conditions": ["not_starting", "battery_low"], "conclusion": "Charge the battery."},
            {"conditions": ["not_starting", "battery_ok"], "conclusion": "Check the starter motor."},
            {"conditions": ["starting", "stalls_frequently"], "conclusion": "Check the fuel supply."},
            {"conditions": ["starting", "poor_acceleration"], "conclusion": "Check the air filter."},
            {"conditions": ["starting", "unusual_noises"], "conclusion": "Check the engine."},
        ]
`,
  infrence: `
      def diagnose(self, symptoms):
        # Loop through each rule to find a match with the provided symptoms
        for rule in self.rules:
            if all(condition in symptoms for condition in rule["conditions"]):
                return rule["conclusion"]
        return "No diagnosis found. Please consult a professional mechanic."
`,
  flask: `
  from flask import Flask, request, render_template_string

  # Initialize the Flask application
  app = Flask(__name__)
  `,
  routes: `
    @app.route('/', methods=['GET', 'POST'])
def home():
    # Create an instance of the expert system
    system = BajajExpertSystem()
    if request.method == 'POST':
        # Get the list of symptoms from the form submission
        symptoms = request.form.getlist('symptoms')
        # Use the expert system to diagnose based on the symptoms
        diagnosis = system.diagnose(symptoms)
        # Render the diagnosis result
        return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f4f4f4;
                        text-align: center;
                        padding: 50px;
                    }
                    h1 {
                        color: #333;
                    }
                    p {
                        color: #666;
                    }
                    .button {
                        background-color: #4CAF50;
                        color: white;
                        padding: 14px 20px;
                        margin: 8px 0;
                        border: none;
                        cursor: pointer;
                        transition: all 0.3s ease 0s;
                    }
                    .button:hover {
                        background-color: #45a049;
                    }
                    form {
                        background-color: #ffffff;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 0 10px 0 rgba(0,0,0,0.1);
                    }
                    label {
                        margin-right: 10px;
                    }
                </style>
            </head>
            <body>
                <h1>Diagnosis Result</h1>
                <p>{{ diagnosis }}</p>
                <a href="/" class="button">Back</a>
            </body>
            </html>
        """, diagnosis=diagnosis)
    # Render the initial form for users to input symptoms
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background-color: #f4f4f4;
                    text-align: center;
                    padding: 50px;
                }
                h1 {
                    color: #333;
                }
                form {
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px 0 rgba(0,0,0,0.1);
                }
                .button {
                    background-color: #008CBA;
                    color: white;
                    padding: 15px 20px;
                    margin: 10px 0;
                    border: none;
                    cursor: pointer;
                    transition: opacity 0.3s ease-in-out;
                }
                .button:hover {
                    opacity: 0.7;
                }
                label {
                    margin-right: 10px;
                }
            </style>
        </head>
        <body>
            <h1>Welcome to the Bajaj Expert System</h1>
            <form method="post">
                <label><input type="checkbox" name="symptoms" value="not_starting"> Not Starting</label><br>
                <label><input type="checkbox" name="symptoms" value="battery_low"> Battery Low</label><br>
                <label><input type="checkbox" name="symptoms" value="battery_ok"> Battery OK</label><br>
                <label><input type="checkbox" name="symptoms" value="starting"> Starting</label><br>
                <label><input type="checkbox" name="symptoms" value="stalls_frequently"> Stalls Frequently</label><br>
                <label><input type="checkbox" name="symptoms" value="poor_acceleration"> Poor Acceleration</label><br>
                <label><input type="checkbox" name="symptoms" value="unusual_noises"> Unusual Noises</label><br>
                <input type="submit" value="Diagnose" class="button">
            </form>
        </body>
        </html>
    """)
  `,
  html: `
  return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f4f4f4;
                text-align: center;
                padding: 50px;
            }
            h1 {
                color: #333;
            }
            form {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px 0 rgba(0,0,0,0.1);
            }
            .button {
                background-color: #008CBA;
                color: white;
                padding: 15px 20px;
                margin: 10px 0;
                border: none;
                cursor: pointer;
                transition: opacity 0.3s ease-in-out;
            }
            .button:hover {
                opacity: 0.7;
            }
            label {
                margin-right: 10px;
            }
        </style>
    </head>
    <body>
        <h1>Welcome to the Bajaj Expert System</h1>
        <form method="post">
            <label><input type="checkbox" name="symptoms" value="not_starting"> Not Starting</label><br>
            <label><input type="checkbox" name="symptoms" value="battery_low"> Battery Low</label><br>
            <label><input type="checkbox" name="symptoms" value="battery_ok"> Battery OK</label><br>
            <label><input type="checkbox" name="symptoms" value="starting"> Starting</label><br>
            <label><input type="checkbox" name="symptoms" value="stalls_frequently"> Stalls Frequently</label><br>
            <label><input type="checkbox" name="symptoms" value="poor_acceleration"> Poor Acceleration</label><br>
            <label><input type="checkbox" name="symptoms" value="unusual_noises"> Unusual Noises</label><br>
            <input type="submit" value="Diagnose" class="button">
        </form>
    </body>
    </html>
""")

  `,
  run: `
  if __name__ == "__main__":
    app.run(debug=True)
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
        <h2>Simple Expert System for Decision Support</h2> <br />

        <p><b>Overview</b></p>
        <p>This project is a web-based application for diagnosing Bajaj vehicle issues using an expert 
        system and Flask. The application allows users to input symptoms their vehicle is experiencing, 
        and based on predefined rules, it provides a diagnosis.</p> <br />

        <p><b>Expert System</b></p> <br />
        <p><b>Defination:</b> : An expert system is a computer program that mimics the decision-making abilities of a human expert. It uses predefined rules and a knowledge base to diagnose problems or provide solutions.</p> <br />
        <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />

        <p><b>Components of Expert System:</b></p> <br />
        <ol>
          <li>
          <p><strong onClick={() => handleHeadingClick("Knowledge")}>Knowledge Base</strong></p>
          <p>Contains domain-specific knowledge, in this case, rules for diagnosing issues with Bajaj vehicles.</p>
          </li> 
          <br />
          <li>
            <p><strong onClick={() => handleHeadingClick("infrence")}>Infrence Engine</strong></p>
            <p>Applies logical rules to the knowledge base to deduce conclusions from the given facts (symptoms).</p>
          </li>
          <br />
          <li>
            <p><b>User Interface</b></p>
            <p>Allows users to interact with the system, input their symptoms, and receive diagnoses. 
              This is achieved through the web interface created using Flask (explained below).</p>
          </li>
        </ol> <br />

        <p><b>Web Devlopment With Flask</b></p> <br />
        <p><b>Defination:</b> Flask is a lightweight web framework for Python. It allows developers 
        to create web applications quickly and with minimal code.</p> <br />

        <p><b>Components Used in Code:</b></p> <br />
        <ol>
          <li>
            <p><strong  onClick={() => handleHeadingClick("flask")}>Flask Application:</strong></p>
            <ul>
              <li><p>The <b>Flask</b> object is initialized to create the application.</p></li>
              <li><p>Routes are defined using decorators to handle different URL endpoints.</p></li>
            </ul>
          </li>
          <br />
          <li>
            <p><strong onClick={() => handleHeadingClick("routes")}>Routes:</strong></p>
            <ul>
              <li><p><b>Home Route ('/'):</b> Handles both GET and POST requests.</p></li>
              <li><p><b>GET Request:</b> Displays a form where users can select symptoms.</p></li>
              <li><p><b>POST Request:</b> Processes the form data (selected symptoms), uses the expert system to diagnose, and displays the diagnosis.</p></li>
            </ul>
          </li>
          <br />
          <li>
            <p><strong onClick={() => handleHeadingClick("html")}>HTML Template:</strong></p>
            <ul>
              <li><p>Rendered using <u><b>render_template_string</b></u>.</p></li>
              <li><p>Contains forms, checkboxes for symptoms, and buttons for submission.</p></li>
              <li>Displays the diagnosis result or the form based on the request type.</li>
            </ul>
          </li>
          <br />
          <li>
            <p><strong onClick={() => handleHeadingClick("run")}>Running the Application:</strong></p>
            <ul><li><p>Runs the Flask application in debug mode.</p></li></ul>
          </li>
        </ol> <br />
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
            <button className="button" style={{marginRight: "20px"}}>
            <a href="https://www.kaggle.com/code/priyansh2904/unit4-lab1?scriptVersionId=188475084" target="_blank"> View Runable code</a>
            </button>
            <button className="button" style={{padding: "10px"}}>
            <a href="/Unit4Lab1.py" download={"Unit4Lab1.py"}> Downlaod code</a>
            </button>
    </div>
    </div>
    );
};
export default Lab2;
