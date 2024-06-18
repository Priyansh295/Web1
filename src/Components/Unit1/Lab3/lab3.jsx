import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.gif';
import Img2 from './imgs/image2.gif';
import Img3 from './imgs/image3.jpg';
import Img4 from './imgs/image4.png';
import Img5 from './imgs/image5.png';
import './lab3.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `
import cv2
import numpy as np

# Function to classify color
def classify_color(hsv_value):
    h, s, v = hsv_value
    if (0 <= h <= 10) or (160 <= h <= 180):
        return "Hot"  # Red
    elif 100 <= h <= 140:
        return "Cold"  # Blue
    elif 35 <= h <= 85:
        return "Natural"  # Green
    elif 20 <= h <= 30:
        return "Warm"  # Yellow
    else:
        return "Unknown"

# Capture image from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the histogram to find the most predominant color
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    predominant_color_hue = np.argmax(hist)

    # Classify the color
    classification = classify_color((predominant_color_hue, 255, 255))

    # Display the classification on the image
    cv2.putText(frame, f"Classification: {classification}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

`;
const example = `
# Define virtual objects as an array of HSV values
virtual_objects = [
(0, 255, 255), # Red
(120, 255, 255), # Blue
(60, 255, 255), # Green
(25, 255, 255), # Yellow
(90, 255, 255) # Cyan (not classified)
]
# Sort virtual objects based on classification rules
sorted_objects = sorted(virtual_objects, key=lambda x: classify_color(x))
# Print out the classifications
for obj in sorted_objects:
classification = classify_color(obj)
print(f"Object {obj} is classified as: {classification}")
`;

const codeSections = {
  step1: `
def classify_color(hsv_value):
    h, s, v = hsv_value
    if (0 <= h <= 10) or (160 <= h <= 180):
        return "Hot"  # Red
    elif 100 <= h <= 140:
        return "Cold"  # Blue
    elif 35 <= h <= 85:
        return "Natural"  # Green
    elif 20 <= h <= 30:
        return "Warm"  # Yellow
    else:
        return "Unknown"
`,
  color: `
# Capture image from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the histogram to find the most predominant color
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    predominant_color_hue = np.argmax(hist)

    # Classify the color
    classification = classify_color((predominant_color_hue, 255, 255))

    # Display the classification on the image
    cv2.putText(frame, f"Classification: {classification}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
`,
  hsv: `
# Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

`,
  step3: `
  cv2.putText(frame, f"Classification: {classification}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
  `
};

const Lab2 = () => {
  const [activeTab, setActiveTab] = useState('step1');
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
  <h2>Color Classification and Sorting Theory</h2> <br /> 
  <p><strong onClick={() => handleHeadingClick("step1")}>Step 1: Defining Color Classification Rules</strong></p> <br />
  <p>Red Objects: Classified as "Hot"</p>
  <p>Blue Objects: Classified as "Cold"</p>
  <p>Green Objects: Classified as "Natural"</p>
  <p>Yellow Objects: Classified as "Warm"</p>
  <p>Other Colors: Classified as "Unknown"</p> <br />
  <img style={{width: '100%'}} src={Img1} alt="image1" /> <br />
  <img style={{width: '100%'}} src={Img2} alt="image2" /> <br /> <br />

  <p><strong onClick={() => handleHeadingClick("color")}>Step 2: Using OpenCV to Capture Images and Detect Objects' Colors</strong></p> <br />
  <p>Use the provided Python code to capture an image from a webcam, detect the predominant color of an object in the image, and classify the object based on the predefined rules.</p>
  <p>Apply the rules to the detected objects and print out the classification.</p> <br />
  <img style={{width: '100%'}} src={Img5} alt="image5" /> <br /> <br />
  <img style={{width: '100%'}} src={Img4} alt="image4" /> <br /> <br />
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
          <a href="https://github.com/Priyansh295/Lab-Portal/blob/main/Lab3.ipynb" target="_blank"> View Runable code</a>
          </button>
        </div>
      </div>
  );
};
export default Lab2;
