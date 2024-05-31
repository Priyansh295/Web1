import React, { useEffect, useState, useRef } from "react";
import './Dashboard.css';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css'; // Import a Highlight.js style
import ace from 'ace-builds';
import 'ace-builds/webpack-resolver'; // This allows ACE to find its modes and themes
// Register the Python language for Highlight.js
hljs.registerLanguage('python', python);

const codeSnippet = `
import cv2

# Attempt to use the DirectShow backend for capturing video on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

try:
    # Looping continuously to get frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Failed to capture image")
            break

        # Display the resulting frame
        cv2.imshow('Webcam Video', frame)

        # Break the loop on 'q' key press (waitKey returns a 32-bit integer)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
`;

const codeSnippet2 = `
import cv2
import numpy as np

def detect_shapes(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny
    edged = cv2.Canny(blurred, 30, 100)  # Adjusted thresholds for Canny
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) < 500:
            continue
        
        # Approximate the contour
        peri = cv2.arcLength(contour, True);
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True);

        # Determine the shape of the contour based on the number of vertices
        shape = "unidentified";
        if len(approx) == 3:
            shape = "Triangle";
        elif len(approx) == 4:
            shape = "Rectangle";
        elif len(approx) == 5:
            shape = "Pentagon";
        elif len(approx) == 6:
            shape = "Hexagon";
        else:
            shape = "Circle";

        # Draw the contour and the name of the shape on the image
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2);
        M = cv2.moments(contour);
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"]);
            cY = int(M["m01"] / M["m00"]);
            cv2.putText(frame, shape, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2);
        
    return frame;

def main():
    # Capture video from the webcam
    cap = cv2.VideoCapture(0);
    
    while True:
        ret, frame = cap.read();
        if not ret:
            break;
        
        # Detect and label shapes in the frame
        frame = detect_shapes(frame);
        
        # Display the frame
        cv2.imshow("Shape Detection", frame);
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    
    # Release the capture and close any OpenCV windows
    cap.release();
    cv2.destroyAllWindows();

if __name__ == "__main__":
    main();

`;

const Lab2 = () => {
  const [activeTab, setActiveTab] = useState('step1');

  useEffect(() => {
    hljs.highlightAll();

    // Initialize ACE Editor for the first box
    const editor1 = ace.edit("editor1");
    editor1.setTheme("ace/theme/twilight");
    editor1.session.setMode("ace/mode/python");
    editor1.setFontSize(16); // Set the font size for the first editor
    editor1.session.setValue(`#PART 1:
# Step 1: Open a terminal or command prompt.

# Step 2: Check if pip is installed
pip --version

# Step 3: Install OpenCV using pip
pip install opencv-python

# Step 4 (Optional): Install additional OpenCV modules
pip install opencv-python-headless
# or
pip install opencv-contrib-python

# Step 5: Verify the installation
python -c "import cv2; print(cv2._version_)"`);

    // Initialize ACE Editor for the second box
    const editor2 = ace.edit("editor2");
    editor2.setTheme("ace/theme/twilight");
    editor2.session.setMode("ace/mode/python");
    editor2.setFontSize(16); // Set the font size for the second editor
    editor2.session.setValue(codeSnippet); // Set the initial content for the second editor

     // Initialize ACE Editor for the third box
    const editor3 = ace.edit("editor3");
    editor3.setTheme("ace/theme/twilight");
    editor3.session.setMode("ace/mode/python");
    editor3.setFontSize(16); // Set the font size for the third editor
    editor3.session.setValue(codeSnippet2); // Set the initial content for the third editor

    // Ensure the editors resize properly
    editor1.resize();
    editor2.resize();
    editor3.resize();
  }, []);

  const tabs = [
    { id: 'step1', label: 'Step 1' },
    { id: 'step2', label: 'Step 2' },
    { id: 'step3', label: 'Step 3' } 
  ];

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
          ctx.fillStyle = '#000000'; // Black color particles
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
  
      // Initialize particles
      for (let i = 0; i < 120; i++) {
        particles.push(new Particle(Math.random() * canvas.width, Math.random() * canvas.height));
      }
  
      drawParticles();
  
      // Cleanup on unmount
      return () => {
        particles = [];
      };
    }, []);
  
    return <canvas ref={canvasRef} style={{ position: 'fixed', zIndex: -1, top: 0, left: 0, width: '100vw', height: '100vh' }} />;
  };
  
  return (
    <div className="dashboard">
      <ParticleCanvas />
      <div className="Layout" style={{ display: "flex", justifyContent: "space-around" }}>
        <div className="box3">
          <h2>Shape Detection Code Explained</h2>
          <p><strong>Imports:</strong></p>
          <p>cv2: This is the OpenCV library, which helps us with computer vision tasks like detecting shapes.</p>
          <p>numpy: A library used for handling arrays (lists of numbers).</p>
          <p><strong>Function: detect_shapes:</strong></p>
          <ul>
            <li><strong>Convert to Grayscale:</strong> Changes the frame from color to black-and-white using cv2.cvtColor.</li>
            <li><strong>Reduce Noise:</strong> Uses cv2.GaussianBlur to make the image smoother, which helps in detecting edges better.</li>
            <li><strong>Find Edges:</strong> Detects the edges in the image using cv2.Canny with two threshold values, 30 and 100.</li>
            <li><strong>Find Shapes:</strong> Uses cv2.findContours to find the outlines of shapes in the edged image.</li>
            <li><strong>Ignore Small Shapes:</strong> Skips very small shapes (area less than 500) to avoid detecting noise.</li>
            <li><strong>Approximate Shapes:</strong> Simplifies the outline of each shape to fewer points using cv2.approxPolyDP.</li>
            <li><strong>Identify Shape:</strong> Counts the points to identify the shape:
              <ul>
                <li>3 points: Triangle</li>
                <li>4 points: Rectangle</li>
                <li>5 points: Pentagon</li>
                <li>6 points: Hexagon</li>
                <li>More than 6 points: Circle</li>
              </ul>
            </li>
            <li><strong>Draw and Label Shapes:</strong> Draws the shape on the frame and writes the name of the shape next to it using cv2.putText.</li>
            <li><strong>Return Frame:</strong> Sends back the frame with the detected shapes and their labels.</li>
          </ul>
          <p><strong>Main Function: main:</strong></p>
          <ul>
            <li><strong>Start Webcam:</strong> Opens the webcam to capture video using cv2.VideoCapture(0).</li>
            <li><strong>Process Each Frame:</strong>
              <ul>
                <li>Continuously reads frames from the webcam.</li>
                <li>If no frame is read, it stops.</li>
              </ul>
            </li>
            <li><strong>Detect Shapes:</strong> Uses the detect_shapes function to find and label shapes in each frame.</li>
            <li><strong>Display Frame:</strong> Shows the processed frame in a window named "Shape Detection".</li>
            <li><strong>Exit on 'q' Key:</strong> Stops the program if the 'q' key is pressed.</li>
            <li><strong>Clean Up:</strong> Releases the webcam and closes all windows using cap.release() and cv2.destroyAllWindows().</li>
            <li><strong>Run the Program:</strong> Calls the main function to start the program when the script is run.</li>
          </ul>
        </div>
        <div className="box4">
          <div className="code-container">
            <pre className="code-snippet">
              <code className="python" style={{color:'#2f3130'}}>
                {codeSnippet2}
              </code>
            </pre>
          </div>
        </div>
      </div>
      <div className="tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="content">
        <div className={`tab-content ${activeTab === 'step1' ? 'active' : ''}`}>
          <div className="box1">
            <div id="editor1">
              {/* ACE editor for step 1 */}
            </div>
          </div>
        </div>
        <div className={`tab-content ${activeTab === 'step2' ? 'active' : ''}`}>
          <div className="box1">
            <div id="editor2">
              {/* ACE editor for step 2 */}
            </div>
          </div>
        </div>
        <div className={`tab-content ${activeTab === 'step3' ? 'active' : ''}`}>
          <div className="box1">
            <div id="editor3">
              {/* ACE editor for step 3 */}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Lab2;
