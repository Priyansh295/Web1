  import React, { useEffect, useState, useRef } from "react";
  import hljs from 'highlight.js/lib/core';
  import python from 'highlight.js/lib/languages/python';
  import 'highlight.js/styles/github.css';
  import ace from 'ace-builds';
  import 'ace-builds/webpack-resolver'; 
  import './Dashboard.css';
  import grayScale1 from './imgs/grayScale1.png';
  import grayScale2 from './imgs/grayScale2.png';
  import gaussianBlur1 from './imgs/gaussianBlur1.png';
  import gaussianBlur2 from './imgs/gaussianBlur2.png';
  import gaussianBlur3 from './imgs/gaussianBlur3.png';
  import canny1 from './imgs/canny1.png';
  import countour1 from './imgs/contour1.png';
  import shapeDetection from './imgs/shapeDetection.png';

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

  const codeSections = {
    grayscale: `
  # Convert the frame to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  `,
    gaussianBlur: `
  # Apply GaussianBlur to reduce noise and improve contour detection
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  `,
    canny: `
  # Detect edges using Canny
  edged = cv2.Canny(blurred, 30, 100)  # Adjusted thresholds for Canny
  `,
    contourDetect: `
    # Find contours in the edged image
      contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      for contour in contours:
          # Filter out small contours
          if cv2.contourArea(contour) < 500:
              continue
          peri = cv2.arcLength(contour, True);
          approx = cv2.approxPolyDP(contour, 0.04 * peri, True);
    `,
    
    shape: `
  # Draw the contour and the name of the shape on the image
          cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2);
          M = cv2.moments(contour);
          if M["m00"] != 0:
              cX = int(M["m10"] / M["m00"]);
              cY = int(M["m01"] / M["m00"]);
              cv2.putText(frame, shape, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2);
  `
  };

  const Lab2 = () => {
    const [activeTab, setActiveTab] = useState('step1');
    const [highlightedCodeSnippet, setHighlightedCodeSnippet] = useState("");


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
          <h2>Shape Detection Theory Explained</h2>
          <p><strong onClick={() => handleHeadingClick("grayscale")}>Grayscale:</strong></p>
          <p>A grayscale image is one where the color information has been removed, leaving only shades of gray. Each pixel in a grayscale image represents an intensity value between 0 (black) and 255 (white).</p> <br />
          <p>Converting to grayscale simplifies the image data, reducing the complexity and computational load for further processing like edge detection and contour detection. Working with a single intensity channel instead of three color channels (Red, Green, Blue) makes these operations more efficient</p><br />
          <p>Achieved by taking a rough average of the R,G,B values</p> <br />
          <img style= {{ width: '100%'}} src={grayScale1} alt="grayScale1" />
          <img style= {{ width: '100%'}} src={grayScale2} alt="grayScale2" />
          
          <p><strong onClick={() => handleHeadingClick("gaussianBlur")}>Gaussian Blur:</strong></p>
          <p>Gaussian blur is a smoothing filter that uses a Gaussian function to calculate the transformation to apply to each pixel in the image. It reduces the noise and detail by averaging out the pixel values in a local neighborhood.</p> <br />
          <p>Applying a Gaussian blur helps to reduce noise and minor variations in the image, which can improve the accuracy of edge detection. By smoothing the image, it becomes easier to detect significant edges and contours, as the algorithm won't be misled by small, irrelevant details.</p> <br />
          <img style= {{ width: '100%'}} src={gaussianBlur1} alt="gaussianBlur1" /> <br />
          <p>Gaussian blur is simply a method of blurring an image through the use of a Gaussian function. Below, you’ll see a 2D Gaussian distribution. Notice that there is a peak in the center and the curve flattens out as you move towards the edges.</p> <br />
          <img style= {{ width: '100%'}} src={gaussianBlur2} alt="gaussianBlur2" /> <br />
          <p>Imagine that this distribution is superimposed over a group of pixels in an image. It should be apparent looking at this graph, that if we took a weighted average of the pixel’s values and the height of the curve at that point, the pixels in the center of the group would contribute most significantly to the resulting value. This is, in essence, how Gaussian blur works.</p> <br />
          <p><i>A Gaussian blur is applied by convolving the image with a Gaussian function. This concept of convolution will be explained more clearly in the upcoming lectures.</i></p> <br />
          <img style= {{ width: '100%'}} src={gaussianBlur3} alt="gaussianBlur3" /> <br />


          <p><strong onClick={() => handleHeadingClick("canny")}>Edge Detection (Canny):</strong></p> <br />
          <p>Edge detection is a technique used to identify points in an image where the brightness changes sharply, indicating the presence of edges. The Canny edge detection algorithm is a popular method that involves several steps, including noise reduction, gradient calculation, non-maximum suppression, and edge tracking by hysteresis.</p> <br />
          <p>Edges represent the boundaries of objects within an image. By detecting edges, the algorithm can identify and isolate the shapes present in the image. The Canny algorithm is effective because it can detect a wide range of edges in images.</p> <br />
          <img style= {{ width: '100%'}} src={canny1} alt="canny1" />

          <p><strong onClick={() => handleHeadingClick("contourDetect")}>Contour Detection and Approximation</strong></p>
          <p>Contours are useful for shape analysis and object detection and recognition. By finding contours, the algorithm can identify the outlines of shapes in the image, which is essential for further processing steps like shape approximation and classification. Uses edge detection as a precursor. The output is a list of contours, where each contour is a curve joining all the continuous points along a boundary with the same color or intensity.</p> <br />
          <p>Contour approximation involves simplifying the contour shape by reducing the number of points on its perimeter while maintaining its overall structure. </p> <br />
          <p>Approximating contours helps in identifying geometric shapes (like triangles, rectangles, pentagons) more easily by reducing noise and insignificant details. It makes it easier to classify the shape based on the number of vertices.</p> <br />
          <img style= {{ width: '100%'}} src={countour1} alt="countour1" />

          <p><strong onClick={() => handleHeadingClick("shape")}>Shape Identification:</strong></p> <br />
          <p>Shape identification involves determining the type of shape based on the characteristics of its contour. In this code, the shape is identified by counting the number of vertices in the approximated contour.</p> <br />
          <p>
            <ol>
              <li>3 vertices: Triangle</li>
              <li>4 vertices: Rectangle</li>
              <li>5 vertices: Pentagon</li>
              <li>6 vertices: Hexagon</li>
            </ol>
          </p> <br />
          <img style= {{ width: '100%'}} src={shapeDetection} alt="shapeDetection" />

          <p><strong>Drawing and Labeling:</strong></p> <br />
          <p>Drawing involves rendering the contours and labels on the original image. The cv2.drawContours function is used to draw the contours, and cv2.putText is used to add text labels.</p> <br />
          <p>Drawing and labeling provide visual feedback on the detected shapes, making it easier to verify the accuracy of the shape detection algorithm. It enhances the interpretability of the results by marking the identified shapes directly on the video feed.</p>
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
            <a href="https://www.kaggle.com/code/pushkarns/lab-2" target="_blank"> View Runable code</a>
            </button>
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
