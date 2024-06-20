import React, { useEffect, useState, useRef } from "react";
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github.css';
import 'ace-builds/webpack-resolver'; 
import Img1 from './imgs/image1.png';
import Img2 from './imgs/image2.png';
import Img3 from './imgs/image3.png';
import Img4 from './imgs/image4.png';
import Img5 from './imgs/image5.png';
import Img6 from './imgs/image6.png';
import Img7 from './imgs/image7.png';
import Img8 from './imgs/image8.png';
import Img9 from './imgs/image9.png';
import Img10 from './imgs/image10.png';
import Img11 from './imgs/image11.png';
import Img12 from './imgs/image12.png';
import Img13 from './imgs/image13.png';
import Img14 from './imgs/image14.png';
import Img15 from './imgs/image15.png';
import './lab10.css';

hljs.registerLanguage('python', python);

const codeSnippet2 = `
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_text
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from IPython.display import Image, display
from sklearn.model_selection import GridSearchCV

np.random.seed(42)  # For reproducibility
num_samples = 1000

# Generate random values for depth, rate, and precision using numpy's uniform distribution
depth = np.random.uniform(1, 100, num_samples)
rate = np.random.uniform(1, 1000, num_samples)
precision = np.random.uniform(0, 1, num_samples)

# Create a DataFrame with the generated data
data = pd.DataFrame({'depth': depth, 'precision': precision, 'rate': rate})

# Print the first few rows of the generated dataset
print("Generated Dataset:")
print(data.head())

# classification function to assign classes based on depth, precision, and rate
def classify(depth, precision, rate):
    if depth > 80 and precision > 0.80:
        return "very good"
    elif depth > 60 and precision > 0.60 and rate > 600:
        return "good"
    elif depth > 40 and precision > 0.40 and rate > 400:
        return "ok"
    elif depth > 20 and precision > 0.20 and rate > 200:
        return "bad"
    else:
        return "very bad"

# Assign 'classes' based on classification function
data['classes'] = data.apply(lambda row: classify(row['depth'], row['precision'], row['rate']), axis=1)

# Print dataset before splitting
print("Dataset before splitting:")
print(data.head())

# Step 2: Split Dataset into Features (X) and Target (y)
X = data[['depth', 'precision', 'rate']]
y = data['classes']

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Implement Pruning (Cost Complexity Pruning)
path = clf.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state = 42, ccp_alpha = ccp_alpha)
    clf.fit(X_train,y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train,y_train) for clf in clfs]
test_scores  = [clf.score(X_test, y_test) for clf in clfs]

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle="steps-post")
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.title('Accuracy vs alpha for training and testing sets')
plt.legend()
plt.show()

optimal_clf = clfs[np.argmax(test_scores)]

# Display the tree structure
plt.figure(figsize=(20, 10))
plot_tree(optimal_clf, filled=True, feature_names=['depth', 'precision', 'rate'], class_names=[str(i) for i in range(1, 6)])
plt.show()

# Print the decision rules
tree_rules = export_text(optimal_clf, feature_names=['depth', 'precision', 'rate'])
print(tree_rules)

# Evaluate the optimal model
y_pred = optimal_clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Generate test data
num_test_samples = 100
test_depth = np.random.uniform(1, 100, num_test_samples)
test_precision = np.random.uniform(0, 1, num_test_samples)
test_speed = np.random.uniform(10, 1000, num_test_samples)

test_data = pd.DataFrame({'depth': test_depth, 'precision': test_precision, 'rate': test_speed})

# Classify test data
test_data['actual_level'] = test_data.apply(lambda row: classify(row['depth'], row['precision'], row['rate']), axis=1)

# Predict using the trained decision tree model
test_X = test_data[['depth', 'precision', 'rate']]
test_data['predicted_level'] = optimal_clf.predict(test_X)

# Display test data with actual and predicted levels
print(test_data.head())


# Evaluate the model on test data
accuracy = accuracy_score(test_data['actual_level'], test_data['predicted_level'])
conf_matrix = confusion_matrix(test_data['actual_level'], test_data['predicted_level'])

print(f"Accuracy on test data: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

test_X = pd.DataFrame({'depth': [80],
                       'precision': [0.9],
                       'rate': [860.228603]})

# Predict using the trained decision tree model
predicted_level = optimal_clf.predict(test_X)

# Print the predicted level
print(f"Predicted Level: {predicted_level[0]}")

# 1. single hyper-parameter pre pruned tree
#Let us implement single hyper-parameter max_depth.
mean_scores = []
depth_range = range(1, 21)

for depth in depth_range:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    mean_scores.append(scores.mean())
    
plt.figure(figsize=(10, 6))
plt.plot(depth_range, mean_scores, marker='o')
plt.xlabel('max_depth')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Accuracy vs max_depth')
plt.show()

# Find the optimal max_depth
optimal_depth = depth_range[np.argmax(mean_scores)]
print(f'Optimal max_depth: {optimal_depth}')

# Train the decision tree with the optimal max_depth
spre_tree=DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
spre_tree.fit(X_train, y_train)

# Display the tree structure
plt.figure(figsize=(20,10))
plot_tree(spre_tree, filled=True, feature_names=['depth', 'precision', 'rate'], class_names=[str(i) for i in range(1, 6)])
plt.show()

spre_pruned_accuracy = spre_tree.score(X_test, y_test)
print(f"Simple Pre-pruned Decision Tree Accuracy: {spre_pruned_accuracy:.4f}")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Define parameters for Grid Search
parameters = {
    'criterion': ['entropy', 'gini'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 1, 2, 3, 4, 5],
    'max_features': ['sqrt', 'log2']
}

# Initialize the DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)

# Setup GridSearchCV
hpre_tree = GridSearchCV(clf, param_grid=parameters, cv=5)

# Fit GridSearchCV
hpre_tree.fit(X_train, y_train)

# Print best parameters and best score
print("Best Parameters found: ", hpre_tree.best_params_)
print("Best Score found: ", hpre_tree.best_score_)

# Use the best estimator found by GridSearchCV to make predictions
best_model = hpre_tree.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the best model if needed
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the best model: {accuracy:.4f}")

# Display the tree structure
plt.figure(figsize=(20,10))
plot_tree(hpre_tree.best_estimator_, filled=True, feature_names=['depth', 'precision', 'rate'], class_names=[str(i) for i in range(1, 6)])
plt.show()
`;

const codeSections = {
  Step2: `
np.random.seed(42) # For reproducibility
num_samples = 1000
# Generate random values for depth, rate, and precision using numpy's uni
form distribution
depth = np.random.uniform(1, 100, num_samples)
rate = np.random.uniform(1, 1000, num_samples)
precision = np.random.uniform(0, 1, num_samples)
# Create a DataFrame with the generated data
data = pd.DataFrame({'depth': depth, 'precision': precision, 'rate': ra
te})
# Print the first few rows of the generated dataset
print("Generated Dataset:")
print(data.head())


# classification function to assign classes based on depth, precision, an
d rate
def classify(depth, precision, rate):
  if depth > 80 and precision > 0.80:
  return "very good"
  elif depth > 60 and precision > 0.60 and rate > 600:
  return "good"
  elif depth > 40 and precision > 0.40 and rate > 400:
  return "ok"
  elif depth > 20 and precision > 0.20 and rate > 200:
  return "bad"
  else:
  return "very bad"
# Assign 'classes' based on classification function
data['classes'] = data.apply(lambda row: classify(row['depth'], row['pr
ecision'], row['rate']), axis=1)
# Print dataset before splitting
print("Dataset before splitting:")
print(data.head())
# Step 2: Split Dataset into Features (X) and Target (y)
X = data[['depth', 'precision', 'rate']]
y = data['classes']
# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3
, random_state=42)
`,
  Step4: `
train_scores = [clf.score(X_train,y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle
="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle="
steps-post")
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.title('Accuracy vs alpha for training and testing sets')
plt.legend()
plt.show()

optimal_clf = clfs[np.argmax(test_scores)]
# Display the tree structure
plt.figure(figsize=(20, 10))

plot_tree(optimal_clf, filled=True, feature_names=['depth', 'precision'
, 'rate'], class_names=[str(i) for i in range(1, 6)])
plt.show()

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
          <h2>Building a Supervised Learning Model to Predict Machining Quality based on Operational Parameters</h2> <br />
          <h3> <strong onClick={() => handleHeadingClick("Step1")}>Step-1: Understanding Decision Trees</strong></h3> <br />
          <p>A decision tree is a popular machine learning algorithm used for classification and regression tasks. It is a tree-like model of decisions and their possible consequences. Here’s a simple breakdown of how it works:</p> <br />
          <ul>
            <li><b>Root Node:</b> This is the topmost node in the tree, representing the entire dataset.</li>
            <li><b>Decision Nodes:</b> These are nodes where the dataset is split into different subsets based on certain conditions.</li>
            <li><b>Leaf Nodes:</b> These are terminal nodes that represent the final decision or output.</li>
          </ul> <br />
          <p>Each internal node of the tree represents a "test" or "decision" on an attribute (e.g., "Is depth {">"} 50?"), and each branch represents the outcome of that decision. The leaves represent the class labels or regression values.</p> <br />
          <img style={{width: '100%'}} src={Img1} alt="image1" /> <br /> <br />
          <img style={{width: '100%'}} src={Img2} alt="image2" /> <br /> <br />

          <h3> <strong onClick={() => handleHeadingClick("Step2")}>Step-2:Generating Dataset and handle the data</strong></h3>
          <p>Generated Dataset:</p>
          <p>depth precision rate</p>
          <ol style={{marginLeft: '25px'}}>
          <li>38.079472 0.261706 185.947796</li>  
          <li>95.120716 0.246979 542.359046</li>  
          <li>73.467400 0.906255 873.072890</li>  
          <li>60.267190 0.249546 732.492662</li>  
          <li>16.445845 0.271950 806.754587</li>  
          </ol> <br />

          <h3> <strong onClick={() => handleHeadingClick("Step3")}>Step-3:Building a Decision Tree Built?</strong></h3> <br />
          <p>When using scikit-learn to build a decision tree for classification, the algorithm splits the data at each node in a way that maximizes the "purity" of the resulting subsets. Gini impurity is one of the metrics used to measure this purity.</p> <br />
          <ol>
            <li><strong>Initialize the Root Node:</strong> The process starts with the entire dataset at the root node.</li>
            <li><strong>Calculate Gini Impurity:</strong> For each possible split, calculate the Gini impurity of the subsets resulting from the split.</li> <br />
            <img style={{width: '100%'}} src={Img3} alt="image3" /> <br /> <br />
            <li><strong>Evaluate Splits:</strong> Divide the dataset based on each feature and split value, calculating the weighted Gini impurity.</li>
            <li><strong>Choose the Best Split:</strong> The algorithm selects the split that results in the lowest weighted Gini impurity and uses this as the decision rule at the current node.</li>
            <li><strong>Repeat for Child Nodes:</strong> Continue recursively until a stopping criterion is met (maximum depth, minimum samples, node purity).</li>
            <li><strong>Final Tree Structure:</strong> The result is a tree structure where each internal node represents a decision based on a feature and a split value, and each leaf node represents a class label (for classification tasks).</li> <br />
            <img style={{width: '100%'}} src={Img4} alt="image4" /> <br /> <br />
            <img style={{width: '100%'}} src={Img5} alt="image5" /> <br /> <br />
            <img style={{width: '100%'}} src={Img6} alt="image6" /> <br /> <br />
          </ol> <br />

          <h3><strong onClick={() => handleHeadingClick("Step4")}>Step-4:Understanding Pruning?</strong></h3> <br />
          <p>Pruning in the context of decision trees refers to the process of reducing the size of the tree by removing specific parts of it. This technique aims to improve the tree's ability to generalize to new, unseen data while avoiding overfitting to the training data.</p> <br />
          <p>Pruning is necessary to prevent overfitting, where decision trees become overly complex and memorize noise or specifics of the training data, leading to poor performance on new data.
          We essentially prune by removing the nodes which have the least amount of information gain.
          </p> <br />
          <img style={{width: '100%'}} src={Img7} alt="image7" /> <br /> <br />
          <img style={{width: '100%'}} src={Img8} alt="image8" /> <br /> <br />

          <p> <b> Types of Pruning:</b></p> <br />
          <ul>
            <li><b>Cost Complexity Pruning (ccp): </b> Pruning in the context of decision trees refers to the process of reducing the size of the tree by removing specific parts of it. <br /> <br /> This technique aims to improve the tree's ability to generalize to new, unseen data while avoiding overfitting to the training data.
            Cost Complexity Pruning (ccp) balances tree complexity and training accuracy. Higher ccp_alpha values (𝛼) lead to more aggressive pruning, resulting in simpler trees with fewer nodes.</li> <br />
            <img style={{width: '100%'}} src={Img11} alt="image11" /> <br /> <br />
            <img style={{width: '100%'}} src={Img12} alt="image12" /> <br /> <br />
            <img style={{width: '100%'}} src={Img13} alt="image13" /> <br /> <br />
            <li><b>Pre-pruning:</b> This involves setting stopping criteria before the tree is fully grown. It stops splitting nodes when further splitting does not lead to an improvement in model accuracy or when certain conditions are met.</li> <br />
            <img style={{width: '100%'}} src={Img14} alt="image14" /> <br /> <br />
            <li><b>Post-Pruning (Reduced Error Pruning):</b> This technique involves growing the decision tree to its maximum size (fully grown) and then pruning back the nodes that do not provide significant improvements to the model's accuracy or validation performance.</li> <br />
            <img style={{width: '100%'}} src={Img15} alt="image15" /> <br /> <br />
          </ul>

          <h3><strong onClick={() => handleHeadingClick("Step5")}>Step-5:Reading and evaluating the tree</strong></h3> <br />
          <img style={{width: '100%'}} src={Img9} alt="image9" /> <br /> <br />
          <p>This is a representation of the decision rules the model has made. This allows us to easily see 
              how the algorithm make each decision. Its just a representation of a decision tree in a easier to 
              read format as decision trees can become very complicated to pictorially represent and read.</p>
          <p>How to read:</p>
          <ul>
            <li>Each line represents a decision node in the tree. It specifies a condition based on a 
            feature.</li>
            <li>Indentation indicates the level of the node in the tree. More indentation means the node 
            is deeper in the tree.</li>
            <li>The condition (e.g., depth {"<"}= 50) is evaluated for each data point. If true, follow the 
                branch; if false, go to the next condition.
                </li>
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
          <a href="https://www.kaggle.com/code/percival224/unit-3-lab-4/notebook" target="_blank"> View Runable code</a>
          </button>
        </div>
      </div>
  );
};
export default Lab2;
