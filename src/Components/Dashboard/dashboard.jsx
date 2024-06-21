import React from 'react';
import './dashboard.css';
import './vscode-theme.css';
import Img1 from './imgs/mainpagelogo.png';

function Dashboard() {
  return (
    <div className="dashboard">
      <div className="image-container">
      <img style={{width: '80%'}} src={Img1} alt="image1" /> <br /> <br />
      </div>
    </div>
  );
}

export default Dashboard;
