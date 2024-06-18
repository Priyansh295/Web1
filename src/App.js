import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useHistory } from 'react-router-dom';
import './App.css';
import './Components/Dashboard/vscode-theme.css';
import Lab1 from './Components/Unit1/Lab1/lab1';
import Lab2 from './Components/Unit1/Lab2/lab2';
import Lab3 from './Components/Unit1/Lab3/lab3';
import Lab4 from './Components/Unit1/Lab4/lab4';
import Lab5 from './Components/Unit1/Lab5/lab5';
import Lab6 from './Components/Unit1/Lab6/lab6';
import Lab7 from './Components/Unit2/Lab1/lab7';
import Lab8 from './Components/Unit2/Lab2/lab8';
import Lab9 from './Components/Unit2/Lab3/lab9';
import Lab10 from './Components/Unit2/Lab4/lab10';
import Lab11 from './Components/Unit2/Lab5/lab11';
import Lab12 from './Components/Unit2/Lab6/lab12';
import Topbar from './Components/Topbar/Topbar';
import Sidebar from './Components/Sidebar/Sidebar';
import Dashboard from './Components/Dashboard/dashboard';
import Login from './Components/login/login';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const handleLogin = () => {
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
  };

  return (
    <Router>
      <div className="app-container">
        {isAuthenticated ? (
          <>
            <Sidebar username="Priyansh"  onLogout={handleLogout} />
            <div className="content-container">
              <Topbar/>
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/lab1" element={<Lab1 />} />
                <Route path="/lab2" element={<Lab2 />} />
                <Route path="/lab3" element={<Lab3 />} />
                <Route path="/lab4" element={<Lab4 />} />
                <Route path="/lab5" element={<Lab5 />} />
                <Route path="/lab6" element={<Lab6 />} />
                <Route path="/lab7" element={<Lab7 />} />
                <Route path="/lab8" element={<Lab8 />} />
                <Route path="/lab9" element={<Lab9 />} />
                <Route path="/lab10" element={<Lab10 />} />
                <Route path="/lab11" element={<Lab11 />} />
                <Route path="/lab12" element={<Lab12/>} />
              </Routes>
            </div>
          </>
        ) : (
          <Routes>
            <Route path="/login" element={<Login onLogin={handleLogin} />} />
            <Route path="*" element={<Navigate to="/login" />} />
          </Routes>
        )}
      </div>
    </Router>
  );
}

export default App;
