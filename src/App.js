import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import './App.css';
import Lab1 from './Components/Lab1/lab1';
import Lab2 from './Components/Lab2/lab2';
import Topbar from './Components/Topbar/Topbar';
import Sidebar from './Components/Sidebar/Sidebar';
import Dashboard from "./Components/Dashboard/dashboard";

function App() {
  return (
    <Router>
      <div className="app-container">
        <Sidebar username="Priyansh" />
        <div className="content-container">
          <Topbar />
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/lab1" element={<Lab1 />} />
            <Route path="/lab2" element={<Lab2 />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
