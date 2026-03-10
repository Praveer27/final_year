import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import Home from './pages/Home';
import WebcamRecognition from './pages/WebcamRecognition';
import UploadImage from './pages/UploadImage';
import UploadVideo from './pages/UploadVideo';
import Dashboard from './pages/Dashboard';
import About from './pages/About';

function App() {
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => {
        setBackendStatus(data.status === 'healthy' ? 'connected' : 'error');
      })
      .catch(() => setBackendStatus('disconnected'));
  }, []);

  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="nav-container">
            <Link to="/" className="nav-logo">
              Gesture2Speech
            </Link>

            <ul className="nav-menu">
              <li className="nav-item"><Link to="/" className="nav-link">Home</Link></li>
              <li className="nav-item"><Link to="/webcam" className="nav-link">Webcam</Link></li>
              <li className="nav-item"><Link to="/upload-image" className="nav-link">Upload Image</Link></li>
              <li className="nav-item"><Link to="/upload-video" className="nav-link">Upload Video</Link></li>
              <li className="nav-item"><Link to="/dashboard" className="nav-link">Dashboard</Link></li>
              <li className="nav-item"><Link to="/about" className="nav-link">About</Link></li>
            </ul>

            <div className={`status-indicator ${backendStatus}`}>
              {backendStatus === 'connected' && 'Connected'}
              {backendStatus === 'disconnected' && 'Disconnected'}
              {backendStatus === 'checking' && 'Checking...'}
              {backendStatus === 'error' && 'Error'}
            </div>
          </div>
        </nav>

        <div className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/webcam" element={<WebcamRecognition />} />
            <Route path="/upload-image" element={<UploadImage />} />
            <Route path="/upload-video" element={<UploadVideo />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>

        <footer className="footer">
          <p>© 2024 Gesture2Speech - Indian Sign Language Recognition System</p>
          <p>Built for accessibility</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;