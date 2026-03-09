import React from 'react';

function Home() {
  return (
    <div className="page-container">
      <h1 className="page-title">🤟 Gesture2Speech</h1>
      <p className="page-subtitle">Indian Sign Language Recognition System</p>
      
      <div className="card-grid">
        <div className="card">
          <h2>📹 Real-time Recognition</h2>
          <p>Use your webcam for instant gesture recognition with live feedback and confidence scores.</p>
          <a href="/webcam" className="btn btn-primary mt-2">Try Webcam</a>
        </div>
        
        <div className="card">
          <h2>📤 Upload & Predict</h2>
          <p>Upload images or videos of sign language gestures for accurate predictions.</p>
          <a href="/upload-image" className="btn btn-primary mt-2">Upload Image</a>
        </div>
        
        <div className="card">
          <h2>📊 Dashboard</h2>
          <p>View training metrics, model performance, and dataset statistics.</p>
          <a href="/dashboard" className="btn btn-primary mt-2">View Dashboard</a>
        </div>
      </div>

      <div className="mt-4" style={{background: 'white', padding: '2rem', borderRadius: '15px'}}>
        <h2>About This Project</h2>
        <p>Gesture2Speech is an advanced Indian Sign Language (ISL) recognition system that uses deep learning to translate sign language gestures into text and speech. Built with modern technologies including:</p>
        <ul style={{textAlign: 'left', maxWidth: '600px', margin: '1rem auto'}}>
          <li>✅ MediaPipe for hand landmark detection</li>
          <li>✅ PyTorch for deep learning models (LSTM, GRU, Transformer)</li>
          <li>✅ React for modern web interface</li>
          <li>✅ Flask REST API for backend services</li>
          <li>✅ Real-time gesture recognition</li>
        </ul>
        
        <div className="mt-3">
          <h3>Features</h3>
          <div className="card-grid mt-2">
            <div className="card">
              <h4>🎯 High Accuracy</h4>
              <p>94-96% accuracy with state-of-the-art models</p>
            </div>
            <div className="card">
              <h4>⚡ Real-time</h4>
              <p>30+ FPS for smooth recognition</p>
            </div>
            <div className="card">
              <h4>🌐 Web-based</h4>
              <p>Access from any device with a browser</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;

// Made with Bob
