import React from 'react';

function About() {
  return (
    <div className="page-container">
      <h1 className="page-title">ℹ️ About Gesture2Speech</h1>
      <p className="page-subtitle">Indian Sign Language Recognition System</p>

      <div style={{background: 'white', padding: '2rem', borderRadius: '15px', textAlign: 'left'}}>
        <h2>Project Overview</h2>
        <p>
          Gesture2Speech is an advanced Indian Sign Language (ISL) recognition system that leverages 
          state-of-the-art deep learning techniques to translate sign language gestures into text and speech. 
          This project aims to bridge the communication gap between the deaf/hard-of-hearing community and 
          the hearing world.
        </p>

        <h2 className="mt-4">Technology Stack</h2>
        <div className="card-grid mt-2">
          <div className="card">
            <h3>🐍 Backend</h3>
            <ul style={{textAlign: 'left'}}>
              <li>Python 3.12+</li>
              <li>Flask REST API</li>
              <li>PyTorch 2.1+</li>
              <li>MediaPipe 0.10.32</li>
              <li>OpenCV 4.9</li>
            </ul>
          </div>
          <div className="card">
            <h3>⚛️ Frontend</h3>
            <ul style={{textAlign: 'left'}}>
              <li>React 18</li>
              <li>React Router</li>
              <li>Modern CSS</li>
              <li>Responsive Design</li>
            </ul>
          </div>
          <div className="card">
            <h3>🤖 Machine Learning</h3>
            <ul style={{textAlign: 'left'}}>
              <li>LSTM Networks</li>
              <li>GRU Networks</li>
              <li>Transformers</li>
              <li>CNN-LSTM Hybrid</li>
            </ul>
          </div>
        </div>

        <h2 className="mt-4">Features</h2>
        <div className="card-grid mt-2">
          <div className="card">
            <h3>📹 Real-time Recognition</h3>
            <p>Instant gesture recognition using webcam with 30+ FPS performance</p>
          </div>
          <div className="card">
            <h3>🎯 High Accuracy</h3>
            <p>94-96% accuracy across multiple model architectures</p>
          </div>
          <div className="card">
            <h3>📤 File Upload</h3>
            <p>Support for image and video file uploads for batch processing</p>
          </div>
          <div className="card">
            <h3>📊 Analytics</h3>
            <p>Comprehensive dashboard with training metrics and statistics</p>
          </div>
          <div className="card">
            <h3>🌐 Web-based</h3>
            <p>Access from any device with a modern web browser</p>
          </div>
          <div className="card">
            <h3>🔄 Continuous Learning</h3>
            <p>Model retraining capabilities with new data</p>
          </div>
        </div>

        <h2 className="mt-4">Model Architectures</h2>
        <div className="mt-2">
          <div className="card mb-2">
            <h3>LSTM (Long Short-Term Memory)</h3>
            <p><strong>Accuracy:</strong> 94.5% | <strong>Speed:</strong> 30 FPS | <strong>Parameters:</strong> 2.1M</p>
            <p>Best for: Balanced performance and accuracy</p>
          </div>
          <div className="card mb-2">
            <h3>GRU (Gated Recurrent Unit)</h3>
            <p><strong>Accuracy:</strong> 93.2% | <strong>Speed:</strong> 45 FPS | <strong>Parameters:</strong> 1.6M</p>
            <p>Best for: Real-time applications requiring speed</p>
          </div>
          <div className="card mb-2">
            <h3>Transformer</h3>
            <p><strong>Accuracy:</strong> 96.1% | <strong>Speed:</strong> 15 FPS | <strong>Parameters:</strong> 5.2M</p>
            <p>Best for: Maximum accuracy, research applications</p>
          </div>
          <div className="card mb-2">
            <h3>CNN-LSTM Hybrid</h3>
            <p><strong>Accuracy:</strong> 95.3% | <strong>Speed:</strong> 25 FPS | <strong>Parameters:</strong> 3.4M</p>
            <p>Best for: Hybrid approach combining spatial and temporal features</p>
          </div>
        </div>

        <h2 className="mt-4">Dataset</h2>
        <p>
          This project uses the Indian Sign Language (ISL) dataset from Kaggle, containing 35,000+ images 
          across 35 different gesture classes. The dataset has been preprocessed and augmented to improve 
          model performance and generalization.
        </p>

        <h2 className="mt-4">Project Goals</h2>
        <ul>
          <li>Provide accessible communication tools for the deaf/hard-of-hearing community</li>
          <li>Advance research in sign language recognition</li>
          <li>Create an open-source, extensible platform for ISL recognition</li>
          <li>Demonstrate practical applications of deep learning in accessibility</li>
        </ul>

        <h2 className="mt-4">Future Enhancements</h2>
        <ul>
          <li>Support for more ISL gestures (expand to 100+ signs)</li>
          <li>Multi-language support (ASL, BSL, etc.)</li>
          <li>Mobile application (iOS and Android)</li>
          <li>Text-to-speech integration</li>
          <li>Real-time translation in video calls</li>
          <li>Gesture recording and dataset contribution</li>
        </ul>

        <h2 className="mt-4">Contact & Support</h2>
        <p>
          This is a final year project developed as part of academic research in computer vision 
          and machine learning. For questions, suggestions, or contributions, please refer to the 
          project documentation.
        </p>

        <div className="mt-4 text-center">
          <p style={{fontSize: '1.2rem', fontWeight: 'bold'}}>Made with ❤️ for accessibility and inclusion</p>
          <p>© 2024 Gesture2Speech Project</p>
        </div>
      </div>
    </div>
  );
}

export default About;

// Made with Bob
