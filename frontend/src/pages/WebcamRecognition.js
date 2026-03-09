import React, { useState, useRef } from 'react';

function WebcamRecognition() {
  const [isActive, setIsActive] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const videoRef = useRef(null);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsActive(true);
      }
    } catch (err) {
      alert('Error accessing webcam: ' + err.message);
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      setIsActive(false);
    }
  };

  return (
    <div className="page-container">
      <h1 className="page-title">📹 Real-time Gesture Recognition</h1>
      <p className="page-subtitle">Use your webcam to recognize sign language gestures in real-time</p>

      <div className="webcam-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{
            width: '100%',
            borderRadius: '15px',
            background: '#000'
          }}
        />
      </div>

      <div className="flex-center gap-2 mt-3">
        {!isActive ? (
          <button onClick={startWebcam} className="btn btn-primary">
            Start Webcam
          </button>
        ) : (
          <button onClick={stopWebcam} className="btn btn-danger">
            Stop Webcam
          </button>
        )}
      </div>

      {prediction && (
        <div className="result-container">
          <h2>Prediction</h2>
          <div className="result-prediction">{prediction.gesture}</div>
          <div className="result-confidence">
            Confidence: {(prediction.confidence * 100).toFixed(1)}%
          </div>
        </div>
      )}

      <div className="mt-4" style={{background: 'white', padding: '1.5rem', borderRadius: '15px'}}>
        <h3>Instructions:</h3>
        <ol style={{textAlign: 'left', maxWidth: '600px', margin: '1rem auto'}}>
          <li>Click "Start Webcam" to begin</li>
          <li>Position your hand in front of the camera</li>
          <li>Make a sign language gesture</li>
          <li>The system will detect and predict the gesture</li>
          <li>Results will appear below the video</li>
        </ol>
        <p className="mt-2"><strong>Note:</strong> Make sure you have good lighting and your hand is clearly visible.</p>
      </div>
    </div>
  );
}

export default WebcamRecognition;

// Made with Bob
