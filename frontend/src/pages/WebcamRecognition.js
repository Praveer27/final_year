import React, { useEffect, useRef, useState } from 'react';

function WebcamRecognition() {
  const [isActive, setIsActive] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  const startWebcam = async () => {
    setError(null);
    setPrediction(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsActive(true);
      }
    } catch (err) {
      setError('Error accessing webcam: ' + err.message);
    }
  };

  const stopWebcam = () => {
    // stop polling
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // stop camera tracks
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    setIsActive(false);
  };

  // Capture a frame and send to backend
  const sendFrameForPrediction = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;

    // wait until video has dimensions
    if (!video.videoWidth || !video.videoHeight) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // keep a reasonable resolution (faster + better than tiny)
    const targetW = 640;
    const scale = targetW / video.videoWidth;
    const targetH = Math.round(video.videoHeight * scale);

    canvas.width = targetW;
    canvas.height = targetH;

    ctx.drawImage(video, 0, 0, targetW, targetH);

    // convert to JPEG blob
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', 0.85)
    );

    if (!blob) return;

    const formData = new FormData();
    // IMPORTANT: backend expects request.files["image"]
    formData.append('image', blob, 'frame.jpg');

    try {
      const res = await fetch('/api/predict/image', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data?.error || 'Prediction request failed');
        return;
      }

      // Backend returns: { prediction, confidence, hands_detected }
      setPrediction({
        gesture: data.prediction,
        confidence: data.confidence ?? 0,
        handsDetected: data.hands_detected ?? 0,
      });
      setError(null);
    } catch (e) {
      setError('Failed to reach backend: ' + e.message);
    }
  };

  // Start/stop polling when webcam active changes
  useEffect(() => {
    if (!isActive) return;

    // send a frame every 800ms (adjust for speed)
    intervalRef.current = setInterval(sendFrameForPrediction, 800);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

  return (
    <div className="page-container">
      <h1 className="page-title">Real-time Gesture Recognition</h1>
      <p className="page-subtitle">
        Use your webcam to recognize sign language gestures in real-time
      </p>

      <div className="webcam-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            width: '100%',
            borderRadius: '15px',
            background: '#000',
          }}
        />
        {/* hidden canvas used for capturing frames */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
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

      {error && (
        <div className="result-container" style={{ borderColor: '#d9534f' }}>
          <h2>Error</h2>
          <div style={{ color: '#d9534f' }}>{error}</div>
        </div>
      )}

      {prediction && (
        <div className="result-container">
          <h2>Prediction</h2>
          <div className="result-prediction">{prediction.gesture}</div>
          <div className="result-confidence">
            Confidence: {(prediction.confidence * 100).toFixed(1)}% <br />
            Hands detected: {prediction.handsDetected}
          </div>
        </div>
      )}

      <div
        className="mt-4"
        style={{ background: 'white', padding: '1.5rem', borderRadius: '15px' }}
      >
        <h3>Instructions:</h3>
        <ol style={{ textAlign: 'left', maxWidth: '600px', margin: '1rem auto' }}>
          <li>Click "Start Webcam" to begin</li>
          <li>Position your hand in front of the camera</li>
          <li>Make a sign language gesture</li>
          <li>The system will detect and predict the gesture</li>
          <li>Results will appear below the video</li>
        </ol>
        <p className="mt-2">
          <strong>Note:</strong> Good lighting and a close-up of the hand improves detection.
        </p>
      </div>
    </div>
  );
}

export default WebcamRecognition;